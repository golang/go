// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"strings"

	"golang.org/x/tools/internal/lsp/mod"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
	"golang.org/x/tools/internal/xcontext"
)

func (s *Server) diagnoseDetached(snapshot source.Snapshot) {
	ctx := snapshot.View().BackgroundContext()
	ctx = xcontext.Detach(ctx)

	s.diagnose(ctx, snapshot)
}

func (s *Server) diagnoseSnapshot(snapshot source.Snapshot) {
	ctx := snapshot.View().BackgroundContext()

	s.diagnose(ctx, snapshot)
}

// diagnose is a helper function for running diagnostics with a given context.
// Do not call it directly.
func (s *Server) diagnose(ctx context.Context, snapshot source.Snapshot) {
	ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
	defer done()

	ctx = telemetry.Snapshot.With(ctx, snapshot.ID())

	// Diagnose all of the packages in the workspace.
	go func() {
		wsPackages, err := snapshot.WorkspacePackages(ctx)
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			log.Error(ctx, "diagnose: no workspace packages", err, telemetry.Directory.Of(snapshot.View().Folder))
			return
		}
		for _, ph := range wsPackages {
			go func(ph source.PackageHandle) {
				// Only run analyses for packages with open files.
				var withAnalyses bool
				for _, fh := range ph.CompiledGoFiles() {
					if s.session.IsOpen(fh.File().Identity().URI) {
						withAnalyses = true
					}
				}
				reports, warn, err := source.Diagnostics(ctx, snapshot, ph, withAnalyses)
				// Check if might want to warn the user about their build configuration.
				if warn && !snapshot.View().ValidBuildConfiguration() {
					s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
						Type: protocol.Warning,
						// TODO(rstambler): We should really be able to point to a link on the website.
						Message: `You are neither in a module nor in your GOPATH. Please see https://github.com/golang/go/wiki/Modules for information on how to set up your Go project.`,
					})
				}
				if ctx.Err() != nil {
					return
				}
				if err != nil {
					log.Error(ctx, "diagnose: could not generate diagnostics for package", err, telemetry.Package.Of(ph.ID()))
					return
				}
				s.publishReports(ctx, snapshot, reports, withAnalyses)
			}(ph)
		}
	}()

	// Diagnose the go.mod file.
	go func() {
		reports, err := mod.Diagnostics(ctx, snapshot)
		if ctx.Err() != nil {
			return
		}
		if err != nil {
			log.Error(ctx, "diagnose: could not generate diagnostics for go.mod file", err)
			return
		}
		s.publishReports(ctx, snapshot, reports, false)
	}()
}

func (s *Server) publishReports(ctx context.Context, snapshot source.Snapshot, reports map[source.FileIdentity][]source.Diagnostic, withAnalysis bool) {
	// Check for context cancellation before publishing diagnostics.
	if ctx.Err() != nil {
		return
	}

	s.deliveredMu.Lock()
	defer s.deliveredMu.Unlock()

	for fileID, diagnostics := range reports {
		// Don't deliver diagnostics if the context has already been canceled.
		if ctx.Err() != nil {
			break
		}

		// Pre-sort diagnostics to avoid extra work when we compare them.
		source.SortDiagnostics(diagnostics)
		toSend := sentDiagnostics{
			version:      fileID.Version,
			identifier:   fileID.Identifier,
			sorted:       diagnostics,
			withAnalysis: withAnalysis,
			snapshotID:   snapshot.ID(),
		}

		// We use the zero values if this is an unknown file.
		delivered := s.delivered[fileID.URI]

		// Snapshot IDs are always increasing, so we use them instead of file
		// versions to create the correct order for diagnostics.

		// If we've already delivered diagnostics for a future snapshot for this file,
		// do not deliver them.
		if delivered.snapshotID > toSend.snapshotID {
			// Do not update the delivered map since it already contains newer diagnostics.
			continue
		}

		// Check if we should reuse the cached diagnostics.
		if equalDiagnostics(delivered.sorted, diagnostics) {
			// Make sure to update the delivered map.
			s.delivered[fileID.URI] = toSend
			continue
		}

		// If we've already delivered diagnostics for this file, at this
		// snapshot, with analyses, do not send diagnostics without analyses.
		if delivered.snapshotID == toSend.snapshotID && delivered.version == toSend.version &&
			delivered.withAnalysis && !toSend.withAnalysis {
			// Do not update the delivered map since it already contains better diagnostics.
			continue
		}

		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			Diagnostics: toProtocolDiagnostics(diagnostics),
			URI:         protocol.NewURI(fileID.URI),
			Version:     fileID.Version,
		}); err != nil {
			if ctx.Err() == nil {
				log.Error(ctx, "publishReports: failed to deliver diagnostic", err, telemetry.File)
			}
			continue
		}
		// Update the delivered map.
		s.delivered[fileID.URI] = toSend
	}
}

// equalDiagnostics returns true if the 2 lists of diagnostics are equal.
// It assumes that both a and b are already sorted.
func equalDiagnostics(a, b []source.Diagnostic) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if source.CompareDiagnostic(a[i], b[i]) != 0 {
			return false
		}
	}
	return true
}

func toProtocolDiagnostics(diagnostics []source.Diagnostic) []protocol.Diagnostic {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		related := make([]protocol.DiagnosticRelatedInformation, 0, len(diag.Related))
		for _, rel := range diag.Related {
			related = append(related, protocol.DiagnosticRelatedInformation{
				Location: protocol.Location{
					URI:   protocol.NewURI(rel.URI),
					Range: rel.Range,
				},
				Message: rel.Message,
			})
		}
		reports = append(reports, protocol.Diagnostic{
			Message:            strings.TrimSpace(diag.Message), // go list returns errors prefixed by newline
			Range:              diag.Range,
			Severity:           diag.Severity,
			Source:             diag.Source,
			Tags:               diag.Tags,
			RelatedInformation: related,
		})
	}
	return reports
}
