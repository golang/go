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
)

func (s *Server) diagnoseSnapshot(ctx context.Context, snapshot source.Snapshot) {
	ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
	defer done()

	wsPackages, err := snapshot.View().WorkspacePackageIDs(ctx)
	if err != nil {
		log.Error(ctx, "diagnoseSnapshot: no workspace packages", err, telemetry.Directory.Of(snapshot.View().Folder))
		return
	}
	for _, id := range wsPackages {
		go func(id string) {
			ph, err := snapshot.PackageHandle(ctx, id)
			if err != nil {
				log.Error(ctx, "diagnoseSnapshot: no PackageHandle for workspace package", err, telemetry.Package.Of(id))
				return
			}
			reports, _, err := source.PackageDiagnostics(ctx, snapshot, ph, false, snapshot.View().Options().DisabledAnalyses)
			if err != nil {
				log.Error(ctx, "diagnoseSnapshot: no diagnostics", err, telemetry.Package.Of(ph.ID()))
				return
			}
			// Don't publish empty diagnostics.
			s.publishReports(ctx, reports, false)
		}(id)
	}
	// Run diagnostics on the go.mod file.
	s.diagnoseModfile(snapshot)
}

func (s *Server) diagnoseFile(snapshot source.Snapshot, fh source.FileHandle) {
	ctx := snapshot.View().BackgroundContext()
	ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
	defer done()

	ctx = telemetry.File.With(ctx, fh.Identity().URI)

	reports, warningMsg, err := source.FileDiagnostics(ctx, snapshot, fh, true, snapshot.View().Options().DisabledAnalyses)
	// Check the warning message first.
	if warningMsg != "" {
		s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Info,
			Message: warningMsg,
		})
	}
	if err != nil {
		if err != context.Canceled {
			log.Error(ctx, "diagnoseFile: could not generate diagnostics", err)
		}
		return
	}
	// Publish empty diagnostics for files.
	s.publishReports(ctx, reports, true)
}

func (s *Server) diagnoseModfile(snapshot source.Snapshot) {
	ctx := snapshot.View().BackgroundContext()
	ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
	defer done()

	f, diags, err := mod.Diagnostics(ctx, snapshot)
	if err != nil {
		if err != context.Canceled {
			log.Error(ctx, "diagnoseModfile: could not generate diagnostics", err)
		}
		return
	}
	reports := map[source.FileIdentity][]source.Diagnostic{f: diags}
	// Publish empty diagnostics for files.
	s.publishReports(ctx, reports, true)
}

func (s *Server) publishReports(ctx context.Context, reports map[source.FileIdentity][]source.Diagnostic, publishEmpty bool) {
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
			version:    fileID.Version,
			identifier: fileID.Identifier,
			sorted:     diagnostics,
		}
		delivered, ok := s.delivered[fileID.URI]
		// If diagnostics are empty and not previously delivered,
		// only send them if we are publishing empty diagnostics.
		if !ok && len(diagnostics) == 0 && !publishEmpty {
			// Update the delivered map to cache the diagnostics.
			s.delivered[fileID.URI] = toSend
			continue
		}
		// Reuse equivalent cached diagnostics for subsequent file versions (if known),
		// or identical files (if versions are not known).
		if ok {
			// If the file is open, and we've already delivered diagnostics for
			// a later version, do nothing. This only works for open files,
			// since their contents in the editor are the source of truth.
			if s.session.IsOpen(fileID.URI) && fileID.Version < delivered.version {
				continue
			}
			geqVersion := fileID.Version >= delivered.version && delivered.version > 0
			noVersions := (fileID.Version == 0 && delivered.version == 0) && delivered.identifier == fileID.Identifier
			if (geqVersion || noVersions) && equalDiagnostics(delivered.sorted, diagnostics) {
				// Update the delivered map even if we reuse cached diagnostics.
				s.delivered[fileID.URI] = toSend
				continue
			}
		}

		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			Diagnostics: toProtocolDiagnostics(ctx, diagnostics),
			URI:         protocol.NewURI(fileID.URI),
			Version:     fileID.Version,
		}); err != nil {
			log.Error(ctx, "failed to deliver diagnostic", err, telemetry.File)
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

func toProtocolDiagnostics(ctx context.Context, diagnostics []source.Diagnostic) []protocol.Diagnostic {
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
