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

	wsPackages, err := snapshot.WorkspacePackages(ctx)
	if err != nil {
		log.Error(ctx, "diagnoseSnapshot: no workspace packages", err, telemetry.Directory.Of(snapshot.View().Folder))
		return
	}
	for _, ph := range wsPackages {
		go func(ph source.PackageHandle) {
			reports, _, err := source.PackageDiagnostics(ctx, snapshot, ph, false, snapshot.View().Options().DisabledAnalyses)
			if err != nil {
				log.Error(ctx, "diagnoseSnapshot: no diagnostics", err, telemetry.Package.Of(ph.ID()))
				return
			}
			s.publishReports(ctx, snapshot, reports, false)
		}(ph)
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
	s.publishReports(ctx, snapshot, reports, true)
}

func (s *Server) diagnoseModfile(snapshot source.Snapshot) {
	ctx := snapshot.View().BackgroundContext()
	ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
	defer done()

	reports, err := mod.Diagnostics(ctx, snapshot)
	if err != nil {
		if err != context.Canceled {
			log.Error(ctx, "diagnoseModfile: could not generate diagnostics", err)
		}
		return
	}
	s.publishReports(ctx, snapshot, reports, false)
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

		// Reuse cached diagnostics and update the delivered map.
		if fileID.Version >= delivered.version && equalDiagnostics(delivered.sorted, diagnostics) {
			s.delivered[fileID.URI] = toSend
			continue
		}
		// If we've already delivered diagnostics with analyses for this file, for this snapshot,
		// at this version, do not send diagnostics without analyses.
		if delivered.snapshotID == toSend.snapshotID && delivered.version == toSend.version &&
			delivered.withAnalysis && !toSend.withAnalysis {
			continue
		}
		if err := s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
			Diagnostics: toProtocolDiagnostics(diagnostics),
			URI:         protocol.NewURI(fileID.URI),
			Version:     fileID.Version,
		}); err != nil {
			log.Error(ctx, "publishReports: failed to deliver diagnostic", err, telemetry.File)
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
