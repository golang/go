// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/lsp/telemetry"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
	"golang.org/x/tools/internal/telemetry/trace"
)

func (s *Server) diagnoseSnapshot(snapshot source.Snapshot, cphs []source.CheckPackageHandle) {
	for _, cph := range cphs {
		if len(cph.CompiledGoFiles()) == 0 {
			continue
		}
		f := cph.CompiledGoFiles()[0]

		// Run diagnostics on the workspace package.
		go func(snapshot source.Snapshot, uri span.URI) {
			s.diagnostics(snapshot, uri)
		}(snapshot, f.File().Identity().URI)
	}
}

func (s *Server) diagnostics(snapshot source.Snapshot, uri span.URI) error {
	ctx := snapshot.View().BackgroundContext()
	ctx, done := trace.StartSpan(ctx, "lsp:background-worker")
	defer done()

	ctx = telemetry.File.With(ctx, uri)

	f, err := snapshot.View().GetFile(ctx, uri)
	if err != nil {
		return err
	}
	reports, warningMsg, err := source.Diagnostics(ctx, snapshot, f, snapshot.View().Options().DisabledAnalyses)
	if err != nil {
		return err
	}
	if warningMsg != "" {
		s.client.ShowMessage(ctx, &protocol.ShowMessageParams{
			Type:    protocol.Info,
			Message: warningMsg,
		})
	}

	s.undeliveredMu.Lock()
	defer s.undeliveredMu.Unlock()

	for fileID, diagnostics := range reports {
		if err := s.publishDiagnostics(ctx, fileID, diagnostics); err != nil {
			if s.undelivered == nil {
				s.undelivered = make(map[source.FileIdentity][]source.Diagnostic)
			}
			s.undelivered[fileID] = diagnostics

			log.Error(ctx, "failed to deliver diagnostic (will retry)", err, telemetry.File)
			continue
		}
		// In case we had old, undelivered diagnostics.
		delete(s.undelivered, fileID)
	}
	// Anytime we compute diagnostics, make sure to also send along any
	// undelivered ones (only for remaining URIs).
	for uri, diagnostics := range s.undelivered {
		if err := s.publishDiagnostics(ctx, uri, diagnostics); err != nil {
			log.Error(ctx, "failed to deliver diagnostic for (will not retry)", err, telemetry.File)
		}

		// If we fail to deliver the same diagnostics twice, just give up.
		delete(s.undelivered, uri)
	}
	return nil
}

func (s *Server) publishDiagnostics(ctx context.Context, fileID source.FileIdentity, diagnostics []source.Diagnostic) error {
	s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
		Diagnostics: toProtocolDiagnostics(ctx, diagnostics),
		URI:         protocol.NewURI(fileID.URI),
		Version:     fileID.Version,
	})
	return nil
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
