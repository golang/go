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
	"golang.org/x/tools/internal/lsp/telemetry/log"
	"golang.org/x/tools/internal/span"
)

func (s *Server) Diagnostics(ctx context.Context, view source.View, uri span.URI) {
	ctx = telemetry.File.With(ctx, uri)
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		log.Error(ctx, "no file", err, telemetry.File)
		return
	}
	// For non-Go files, don't return any diagnostics.
	gof, ok := f.(source.GoFile)
	if !ok {
		return
	}
	reports, err := source.Diagnostics(ctx, view, gof, s.disabledAnalyses)
	if err != nil {
		log.Error(ctx, "failed to compute diagnostics", err, telemetry.File)
		return
	}

	s.undeliveredMu.Lock()
	defer s.undeliveredMu.Unlock()

	for uri, diagnostics := range reports {
		if err := s.publishDiagnostics(ctx, view, uri, diagnostics); err != nil {
			if s.undelivered == nil {
				s.undelivered = make(map[span.URI][]source.Diagnostic)
			}
			log.Error(ctx, "failed to deliver diagnostic (will retry)", err, telemetry.File)
			s.undelivered[uri] = diagnostics
			continue
		}
		// In case we had old, undelivered diagnostics.
		delete(s.undelivered, uri)
	}
	// Anytime we compute diagnostics, make sure to also send along any
	// undelivered ones (only for remaining URIs).
	for uri, diagnostics := range s.undelivered {
		if err := s.publishDiagnostics(ctx, view, uri, diagnostics); err != nil {
			log.Error(ctx, "failed to deliver diagnostic for (will not retry)", err, telemetry.File)
		}
		// If we fail to deliver the same diagnostics twice, just give up.
		delete(s.undelivered, uri)
	}
}

func (s *Server) publishDiagnostics(ctx context.Context, view source.View, uri span.URI, diagnostics []source.Diagnostic) error {
	protocolDiagnostics, err := toProtocolDiagnostics(ctx, view, diagnostics)
	if err != nil {
		return err
	}
	s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
		Diagnostics: protocolDiagnostics,
		URI:         protocol.NewURI(uri),
	})
	return nil
}

func toProtocolDiagnostics(ctx context.Context, v source.View, diagnostics []source.Diagnostic) ([]protocol.Diagnostic, error) {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		diagnostic, err := toProtocolDiagnostic(ctx, v, diag)
		if err != nil {
			return nil, err
		}
		reports = append(reports, diagnostic)
	}
	return reports, nil
}

func toProtocolDiagnostic(ctx context.Context, v source.View, diag source.Diagnostic) (protocol.Diagnostic, error) {
	_, m, err := getSourceFile(ctx, v, diag.Span.URI())
	if err != nil {
		return protocol.Diagnostic{}, err
	}
	var severity protocol.DiagnosticSeverity
	switch diag.Severity {
	case source.SeverityError:
		severity = protocol.SeverityError
	case source.SeverityWarning:
		severity = protocol.SeverityWarning
	}
	rng, err := m.Range(diag.Span)
	if err != nil {
		return protocol.Diagnostic{}, err
	}
	return protocol.Diagnostic{
		Message:  strings.TrimSpace(diag.Message), // go list returns errors prefixed by newline
		Range:    rng,
		Severity: severity,
		Source:   diag.Source,
	}, nil
}
