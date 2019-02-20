// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"golang.org/x/tools/internal/span"
)

func (s *server) cacheAndDiagnose(ctx context.Context, uri span.URI, content string) error {
	if err := s.setContent(ctx, uri, []byte(content)); err != nil {
		return err
	}
	go func() {
		ctx := s.view.BackgroundContext()
		if ctx.Err() != nil {
			return
		}
		reports, err := source.Diagnostics(ctx, s.view, uri)
		if err != nil {
			return // handle error?
		}
		for uri, diagnostics := range reports {
			s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				Diagnostics: toProtocolDiagnostics(ctx, s.view, diagnostics),
				URI:         protocol.NewURI(uri),
			})
		}
	}()
	return nil
}

func (s *server) setContent(ctx context.Context, uri span.URI, content []byte) error {
	return s.view.SetContent(ctx, uri, content)
}

func toProtocolDiagnostics(ctx context.Context, v source.View, diagnostics []source.Diagnostic) []protocol.Diagnostic {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		_, m, err := newColumnMap(ctx, v, diag.Span.URI)
		if err != nil {
			//TODO: if we can't find the file we cannot map
			//the diagnostic, but also this should never happen
			continue
		}
		src := diag.Source
		if src == "" {
			src = "LSP"
		}
		var severity protocol.DiagnosticSeverity
		switch diag.Severity {
		case source.SeverityError:
			severity = protocol.SeverityError
		case source.SeverityWarning:
			severity = protocol.SeverityWarning
		}
		reports = append(reports, protocol.Diagnostic{
			Message:  diag.Message,
			Range:    m.Range(diag.Span),
			Severity: severity,
			Source:   src,
		})
	}
	return reports
}

func sorted(d []protocol.Diagnostic) {
	sort.Slice(d, func(i int, j int) bool {
		if d[i].Range.Start.Line == d[j].Range.Start.Line {
			if d[i].Range.Start.Character == d[j].Range.Start.Character {
				return d[i].Message < d[j].Message
			}
			return d[i].Range.Start.Character < d[j].Range.Start.Character
		}
		return d[i].Range.Start.Line < d[j].Range.Start.Line
	})
}
