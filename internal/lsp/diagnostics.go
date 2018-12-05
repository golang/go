// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"context"
	"sort"

	"golang.org/x/tools/internal/lsp/cache"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func (s *server) CacheAndDiagnose(ctx context.Context, uri protocol.DocumentURI, text string) {
	f := s.view.GetFile(source.URI(uri))
	f.SetContent([]byte(text))

	go func() {
		reports, err := source.Diagnostics(ctx, f)
		if err != nil {
			return // handle error?
		}
		for filename, diagnostics := range reports {
			s.client.PublishDiagnostics(ctx, &protocol.PublishDiagnosticsParams{
				URI:         protocol.DocumentURI(source.ToURI(filename)),
				Diagnostics: toProtocolDiagnostics(s.view, diagnostics),
			})
		}
	}()
}

func toProtocolDiagnostics(v *cache.View, diagnostics []source.Diagnostic) []protocol.Diagnostic {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		f := v.GetFile(source.ToURI(diag.Filename))
		tok, err := f.GetToken()
		if err != nil {
			continue // handle error?
		}
		pos := fromTokenPosition(tok, diag.Position)
		if !pos.IsValid() {
			continue // handle error?
		}
		reports = append(reports, protocol.Diagnostic{
			Message: diag.Message,
			Range: toProtocolRange(tok, source.Range{
				Start: pos,
				End:   pos,
			}),
			Severity: protocol.SeverityError, // all diagnostics have error severity for now
			Source:   "LSP",
		})
	}
	return reports
}

func sorted(d []protocol.Diagnostic) {
	sort.Slice(d, func(i int, j int) bool {
		if d[i].Range.Start.Line == d[j].Range.Start.Line {
			if d[i].Range.Start.Character == d[j].Range.End.Character {
				return d[i].Message < d[j].Message
			}
			return d[i].Range.Start.Character < d[j].Range.End.Character
		}
		return d[i].Range.Start.Line < d[j].Range.Start.Line
	})
}
