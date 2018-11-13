// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
	"sort"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
)

func toProtocolDiagnostics(v *source.View, diagnostics []source.Diagnostic) []protocol.Diagnostic {
	reports := []protocol.Diagnostic{}
	for _, diag := range diagnostics {
		tok := v.Config.Fset.File(diag.Range.Start)
		reports = append(reports, protocol.Diagnostic{
			Message:  diag.Message,
			Range:    toProtocolRange(tok, diag.Range),
			Severity: toProtocolSeverity(diag.Severity),
			Source:   "LSP",
		})
	}
	return reports
}

func toProtocolSeverity(severity source.DiagnosticSeverity) protocol.DiagnosticSeverity {
	switch severity {
	case source.SeverityError:
		return protocol.SeverityError
	case source.SeverityWarning:
		return protocol.SeverityWarning
	case source.SeverityHint:
		return protocol.SeverityHint
	case source.SeverityInformation:
		return protocol.SeverityInformation
	}
	return protocol.SeverityError // default
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
