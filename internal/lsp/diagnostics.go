// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lsp

import (
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
