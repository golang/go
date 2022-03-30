// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.17
// +build go1.17

package hooks

import (
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/source"
	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/quickfix"
	"honnef.co/go/tools/simple"
	"honnef.co/go/tools/staticcheck"
	"honnef.co/go/tools/stylecheck"
)

func updateAnalyzers(options *source.Options) {
	options.StaticcheckSupported = true

	mapSeverity := func(severity lint.Severity) protocol.DiagnosticSeverity {
		switch severity {
		case lint.SeverityError:
			return protocol.SeverityError
		case lint.SeverityDeprecated:
			// TODO(dh): in LSP, deprecated is a tag, not a severity.
			//   We'll want to support this once we enable SA5011.
			return protocol.SeverityWarning
		case lint.SeverityWarning:
			return protocol.SeverityWarning
		case lint.SeverityInfo:
			return protocol.SeverityInformation
		case lint.SeverityHint:
			return protocol.SeverityHint
		default:
			return protocol.SeverityWarning
		}
	}
	add := func(analyzers []*lint.Analyzer, skip map[string]struct{}) {
		for _, a := range analyzers {
			if _, ok := skip[a.Analyzer.Name]; ok {
				continue
			}

			enabled := !a.Doc.NonDefault
			options.AddStaticcheckAnalyzer(a.Analyzer, enabled, mapSeverity(a.Doc.Severity))
		}
	}

	add(simple.Analyzers, nil)
	add(staticcheck.Analyzers, map[string]struct{}{
		// This check conflicts with the vet printf check (golang/go#34494).
		"SA5009": {},
		// This check relies on facts from dependencies, which
		// we don't currently compute.
		"SA5011": {},
	})
	add(stylecheck.Analyzers, nil)
	add(quickfix.Analyzers, nil)
}
