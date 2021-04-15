// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.15
// +build go1.15

package hooks

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/source"
	"honnef.co/go/tools/analysis/lint"
	"honnef.co/go/tools/simple"
	"honnef.co/go/tools/staticcheck"
	"honnef.co/go/tools/stylecheck"
)

func updateAnalyzers(options *source.Options) {
	add := func(analyzers map[string]*analysis.Analyzer, docs map[string]*lint.Documentation, skip map[string]struct{}) {
		for check, a := range analyzers {
			if _, ok := skip[check]; ok {
				continue
			}

			enabled := !docs[check].NonDefault
			options.AddStaticcheckAnalyzer(a, enabled)
		}
	}

	add(simple.Analyzers, simple.Docs, nil)
	add(staticcheck.Analyzers, staticcheck.Docs, map[string]struct{}{
		// This check conflicts with the vet printf check (golang/go#34494).
		"SA5009": {},
		// This check relies on facts from dependencies, which
		// we don't currently compute.
		"SA5011": {},
	})
	add(stylecheck.Analyzers, stylecheck.Docs, nil)
}
