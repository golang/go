// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hooks

import (
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/internal/lsp/source"
	"honnef.co/go/tools/simple"
	"honnef.co/go/tools/staticcheck"
	"honnef.co/go/tools/stylecheck"
)

func updateAnalyzers(options *source.Options) {
	var analyzers []*analysis.Analyzer
	for _, a := range simple.Analyzers {
		analyzers = append(analyzers, a)
	}
	for _, a := range staticcheck.Analyzers {
		switch a.Name {
		case "SA5009":
			// This check conflicts with the vet printf check (golang/go#34494).
		case "SA5011":
			// This check relies on facts from dependencies, which
			// we don't currently compute.
		default:
			analyzers = append(analyzers, a)
		}
	}
	for _, a := range stylecheck.Analyzers {
		analyzers = append(analyzers, a)
	}
	// Always add hooks for all available analyzers, but disable them if the
	// user does not have staticcheck enabled (they may enable it later on).
	for _, a := range analyzers {
		options.AddStaticcheckAnalyzer(a)
	}
}
