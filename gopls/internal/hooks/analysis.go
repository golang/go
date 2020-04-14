// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package hooks

import (
	"golang.org/x/tools/internal/lsp/source"
	"honnef.co/go/tools/simple"
	"honnef.co/go/tools/staticcheck"
	"honnef.co/go/tools/stylecheck"
)

func updateAnalyzers(options *source.Options) {
	if options.StaticCheck {
		for _, a := range simple.Analyzers {
			options.AddDefaultAnalyzer(a)
		}
		for _, a := range staticcheck.Analyzers {
			switch a.Name {
			case "SA5009":
				// This check conflicts with the vet printf check (golang/go#34494).
			case "SA5011":
				// This check relies on facts from dependencies, which
				// we don't currently compute.
			default:
				options.AddDefaultAnalyzer(a)
			}
		}
		for _, a := range stylecheck.Analyzers {
			options.AddDefaultAnalyzer(a)
		}
	}
}
