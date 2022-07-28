// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/protocol"

	. "golang.org/x/tools/internal/lsp/regtest"
)

var symbolOptions struct {
	workdir, query, matcher, style string
	printResults                   bool
}

func init() {
	flag.StringVar(&symbolOptions.workdir, "symbol_workdir", "", "if set, run symbol benchmark in this directory")
	flag.StringVar(&symbolOptions.query, "symbol_query", "test", "symbol query to use in benchmark")
	flag.StringVar(&symbolOptions.matcher, "symbol_matcher", "", "symbol matcher to use in benchmark")
	flag.StringVar(&symbolOptions.style, "symbol_style", "", "symbol style to use in benchmark")
	flag.BoolVar(&symbolOptions.printResults, "symbol_print_results", false, "whether to print symbol query results")
}

func TestBenchmarkSymbols(t *testing.T) {
	if symbolOptions.workdir == "" {
		t.Skip("-symbol_workdir not configured")
	}

	opts := benchmarkOptions(symbolOptions.workdir)
	settings := make(Settings)
	if symbolOptions.matcher != "" {
		settings["symbolMatcher"] = symbolOptions.matcher
	}
	if symbolOptions.style != "" {
		settings["symbolStyle"] = symbolOptions.style
	}
	opts = append(opts, settings)

	WithOptions(opts...).Run(t, "", func(t *testing.T, env *Env) {
		// We can't Await in this test, since we have disabled hooks. Instead, run
		// one symbol request to completion to ensure all necessary cache entries
		// are populated.
		symbols, err := env.Editor.Server.Symbol(env.Ctx, &protocol.WorkspaceSymbolParams{
			Query: symbolOptions.query,
		})
		if err != nil {
			t.Fatal(err)
		}

		if symbolOptions.printResults {
			fmt.Println("Results:")
			for i := 0; i < len(symbols); i++ {
				fmt.Printf("\t%d. %s (%s)\n", i, symbols[i].Name, symbols[i].ContainerName)
			}
		}

		results := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := env.Editor.Server.Symbol(env.Ctx, &protocol.WorkspaceSymbolParams{
					Query: symbolOptions.query,
				}); err != nil {
					t.Fatal(err)
				}
			}
		})
		printBenchmarkResults(results)
	})
}
