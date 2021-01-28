// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"fmt"
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"

	"golang.org/x/tools/internal/lsp/protocol"
)

func TestMain(m *testing.M) {
	Main(m)
}

func printBenchmarkResults(result testing.BenchmarkResult) {
	fmt.Println("Benchmark Statistics:")
	fmt.Println(result.String())
	fmt.Println(result.MemString())
}

var iwlOptions struct {
	workdir string
}

func init() {
	flag.StringVar(&iwlOptions.workdir, "iwl_workdir", "", "if set, run IWL benchmark in this directory")
}

func TestBenchmarkIWL(t *testing.T) {
	if iwlOptions.workdir == "" {
		t.Skip("-iwl_workdir not configured")
	}

	opts := stressTestOptions(iwlOptions.workdir)
	// Don't skip hooks, so that we can wait for IWL.
	opts = append(opts, SkipHooks(false))

	results := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			WithOptions(opts...).Run(t, "", func(t *testing.T, env *Env) {})
		}
	})

	printBenchmarkResults(results)
}

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

	opts := stressTestOptions(symbolOptions.workdir)
	conf := EditorConfig{}
	if symbolOptions.matcher != "" {
		conf.SymbolMatcher = &symbolOptions.matcher
	}
	if symbolOptions.style != "" {
		conf.SymbolStyle = &symbolOptions.style
	}
	opts = append(opts, conf)

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
