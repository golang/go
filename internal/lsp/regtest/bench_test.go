// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"flag"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

var symbolBench = struct {
	workdir, query, matcher, style string
	printResults                   bool
}{}

func init() {
	flag.StringVar(&symbolBench.workdir, "symbol_workdir", "", "if set, run symbol benchmark in this directory")
	flag.StringVar(&symbolBench.query, "symbol_query", "test", "symbol query to use in benchmark")
	flag.StringVar(&symbolBench.matcher, "symbol_matcher", "", "symbol matcher to use in benchmark")
	flag.StringVar(&symbolBench.style, "symbol_style", "", "symbol style to use in benchmark")
	flag.BoolVar(&symbolBench.printResults, "symbol_print_results", false, "symbol style to use in benchmark")
}

func TestBenchmarkSymbols(t *testing.T) {
	if symbolBench.workdir == "" {
		t.Skip("-symbol_workdir not configured")
	}
	opts := stressTestOptions(symbolBench.workdir)
	conf := fake.EditorConfig{}
	if symbolBench.matcher != "" {
		conf.SymbolMatcher = &symbolBench.matcher
	}
	if symbolBench.style != "" {
		conf.SymbolStyle = &symbolBench.style
	}
	opts = append(opts, WithEditorConfig(conf))
	withOptions(opts...).run(t, "", func(t *testing.T, env *Env) {
		// We can't Await in this test, since we have disabled hooks. Instead, run
		// one symbol request to completion to ensure all necessary cache entries
		// are populated.
		results, err := env.Editor.Server.Symbol(env.Ctx, &protocol.WorkspaceSymbolParams{
			Query: symbolBench.query,
		})
		if err != nil {
			t.Fatal(err)
		}
		if symbolBench.printResults {
			fmt.Println("Results:")
			for i := 0; i < len(results); i++ {
				fmt.Printf("\t%d. %s\n", i, results[i].Name)
			}
		}
		b := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				if _, err := env.Editor.Server.Symbol(env.Ctx, &protocol.WorkspaceSymbolParams{
					Query: symbolBench.query,
				}); err != nil {
					t.Fatal(err)
				}
			}
		})
		fmt.Println("Benchmark stats:")
		fmt.Println(b.String())
		fmt.Println(b.MemString())
	})
}
