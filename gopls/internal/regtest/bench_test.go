// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package regtest

import (
	"flag"
	"fmt"
	"testing"

	"golang.org/x/tools/internal/lsp"
	"golang.org/x/tools/internal/lsp/fake"
	"golang.org/x/tools/internal/lsp/protocol"
)

var iwlBench = struct {
	workdir string
}{}

func init() {
	flag.StringVar(&iwlBench.workdir, "iwl_workdir", "", "if set, run IWL benchmark in this directory")
}

func TestBenchmarkIWL(t *testing.T) {
	if iwlBench.workdir == "" {
		t.Skip("-iwl_workdir not configured")
	}
	opts := stressTestOptions(iwlBench.workdir)
	// Don't skip hooks, so that we can wait for IWL.
	opts = append(opts, SkipHooks(false))
	b := testing.Benchmark(func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			withOptions(opts...).run(t, "", func(t *testing.T, env *Env) {
				env.Await(
					CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1),
				)
			})
		}
	})
	printBench(b)
}

var symbolBench = struct {
	workdir, query, matcher, style string
	printResults                   bool
}{}

func init() {
	flag.StringVar(&symbolBench.workdir, "symbol_workdir", "", "if set, run symbol benchmark in this directory")
	flag.StringVar(&symbolBench.query, "symbol_query", "test", "symbol query to use in benchmark")
	flag.StringVar(&symbolBench.matcher, "symbol_matcher", "", "symbol matcher to use in benchmark")
	flag.StringVar(&symbolBench.style, "symbol_style", "", "symbol style to use in benchmark")
	flag.BoolVar(&symbolBench.printResults, "symbol_print_results", false, "whether to print symbol query results")
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
		printBench(b)
	})
}

func printBench(b testing.BenchmarkResult) {
	fmt.Println("Benchmark stats:")
	fmt.Println(b.String())
	fmt.Println(b.MemString())
}

func dummyCompletionBenchmarkFunction() { const s = "placeholder"; fmt.Printf("%s", s) }

var completionBench = struct {
	workdir, fileName, locationRegexp string
	printResults                      bool
}{}

func init() {
	flag.StringVar(&completionBench.workdir, "completion_workdir", "", "if set run completion benchmark in this directory (other benchmark flags expect an x/tools dir)")
	flag.StringVar(&completionBench.fileName, "completion_file", "internal/lsp/regtest/bench_test.go", "relative path to the file to complete")
	flag.StringVar(&completionBench.locationRegexp, "completion_regexp", `dummyCompletionBenchmarkFunction.*fmt\.Printf\("%s", s(\))`, "regexp location to complete at")
	flag.BoolVar(&completionBench.printResults, "completion_print_results", false, "whether to print completion results")
}

func TestBenchmarkCompletion(t *testing.T) {
	if completionBench.workdir == "" {
		t.Skip("-completion_workdir not configured")
	}
	opts := stressTestOptions(completionBench.workdir)
	// Completion gives bad results if IWL is not yet complete, so we must await
	// it first (and therefore need hooks).
	opts = append(opts, SkipHooks(false))
	withOptions(opts...).run(t, "", func(t *testing.T, env *Env) {
		env.Await(
			CompletedWork(lsp.DiagnosticWorkTitle(lsp.FromInitialWorkspaceLoad), 1),
		)
		env.OpenFile(completionBench.fileName)
		params := &protocol.CompletionParams{}
		params.Context.TriggerCharacter = "s"
		params.Context.TriggerKind = protocol.TriggerCharacter
		params.TextDocument.URI = env.Sandbox.Workdir.URI(completionBench.fileName)
		params.Position = env.RegexpSearch(completionBench.fileName, completionBench.locationRegexp).ToProtocolPosition()

		// Run one completion to make sure everything is warm.
		list, err := env.Editor.Server.Completion(env.Ctx, params)
		if err != nil {
			t.Fatal(err)
		}
		if completionBench.printResults {
			fmt.Println("Results:")
			for i := 0; i < len(list.Items); i++ {
				fmt.Printf("\t%d. %v\n", i, list.Items[i])
			}
		}
		b := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				_, err := env.Editor.Server.Completion(env.Ctx, params)
				if err != nil {
					t.Fatal(err)
				}
			}
		})
		printBench(b)
	})
}
