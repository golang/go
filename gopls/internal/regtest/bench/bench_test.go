// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"fmt"
	"testing"
	"time"

	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/internal/lsp/fake"
	. "golang.org/x/tools/internal/lsp/regtest"

	"golang.org/x/tools/internal/lsp/protocol"
)

func TestMain(m *testing.M) {
	Main(m, hooks.Options)
}

func benchmarkOptions(dir string) []RunOption {
	return []RunOption{
		// Run in an existing directory, since we're trying to simulate known cases
		// that cause gopls memory problems.
		InExistingDir(dir),
		// Skip logs as they buffer up memory unnaturally.
		SkipLogs(),
		// The Debug server only makes sense if running in singleton mode.
		Modes(Singleton),
		// Set a generous timeout. Individual tests should control their own
		// graceful termination.
		Timeout(20 * time.Minute),

		// Use the actual proxy, since we want our builds to succeed.
		GOPROXY("https://proxy.golang.org"),
	}
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

var (
	benchDir  = flag.String("didchange_dir", "", "If set, run benchmarks in this dir. Must also set regtest_bench_file.")
	benchFile = flag.String("didchange_file", "", "The file to modify")
)

// TestBenchmarkDidChange benchmarks modifications of a single file by making
// synthetic modifications in a comment. It controls pacing by waiting for the
// server to actually start processing the didChange notification before
// proceeding. Notably it does not wait for diagnostics to complete.
//
// Run it by passing -didchange_dir and -didchange_file, where -didchange_dir
// is the path to a workspace root, and -didchange_file is the
// workspace-relative path to a file to modify. e.g.:
//
//  go test -run=TestBenchmarkDidChange \
//   -didchange_dir=path/to/kubernetes \
//   -didchange_file=pkg/util/hash/hash.go
func TestBenchmarkDidChange(t *testing.T) {
	if *benchDir == "" {
		t.Skip("-didchange_dir is not set")
	}
	if *benchFile == "" {
		t.Fatal("-didchange_file must be set if -didchange_dir is set")
	}

	opts := benchmarkOptions(*benchDir)
	WithOptions(opts...).Run(t, "", func(_ *testing.T, env *Env) {
		env.OpenFile(*benchFile)
		env.Await(env.DoneWithOpen())
		// Insert the text we'll be modifying at the top of the file.
		env.EditBuffer(*benchFile, fake.Edit{Text: "// __REGTEST_PLACEHOLDER_0__\n"})
		result := testing.Benchmark(func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				env.EditBuffer(*benchFile, fake.Edit{
					Start: fake.Pos{Line: 0, Column: 0},
					End:   fake.Pos{Line: 1, Column: 0},
					// Increment
					Text: fmt.Sprintf("// __REGTEST_PLACEHOLDER_%d__\n", i+1),
				})
				env.Await(StartedChange(uint64(i + 1)))
			}
			b.StopTimer()
		})
		printBenchmarkResults(result)
	})
}
