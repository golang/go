// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"fmt"
	"runtime"
	"strings"
	"testing"

	. "golang.org/x/tools/gopls/internal/regtest"

	"golang.org/x/tools/internal/lsp/fake"
)

// dummyCompletionFunction to test manually configured completion using CLI.
func dummyCompletionFunction() { const s = "placeholder"; fmt.Printf("%s", s) }

type completionBenchOptions struct {
	workdir, file, locationRegexp string
	printResults                  bool
	// hook to run edits before initial completion, not supported for manually
	// configured completions.
	preCompletionEdits func(*Env)
}

var completionOptions = completionBenchOptions{}

func init() {
	flag.StringVar(&completionOptions.workdir, "completion_workdir", "", "directory to run completion benchmarks in")
	flag.StringVar(&completionOptions.file, "completion_file", "", "relative path to the file to complete in")
	flag.StringVar(&completionOptions.locationRegexp, "completion_regexp", "", "regexp location to complete at")
	flag.BoolVar(&completionOptions.printResults, "completion_print_results", false, "whether to print completion results")
}

func benchmarkCompletion(options completionBenchOptions, t *testing.T) {
	if completionOptions.workdir == "" {
		t.Skip("-completion_workdir not configured, skipping benchmark")
	}

	opts := stressTestOptions(options.workdir)

	// Completion gives bad results if IWL is not yet complete, so we must await
	// it first (and therefore need hooks).
	opts = append(opts, SkipHooks(false))

	WithOptions(opts...).Run(t, "", func(t *testing.T, env *Env) {
		env.OpenFile(options.file)

		// Run edits required for this completion.
		if options.preCompletionEdits != nil {
			options.preCompletionEdits(env)
		}

		// Add a comment as a marker at the start of the file, we'll replace
		// this in every iteration to trigger type checking and hence emulate
		// a more real world scenario.
		env.EditBuffer(options.file, fake.Edit{Text: "// 0\n"})

		// Run a completion to make sure the system is warm.
		pos := env.RegexpSearch(options.file, options.locationRegexp)
		completions := env.Completion(options.file, pos)

		if options.printResults {
			fmt.Println("Results:")
			for i := 0; i < len(completions.Items); i++ {
				fmt.Printf("\t%d. %v\n", i, completions.Items[i])
			}
		}

		results := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				b.StopTimer()
				env.RegexpReplace(options.file, `\/\/ \d*`, fmt.Sprintf("// %d", i))

				// explicitly garbage collect since we don't want to count this
				// time in completion benchmarks.
				if i%10 == 0 {
					runtime.GC()
				}
				b.StartTimer()

				env.Completion(options.file, pos)
			}
		})

		printBenchmarkResults(results)
	})
}

// endPosInBuffer returns the position for last character in the buffer for
// the given file.
func endPosInBuffer(env *Env, name string) fake.Pos {
	buffer := env.Editor.BufferText(name)
	lines := strings.Split(buffer, "\n")
	numLines := len(lines)

	return fake.Pos{
		Line:   numLines - 1,
		Column: len([]rune(lines[numLines-1])),
	}
}

// Benchmark completion at a specified file and location. When no CLI options
// are specified, this test is skipped.
// To Run (from x/tools/gopls) against the dummy function above:
// 	go test -v ./internal/regtest -run=TestBenchmarkConfiguredCompletion
// 	-completion_workdir="$HOME/Developer/tools"
// 	-completion_file="gopls/internal/regtest/completion_bench_test.go"
// 	-completion_regexp="dummyCompletionFunction.*fmt\.Printf\(\"%s\", s(\))"
func TestBenchmarkConfiguredCompletion(t *testing.T) {
	benchmarkCompletion(completionOptions, t)
}

// To run (from x/tools/gopls):
// 	go test -v ./internal/regtest -run TestBenchmark<>Completion
//	-completion_workdir="$HOME/Developer/tools"
// where <> is one of the tests below. completion_workdir should be path to
// x/tools on your system.

// Benchmark struct completion in tools codebase.
func TestBenchmarkStructCompletion(t *testing.T) {
	file := "internal/lsp/cache/session.go"

	preCompletionEdits := func(env *Env) {
		env.OpenFile(file)
		originalBuffer := env.Editor.BufferText(file)
		env.EditBuffer(file, fake.Edit{
			End:  endPosInBuffer(env, file),
			Text: originalBuffer + "\nvar testVariable map[string]bool = Session{}.\n",
		})
	}

	benchmarkCompletion(completionBenchOptions{
		workdir:            completionOptions.workdir,
		file:               file,
		locationRegexp:     `var testVariable map\[string\]bool = Session{}(\.)`,
		preCompletionEdits: preCompletionEdits,
		printResults:       completionOptions.printResults,
	}, t)
}

// Benchmark import completion in tools codebase.
func TestBenchmarkImportCompletion(t *testing.T) {
	benchmarkCompletion(completionBenchOptions{
		workdir:        completionOptions.workdir,
		file:           "internal/lsp/source/completion/completion.go",
		locationRegexp: `go\/()`,
		printResults:   completionOptions.printResults,
	}, t)
}

// Benchmark slice completion in tools codebase.
func TestBenchmarkSliceCompletion(t *testing.T) {
	file := "internal/lsp/cache/session.go"

	preCompletionEdits := func(env *Env) {
		env.OpenFile(file)
		originalBuffer := env.Editor.BufferText(file)
		env.EditBuffer(file, fake.Edit{
			End:  endPosInBuffer(env, file),
			Text: originalBuffer + "\nvar testVariable []byte = \n",
		})
	}

	benchmarkCompletion(completionBenchOptions{
		workdir:            completionOptions.workdir,
		file:               file,
		locationRegexp:     `var testVariable \[\]byte (=)`,
		preCompletionEdits: preCompletionEdits,
		printResults:       completionOptions.printResults,
	}, t)
}

// Benchmark deep completion in function call in tools codebase.
func TestBenchmarkFuncDeepCompletion(t *testing.T) {
	file := "internal/lsp/source/completion/completion.go"
	fileContent := `
func (c *completer) _() {
	c.inference.kindMatches(c.)
}
`
	preCompletionEdits := func(env *Env) {
		env.OpenFile(file)
		originalBuffer := env.Editor.BufferText(file)
		env.EditBuffer(file, fake.Edit{
			End:  endPosInBuffer(env, file),
			Text: originalBuffer + fileContent,
		})
	}

	benchmarkCompletion(completionBenchOptions{
		workdir:            completionOptions.workdir,
		file:               file,
		locationRegexp:     `func \(c \*completer\) _\(\) {\n\tc\.inference\.kindMatches\((c)`,
		preCompletionEdits: preCompletionEdits,
		printResults:       completionOptions.printResults,
	}, t)
}
