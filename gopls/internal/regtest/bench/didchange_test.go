// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"flag"
	"fmt"
	"os"
	"runtime/pprof"
	"testing"

	"golang.org/x/tools/internal/lsp/fake"

	. "golang.org/x/tools/internal/lsp/regtest"
)

var (
	benchDir     = flag.String("didchange_dir", "", "If set, run benchmarks in this dir. Must also set didchange_file.")
	benchFile    = flag.String("didchange_file", "", "The file to modify")
	benchProfile = flag.String("didchange_cpuprof", "", "file to write cpu profiling data to")
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
//	go test -run=TestBenchmarkDidChange \
//	 -didchange_dir=path/to/kubernetes \
//	 -didchange_file=pkg/util/hash/hash.go
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

		// Run the profiler after the initial load,
		// across all benchmark iterations.
		if *benchProfile != "" {
			profile, err := os.Create(*benchProfile)
			if err != nil {
				t.Fatal(err)
			}
			defer profile.Close()
			if err := pprof.StartCPUProfile(profile); err != nil {
				t.Fatal(err)
			}
			defer pprof.StopCPUProfile()
		}

		result := testing.Benchmark(func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				env.EditBuffer(*benchFile, fake.Edit{
					Start: fake.Pos{Line: 0, Column: 0},
					End:   fake.Pos{Line: 1, Column: 0},
					// Increment
					Text: fmt.Sprintf("// __REGTEST_PLACEHOLDER_%d__\n", i+1),
				})
				env.Await(StartedChange(uint64(i + 1)))
			}
		})
		printBenchmarkResults(result)
	})
}
