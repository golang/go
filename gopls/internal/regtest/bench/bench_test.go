// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"fmt"
	"testing"

	"golang.org/x/tools/gopls/internal/hooks"
	"golang.org/x/tools/internal/lsp/bug"

	. "golang.org/x/tools/internal/lsp/regtest"
)

func TestMain(m *testing.M) {
	bug.PanicOnBugs = true
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
		Modes(Default),
		// Remove the default timeout. Individual tests should control their
		// own graceful termination.
		NoDefaultTimeout(),

		// Use the actual proxy, since we want our builds to succeed.
		GOPROXY("https://proxy.golang.org"),
	}
}

func printBenchmarkResults(result testing.BenchmarkResult) {
	fmt.Printf("BenchmarkStatistics\t%s\t%s\n", result.String(), result.MemString())
}
