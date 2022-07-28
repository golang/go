// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"runtime"
	"testing"

	. "golang.org/x/tools/internal/lsp/regtest"
)

// TestPrintMemStats measures the memory usage of loading a project.
// It uses the same -didchange_dir flag as above.
// Always run it in isolation since it measures global heap usage.
//
// Kubernetes example:
//
//	$ go test -v -run=TestPrintMemStats -didchange_dir=$HOME/w/kubernetes
//	TotalAlloc:      5766 MB
//	HeapAlloc:       1984 MB
//
// Both figures exhibit variance of less than 1%.
func TestPrintMemStats(t *testing.T) {
	if *benchDir == "" {
		t.Skip("-didchange_dir is not set")
	}

	// Load the program...
	opts := benchmarkOptions(*benchDir)
	WithOptions(opts...).Run(t, "", func(_ *testing.T, env *Env) {
		// ...and print the memory usage.
		runtime.GC()
		runtime.GC()
		var mem runtime.MemStats
		runtime.ReadMemStats(&mem)
		t.Logf("TotalAlloc:\t%d MB", mem.TotalAlloc/1e6)
		t.Logf("HeapAlloc:\t%d MB", mem.HeapAlloc/1e6)
	})
}
