// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bench

import (
	"runtime"
	"testing"
)

// TestPrintMemStats measures the memory usage of loading a project.
// It uses the same -didchange_dir flag as above.
// Always run it in isolation since it measures global heap usage.
//
// Kubernetes example:
//
//	$ go test -v -run=TestPrintMemStats -workdir=$HOME/w/kubernetes
//	TotalAlloc:      5766 MB
//	HeapAlloc:       1984 MB
//
// Both figures exhibit variance of less than 1%.
func TestPrintMemStats(t *testing.T) {
	// This test only makes sense when run in isolation, so for now it is
	// manually skipped.
	//
	// TODO(rfindley): figure out a better way to capture memstats as a benchmark
	// metric.
	t.Skip("unskip to run this test manually")

	_ = benchmarkEnv(t)

	runtime.GC()
	runtime.GC()
	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	t.Logf("TotalAlloc:\t%d MB", mem.TotalAlloc/1e6)
	t.Logf("HeapAlloc:\t%d MB", mem.HeapAlloc/1e6)
}
