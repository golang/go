// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test heap sampling logic.

package main

import (
	"fmt"
	"math"
	"runtime"
)

var a16 *[16]byte
var a512 *[512]byte
var a256 *[256]byte
var a1k *[1024]byte
var a64k *[64 * 1024]byte

// This test checks that heap sampling produces reasonable
// results. Note that heap sampling uses randomization, so the results
// vary for run to run. This test only checks that the resulting
// values appear reasonable.
func main() {
        // Sample at 5K instead of default 512K to exercise sampling more.
        runtime.MemProfileRate = 64 * 1024

	const countInterleaved = 100000
	allocInterleaved(countInterleaved)
	checkAllocations(getMemProfileRecords(), "main.allocInterleaved", countInterleaved, 
	[]int64{64 * 1024, 1024, 64 * 1024, 512, 64 * 1024, 256})

	const count = 1000000
	alloc(count)
	checkAllocations(getMemProfileRecords(), "main.alloc", count, []int64{1024, 512, 256})
}

// allocInterleaved stress-tests the heap sampling logic by
// interleaving large and small allocations.
func allocInterleaved(n int) {
	for i := 0; i < n; i++ {
		// Test verification depends on these lines being contiguous.
		a64k = new([64 * 1024]byte)
		a1k = new([1024]byte)
		a64k = new([64 * 1024]byte)
		a512 = new([512]byte)
		a64k = new([64 * 1024]byte)
		a256 = new([256]byte)
	}
}

// alloc performs only small allocations for sanity testing.
func alloc(n int) {
	for i := 0; i < n; i++ {
		// Test verification depends on these lines being contiguous.
		a1k = new([1024]byte)
		a512 = new([512]byte)
		a256 = new([256]byte)
	}
}

// checkAllocations validates that the profile records collected for
// the named function are consistent with count contiguous allocations
// of the specified sizes.
func checkAllocations(records []runtime.MemProfileRecord, fname string, count int64, size []int64) {
	a := allocObjects(records, fname)
	firstLine := 0
	for ln := range a {
		if firstLine == 0 || firstLine > ln {
			firstLine = ln
		}
	}
	var totalcount int64
	for i, w := range size {
		ln := firstLine + i
		s := a[ln]
		checkValue(fname, ln, "objects", count, s.objects)
		checkValue(fname, ln, "bytes", count*w, s.bytes)
		totalcount += s.objects
	}
	// Check the total number of allocations, to ensure some sampling occurred.
	checkValue(fname, 0, "total", count * int64(len(size)), totalcount)
}

// checkValue checks an unsampled value against a range.
func checkValue(fname string, ln int, name string, want, got int64) {
        margin := want / 10
	if got < want - margin || got > want + margin {
		panic(fmt.Sprintf("%s:%d want %s >= %d && <= %d, got %d", fname, ln, name, want-margin, want+margin, got))
	}
}

func getMemProfileRecords() []runtime.MemProfileRecord {
	// Force the runtime to update the object and byte counts.
	// This can take up to two GC cycles to get a complete
	// snapshot of the current point in time.
	runtime.GC()
	runtime.GC()

	// Find out how many records there are (MemProfile(nil, true)),
	// allocate that many records, and get the data.
	// There's a race—more records might be added between
	// the two calls—so allocate a few extra records for safety
	// and also try again if we're very unlucky.
	// The loop should only execute one iteration in the common case.
	var p []runtime.MemProfileRecord
	n, ok := runtime.MemProfile(nil, true)
	for {
		// Allocate room for a slightly bigger profile,
		// in case a few more entries have been added
		// since the call to MemProfile.
		p = make([]runtime.MemProfileRecord, n+50)
		n, ok = runtime.MemProfile(p, true)
		if ok {
			p = p[0:n]
			break
		}
		// Profile grew; try again.
	}
	return p
}

type allocStat struct {
	bytes, objects int64
}

// allocObjects examines the profile records for the named function
// and returns the allocation stats aggregated by source line number.
func allocObjects(records []runtime.MemProfileRecord, function string) map[int]allocStat {
	a := make(map[int]allocStat)
	for _, r := range records {
		for _, s := range r.Stack0 {
			if s == 0 {
				break
			}
			if f := runtime.FuncForPC(s); f != nil {
				name := f.Name()
				_, line := f.FileLine(s)
				if name == function {
					allocStat := a[line]
					allocStat.bytes += r.AllocBytes
					allocStat.objects += r.AllocObjects
					a[line] = allocStat
				}
			}
		}
	}
	for line, stats := range a {
		objects, bytes := scaleHeapSample(stats.objects, stats.bytes, int64(runtime.MemProfileRate))
		a[line] = allocStat{bytes, objects}
	}
	return a
}

// scaleHeapSample unsamples heap allocations.
// Taken from src/cmd/pprof/internal/profile/legacy_profile.go
func scaleHeapSample(count, size, rate int64) (int64, int64) {
	if count == 0 || size == 0 {
		return 0, 0
	}

	if rate <= 1 {
		// if rate==1 all samples were collected so no adjustment is needed.
		// if rate<1 treat as unknown and skip scaling.
		return count, size
	}

	avgSize := float64(size) / float64(count)
	scale := 1 / (1 - math.Exp(-avgSize/float64(rate)))

	return int64(float64(count) * scale), int64(float64(size) * scale)
}
