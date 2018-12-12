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
var a16k *[16 * 1024]byte
var a32k *[32 * 1024]byte
var a64k *[64 * 1024]byte

// This test checks that heap sampling produces reasonable
// results. Note that heap sampling uses randomization, so the results
// vary for run to run. To avoid flakes, this test performs multiple
// experiments and only complains if all of them consistently fail.
func main() {
	// Sample at 16K instead of default 512K to exercise sampling more heavily.
	runtime.MemProfileRate = 16 * 1024

	const countInterleaved = 100000
	allocInterleaved1(countInterleaved)
	allocInterleaved2(countInterleaved)
	allocInterleaved3(countInterleaved)
	allocInterleavedNames := []string{
		"main.allocInterleaved1",
		"main.allocInterleaved2",
		"main.allocInterleaved3",
	}
	checkAllocations(getMemProfileRecords(), allocInterleavedNames, countInterleaved,
		[]int64{64 * 1024, 1024, 32 * 1024, 512, 16 * 1024, 256})

	const count = 1000000
	allocSmall1(count)
	allocSmall2(count)
	allocSmall3(count)
	allocSmallNames := []string{
		"main.allocSmall1",
		"main.allocSmall2",
		"main.allocSmall3",
	}
	checkAllocations(getMemProfileRecords(), allocSmallNames, count, []int64{1024, 512, 256})
}

// allocInterleaved stress-tests the heap sampling logic by
// interleaving large and small allocations.
func allocInterleaved(n int) {
	for i := 0; i < n; i++ {
		// Test verification depends on these lines being contiguous.
		a64k = new([64 * 1024]byte)
		a1k = new([1024]byte)
		a32k = new([32 * 1024]byte)
		a512 = new([512]byte)
		a16k = new([16 * 1024]byte)
		a256 = new([256]byte)
	}
}

// Three separate instances of testing to avoid flakes.
func allocInterleaved1(n int) {
	allocInterleaved(n)
}

func allocInterleaved2(n int) {
	allocInterleaved(n)
}

func allocInterleaved3(n int) {
	allocInterleaved(n)
}

// allocSmall performs only small allocations for sanity testing.
func allocSmall(n int) {
	for i := 0; i < n; i++ {
		// Test verification depends on these lines being contiguous.
		a1k = new([1024]byte)
		a512 = new([512]byte)
		a256 = new([256]byte)
	}
}

// Three separate instances of testing to avoid flakes. Will report an error
// only if they all consistently report failures.
func allocSmall1(n int) {
	allocSmall(n)
}

func allocSmall2(n int) {
	allocSmall(n)
}

func allocSmall3(n int) {
	allocSmall(n)
}

// checkAllocations validates that the profile records collected for
// the named function are consistent with count contiguous allocations
// of the specified sizes.
// Check multiple functions and only report consistent failures across
// multiple tests.
// Look only at samples that contain a frame in fnames, and group the
// allocations by their line number. All these allocations are done from
// the same leaf function, so their line numbers are the same.
func checkAllocations(records []runtime.MemProfileRecord, fnames []string, count int64, size []int64) {
	objectsPerLine := map[int][]int64{}
	bytesPerLine := map[int][]int64{}
	totalCount := []int64{}
	// Compute the line number of the first allocation. All the
	// allocations are from the same leaf, so pick the first one.
	var firstLine int
	for ln := range allocObjects(records, fnames[0]) {
		if firstLine == 0 || firstLine > ln {
			firstLine = ln
		}
	}
	for _, fname := range fnames {
		var objectCount int64
		a := allocObjects(records, fname)
		for s := range size {
			// Allocations of size size[s] are  from line firstLine + s.
			ln := firstLine + s
			objectsPerLine[ln] = append(objectsPerLine[ln], a[ln].objects)
			bytesPerLine[ln] = append(bytesPerLine[ln], a[ln].bytes)
			objectCount += a[ln].objects
		}
		totalCount = append(totalCount, objectCount)
	}
	for i, w := range size {
		ln := firstLine + i
		checkValue(fnames[0], ln, "objects", count, objectsPerLine[ln])
		checkValue(fnames[0], ln, "bytes", count*w, bytesPerLine[ln])
	}
	checkValue(fnames[0], 0, "total", count*int64(len(size)), totalCount)
}

// checkValue checks an unsampled value against its expected value.
// Given that this is a sampled value, it will be unexact and will change
// from run to run. Only report it as a failure if all the values land
// consistently far from the expected value.
func checkValue(fname string, ln int, testName string, want int64, got []int64) {
	if got == nil {
		panic("Unexpected empty result")
	}
	min, max := got[0], got[0]
	for _, g := range got[1:] {
		if g < min {
			min = g
		}
		if g > max {
			max = g
		}
	}
	margin := want / 10 // 10% margin.
	if min > want+margin {
		panic(fmt.Sprintf("%s:%d want %s <= %d, got %v", fname, ln, testName, want+margin, got))
	}
	if max < want-margin {
		panic(fmt.Sprintf("%s:%d want %s >= %d, got %d", fname, ln, testName, want-margin, got))

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

// allocObjects examines the profile records for samples including the
// named function and returns the allocation stats aggregated by
// source line number of the allocation (at the leaf frame).
func allocObjects(records []runtime.MemProfileRecord, function string) map[int]allocStat {
	a := make(map[int]allocStat)
	for _, r := range records {
		var pcs []uintptr
		for _, s := range r.Stack0 {
			if s == 0 {
				break
			}
			pcs = append(pcs, s)
		}
		frames := runtime.CallersFrames(pcs)
		line := 0
		for {
			frame, more := frames.Next()
			name := frame.Function
			if line == 0 {
				line = frame.Line
			}
			if name == function {
				allocStat := a[line]
				allocStat.bytes += r.AllocBytes
				allocStat.objects += r.AllocObjects
				a[line] = allocStat
			}
			if !more {
				break
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
