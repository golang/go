// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sort_test

import (
	"runtime"
	. "sort"
	"testing"
)

func f(a []int, x int) func(int) bool {
	return func(i int) bool {
		return a[i] >= x
	}
}

var data = []int{0: -10, 1: -5, 2: 0, 3: 1, 4: 2, 5: 3, 6: 5, 7: 7, 8: 11, 9: 100, 10: 100, 11: 100, 12: 1000, 13: 10000}

var tests = []struct {
	name string
	n    int
	f    func(int) bool
	i    int
}{
	{"empty", 0, nil, 0},
	{"1 1", 1, func(i int) bool { return i >= 1 }, 1},
	{"1 true", 1, func(i int) bool { return true }, 0},
	{"1 false", 1, func(i int) bool { return false }, 1},
	{"1e9 991", 1e9, func(i int) bool { return i >= 991 }, 991},
	{"1e9 true", 1e9, func(i int) bool { return true }, 0},
	{"1e9 false", 1e9, func(i int) bool { return false }, 1e9},
	{"data -20", len(data), f(data, -20), 0},
	{"data -10", len(data), f(data, -10), 0},
	{"data -9", len(data), f(data, -9), 1},
	{"data -6", len(data), f(data, -6), 1},
	{"data -5", len(data), f(data, -5), 1},
	{"data 3", len(data), f(data, 3), 5},
	{"data 11", len(data), f(data, 11), 8},
	{"data 99", len(data), f(data, 99), 9},
	{"data 100", len(data), f(data, 100), 9},
	{"data 101", len(data), f(data, 101), 12},
	{"data 10000", len(data), f(data, 10000), 13},
	{"data 10001", len(data), f(data, 10001), 14},
	{"descending a", 7, func(i int) bool { return []int{99, 99, 59, 42, 7, 0, -1, -1}[i] <= 7 }, 4},
	{"descending 7", 1e9, func(i int) bool { return 1e9-i <= 7 }, 1e9 - 7},
	{"overflow", 2e9, func(i int) bool { return false }, 2e9},
}

func TestSearch(t *testing.T) {
	for _, e := range tests {
		i := Search(e.n, e.f)
		if i != e.i {
			t.Errorf("%s: expected index %d; got %d", e.name, e.i, i)
		}
	}
}

// log2 computes the binary logarithm of x, rounded up to the next integer.
// (log2(0) == 0, log2(1) == 0, log2(2) == 1, log2(3) == 2, etc.)
//
func log2(x int) int {
	n := 0
	for p := 1; p < x; p += p {
		// p == 2**n
		n++
	}
	// p/2 < x <= p == 2**n
	return n
}

func TestSearchEfficiency(t *testing.T) {
	n := 100
	step := 1
	for exp := 2; exp < 10; exp++ {
		// n == 10**exp
		// step == 10**(exp-2)
		max := log2(n)
		for x := 0; x < n; x += step {
			count := 0
			i := Search(n, func(i int) bool { count++; return i >= x })
			if i != x {
				t.Errorf("n = %d: expected index %d; got %d", n, x, i)
			}
			if count > max {
				t.Errorf("n = %d, x = %d: expected <= %d calls; got %d", n, x, max, count)
			}
		}
		n *= 10
		step *= 10
	}
}

// Smoke tests for convenience wrappers - not comprehensive.

var fdata = []float64{0: -3.14, 1: 0, 2: 1, 3: 2, 4: 1000.7}
var sdata = []string{0: "f", 1: "foo", 2: "foobar", 3: "x"}

var wrappertests = []struct {
	name   string
	result int
	i      int
}{
	{"SearchInts", SearchInts(data, 11), 8},
	{"SearchFloat64s", SearchFloat64s(fdata, 2.1), 4},
	{"SearchStrings", SearchStrings(sdata, ""), 0},
	{"IntSlice.Search", IntSlice(data).Search(0), 2},
	{"Float64Slice.Search", Float64Slice(fdata).Search(2.0), 3},
	{"StringSlice.Search", StringSlice(sdata).Search("x"), 3},
}

func TestSearchWrappers(t *testing.T) {
	for _, e := range wrappertests {
		if e.result != e.i {
			t.Errorf("%s: expected index %d; got %d", e.name, e.i, e.result)
		}
	}
}

func runSearchWrappers() {
	SearchInts(data, 11)
	SearchFloat64s(fdata, 2.1)
	SearchStrings(sdata, "")
	IntSlice(data).Search(0)
	Float64Slice(fdata).Search(2.0)
	StringSlice(sdata).Search("x")
}

func TestSearchWrappersDontAlloc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}
	allocs := testing.AllocsPerRun(100, runSearchWrappers)
	if allocs != 0 {
		t.Errorf("expected no allocs for runSearchWrappers, got %v", allocs)
	}
}

func BenchmarkSearchWrappers(b *testing.B) {
	for i := 0; i < b.N; i++ {
		runSearchWrappers()
	}
}

// Abstract exhaustive test: all sizes up to 100,
// all possible return values.  If there are any small
// corner cases, this test exercises them.
func TestSearchExhaustive(t *testing.T) {
	for size := 0; size <= 100; size++ {
		for targ := 0; targ <= size; targ++ {
			i := Search(size, func(i int) bool { return i >= targ })
			if i != targ {
				t.Errorf("Search(%d, %d) = %d", size, targ, i)
			}
		}
	}
}
