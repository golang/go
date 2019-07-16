// +build darwin linux
// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that maps don't go quadratic for NaNs and other values.

package main

import (
	"fmt"
	"math"
	"time"
)

// checkLinear asserts that the running time of f(n) is in O(n).
// tries is the initial number of iterations.
func checkLinear(typ string, tries int, f func(n int)) {
	// Depending on the machine and OS, this test might be too fast
	// to measure with accurate enough granularity. On failure,
	// make it run longer, hoping that the timing granularity
	// is eventually sufficient.

	timeF := func(n int) time.Duration {
		t1 := time.Now()
		f(n)
		return time.Since(t1)
	}

	t0 := time.Now()

	n := tries
	fails := 0
	for {
		t1 := timeF(n)
		t2 := timeF(2 * n)

		// should be 2x (linear); allow up to 3x
		if t2 < 3*t1 {
			if false {
				fmt.Println(typ, "\t", time.Since(t0))
			}
			return
		}
		// If n ops run in under a second and the ratio
		// doesn't work out, make n bigger, trying to reduce
		// the effect that a constant amount of overhead has
		// on the computed ratio.
		if t1 < 1*time.Second {
			n *= 2
			continue
		}
		// Once the test runs long enough for n ops,
		// try to get the right ratio at least once.
		// If five in a row all fail, give up.
		if fails++; fails >= 5 {
			panic(fmt.Sprintf("%s: too slow: %d inserts: %v; %d inserts: %v\n",
				typ, n, t1, 2*n, t2))
		}
	}
}

type I interface {
	f()
}

type C int

func (C) f() {}

func main() {
	// NaNs. ~31ms on a 1.6GHz Zeon.
	checkLinear("NaN", 30000, func(n int) {
		m := map[float64]int{}
		nan := math.NaN()
		for i := 0; i < n; i++ {
			m[nan] = 1
		}
		if len(m) != n {
			panic("wrong size map after nan insertion")
		}
	})

	// ~6ms on a 1.6GHz Zeon.
	checkLinear("eface", 10000, func(n int) {
		m := map[interface{}]int{}
		for i := 0; i < n; i++ {
			m[i] = 1
		}
	})

	// ~7ms on a 1.6GHz Zeon.
	// Regression test for CL 119360043.
	checkLinear("iface", 10000, func(n int) {
		m := map[I]int{}
		for i := 0; i < n; i++ {
			m[C(i)] = 1
		}
	})

	// ~6ms on a 1.6GHz Zeon.
	checkLinear("int", 10000, func(n int) {
		m := map[int]int{}
		for i := 0; i < n; i++ {
			m[i] = 1
		}
	})

	// ~18ms on a 1.6GHz Zeon.
	checkLinear("string", 10000, func(n int) {
		m := map[string]int{}
		for i := 0; i < n; i++ {
			m[fmt.Sprint(i)] = 1
		}
	})

	// ~6ms on a 1.6GHz Zeon.
	checkLinear("float32", 10000, func(n int) {
		m := map[float32]int{}
		for i := 0; i < n; i++ {
			m[float32(i)] = 1
		}
	})

	// ~6ms on a 1.6GHz Zeon.
	checkLinear("float64", 10000, func(n int) {
		m := map[float64]int{}
		for i := 0; i < n; i++ {
			m[float64(i)] = 1
		}
	})

	// ~22ms on a 1.6GHz Zeon.
	checkLinear("complex64", 10000, func(n int) {
		m := map[complex64]int{}
		for i := 0; i < n; i++ {
			m[complex(float32(i), float32(i))] = 1
		}
	})

	// ~32ms on a 1.6GHz Zeon.
	checkLinear("complex128", 10000, func(n int) {
		m := map[complex128]int{}
		for i := 0; i < n; i++ {
			m[complex(float64(i), float64(i))] = 1
		}
	})

	// ~70ms on a 1.6GHz Zeon.
	// The iterate/delete idiom currently takes expected
	// O(n lg n) time.  Fortunately, the checkLinear test
	// leaves enough wiggle room to include n lg n time
	// (it actually tests for O(n^log_2(3)).
	// To prevent false positives, average away variation
	// by doing multiple rounds within a single run.
	checkLinear("iterdelete", 2500, func(n int) {
		for round := 0; round < 4; round++ {
			m := map[int]int{}
			for i := 0; i < n; i++ {
				m[i] = i
			}
			for i := 0; i < n; i++ {
				for k := range m {
					delete(m, k)
					break
				}
			}
		}
	})
}
