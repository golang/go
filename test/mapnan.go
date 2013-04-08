// +build darwin linux
// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that NaNs in maps don't go quadratic.

package main

import (
	"fmt"
	"math"
	"time"
	"syscall"
)

func main() {

	// Test that NaNs in maps don't go quadratic.
	t := func(n int) time.Duration {
		var u0 syscall.Rusage
		if err := syscall.Getrusage(0,  &u0); err != nil {
			panic(err)
		}
		m := map[float64]int{}
		nan := math.NaN()
		for i := 0; i < n; i++ {
			m[nan] = 1
		}
		if len(m) != n {
			panic("wrong size map after nan insertion")
		}
		var u1 syscall.Rusage
		if err := syscall.Getrusage(0,  &u1); err != nil {
			panic(err)
		}
		return time.Duration(u1.Utime.Nano() - u0.Utime.Nano())
	}

	// Depending on the machine and OS, this test might be too fast
	// to measure with accurate enough granularity. On failure,
	// make it run longer, hoping that the timing granularity
	// is eventually sufficient.

	n := 30000 // ~8ms user time on a Mid 2011 MacBook Air (1.8 GHz Core i7)
	fails := 0
	for {
		t1 := t(n)
		t2 := t(2 * n)
		// should be 2x (linear); allow up to 3x
		if t2 < 3*t1 {
			return
		}
		fails++
		if fails == 6 {
			panic(fmt.Sprintf("too slow: %d inserts: %v; %d inserts: %v\n", n, t1, 2*n, t2))
		}
		if fails < 4 {
			n *= 2
		}
	}
}
