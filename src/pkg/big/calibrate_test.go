// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file computes the Karatsuba threshold as a "test".
// Usage: gotest -calibrate

package big

import (
	"flag"
	"fmt"
	"testing"
	"time"
	"unsafe" // for Sizeof
)


var calibrate = flag.Bool("calibrate", false, "run calibration test")


// makeNumber creates an n-word number 0xffff...ffff
func makeNumber(n int) *Int {
	var w Word
	b := make([]byte, n*unsafe.Sizeof(w))
	for i := range b {
		b[i] = 0xff
	}
	var x Int
	x.SetBytes(b)
	return &x
}


// measure returns the time to compute x*x in nanoseconds
func measure(f func()) int64 {
	const N = 100
	start := time.Nanoseconds()
	for i := N; i > 0; i-- {
		f()
	}
	stop := time.Nanoseconds()
	return (stop - start) / N
}


func computeThreshold(t *testing.T) int {
	// use a mix of numbers as work load
	x := make([]*Int, 20)
	for i := range x {
		x[i] = makeNumber(10 * (i + 1))
	}

	threshold := -1
	for n := 8; threshold < 0 || n <= threshold+20; n += 2 {
		// set work load
		f := func() {
			var t Int
			for _, x := range x {
				t.Mul(x, x)
			}
		}

		karatsubaThreshold = 1e9 // disable karatsuba
		t1 := measure(f)

		karatsubaThreshold = n // enable karatsuba
		t2 := measure(f)

		c := '<'
		mark := ""
		if t1 > t2 {
			c = '>'
			if threshold < 0 {
				threshold = n
				mark = " *"
			}
		}

		fmt.Printf("%4d: %8d %c %8d%s\n", n, t1, c, t2, mark)
	}
	return threshold
}


func TestCalibrate(t *testing.T) {
	if *calibrate {
		fmt.Printf("Computing Karatsuba threshold\n")
		fmt.Printf("threshold = %d\n", computeThreshold(t))
	}
}
