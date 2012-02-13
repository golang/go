// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file prints execution times for the Mul benchmark
// given different Karatsuba thresholds. The result may be
// used to manually fine-tune the threshold constant. The
// results are somewhat fragile; use repeated runs to get
// a clear picture.

// Usage: go test -run=TestCalibrate -calibrate

package big

import (
	"flag"
	"fmt"
	"testing"
	"time"
)

var calibrate = flag.Bool("calibrate", false, "run calibration test")

// measure returns the time to run f
func measure(f func()) time.Duration {
	const N = 100
	start := time.Now()
	for i := N; i > 0; i-- {
		f()
	}
	stop := time.Now()
	return stop.Sub(start) / N
}

func computeThresholds() {
	fmt.Printf("Multiplication times for varying Karatsuba thresholds\n")
	fmt.Printf("(run repeatedly for good results)\n")

	// determine Tk, the work load execution time using basic multiplication
	karatsubaThreshold = 1e9 // disable karatsuba
	Tb := measure(benchmarkMulLoad)
	fmt.Printf("Tb = %dns\n", Tb)

	// thresholds
	n := 8 // any lower values for the threshold lead to very slow multiplies
	th1 := -1
	th2 := -1

	var deltaOld time.Duration
	for count := -1; count != 0; count-- {
		// determine Tk, the work load execution time using Karatsuba multiplication
		karatsubaThreshold = n // enable karatsuba
		Tk := measure(benchmarkMulLoad)

		// improvement over Tb
		delta := (Tb - Tk) * 100 / Tb

		fmt.Printf("n = %3d  Tk = %8dns  %4d%%", n, Tk, delta)

		// determine break-even point
		if Tk < Tb && th1 < 0 {
			th1 = n
			fmt.Print("  break-even point")
		}

		// determine diminishing return
		if 0 < delta && delta < deltaOld && th2 < 0 {
			th2 = n
			fmt.Print("  diminishing return")
		}
		deltaOld = delta

		fmt.Println()

		// trigger counter
		if th1 >= 0 && th2 >= 0 && count < 0 {
			count = 20 // this many extra measurements after we got both thresholds
		}

		n++
	}
}

func TestCalibrate(t *testing.T) {
	if *calibrate {
		computeThresholds()
	}
}
