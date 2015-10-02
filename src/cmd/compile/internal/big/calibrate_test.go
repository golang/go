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

func karatsubaLoad(b *testing.B) {
	BenchmarkMul(b)
}

// measureKaratsuba returns the time to run a Karatsuba-relevant benchmark
// given Karatsuba threshold th.
func measureKaratsuba(th int) time.Duration {
	th, karatsubaThreshold = karatsubaThreshold, th
	res := testing.Benchmark(karatsubaLoad)
	karatsubaThreshold = th
	return time.Duration(res.NsPerOp())
}

func computeThresholds() {
	fmt.Printf("Multiplication times for varying Karatsuba thresholds\n")
	fmt.Printf("(run repeatedly for good results)\n")

	// determine Tk, the work load execution time using basic multiplication
	Tb := measureKaratsuba(1e9) // th == 1e9 => Karatsuba multiplication disabled
	fmt.Printf("Tb = %10s\n", Tb)

	// thresholds
	th := 4
	th1 := -1
	th2 := -1

	var deltaOld time.Duration
	for count := -1; count != 0 && th < 128; count-- {
		// determine Tk, the work load execution time using Karatsuba multiplication
		Tk := measureKaratsuba(th)

		// improvement over Tb
		delta := (Tb - Tk) * 100 / Tb

		fmt.Printf("th = %3d  Tk = %10s  %4d%%", th, Tk, delta)

		// determine break-even point
		if Tk < Tb && th1 < 0 {
			th1 = th
			fmt.Print("  break-even point")
		}

		// determine diminishing return
		if 0 < delta && delta < deltaOld && th2 < 0 {
			th2 = th
			fmt.Print("  diminishing return")
		}
		deltaOld = delta

		fmt.Println()

		// trigger counter
		if th1 >= 0 && th2 >= 0 && count < 0 {
			count = 10 // this many extra measurements after we got both thresholds
		}

		th++
	}
}

func TestCalibrate(t *testing.T) {
	if *calibrate {
		computeThresholds()
	}
}
