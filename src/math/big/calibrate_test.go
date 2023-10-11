// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Calibration used to determine thresholds for using
// different algorithms.  Ideally, this would be converted
// to go generate to create thresholds.go

// This file prints execution times for the Mul benchmark
// given different Karatsuba thresholds. The result may be
// used to manually fine-tune the threshold constant. The
// results are somewhat fragile; use repeated runs to get
// a clear picture.

// Calculates lower and upper thresholds for when basicSqr
// is faster than standard multiplication.

// Usage: go test -run='^TestCalibrate$' -v -calibrate

package big

import (
	"flag"
	"fmt"
	"testing"
	"time"
)

var calibrate = flag.Bool("calibrate", false, "run calibration test")

const (
	sqrModeMul       = "mul(x, x)"
	sqrModeBasic     = "basicSqr(x)"
	sqrModeKaratsuba = "karatsubaSqr(x)"
)

func TestCalibrate(t *testing.T) {
	if !*calibrate {
		return
	}

	computeKaratsubaThresholds()

	// compute basicSqrThreshold where overhead becomes negligible
	minSqr := computeSqrThreshold(10, 30, 1, 3, sqrModeMul, sqrModeBasic)
	// compute karatsubaSqrThreshold where karatsuba is faster
	maxSqr := computeSqrThreshold(200, 500, 10, 3, sqrModeBasic, sqrModeKaratsuba)
	if minSqr != 0 {
		fmt.Printf("found basicSqrThreshold = %d\n", minSqr)
	} else {
		fmt.Println("no basicSqrThreshold found")
	}
	if maxSqr != 0 {
		fmt.Printf("found karatsubaSqrThreshold = %d\n", maxSqr)
	} else {
		fmt.Println("no karatsubaSqrThreshold found")
	}
}

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

func computeKaratsubaThresholds() {
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

func measureSqr(words, nruns int, mode string) time.Duration {
	// more runs for better statistics
	initBasicSqr, initKaratsubaSqr := basicSqrThreshold, karatsubaSqrThreshold

	switch mode {
	case sqrModeMul:
		basicSqrThreshold = words + 1
	case sqrModeBasic:
		basicSqrThreshold, karatsubaSqrThreshold = words-1, words+1
	case sqrModeKaratsuba:
		karatsubaSqrThreshold = words - 1
	}

	var testval int64
	for i := 0; i < nruns; i++ {
		res := testing.Benchmark(func(b *testing.B) { benchmarkNatSqr(b, words) })
		testval += res.NsPerOp()
	}
	testval /= int64(nruns)

	basicSqrThreshold, karatsubaSqrThreshold = initBasicSqr, initKaratsubaSqr

	return time.Duration(testval)
}

func computeSqrThreshold(from, to, step, nruns int, lower, upper string) int {
	fmt.Printf("Calibrating threshold between %s and %s\n", lower, upper)
	fmt.Printf("Looking for a timing difference for x between %d - %d words by %d step\n", from, to, step)
	var initPos bool
	var threshold int
	for i := from; i <= to; i += step {
		baseline := measureSqr(i, nruns, lower)
		testval := measureSqr(i, nruns, upper)
		pos := baseline > testval
		delta := baseline - testval
		percent := delta * 100 / baseline
		fmt.Printf("words = %3d deltaT = %10s (%4d%%) is %s better: %v", i, delta, percent, upper, pos)
		if i == from {
			initPos = pos
		}
		if threshold == 0 && pos != initPos {
			threshold = i
			fmt.Printf("  threshold  found")
		}
		fmt.Println()

	}
	if threshold != 0 {
		fmt.Printf("Found threshold = %d between %d - %d\n", threshold, from, to)
	} else {
		fmt.Printf("Found NO threshold between %d - %d\n", from, to)
	}
	return threshold
}
