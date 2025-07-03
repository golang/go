// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TestCalibrate determines appropriate thresholds for when to use
// different calculation algorithms. To run it, use:
//
//	go test -run=Calibrate -calibrate >cal.log
//
// Calibration data is printed in CSV format, along with the normal test output.
// See calibrate.md for more details about using the output.

package big

import (
	"flag"
	"fmt"
	"internal/sysinfo"
	"math"
	"runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

var calibrate = flag.Bool("calibrate", false, "run calibration test")
var calibrateOnce sync.Once

func TestCalibrate(t *testing.T) {
	if !*calibrate {
		return
	}

	t.Run("KaratsubaMul", computeKaratsubaThreshold)
	t.Run("BasicSqr", computeBasicSqrThreshold)
	t.Run("KaratsubaSqr", computeKaratsubaSqrThreshold)
	t.Run("DivRecursive", computeDivRecursiveThreshold)
}

func computeKaratsubaThreshold(t *testing.T) {
	set := func(n int) { karatsubaThreshold = n }
	computeThreshold(t, "karatsuba", set, 0, 4, 200, benchMul, 200, 8, 400)
}

func benchMul(size int) func() {
	x := rndNat(size)
	y := rndNat(size)
	var z nat
	return func() {
		z.mul(nil, x, y)
	}
}

func computeBasicSqrThreshold(t *testing.T) {
	setDuringTest(t, &karatsubaSqrThreshold, 1e9)
	set := func(n int) { basicSqrThreshold = n }
	computeThreshold(t, "basicSqr", set, 2, 1, 40, benchBasicSqr, 1, 1, 40)
}

func benchBasicSqr(size int) func() {
	x := rndNat(size)
	var z nat
	return func() {
		// Run 100 squarings because 1 is too fast at the small sizes we consider.
		// Some systems don't even have precise enough clocks to measure it accurately.
		for range 100 {
			z.sqr(nil, x)
		}
	}
}

func computeKaratsubaSqrThreshold(t *testing.T) {
	set := func(n int) { karatsubaSqrThreshold = n }
	computeThreshold(t, "karatsubaSqr", set, 0, 4, 200, benchSqr, 200, 8, 400)
}

func benchSqr(size int) func() {
	x := rndNat(size)
	var z nat
	return func() {
		z.sqr(nil, x)
	}
}

func computeDivRecursiveThreshold(t *testing.T) {
	set := func(n int) { divRecursiveThreshold = n }
	computeThreshold(t, "divRecursive", set, 4, 4, 200, benchDiv, 200, 8, 400)
}

func benchDiv(size int) func() {
	divx := rndNat(2 * size)
	divy := rndNat(size)
	var z, r nat
	return func() {
		z.div(nil, r, divx, divy)
	}
}

func computeThreshold(t *testing.T, name string, set func(int), thresholdLo, thresholdStep, thresholdHi int, bench func(int) func(), sizeLo, sizeStep, sizeHi int) {
	// Start CSV output; wrapped in txtar framing to separate CSV from other test ouptut.
	fmt.Printf("-- calibrate-%s.csv --\n", name)
	defer fmt.Printf("-- eof --\n")

	fmt.Printf("goos,%s\n", runtime.GOOS)
	fmt.Printf("goarch,%s\n", runtime.GOARCH)
	fmt.Printf("cpu,%s\n", sysinfo.CPUName())
	fmt.Printf("calibrate,%s\n", name)

	// Expand lists of sizes and thresholds we will test.
	var sizes, thresholds []int
	for size := sizeLo; size <= sizeHi; size += sizeStep {
		sizes = append(sizes, size)
	}
	for thresh := thresholdLo; thresh <= thresholdHi; thresh += thresholdStep {
		thresholds = append(thresholds, thresh)
	}

	fmt.Printf("%s\n", csv("size \\ threshold", thresholds))

	// Track minimum time observed for each size, threshold pair.
	times := make([][]float64, len(sizes))
	for i := range sizes {
		times[i] = make([]float64, len(thresholds))
		for j := range thresholds {
			times[i][j] = math.Inf(+1)
		}
	}

	// For each size, run at most MaxRounds of considering every threshold.
	// If we run a threshold Stable times in a row without seeing more
	// than a 1% improvement in the observed minimum, move on to the next one.
	// After we run Converged rounds (not necessarily in a row)
	// without seeing any threshold improve by more than 1%, stop.
	const (
		MaxRounds = 1600
		Stable    = 20
		Converged = 200
	)

	for i, size := range sizes {
		b := bench(size)
		same := 0
		for range MaxRounds {
			better := false
			for j, threshold := range thresholds {
				// No point if threshold is far beyond size
				if false && threshold > size+2*sizeStep {
					continue
				}

				// BasicSqr is different from the recursive thresholds: it either applies or not,
				// without any question of recursive subproblems. Only try the thresholds
				//	size-1, size, size+1, size+2
				// to get two data points using basic multiplication and two using basic squaring.
				// This avoids gathering many redundant data points.
				// (The others have redundant data points as well, but for them the math is less trivial
				// and best not duplicated in the calibration code.)
				if false && name == "basicSqr" && (threshold < size-1 || threshold > size+3) {
					continue
				}

				set(threshold)
				b() // warm up
				b()
				tmin := times[i][j]
				for k := 0; k < Stable; k++ {
					start := time.Now()
					b()
					t := float64(time.Since(start))
					if t < tmin {
						if t < tmin*99/100 {
							better = true
							k = 0
						}
						tmin = t
					}
				}
				times[i][j] = tmin
			}
			if !better {
				if same++; same >= Converged {
					break
				}
			}
		}

		fmt.Printf("%s\n", csv(fmt.Sprint(size), times[i]))
	}

	// For each size, normalize timings by the minimum achieved for that size.
	fmt.Printf("%s\n", csv("size \\ threshold", thresholds))
	norms := make([][]float64, len(sizes))
	for i, times := range times {
		m := min(1e100, slices.Min(times)) // make finite so divide preserves inf values
		norms[i] = make([]float64, len(times))
		for j, d := range times {
			norms[i][j] = d / m
		}
		fmt.Printf("%s\n", csv(fmt.Sprint(sizes[i]), norms[i]))
	}

	// For each threshold, compute geomean of normalized timings across all sizes.
	geomeans := make([]float64, len(thresholds))
	for j := range thresholds {
		p := 1.0
		n := 0
		for i := range sizes {
			if v := norms[i][j]; !math.IsInf(v, +1) {
				p *= v
				n++
			}
		}
		if n == 0 {
			geomeans[j] = math.Inf(+1)
		} else {
			geomeans[j] = math.Pow(p, 1/float64(n))
		}
	}
	fmt.Printf("%s\n", csv("geomean", geomeans))

	// Add best threshold and smallest, largest within 10% and 5% of best.
	var lo10, lo5, best, hi5, hi10 int
	for i, g := range geomeans {
		if g < geomeans[best] {
			best = i
		}
	}
	lo5 = best
	for lo5 > 0 && geomeans[lo5-1] <= 1.05 {
		lo5--
	}
	lo10 = lo5
	for lo10 > 0 && geomeans[lo10-1] <= 1.10 {
		lo10--
	}
	hi5 = best
	for hi5+1 < len(geomeans) && geomeans[hi5+1] <= 1.05 {
		hi5++
	}
	hi10 = hi5
	for hi10+1 < len(geomeans) && geomeans[hi10+1] <= 1.10 {
		hi10++
	}
	fmt.Printf("lo10%%,%d\n", thresholds[lo10])
	fmt.Printf("lo5%%,%d\n", thresholds[lo5])
	fmt.Printf("min,%d\n", thresholds[best])
	fmt.Printf("hi5%%,%d\n", thresholds[hi5])
	fmt.Printf("hi10%%,%d\n", thresholds[hi10])

	set(thresholds[best])
}

// csv returns a single csv line starting with name and followed by the values.
// Values that are float64 +infinity, denoting missing data, are replaced by an empty string.
func csv[T int | float64](name string, values []T) string {
	line := []string{name}
	for _, v := range values {
		if math.IsInf(float64(v), +1) {
			line = append(line, "")
		} else {
			line = append(line, fmt.Sprint(v))
		}
	}
	return strings.Join(line, ",")
}
