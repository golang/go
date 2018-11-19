// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"math"
	"runtime"
	"testing"
	"time"
)

// aeq returns true if x and y are equal up to 8 digits (1 part in 100
// million).
func aeq(x, y float64) bool {
	if x < 0 && y < 0 {
		x, y = -x, -y
	}
	const digits = 8
	factor := 1 - math.Pow(10, -digits+1)
	return x*factor <= y && y*factor <= x
}

func TestMMU(t *testing.T) {
	t.Parallel()

	// MU
	// 1.0  *****   *****   *****
	// 0.5      *   *   *   *
	// 0.0      *****   *****
	//      0   1   2   3   4   5
	util := [][]MutatorUtil{{
		{0e9, 1},
		{1e9, 0},
		{2e9, 1},
		{3e9, 0},
		{4e9, 1},
		{5e9, 0},
	}}
	mmuCurve := NewMMUCurve(util)

	for _, test := range []struct {
		window time.Duration
		want   float64
		worst  []float64
	}{
		{0, 0, []float64{}},
		{time.Millisecond, 0, []float64{0, 0}},
		{time.Second, 0, []float64{0, 0}},
		{2 * time.Second, 0.5, []float64{0.5, 0.5}},
		{3 * time.Second, 1 / 3.0, []float64{1 / 3.0}},
		{4 * time.Second, 0.5, []float64{0.5}},
		{5 * time.Second, 3 / 5.0, []float64{3 / 5.0}},
		{6 * time.Second, 3 / 5.0, []float64{3 / 5.0}},
	} {
		if got := mmuCurve.MMU(test.window); !aeq(test.want, got) {
			t.Errorf("for %s window, want mu = %f, got %f", test.window, test.want, got)
		}
		worst := mmuCurve.Examples(test.window, 2)
		// Which exact windows are returned is unspecified
		// (and depends on the exact banding), so we just
		// check that we got the right number with the right
		// utilizations.
		if len(worst) != len(test.worst) {
			t.Errorf("for %s window, want worst %v, got %v", test.window, test.worst, worst)
		} else {
			for i := range worst {
				if worst[i].MutatorUtil != test.worst[i] {
					t.Errorf("for %s window, want worst %v, got %v", test.window, test.worst, worst)
					break
				}
			}
		}
	}
}

func TestMMUTrace(t *testing.T) {
	if runtime.GOOS == "darwin" && (runtime.GOARCH == "arm" || runtime.GOARCH == "arm64") {
		t.Skipf("files from outside the package are not available on %s/%s", runtime.GOOS, runtime.GOARCH)
	}
	// Can't be t.Parallel() because it modifies the
	// testingOneBand package variable.

	p, err := New("../trace/testdata/stress_1_10_good")
	if err != nil {
		t.Fatalf("failed to read input file: %v", err)
	}
	if err := p.Parse(0, 1<<62, nil); err != nil {
		t.Fatalf("failed to parse trace: %s", err)
	}
	mu := p.MutatorUtilization(UtilSTW | UtilBackground | UtilAssist)
	mmuCurve := NewMMUCurve(mu)

	// Test the optimized implementation against the "obviously
	// correct" implementation.
	for window := time.Nanosecond; window < 10*time.Second; window *= 10 {
		want := mmuSlow(mu[0], window)
		got := mmuCurve.MMU(window)
		if !aeq(want, got) {
			t.Errorf("want %f, got %f mutator utilization in window %s", want, got, window)
		}
	}

	// Test MUD with band optimization against MUD without band
	// optimization. We don't have a simple testing implementation
	// of MUDs (the simplest implementation is still quite
	// complex), but this is still a pretty good test.
	defer func(old int) { bandsPerSeries = old }(bandsPerSeries)
	bandsPerSeries = 1
	mmuCurve2 := NewMMUCurve(mu)
	quantiles := []float64{0, 1 - .999, 1 - .99}
	for window := time.Microsecond; window < time.Second; window *= 10 {
		mud1 := mmuCurve.MUD(window, quantiles)
		mud2 := mmuCurve2.MUD(window, quantiles)
		for i := range mud1 {
			if !aeq(mud1[i], mud2[i]) {
				t.Errorf("for quantiles %v at window %v, want %v, got %v", quantiles, window, mud2, mud1)
				break
			}
		}
	}
}

func BenchmarkMMU(b *testing.B) {
	p, err := New("../trace/testdata/stress_1_10_good")
	if err != nil {
		b.Fatalf("failed to read input file: %v", err)
	}
	if err := p.Parse(0, 1<<62, nil); err != nil {
		b.Fatalf("failed to parse trace: %s", err)
	}
	mu := p.MutatorUtilization(UtilSTW | UtilBackground | UtilAssist | UtilSweep)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		mmuCurve := NewMMUCurve(mu)
		xMin, xMax := time.Microsecond, time.Second
		logMin, logMax := math.Log(float64(xMin)), math.Log(float64(xMax))
		const samples = 100
		for i := 0; i < samples; i++ {
			window := time.Duration(math.Exp(float64(i)/(samples-1)*(logMax-logMin) + logMin))
			mmuCurve.MMU(window)
		}
	}
}

func mmuSlow(util []MutatorUtil, window time.Duration) (mmu float64) {
	if max := time.Duration(util[len(util)-1].Time - util[0].Time); window > max {
		window = max
	}

	mmu = 1.0

	// muInWindow returns the mean mutator utilization between
	// util[0].Time and end.
	muInWindow := func(util []MutatorUtil, end int64) float64 {
		total := 0.0
		var prevU MutatorUtil
		for _, u := range util {
			if u.Time > end {
				total += prevU.Util * float64(end-prevU.Time)
				break
			}
			total += prevU.Util * float64(u.Time-prevU.Time)
			prevU = u
		}
		return total / float64(end-util[0].Time)
	}
	update := func() {
		for i, u := range util {
			if u.Time+int64(window) > util[len(util)-1].Time {
				break
			}
			mmu = math.Min(mmu, muInWindow(util[i:], u.Time+int64(window)))
		}
	}

	// Consider all left-aligned windows.
	update()
	// Reverse the trace. Slightly subtle because each MutatorUtil
	// is a *change*.
	rutil := make([]MutatorUtil, len(util))
	if util[len(util)-1].Util != 0 {
		panic("irreversible trace")
	}
	for i, u := range util {
		util1 := 0.0
		if i != 0 {
			util1 = util[i-1].Util
		}
		rutil[len(rutil)-i-1] = MutatorUtil{Time: -u.Time, Util: util1}
	}
	util = rutil
	// Consider all right-aligned windows.
	update()
	return
}
