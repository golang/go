// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package traceparser

import (
	"math"
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
	util := []MutatorUtil{
		{0e9, 1},
		{1e9, 0},
		{2e9, 1},
		{3e9, 0},
		{4e9, 1},
		{5e9, 0},
	}
	mmuCurve := NewMMUCurve(util)

	for _, test := range []struct {
		window time.Duration
		want   float64
	}{
		{0, 0},
		{time.Millisecond, 0},
		{time.Second, 0},
		{2 * time.Second, 0.5},
		{3 * time.Second, 1 / 3.0},
		{4 * time.Second, 0.5},
		{5 * time.Second, 3 / 5.0},
		{6 * time.Second, 3 / 5.0},
	} {
		if got := mmuCurve.MMU(test.window); !aeq(test.want, got) {
			t.Errorf("for %s window, want mu = %f, got %f", test.window, test.want, got)
		}
	}
}

func TestMMUTrace(t *testing.T) {
	t.Parallel()

	p, err := New("../trace/testdata/stress_1_10_good")
	if err != nil {
		t.Fatalf("failed to read input file: %v", err)
	}
	if err := p.Parse(0, 1<<62, nil); err != nil {
		t.Fatalf("failed to parse trace: %s", err)
	}
	mu := p.MutatorUtilization()
	mmuCurve := NewMMUCurve(mu)

	// Test the optimized implementation against the "obviously
	// correct" implementation.
	for window := time.Nanosecond; window < 10*time.Second; window *= 10 {
		want := mmuSlow(mu, window)
		got := mmuCurve.MMU(window)
		if !aeq(want, got) {
			t.Errorf("want %f, got %f mutator utilization in window %s", want, got, window)
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
	mu := p.MutatorUtilization()
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
