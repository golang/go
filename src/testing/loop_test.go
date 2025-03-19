// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

func TestBenchmarkBLoop(t *T) {
	var initialStart highPrecisionTime
	var firstStart highPrecisionTime
	var scaledStart highPrecisionTime
	var runningEnd bool
	runs := 0
	iters := 0
	firstBN := 0
	restBN := 0
	finalBN := 0
	bRet := Benchmark(func(b *B) {
		initialStart = b.start
		runs++
		for b.Loop() {
			if iters == 0 {
				firstStart = b.start
				firstBN = b.N
			} else {
				restBN = max(restBN, b.N)
			}
			if iters == 1 {
				scaledStart = b.start
			}
			iters++
		}
		finalBN = b.N
		runningEnd = b.timerOn
	})
	// Verify that a b.Loop benchmark is invoked just once.
	if runs != 1 {
		t.Errorf("want runs == 1, got %d", runs)
	}
	// Verify that at least one iteration ran.
	if iters == 0 {
		t.Fatalf("no iterations ran")
	}
	// Verify that b.N, bRet.N, and the b.Loop() iteration count match.
	if finalBN != iters || bRet.N != iters {
		t.Errorf("benchmark iterations mismatch: %d loop iterations, final b.N=%d, bRet.N=%d", iters, finalBN, bRet.N)
	}
	// Verify that b.N was 0 inside the loop
	if firstBN != 0 {
		t.Errorf("want b.N == 0 on first iteration, got %d", firstBN)
	}
	if restBN != 0 {
		t.Errorf("want b.N == 0 on subsequent iterations, got %d", restBN)
	}
	// Make sure the benchmark ran for an appropriate amount of time.
	if bRet.T < benchTime.d {
		t.Fatalf("benchmark ran for %s, want >= %s", bRet.T, benchTime.d)
	}
	// Verify that the timer is reset on the first loop, and then left alone.
	if firstStart == initialStart {
		t.Errorf("b.Loop did not reset the timer")
	}
	if scaledStart != firstStart {
		t.Errorf("b.Loop stops and restarts the timer during iteration")
	}
	// Verify that it stopped the timer after the last loop.
	if runningEnd {
		t.Errorf("timer was still running after last iteration")
	}
}

// See also TestBenchmarkBLoop* in other files.
