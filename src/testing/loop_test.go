// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"strings"
	"time"
)

// See also TestBenchmarkBLoop* in other files.

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

func TestBenchmarkBLoopCheapEarlyTerminate(t *T) {
	if Short() {
		t.Skip("B.Loop test needs to run for > 1s to saturate 1e9 iterations")
	}
	runCnt := 0
	// Set the benchmark time high enough that we're likely to hit the 1B
	// iteration limit even on very slow hardware.
	// (on an AMD Ryzen 5900X, this benchmark runs in just over a second)
	//
	// Notably, the assertions below shouldn't fail if a test-run is slow
	// enough that it doesn't saturate the limit.
	const maxBenchTime = time.Second * 30
	res := Benchmark(func(b *B) {
		// Set the benchmark time _much_ higher than required to hit 1e9 iterations.
		b.benchTime.d = maxBenchTime
		for b.Loop() {
			runCnt++
		}
	})
	if runCnt > maxBenchPredictIters {
		t.Errorf("loop body ran more than max (%d) times: %d", maxBenchPredictIters, runCnt)
		if res.T >= maxBenchTime {
			t.Logf("cheap benchmark exhausted time budget: %s; ran for %s", maxBenchTime, res.T)
		}
	}

	if res.N != runCnt {
		t.Errorf("disagreeing loop counts: res.N reported %d, while b.Loop() iterated %d times", res.N, runCnt)
	}

	if res.N > maxBenchPredictIters {
		t.Errorf("benchmark result claims more runs than max (%d) times: %d", maxBenchPredictIters, res.N)
	}

}

func TestBenchmarkBLoopBreak(t *T) {
	var bState *B
	var bLog bytes.Buffer
	bRet := Benchmark(func(b *B) {
		// The Benchmark function provides no access to the failure state and
		// discards the log, so capture the B and save its log.
		bState = b
		b.common.w = &bLog

		for i := 0; b.Loop(); i++ {
			if i == 2 {
				break
			}
		}
	})
	if !bState.failed {
		t.Errorf("benchmark should have failed")
	}
	const wantLog = "benchmark function returned without B.Loop"
	if log := bLog.String(); !strings.Contains(log, wantLog) {
		t.Errorf("missing error %q in output:\n%s", wantLog, log)
	}
	// A benchmark that exits early should not report its target iteration count
	// because it's not meaningful.
	if bRet.N != 0 {
		t.Errorf("want N == 0, got %d", bRet.N)
	}
}

func TestBenchmarkBLoopError(t *T) {
	// Test that a benchmark that exits early because of an error doesn't *also*
	// complain that the benchmark exited early.
	var bState *B
	var bLog bytes.Buffer
	bRet := Benchmark(func(b *B) {
		bState = b
		b.common.w = &bLog

		for i := 0; b.Loop(); i++ {
			b.Error("error")
			return
		}
	})
	if !bState.failed {
		t.Errorf("benchmark should have failed")
	}
	const noWantLog = "benchmark function returned without B.Loop"
	if log := bLog.String(); strings.Contains(log, noWantLog) {
		t.Errorf("unexpected error %q in output:\n%s", noWantLog, log)
	}
	if bRet.N != 0 {
		t.Errorf("want N == 0, got %d", bRet.N)
	}
}

func TestBenchmarkBLoopStop(t *T) {
	var bState *B
	var bLog bytes.Buffer
	bRet := Benchmark(func(b *B) {
		bState = b
		b.common.w = &bLog

		for i := 0; b.Loop(); i++ {
			b.StopTimer()
		}
	})
	if !bState.failed {
		t.Errorf("benchmark should have failed")
	}
	const wantLog = "B.Loop called with timer stopped"
	if log := bLog.String(); !strings.Contains(log, wantLog) {
		t.Errorf("missing error %q in output:\n%s", wantLog, log)
	}
	if bRet.N != 0 {
		t.Errorf("want N == 0, got %d", bRet.N)
	}
}
