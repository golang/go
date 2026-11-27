// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testing

import (
	"bytes"
	"strings"
	"sync/atomic"
)

// TestRetryBasic tests that a test calling Retry is re-run and succeeds
// on the second attempt.
func TestRetryBasic(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		n := attempt.Add(1)
		if n == 1 {
			t.Retry("transient failure")
		}
	})
	<-t1.signal

	if t1.Failed() {
		t.Errorf("test should have passed after retry, but failed")
	}
	if attempt.Load() != 2 {
		t.Errorf("expected 2 attempts, got %d", attempt.Load())
	}
}

// TestRetryAllAttemptsFail tests that when all retry attempts fail,
// the test is marked as permanently failed.
func TestRetryAllAttemptsFail(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		attempt.Add(1)
		t.Retry("always fails")
	})
	<-t1.signal

	if !t1.Failed() {
		t.Errorf("test should have failed after exhausting retries")
	}
	if attempt.Load() != 2 {
		t.Errorf("expected 2 attempts (1 original + 1 retry), got %d", attempt.Load())
	}
}

// TestRetriesCustomCount tests that Retries(n) sets a custom retry count.
func TestRetriesCustomCount(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		t.Retries(3)
		n := attempt.Add(1)
		if n <= 3 {
			t.Retry("transient failure")
		}
		// Succeeds on 4th attempt
	})
	<-t1.signal

	if t1.Failed() {
		t.Errorf("test should have passed after 3 retries, but failed")
	}
	if attempt.Load() != 4 {
		t.Errorf("expected 4 attempts (1 original + 3 retries), got %d", attempt.Load())
	}
}

// TestRetriesExhausted tests that when Retries(n) is set and all n retries
// fail, the test is permanently failed.
func TestRetriesExhausted(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		t.Retries(2)
		attempt.Add(1)
		t.Retry("persistent failure")
	})
	<-t1.signal

	if !t1.Failed() {
		t.Errorf("test should have failed after exhausting all retries")
	}
	if attempt.Load() != 3 {
		t.Errorf("expected 3 attempts (1 original + 2 retries), got %d", attempt.Load())
	}
}

// TestRetrySubtest tests that only the failing subtest is retried,
// not the parent test.
func TestRetrySubtest(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var parentRuns atomic.Int32
	var subtestRuns atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		parentRuns.Add(1)
		t.Run("flaky", func(t *T) {
			n := subtestRuns.Add(1)
			if n == 1 {
				t.Retry("flaky subtest")
			}
		})
	})
	<-t1.signal

	if parentRuns.Load() != 1 {
		t.Errorf("parent should run once, ran %d times", parentRuns.Load())
	}
	if subtestRuns.Load() != 2 {
		t.Errorf("subtest should run twice, ran %d times", subtestRuns.Load())
	}
	if t1.Failed() {
		t.Errorf("parent test should pass after subtest retry")
	}
}

// TestRetryWithChatty tests that the retry event is emitted in chatty mode.
func TestRetryWithChatty(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
			chatty: &chattyPrinter{w: &buf},
			name:   "TestFlaky",
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		n := attempt.Add(1)
		if n == 1 {
			t.Retry("network timeout")
		}
	})
	<-t1.signal

	output := buf.String()
	if !strings.Contains(output, "=== RETRY TestFlaky") {
		t.Errorf("expected RETRY event in output, got: %s", output)
	}
	if !strings.Contains(output, "retry reason: network timeout") {
		t.Errorf("expected retry reason in output, got: %s", output)
	}
}

// TestRetryDisabledByZeroRetries tests that Retries(0) explicitly disables retries.
func TestRetryDisabledByZeroRetries(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		t.Retries(0) // Explicitly disable retries
		attempt.Add(1)
		t.Retry("should not retry")
	})
	<-t1.signal

	if !t1.Failed() {
		t.Errorf("test should have failed (retries disabled with 0)")
	}
	if attempt.Load() != 1 {
		t.Errorf("expected 1 attempt (no retries), got %d", attempt.Load())
	}
}

// TestRetryDisabledByNegativeRetries tests that Retries(-1) disables retries.
func TestRetryDisabledByNegativeRetries(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		t.Retries(-1) // Explicitly disable retries
		attempt.Add(1)
		t.Retry("should not retry")
	})
	<-t1.signal

	if !t1.Failed() {
		t.Errorf("test should have failed (retries disabled)")
	}
	if attempt.Load() != 1 {
		t.Errorf("expected 1 attempt (no retries), got %d", attempt.Load())
	}
}

// TestRetryNoRetryWithoutCall tests that a test that does not call Retry
// is not retried even if Retries was called.
func TestRetryNoRetryWithoutCall(t *T) {
	tstate := newTestState(1, allMatcher())
	var buf bytes.Buffer
	var attempt atomic.Int32

	t1 := &T{
		common: common{
			signal: make(chan bool, 1),
			w:      &buf,
		},
		tstate: tstate,
	}

	tRunner(t1, func(t *T) {
		t.Retries(5)
		attempt.Add(1)
		// Don't call Retry — test passes normally
	})
	<-t1.signal

	if t1.Failed() {
		t.Errorf("test should have passed")
	}
	if attempt.Load() != 1 {
		t.Errorf("expected 1 attempt (no retry called), got %d", attempt.Load())
	}
}
