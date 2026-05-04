package executor_test

import (
	"context"
	"runtime/executor"
	"strings"
	"testing"
)

// TestNewReturnsNonNil exercises T018 from the task plan.
func TestNewReturnsNonNil(t *testing.T) {
	if got := executor.New(); got == nil {
		t.Fatal("executor.New() = nil; want non-nil *Executor")
	}
}

// TestCoNilPanics confirms the input-validation path.
func TestCoNilPanics(t *testing.T) {
	assertPanicContains(t, "nil function", func() {
		executor.New().Co(nil)
	})
}

// TestYieldOutsideTaskPanics confirms FR-013a's misuse-detection
// behavior: Yield called from a non-executor goroutine panics with
// the documented sentinel.
func TestYieldOutsideTaskPanics(t *testing.T) {
	assertPanicContains(t, "Yield called outside an executor task", func() {
		executor.Yield()
	})
}

// TestEmptyPulseReturnsNilQuickly exercises FR-015 / SC-007: a
// Pulse on an Executor with no tasks returns nil promptly.
func TestEmptyPulseReturnsNilQuickly(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("empty Pulse returned %v; want nil", err)
	}
}

// TestPulseExpiredContextReturnsErr exercises FR-003's pre-expired
// context fast path.
func TestPulseExpiredContextReturnsErr(t *testing.T) {
	ex := executor.New()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := ex.Pulse(ctx); err == nil {
		t.Fatal("Pulse with cancelled context returned nil; want non-nil")
	}
}

func assertPanicContains(t *testing.T, want string, fn func()) {
	t.Helper()
	defer func() {
		r := recover()
		if r == nil {
			t.Fatalf("expected panic containing %q, got no panic", want)
		}
		msg, ok := r.(string)
		if !ok {
			if e, isErr := r.(error); isErr {
				msg = e.Error()
			} else {
				t.Fatalf("expected panic with string or error, got %T: %v", r, r)
			}
		}
		if !strings.Contains(msg, want) {
			t.Fatalf("expected panic message to contain %q, got %q", want, msg)
		}
	}()
	fn()
}
