// Behavioral tests for runtime/executor, organized by user story
// (T019–T023 for US1, T030–T036 for US2, T043–T047 for US3,
// T051–T053 for US4).

package executor_test

import (
	"context"
	"runtime"
	"runtime/executor"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// =====================================================================
// US1: Run a cooperative task to completion on the calling thread (P1).
// =====================================================================

// T019: owner-thread inline Co — the function runs on the calling
// goroutine's OS thread. We establish ownership with a no-op Pulse,
// then submit a task and observe its thread identity.
func TestUS1_OwnerThreadCoRunsInline(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ex := executor.New()
	// First Pulse establishes the owner thread (with the goroutine
	// pinned for stable cross-Pulse ownership).
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("initial empty Pulse: %v", err)
	}

	caller := currentTID()
	var observed uintptr
	var ran bool

	ex.Co(func() {
		observed = currentTID()
		ran = true
	})

	if !ran {
		t.Fatal("Co did not run inline; ran=false after Co returned")
	}
	if observed != caller {
		t.Fatalf("task observed thread id %x; want %x (Co should run inline on caller's thread)", observed, caller)
	}
}

// T020: NumGoroutine invariant — spawning many non-suspending tasks
// does not change runtime.NumGoroutine().
func TestUS1_NumGoroutineUnchanged(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	assertNumGoroutineUnchanged(t, func() {
		for i := 0; i < 1000; i++ {
			ex.Co(func() {})
		}
	})
}

// T021: cross-thread Co does not run on the producer's thread; it
// runs on the owner thread at the next Pulse.
func TestUS1_CrossThreadCoEnqueues(t *testing.T) {
	ex := executor.New()
	// Establish ownership on this goroutine's thread.
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ownerTID := currentTID()

	var observed uintptr
	var ran atomic.Bool

	// Producer goroutine on a different OS thread.
	done := make(chan struct{})
	go func() {
		runtime.LockOSThread()
		defer runtime.UnlockOSThread()
		producerTID := currentTID()
		if producerTID == ownerTID {
			// LockOSThread can occasionally produce the same OS
			// thread on small machines; if so, skip the producer-side
			// assertion and rely on the post-Pulse check.
		}
		ex.Co(func() {
			observed = currentTID()
			ran.Store(true)
		})
		close(done)
	}()
	<-done

	if ran.Load() {
		t.Fatal("cross-thread Co ran on producer's thread; want enqueued for next Pulse")
	}

	// Drive the task by Pulsing on the owner thread.
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse drive: %v", err)
	}

	if !ran.Load() {
		t.Fatal("task did not run after Pulse; want it to have been picked up from submitQ")
	}
	if observed != ownerTID {
		t.Fatalf("task ran on thread %x; want owner thread %x", observed, ownerTID)
	}
}

// T022: concurrent cross-thread Co — N producers, N tasks, each
// runs exactly once.
func TestUS1_ConcurrentCrossThreadCo(t *testing.T) {
	const N = 64
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var ran atomic.Int64
	var wg sync.WaitGroup
	wg.Add(N)
	for i := 0; i < N; i++ {
		go func() {
			defer wg.Done()
			ex.Co(func() { ran.Add(1) })
		}()
	}
	wg.Wait()

	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse drive: %v", err)
	}
	if got := ran.Load(); got != N {
		t.Fatalf("ran = %d; want %d", got, N)
	}
}

// T023: re-entrant Co inside a task runs inline.
func TestUS1_NestedCoInsideTask(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var order []string
	ex.Co(func() {
		order = append(order, "outer-pre")
		ex.Co(func() {
			order = append(order, "inner")
		})
		order = append(order, "outer-post")
	})

	want := []string{"outer-pre", "inner", "outer-post"}
	if len(order) != 3 {
		t.Fatalf("order = %v; want %v", order, want)
	}
	for i, v := range want {
		if order[i] != v {
			t.Fatalf("order = %v; want %v", order, want)
		}
	}
}

// =====================================================================
// US2: Suspend and later resume via Pulse + Yield + panic propagation.
// =====================================================================

// T031: Yield round-trip — N yields require N+1 Pulse slices.
func TestUS2_YieldRoundTrip(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	const N = 5
	var ticks int
	ex.Co(func() {
		for i := 0; i < N; i++ {
			ticks++
			executor.Yield()
		}
		ticks++ // final increment after last yield
	})

	// After Co (which establishes the task and runs until first
	// Yield), ticks should be 1 (initial increment before Yield #0).
	if ticks != 1 {
		t.Fatalf("after Co, ticks = %d; want 1 (one increment before first Yield)", ticks)
	}
	for p := 0; p < N; p++ {
		if err := ex.Pulse(context.Background()); err != nil {
			t.Fatalf("Pulse %d: %v", p, err)
		}
	}
	if ticks != N+1 {
		t.Fatalf("ticks = %d after %d Pulses; want %d", ticks, N, N+1)
	}
}

// T030: channel suspension — task <-chs, send wakes it on next Pulse.
func TestUS2_ChannelSuspension(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	ch := make(chan int, 1)
	var got int
	ex.Co(func() {
		got = <-ch
	})

	// Task is parked on chanrecv; haven't sent yet.
	if got != 0 {
		t.Fatalf("task ran past <-ch without a send; got=%d", got)
	}

	// Send from another goroutine — this calls goready, which routes
	// through readyHook into our wakeQ.
	go func() { ch <- 42 }()

	// Give the sender goroutine a moment to actually fire goready.
	// We don't strictly need this — the wakeQ drain is observed by
	// Pulse — but it makes the test more deterministic.
	time.Sleep(10 * time.Millisecond)

	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse drive: %v", err)
	}
	if got != 42 {
		t.Fatalf("got = %d; want 42 (task should have completed receive in Pulse)", got)
	}
}

// T032: multiple resumable tasks in one Pulse.
func TestUS2_MultipleResumableInOnePulse(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	const K = 10
	ready := make(chan struct{})
	var advanced atomic.Int64

	for i := 0; i < K; i++ {
		ex.Co(func() {
			<-ready
			advanced.Add(1)
		})
	}

	close(ready)
	time.Sleep(10 * time.Millisecond)

	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	if got := advanced.Load(); got != K {
		t.Fatalf("advanced = %d; want %d (one Pulse should drain all resumable tasks)", got, K)
	}
}

// T035: re-entrant Pulse panics. The panic only fires when the
// task is being driven by a Pulse — i.e., the task must first park,
// then be resumed under Pulse where it calls Pulse on the same
// Executor.
func TestUS2_ReentrantPulsePanics(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}

	wake := make(chan struct{})
	var caught any
	ex.Co(func() {
		<-wake // park; resumed by Pulse below
		defer func() { caught = recover() }()
		_ = ex.Pulse(context.Background())
	})
	close(wake)
	time.Sleep(10 * time.Millisecond)

	// Driving Pulse: it switches into the task, the task calls Pulse,
	// and that inner Pulse must panic. The panic propagates back
	// through the task's deferred recover (which captures it).
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("driving Pulse: %v", err)
	}

	if caught == nil {
		t.Fatal("re-entrant Pulse did not panic")
	}
	msg, _ := caught.(string)
	if !strings.Contains(msg, "re-entrant Pulse") {
		t.Fatalf("panic message = %q; want it to contain %q", msg, "re-entrant Pulse")
	}
}

// T036: panic propagation — task panic inside Pulse-driven slice
// surfaces to the Pulse caller; Executor remains coherent so the
// next Pulse can drive other tasks.
func TestUS2_PanicPropagatesAndExecutorRecovers(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	// Task A: panics on first slice (during Co inline run).
	var caughtA any
	func() {
		defer func() { caughtA = recover() }()
		ex.Co(func() {
			panic("boom-A")
		})
	}()
	if caughtA == nil {
		t.Fatal("inline Co panic did not propagate to Co caller")
	}
	if got, _ := caughtA.(string); got != "boom-A" {
		t.Fatalf("caught = %v; want %q", caughtA, "boom-A")
	}

	// Executor should still be usable: submit a non-panicking task
	// from another goroutine and drive it via Pulse.
	var ranB atomic.Bool
	go func() {
		ex.Co(func() { ranB.Store(true) })
	}()
	time.Sleep(10 * time.Millisecond)

	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("post-panic Pulse: %v", err)
	}
	if !ranB.Load() {
		t.Fatal("post-panic task B did not run; executor state corrupted by panic")
	}
}

// =====================================================================
// US3: Honor context cancellation/timeout in Pulse.
// =====================================================================

// T044: pre-cancelled context returns immediately.
func TestUS3_PreCancelledCtxReturnsErr(t *testing.T) {
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	var ran atomic.Bool
	go func() { ex.Co(func() { ran.Store(true) }) }()
	time.Sleep(10 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := ex.Pulse(ctx)
	if err != context.Canceled {
		t.Fatalf("Pulse returned %v; want context.Canceled", err)
	}
	if ran.Load() {
		t.Fatal("task ran despite pre-cancelled context")
	}
}

// T045: incremental progress across Pulses — each task yields a
// fixed number of times; we Pulse until all tasks complete.
func TestUS3_IncrementalProgressViaYield(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}

	const M = 10
	const yieldsPer = 4
	var done atomic.Int64
	for i := 0; i < M; i++ {
		ex.Co(func() {
			for k := 0; k < yieldsPer; k++ {
				executor.Yield()
			}
			done.Add(1)
		})
	}

	// Drive Pulses until everyone is done. With yieldsPer yields per
	// task, each task takes yieldsPer+1 Pulses (the +1 is the inline
	// Co that ran the body up to the first Yield). Cap iterations at
	// 4× the expected count to detect runaway loops.
	maxPulses := (yieldsPer + 1) * 4
	for p := 0; done.Load() < M && p < maxPulses; p++ {
		if err := ex.Pulse(context.Background()); err != nil {
			t.Fatalf("Pulse %d: %v", p, err)
		}
	}
	if got := done.Load(); got != M {
		t.Fatalf("done = %d after %d Pulses; want %d", got, maxPulses, M)
	}
}

// T043 / FR-008: deadline mid-Pulse — context expiry during a Pulse
// causes ctx.Err() to be returned, and remaining work is preserved.
// We construct a workload where each slice is artificially slow so
// that a sub-millisecond deadline reliably fires mid-Pulse.
func TestUS3_DeadlineMidPulse(t *testing.T) {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	ex := executor.New()
	if err := ex.Pulse(context.Background()); err != nil {
		t.Fatalf("Pulse: %v", err)
	}

	const M = 50
	var done atomic.Int64
	for i := 0; i < M; i++ {
		ex.Co(func() {
			executor.Yield()
			// Burn enough time per slice that 50 of these can't fit
			// in a 1-millisecond budget on any reasonable machine.
			start := time.Now()
			for time.Since(start) < 100*time.Microsecond {
			}
			done.Add(1)
		})
	}

	// First Pulse with a sub-millisecond deadline — must return
	// DeadlineExceeded with some (but not all) tasks done.
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Millisecond)
	err := ex.Pulse(ctx)
	cancel()
	if err != context.DeadlineExceeded {
		t.Fatalf("first Pulse with 1ms deadline returned %v; want context.DeadlineExceeded", err)
	}
	if doneSnapshot := done.Load(); doneSnapshot >= M {
		t.Fatalf("first deadline-bounded Pulse completed all %d tasks (snapshot=%d); test workload too cheap", M, doneSnapshot)
	}

	// Now finish with a generous deadline.
	for done.Load() < M {
		if err := ex.Pulse(context.Background()); err != nil {
			t.Fatalf("follow-up Pulse: %v", err)
		}
	}
	if got := done.Load(); got != M {
		t.Fatalf("after follow-up Pulses, done = %d; want %d", got, M)
	}
}

// =====================================================================
// US4: Multiple independent executor instances.
// =====================================================================

// T051: Pulse on one Executor leaves another's tasks untouched.
func TestUS4_IsolationBetweenInstances(t *testing.T) {
	a := executor.New()
	b := executor.New()
	if err := a.Pulse(context.Background()); err != nil {
		t.Fatal(err)
	}
	if err := b.Pulse(context.Background()); err != nil {
		t.Fatal(err)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	chA, chB := make(chan struct{}), make(chan struct{})
	var ranA, ranB atomic.Bool
	a.Co(func() {
		<-chA
		ranA.Store(true)
	})
	b.Co(func() {
		<-chB
		ranB.Store(true)
	})

	close(chA)
	time.Sleep(10 * time.Millisecond)

	if err := a.Pulse(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !ranA.Load() {
		t.Fatal("a's task did not run after a.Pulse")
	}
	if ranB.Load() {
		t.Fatal("a.Pulse advanced b's task; want isolation")
	}
}
