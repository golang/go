// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
)

var atExitFuncs []func()

func AtExit(f func()) {
	atExitFuncs = append(atExitFuncs, f)
}

func Exit(code int) {
	for i := len(atExitFuncs) - 1; i >= 0; i-- {
		f := atExitFuncs[i]
		atExitFuncs = atExitFuncs[:i]
		f()
	}
	os.Exit(code)
}

// To enable tracing support (-t flag), set EnableTrace to true.
const EnableTrace = false

// forEachGC calls fn each GC cycle until it returns false.
func forEachGC(fn func() bool) {
	type T [32]byte // large enough to avoid runtime's tiny object allocator

	var finalizer func(*T)
	finalizer = func { p -> if fn() {
		runtime.SetFinalizer(p, finalizer)
	} }

	finalizer(new(T))
}

// AdjustStartingHeap modifies GOGC so that GC should not occur until the heap
// grows to the requested size.  This is intended but not promised, though it
// is true-mostly, depending on when the adjustment occurs and on the
// compiler's input and behavior.  Once this size is approximately reached
// GOGC is reset to 100; subsequent GCs may reduce the heap below the requested
// size, but this function does not affect that.
//
// -d=gcadjust=1 enables logging of GOGC adjustment events.
//
// NOTE: If you think this code would help startup time in your own
// application and you decide to use it, please benchmark first to see if it
// actually works for you (it may not: the Go compiler is not typical), and
// whatever the outcome, please leave a comment on bug #56546.  This code
// uses supported interfaces, but depends more than we like on
// current+observed behavior of the garbage collector, so if many people need
// this feature, we should consider/propose a better way to accomplish it.
func AdjustStartingHeap(requestedHeapGoal uint64) {
	logHeapTweaks := Debug.GCAdjust == 1
	mp := runtime.GOMAXPROCS(0)
	gcConcurrency := Flag.LowerC

	const (
		goal   = "/gc/heap/goal:bytes"
		count  = "/gc/cycles/total:gc-cycles"
		allocs = "/gc/heap/allocs:bytes"
		frees  = "/gc/heap/frees:bytes"
	)

	sample := []metrics.Sample{{Name: goal}, {Name: count}, {Name: allocs}, {Name: frees}}
	const (
		GOAL   = 0
		COUNT  = 1
		ALLOCS = 2
		FREES  = 3
	)

	// Assumptions and observations of Go's garbage collector, as of Go 1.17-1.20:

	// - the initial heap goal is 4M, by fiat.  It is possible for Go to start
	//   with a heap as small as 512k, so this may change in the future.

	// - except for the first heap goal, heap goal is a function of
	//   observed-live at the previous GC and current GOGC.  After the first
	//   GC, adjusting GOGC immediately updates GOGC; before the first GC,
	//   adjusting GOGC does not modify goal (but the change takes effect after
	//   the first GC).

	// - the before/after first GC behavior is not guaranteed anywhere, it's
	//   just behavior, and it's a bad idea to rely on it.

	// - we don't know exactly when GC will run, even after we adjust GOGC; the
	//   first GC may not have happened yet, may have already happened, or may
	//   be currently in progress, and GCs can start for several reasons.

	// - forEachGC above will run the provided function at some delay after each
	//   GC's mark phase terminates; finalizers are run after marking as the
	//   spans containing finalizable objects are swept, driven by GC
	//   background activity and allocation demand.

	// - "live at last GC" is not available through the current metrics
	//    interface. Instead, live is estimated by knowing the adjusted value of
	//    GOGC and the new heap goal following a GC (this requires knowing that
	//    at least one GC has occurred):
	//		  estLive = 100 * newGoal / (100 + currentGogc)
	//    this new value of GOGC
	//		  newGogc = 100*requestedHeapGoal/estLive - 100
	//    will result in the desired goal. The logging code checks that the
	//    resulting goal is correct.

	// There's a small risk that the finalizer will be slow to run after a GC
	// that expands the goal to a huge value, and that this will lead to
	// out-of-memory.  This doesn't seem to happen; in experiments on a variety
	// of machines with a variety of extra loads to disrupt scheduling, the
	// worst overshoot observed was 50% past requestedHeapGoal.

	metrics.Read(sample)
	for _, s := range sample {
		if s.Value.Kind() == metrics.KindBad {
			// Just return, a slightly slower compilation is a tolerable outcome.
			if logHeapTweaks {
				fmt.Fprintf(os.Stderr, "GCAdjust: Regret unexpected KindBad for metric %s\n", s.Name)
			}
			return
		}
	}

	// Tinker with GOGC to make the heap grow rapidly at first.
	currentGoal := sample[GOAL].Value.Uint64() // Believe this will be 4MByte or less, perhaps 512k
	myGogc := 100 * requestedHeapGoal / currentGoal
	if myGogc <= 150 {
		return
	}

	if logHeapTweaks {
		sample := append([]metrics.Sample(nil), sample...) // avoid races with GC callback
		AtExit(func() {
			metrics.Read(sample)
			goal := sample[GOAL].Value.Uint64()
			count := sample[COUNT].Value.Uint64()
			oldGogc := debug.SetGCPercent(100)
			if oldGogc == 100 {
				fmt.Fprintf(os.Stderr, "GCAdjust: AtExit goal %d gogc %d count %d maxprocs %d gcConcurrency %d\n",
					goal, oldGogc, count, mp, gcConcurrency)
			} else {
				inUse := sample[ALLOCS].Value.Uint64() - sample[FREES].Value.Uint64()
				overPct := 100 * (int(inUse) - int(requestedHeapGoal)) / int(requestedHeapGoal)
				fmt.Fprintf(os.Stderr, "GCAdjust: AtExit goal %d gogc %d count %d maxprocs %d gcConcurrency %d overPct %d\n",
					goal, oldGogc, count, mp, gcConcurrency, overPct)

			}
		})
	}

	debug.SetGCPercent(int(myGogc))

	adjustFunc := func {

		metrics.Read(sample)
		goal := sample[GOAL].Value.Uint64()
		count := sample[COUNT].Value.Uint64()

		if goal <= requestedHeapGoal { // Stay the course
			if logHeapTweaks {
				fmt.Fprintf(os.Stderr, "GCAdjust: Reuse GOGC adjust, current goal %d, count is %d, current gogc %d\n",
					goal, count, myGogc)
			}
			return true
		}

		// Believe goal has been adjusted upwards, else it would be less-than-or-equal than requestedHeapGoal
		calcLive := 100 * goal / (100 + myGogc)

		if 2*calcLive < requestedHeapGoal { // calcLive can exceed requestedHeapGoal!
			myGogc = 100*requestedHeapGoal/calcLive - 100

			if myGogc > 125 {
				// Not done growing the heap.
				oldGogc := debug.SetGCPercent(int(myGogc))

				if logHeapTweaks {
					// Check that the new goal looks right
					inUse := sample[ALLOCS].Value.Uint64() - sample[FREES].Value.Uint64()
					metrics.Read(sample)
					newGoal := sample[GOAL].Value.Uint64()
					pctOff := 100 * (int64(newGoal) - int64(requestedHeapGoal)) / int64(requestedHeapGoal)
					// Check that the new goal is close to requested.  3% of make.bash fails this test.  Why, TBD.
					if pctOff < 2 {
						fmt.Fprintf(os.Stderr, "GCAdjust: Retry GOGC adjust, current goal %d, count is %d, gogc was %d, is now %d, calcLive %d pctOff %d\n",
							goal, count, oldGogc, myGogc, calcLive, pctOff)
					} else {
						// The GC is being annoying and not giving us the goal that we requested, say more to help understand when/why.
						fmt.Fprintf(os.Stderr, "GCAdjust: Retry GOGC adjust, current goal %d, count is %d, gogc was %d, is now %d, calcLive %d pctOff %d inUse %d\n",
							goal, count, oldGogc, myGogc, calcLive, pctOff, inUse)
					}
				}
				return true
			}
		}

		// In this case we're done boosting GOGC, set it to 100 and don't set a new finalizer.
		oldGogc := debug.SetGCPercent(100)
		// inUse helps estimate how late the finalizer ran; at the instant the previous GC ended,
		// it was (in theory) equal to the previous GC's heap goal.  In a growing heap it is
		// expected to grow to the new heap goal.
		inUse := sample[ALLOCS].Value.Uint64() - sample[FREES].Value.Uint64()
		overPct := 100 * (int(inUse) - int(requestedHeapGoal)) / int(requestedHeapGoal)
		if logHeapTweaks {
			fmt.Fprintf(os.Stderr, "GCAdjust: Reset GOGC adjust, old goal %d, count is %d, gogc was %d, calcLive %d inUse %d overPct %d\n",
				goal, count, oldGogc, calcLive, inUse, overPct)
		}
		return false
	}

	forEachGC(adjustFunc)
}
