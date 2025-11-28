// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package base

import (
	"fmt"
	"os"
	"runtime"
	"runtime/debug"
	"runtime/metrics"
	"sync"
)

// forEachGC calls fn each GC cycle until it returns false.
func forEachGC(fn func() bool) {
	type T [32]byte // large enough to avoid runtime's tiny object allocator
	var finalizer func(*T)
	finalizer = func(p *T) {

		if fn() {
			runtime.SetFinalizer(p, finalizer)
		}
	}

	finalizer(new(T))
}

// AdjustStartingHeap modifies GOGC so that GC should not occur until the heap
// grows to the requested size.  This is intended but not promised, though it
// is true-mostly, depending on when the adjustment occurs and on the
// compiler's input and behavior.  Once the live heap is approximately half
// this size, GOGC is reset to its value when AdjustStartingHeap was called;
// subsequent GCs may reduce the heap below the requested size, but this
// function does not affect that.
//
// logHeapTweaks (-d=gcadjust=1) enables logging of GOGC adjustment events.
//
// The temporarily requested GOGC is derated from what would be the "obvious"
// value necessary to hit the starting heap goal because the obvious
// (goal/live-1)*100 value seems to grow RSS a little more than it "should"
// (compared to GOMEMLIMIT, e.g.) and the assumption is that the GC's control
// algorithms are tuned for GOGC near 100, and not tuned for huge values of
// GOGC.  Different derating factors apply for "lo" and "hi" values of GOGC;
// lo is below derateBreak, hi is above derateBreak.  The derating factors,
// expressed as integer percentages, are derateLoPct and derateHiPct.
// 60-75 is an okay value for derateLoPct, 30-65 seems like a good value for
// derateHiPct, and 600 seems like a good value for derateBreak.  If these
// are zero, defaults are used instead.
//
// NOTE: If you think this code would help startup time in your own
// application and you decide to use it, please benchmark first to see if it
// actually works for you (it may not: the Go compiler is not typical), and
// whatever the outcome, please leave a comment on bug #56546.  This code
// uses supported interfaces, but depends more than we like on
// current+observed behavior of the garbage collector, so if many people need
// this feature, we should consider/propose a better way to accomplish it.
func AdjustStartingHeap(requestedHeapGoal, derateBreak, derateLoPct, derateHiPct uint64, logHeapTweaks bool) {
	mp := runtime.GOMAXPROCS(0)

	const (
		SHgoal   = "/gc/heap/goal:bytes"
		SHcount  = "/gc/cycles/total:gc-cycles"
		SHallocs = "/gc/heap/allocs:bytes"
		SHfrees  = "/gc/heap/frees:bytes"
	)

	var sample = []metrics.Sample{{Name: SHgoal}, {Name: SHcount}, {Name: SHallocs}, {Name: SHfrees}}

	const (
		SH_GOAL   = 0
		SH_COUNT  = 1
		SH_ALLOCS = 2
		SH_FREES  = 3

		MB = 1_000_000
	)

	// These particular magic numbers are designed to make the RSS footprint of -d=-gcstart=2000
	// resemble that of GOMEMLIMIT=2000MiB GOGC=10000 when building large projects
	// (e.g. the Go compiler itself, and the microsoft's typescript AST package),
	// with the further restriction that these magic numbers did a good job of reducing user-cpu
	// for builds at either gcstart=2000 or gcstart=128.
	//
	// The benchmarking to obtain this was (a version of):
	//
	// for i in {1..50} ; do
	//     for what in std cmd/compile cmd/fix cmd/go github.com/microsoft/typescript-go/internal/ast ; do
	//       whatbase=`basename ${what}`
	//       for sh in 128 2000 ; do
	//         for br in 500 600 ; do
	//           for shlo in 65 70; do
	//             for shhi in 55 60 ; do
	//               benchcmd -n=2 ${whatbase} go build -a \
	//               -gcflags=all=-d=gcstart=${sh},gcstartloderate=${shlo},gcstarthiderate=${shhi},gcstartbreak=${br} \
	//               ${what} | tee -a startheap${sh}_${br}_${shhi}_${shlo}.bench
	//             done
	//           done
	//         done
	//       done
	//     done
	// done
	//
	// benchcmd is "go install github.com/aclements/go-misc/benchcmd@latest"

	if derateBreak == 0 {
		derateBreak = 600
	}
	if derateLoPct == 0 {
		derateLoPct = 70
	}
	if derateHiPct == 0 {
		derateHiPct = 55
	}

	gogcDerate := func(myGogc uint64) uint64 {
		if myGogc < derateBreak {
			return (myGogc * derateLoPct) / 100
		}
		return (myGogc * derateHiPct) / 100
	}

	// Assumptions and observations of Go's garbage collector, as of Go 1.17-1.20:

	// - the initial heap goal is 4MiB, by fiat.  It is possible for Go to start
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
	currentGoal := sample[SH_GOAL].Value.Uint64() // Believe this will be 4MByte or less, perhaps 512k
	myGogc := 100 * requestedHeapGoal / currentGoal
	myGogc = gogcDerate(myGogc)
	if myGogc <= 125 {
		return
	}

	if logHeapTweaks {
		sample := append([]metrics.Sample(nil), sample...) // avoid races with GC callback
		AtExit(func() {
			metrics.Read(sample)
			goal := sample[SH_GOAL].Value.Uint64()
			count := sample[SH_COUNT].Value.Uint64()
			oldGogc := debug.SetGCPercent(100)
			if oldGogc == 100 {
				fmt.Fprintf(os.Stderr, "GCAdjust: AtExit goal %dMB gogc %d count %d maxprocs %d\n",
					goal/MB, oldGogc, count, mp)
			} else {
				inUse := sample[SH_ALLOCS].Value.Uint64() - sample[SH_FREES].Value.Uint64()
				overPct := 100 * (int(inUse) - int(requestedHeapGoal)) / int(requestedHeapGoal)
				fmt.Fprintf(os.Stderr, "GCAdjust: AtExit goal %dMB gogc %d count %d maxprocs %d overPct %d\n",
					goal/MB, oldGogc, count, mp, overPct)

			}
		})
	}

	originalGOGC := debug.SetGCPercent(int(myGogc))

	// forEachGC finalizers ought not overlap, but they could run in separate threads.
	// This ought not matter, but just in case it bothers the/a race detector,
	// use this mutex.
	var forEachGCLock sync.Mutex

	adjustFunc := func() bool {

		forEachGCLock.Lock()
		defer forEachGCLock.Unlock()

		metrics.Read(sample)
		goal := sample[SH_GOAL].Value.Uint64()
		count := sample[SH_COUNT].Value.Uint64()

		if goal <= requestedHeapGoal { // Stay the course
			if logHeapTweaks {
				fmt.Fprintf(os.Stderr, "GCAdjust: Reuse GOGC adjust, current goal %dMB, count is %d, current gogc %d\n",
					goal/MB, count, myGogc)
			}
			return true
		}

		// Believe goal has been adjusted upwards, else it would be less-than-or-equal to requestedHeapGoal
		calcLive := 100 * goal / (100 + myGogc)

		if 2*calcLive < requestedHeapGoal { // calcLive can exceed requestedHeapGoal!
			myGogc = 100*requestedHeapGoal/calcLive - 100
			myGogc = gogcDerate(myGogc)

			if myGogc > 125 {
				// Not done growing the heap.
				oldGogc := debug.SetGCPercent(int(myGogc))

				if logHeapTweaks {
					// Check that the new goal looks right
					inUse := sample[SH_ALLOCS].Value.Uint64() - sample[SH_FREES].Value.Uint64()
					metrics.Read(sample)
					newGoal := sample[SH_GOAL].Value.Uint64()
					pctOff := 100 * (int64(newGoal) - int64(requestedHeapGoal)) / int64(requestedHeapGoal)
					// Check that the new goal is close to requested.  3% of make.bash fails this test.  Why, TBD.
					if pctOff < 2 {
						fmt.Fprintf(os.Stderr, "GCAdjust: Retry GOGC adjust, current goal %dMB, count is %d, gogc was %d, is now %d, calcLive %dMB pctOff %d\n",
							goal/MB, count, oldGogc, myGogc, calcLive/MB, pctOff)
					} else {
						// The GC is being annoying and not giving us the goal that we requested, say more to help understand when/why.
						fmt.Fprintf(os.Stderr, "GCAdjust: Retry GOGC adjust, current goal %dMB, count is %d, gogc was %d, is now %d, calcLive %dMB pctOff %d inUse %dMB\n",
							goal/MB, count, oldGogc, myGogc, calcLive/MB, pctOff, inUse/MB)
					}
				}
				return true
			}
		}

		// In this case we're done boosting GOGC, set it to its original value and don't set a new finalizer.
		oldGogc := debug.SetGCPercent(originalGOGC)
		// inUse helps estimate how late the finalizer ran; at the instant the previous GC ended,
		// it was (in theory) equal to the previous GC's heap goal.  In a growing heap it is
		// expected to grow to the new heap goal.
		if logHeapTweaks {
			inUse := sample[SH_ALLOCS].Value.Uint64() - sample[SH_FREES].Value.Uint64()
			overPct := 100 * (int(inUse) - int(requestedHeapGoal)) / int(requestedHeapGoal)
			fmt.Fprintf(os.Stderr, "GCAdjust: Reset GOGC adjust, old goal %dMB, count is %d, gogc was %d, gogc is now %d, calcLive %dMB inUse %dMB overPct %d\n",
				goal/MB, count, oldGogc, originalGOGC, calcLive/MB, inUse/MB, overPct)
		}
		return false
	}

	forEachGC(adjustFunc)
}
