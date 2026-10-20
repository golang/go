// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import _ "unsafe" // for go:linkname

// The constants of the idle mark slices of the wasm collector
// (SPEC Sec 19, T-301.2). js/wasm runs on one thread, so marking stays
// single-threaded and reuses the plain atomics of
// internal/runtime/atomic/atomic_wasm.go. Pointer stores keep the write
// barrier while marking runs.
//
// Upstream lets a background mark worker drain every grey object before
// the scheduler looks at the JS event loop again: the idle worker until no
// work is left, the fractional worker until its utilization share is spent,
// which grows with the age of the cycle. Either blocks the page for a long
// stretch. Under -gc=wasm each background mark worker runs as a slice that
// stops logosWasmSliceNanos after it starts. The idle worker stops through
// the pollWork check of its drain loop, the fractional worker through
// pollFractionalWorkerExit, and gcDrain checks the clock every
// logosWasmCheckWork units of scan work instead of every drainCheckThreshold
// units, because scan work counts bytes, not time. A slice that ends at its
// deadline parks the mark workers: beforeIdle asks the host for a callback
// logosWasmWakeNanos later and returns to the event loop, and handleEvent
// releases the next slice when the callback arrives. Mutator assists stay
// as upstream, so a mutator that allocates faster than the slices mark
// still pays for its own marking.
//
// logosWasmIdleStats is referenced from loGo/s programs through
// //go:linkname pull and carries a push-style //go:linkname mark, so the
// linker's linkname check accepts the pull.
//
// logosWasmSliceNanos is the length of one slice, logosWasmWakeNanos the
// delay of the host callback that follows a parked slice, logosWasmSliceRing
// the number of slice M ids kept for logosWasmIdleStats, and
// logosWasmCheckWork the scan work between two clock checks of a slice.
const (
	logosWasmSliceNanos = 2_000_000
	logosWasmWakeNanos  = 1_000_000
	logosWasmSliceRing  = 32
	logosWasmCheckWork  = gcCreditSlack
)

// logosWasmEventLoop reports whether the target has a JS event loop that
// the scheduler can return to between slices.
const logosWasmEventLoop = GOOS == "js" && GOARCH == "wasm"

// logosWasmSlicer is the state machine of the mark slices. All access
// happens on the single wasm thread.
type logosWasmSlicer struct {
	deadline int64
	started  int64
	longest  int64
	slices   uint64
	parked   bool
	mids     [logosWasmSliceRing]int64
}

var logosWasmIdle logosWasmSlicer

// mayStart reports whether a slice may start: not while the last slice
// ended at its deadline and the event loop has not run since.
func (s *logosWasmSlicer) mayStart() bool {
	return !s.parked
}

// begin starts a slice at now on the M with id mid and records the id.
func (s *logosWasmSlicer) begin(now, mid int64) {
	s.started = now
	s.deadline = now + logosWasmSliceNanos
	s.mids[s.slices%logosWasmSliceRing] = mid
	s.slices++
}

// expired reports whether the running slice is over at now. The first
// call at or after the deadline returns true and parks the mark workers;
// later calls return false until the next slice begins.
func (s *logosWasmSlicer) expired(now int64) bool {
	if s.deadline == 0 || now < s.deadline {
		return false
	}
	s.longest = max(s.longest, now-s.started)
	s.deadline = 0
	s.parked = true
	return true
}

// eventTurn records that the event loop ran, which releases parked mark
// workers.
func (s *logosWasmSlicer) eventTurn() {
	s.parked = false
}

// wakeAt returns the poll deadline beforeIdle hands to the host: the
// earlier of pollUntil and now+logosWasmWakeNanos while the mark workers
// are parked, so the host calls back soon, and pollUntil otherwise.
func (s *logosWasmSlicer) wakeAt(now, pollUntil int64) int64 {
	if !s.parked {
		return pollUntil
	}
	wake := now + logosWasmWakeNanos
	if pollUntil != 0 && pollUntil < wake {
		return pollUntil
	}
	return wake
}

// snapshot copies the recorded M ids into mids and returns the slice
// count and the longest slice that ended at its deadline since the
// previous snapshot.
func (s *logosWasmSlicer) snapshot(mids []int64) (slices uint64, longest int64) {
	copy(mids, s.mids[:min(s.slices, logosWasmSliceRing)])
	longest, s.longest = s.longest, 0
	return s.slices, longest
}

// logosWasmSliceActive reports whether mark slices are in force.
func logosWasmSliceActive() bool {
	return logosWasmEventLoop && logosWasmMode()
}

// logosWasmWorkersMayRun is the gate of the background mark workers in
// findRunnable.
func logosWasmWorkersMayRun() bool {
	return !logosWasmSliceActive() || logosWasmIdle.mayStart()
}

// logosWasmSliceBegin starts a slice for the background mark worker that
// begins at now. It runs on the M that runs the worker.
func logosWasmSliceBegin(now int64) {
	if logosWasmSliceActive() {
		logosWasmIdle.begin(now, getg().m.id)
	}
}

// logosWasmSliceOver is the deadline check of the idle and fractional
// drains.
func logosWasmSliceOver() bool {
	return logosWasmSliceActive() && logosWasmIdle.expired(nanotime())
}

// logosWasmEventTurn records that JavaScript called into Go.
func logosWasmEventTurn() {
	logosWasmIdle.eventTurn()
}

// logosWasmWake returns the clock and poll deadline beforeIdle uses:
// pollUntil, or an earlier wake-up while a parked slice left mark work
// behind. A zero now means upstream saw no timer, and is read here only
// when the wake-up needs it.
func logosWasmWake(now, pollUntil int64) (int64, int64) {
	if !logosWasmSliceActive() || gcBlackenEnabled == 0 || logosWasmIdle.mayStart() {
		return now, pollUntil
	}
	if now == 0 {
		now = nanotime()
	}
	return now, logosWasmIdle.wakeAt(now, pollUntil)
}

// logosDrainCheckThreshold returns the scan work between two clock checks
// of an idle or fractional drain: upstream's threshold, or the credit slack
// under mark slices.
func logosDrainCheckThreshold(threshold int64) int64 {
	if logosWasmSliceActive() {
		return logosClampCheckWork(threshold)
	}
	return threshold
}

// logosWasmDedicatedWorkers returns the number of dedicated mark workers
// the pacer starts a cycle with: none under mark slices, because a
// dedicated worker drains without a deadline and wasm has no preemption to
// interrupt it. The fractional workers of the pacer take over its share.
func logosWasmDedicatedWorkers(needed int64) int64 {
	if logosWasmSliceActive() {
		return 0
	}
	return needed
}

// logosClampCheckWork caps a check threshold at logosWasmCheckWork.
func logosClampCheckWork(threshold int64) int64 {
	return min(threshold, logosWasmCheckWork)
}

// logosWasmIdleStats copies the M ids of the most recent mark slices into
// mids and returns the number of slices started, the longest slice that
// ended at its deadline in nanoseconds since the previous call, and
// whether a mark phase is running.
//
//go:linkname logosWasmIdleStats
func logosWasmIdleStats(mids []int64) (slices uint64, longest int64, marking bool) {
	slices, longest = logosWasmIdle.snapshot(mids)
	return slices, longest, gcBlackenEnabled != 0
}
