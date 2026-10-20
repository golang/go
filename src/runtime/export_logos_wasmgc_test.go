// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Test seam for M19-S3: the idle mark slice state machine, driven with an
// explicit clock so package runtime_test needs neither a JS host nor a
// running collector.

const (
	LogosWasmSliceNanos = logosWasmSliceNanos
	LogosWasmWakeNanos  = logosWasmWakeNanos
	LogosWasmSliceRing  = logosWasmSliceRing
	LogosWasmCheckWork  = logosWasmCheckWork
)

// LogosWasmSliceActive reports whether mark slices are in force.
func LogosWasmSliceActive() bool {
	return logosWasmSliceActive()
}

// LogosDrainCheckThreshold returns the scan work between two clock checks
// of an idle or fractional drain.
func LogosDrainCheckThreshold(threshold int64) int64 {
	return logosDrainCheckThreshold(threshold)
}

// LogosClampCheckWork caps a check threshold at the slice check work.
func LogosClampCheckWork(threshold int64) int64 {
	return logosClampCheckWork(threshold)
}

// LogosWasmDedicatedWorkers returns the number of dedicated mark workers a
// cycle starts with when the pacer asks for needed.
func LogosWasmDedicatedWorkers(needed int64) int64 {
	return logosWasmDedicatedWorkers(needed)
}

// LogosWasmWorkersMayRun reports whether the scheduler may dispatch a
// background mark worker.
func LogosWasmWorkersMayRun() bool {
	return logosWasmWorkersMayRun()
}

// LogosWasmSlicer wraps one slice state machine.
type LogosWasmSlicer struct {
	s logosWasmSlicer
}

// MayStart reports whether a new slice may start.
func (x *LogosWasmSlicer) MayStart() bool {
	return x.s.mayStart()
}

// Begin starts a slice at now on the M with id mid.
func (x *LogosWasmSlicer) Begin(now, mid int64) {
	x.s.begin(now, mid)
}

// Expired reports whether the running slice is over at now.
func (x *LogosWasmSlicer) Expired(now int64) bool {
	return x.s.expired(now)
}

// EventTurn records that the event loop ran.
func (x *LogosWasmSlicer) EventTurn() {
	x.s.eventTurn()
}

// WakeAt returns the poll deadline the idle path hands to the event loop.
func (x *LogosWasmSlicer) WakeAt(now, pollUntil int64) int64 {
	return x.s.wakeAt(now, pollUntil)
}

// Parked reports whether the last slice ended at its deadline and no
// event turn has followed.
func (x *LogosWasmSlicer) Parked() bool {
	return x.s.parked
}

// Stats copies the recorded M ids into mids and returns the slice count
// and the longest slice that ended at its deadline.
func (x *LogosWasmSlicer) Stats(mids []int64) (slices uint64, longest int64) {
	return x.s.snapshot(mids)
}
