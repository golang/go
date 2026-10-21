// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/runtime/math"

// logosGCMode names the collector selected at link time with
// -X runtime.logosGCMode=MODE: greentea, zgc, immix, wasm, batch, or epsilon.
// It is empty under plain go without -gc.
var logosGCMode string

// logosPhysMem is the physical memory in bytes that batch read at gcinit, or
// 0 when the platform reports none or a value below logosMinPhysMem.
var logosPhysMem uint64

// logosMinPhysMem is the smallest physical memory that batch believes. A
// smaller reading comes from a wrong sysctl or a failed parse, and a trigger
// at 90% of it would collect on every few allocated megabytes.
const logosMinPhysMem = 64 << 20

// logosSanePhysMem returns phys, or 0 when it is below logosMinPhysMem.
func logosSanePhysMem(phys uint64) uint64 {
	if phys < logosMinPhysMem {
		return 0
	}
	return phys
}

// logosGCInit returns the initial GOGC and memory limit for the selected
// mode. Batch and epsilon start with GOGC off whatever the environment says.
// Epsilon also starts with no memory limit. Batch keeps the limit and
// forces every collection to stop the world, because the stores of packages
// compiled without write barriers are sound only while no mutator runs
// during mark. The other modes keep the environment's values.
func logosGCInit(gogc int32, limit int64) (int32, int64) {
	switch logosGCMode {
	case "batch":
		logosPhysMem = logosSanePhysMem(logosPhysicalMemory())
		logosForceStopTheWorld()
		return -1, limit
	case "epsilon":
		logosForceStopTheWorld()
		return -1, math.MaxInt64
	}
	return gogc, limit
}

// logosForceStopTheWorld applies gcstoptheworld=1 semantics even when GODEBUG
// asked for concurrent collection. A GODEBUG value of 2, which also sweeps
// with the world stopped, is kept.
func logosForceStopTheWorld() {
	if debug.gcstoptheworld == 0 {
		debug.gcstoptheworld = 1
	}
}

// logosGCRefuses reports whether the mode forbids a collection that t
// would start. Epsilon starts none on its own: only an explicit runtime.GC,
// a gcTriggerCycle, collects. Batch collects only at its heap trigger, so it
// refuses the periodic collection that a GOGC set at run time would enable.
func logosGCRefuses(t gcTrigger) bool {
	return logosModeRefuses(logosGCMode, t)
}

// logosModeRefuses is logosGCRefuses for an explicit mode.
func logosModeRefuses(mode string, t gcTrigger) bool {
	switch mode {
	case "epsilon":
		return t.kind != gcTriggerCycle
	case "batch":
		return t.kind == gcTriggerTime
	}
	return false
}

// logosParseUint parses the decimal number at the start of b after any spaces
// and returns it with the count of digits read; no digits give 0, 0.
func logosParseUint(b []byte) (value uint64, digits int) {
	i := 0
	for i < len(b) && b[i] == ' ' {
		i++
	}
	for ; i < len(b) && b[i] >= '0' && b[i] <= '9'; i++ {
		value = value*10 + uint64(b[i]-'0')
		digits++
	}
	return value, digits
}

// logosBatchLimit returns the memory the batch trigger is a fraction of:
// the memory limit when one is set, else the physical memory phys. With no
// limit and unknown physical memory it returns the largest value, so no
// trigger is ever reached.
func logosBatchLimit(memoryLimit int64, phys uint64) uint64 {
	switch {
	case memoryLimit != math.MaxInt64:
		return uint64(memoryLimit)
	case phys == 0:
		return math.MaxUint64
	}
	return phys
}

// logosBatchTriggerAt returns the heapLive at which batch starts a
// collection: exactly 90% of limit while the heap marked live by the last
// collection is below that mark. Once a collection leaves the live heap at or
// above the mark, the mark is reached already and cannot trigger again, so
// the trigger moves a quarter of the live heap (at least the minimum heap)
// past it. That bounds collection cost per allocated byte; without it every
// span refill would start a collection that frees nothing.
func logosBatchTriggerAt(limit, heapMarked uint64) uint64 {
	mark := limit / 10 * 9
	if heapMarked < mark {
		return mark
	}
	return heapMarked + max(heapMarked/4, defaultHeapMinimum)
}

// logosBatchTrigger is trigger for batch: GOGC plays no part, and the memory
// limit goal of the upstream pacer is replaced by logosBatchTriggerAt. The
// returned goal stays the upstream goal, raised to the trigger, so the
// scavenger keeps pacing against the same goal as before.
func (c *gcControllerState) logosBatchTrigger() (trigger, goal uint64) {
	limit := logosBatchLimit(c.memoryLimit.Load(), logosPhysMem)
	trigger = logosBatchTriggerAt(limit, c.heapMarked)
	goal, _ = c.heapGoalInternal()
	return trigger, max(goal, trigger)
}
