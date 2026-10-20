// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

// Test seam for M17-S2: opaque wrappers over the per-call-site stack
// size table for package runtime_test, which cannot name g or the table.

// LogosFixedStack is the smallest goroutine stack the runtime allocates.
const LogosFixedStack = fixedStack

// LogosStackSlots is the capacity of the per-call-site table.
const LogosStackSlots = logosStackSlots

// LogosStartSize reports the stack size, in bytes, that newproc1 gives a
// goroutine created at the go statement pc: the table's size when pc has
// history and the global starting size otherwise.
func LogosStartSize(pc uintptr) int {
	if size, ok := logosStackStart(pc); ok {
		return int(size)
	}
	return int(startingStackSize)
}

// LogosGlobalStartSize reports the starting stack size the runtime gives
// every goroutine that has no call-site history.
func LogosGlobalStartSize() int {
	return int(startingStackSize)
}

// LogosStackEMA reports the moving average of final stack sizes, in
// bytes, recorded for pc, and whether pc has history.
func LogosStackEMA(pc uintptr) (ema int, ok bool) {
	state, ok := logosStackLookup(pc)
	if !ok {
		return 0, false
	}
	return int(state>>logosStackEMAShift) * fixedStack, true
}

// LogosStackRecord feeds the table the way goexit0 does for a goroutine
// created at pc that exits with a stack of size bytes.
func LogosStackRecord(pc uintptr, size int) {
	logosStackRecord(pc, uintptr(size))
}

// LogosStackOccupied counts the table slots that hold history.
func LogosStackOccupied() int {
	n := 0
	for i := range logosStackTable {
		if logosStackTable[i].Load() != 0 {
			n++
		}
	}
	return n
}

// LogosStackReset empties the table. Callers must not run goroutines
// that exit concurrently.
func LogosStackReset() {
	for i := range logosStackTable {
		logosStackTable[i].Store(0)
	}
}

// LogosStackPick reports the start size, in bytes, that a moving average
// of ema bytes yields when the maximum stack is limit bytes.
func LogosStackPick(ema, limit int) int {
	return int(logosStackPick(uint64(ema/fixedStack), uintptr(limit)))
}

// LogosGoPC reports the program counter of the go statement that created
// the calling goroutine.
func LogosGoPC() uintptr {
	return getg().gopc
}

// LogosCurStackSize reports the size, in bytes, of the calling
// goroutine's stack.
//
//go:nosplit
func LogosCurStackSize() int {
	gp := getg()
	return int(gp.stack.hi - gp.stack.lo)
}
