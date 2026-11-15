// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const LogosFixedStack = fixedStack

const LogosStackSlots = logosStackSlots

func LogosStartSize(pc uintptr) int {
	if size, ok := logosStackStart(pc); ok {
		return int(size)
	}
	return int(startingStackSize)
}

func LogosGlobalStartSize() int {
	return int(startingStackSize)
}

func LogosStackEMA(pc uintptr) (ema int, ok bool) {
	state, ok := logosStackLookup(pc)
	if !ok {
		return 0, false
	}
	return int(state>>logosStackEMAShift) * fixedStack, true
}

func LogosStackRecord(pc uintptr, size int) {
	logosStackRecord(pc, uintptr(size))
}

func LogosStackOccupied() int {
	n := 0
	for i := range logosStackTable {
		if logosStackTable[i].Load() != 0 {
			n++
		}
	}
	return n
}

func LogosStackReset() {
	for i := range logosStackTable {
		logosStackTable[i].Store(0)
	}
}

func LogosStackPick(ema, limit int) int {
	return int(logosStackPick(uint64(ema/fixedStack), uintptr(limit)))
}

func LogosGoPC() uintptr {
	return getg().gopc
}

//go:nosplit
func LogosCurStackSize() int {
	gp := getg()
	return int(gp.stack.hi - gp.stack.lo)
}
