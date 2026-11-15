// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import (
	"internal/goexperiment"
	"internal/runtime/atomic"
	"internal/runtime/sys"
	"unsafe"
)

const (
	logosStackSlotBits  = 10
	logosStackSlots     = 1 << logosStackSlotBits
	logosStackWindow    = 8
	logosStackStateBits = 24
	logosStackStateMask = 1<<logosStackStateBits - 1
	logosStackTagBits   = 64 - logosStackStateBits
	logosStackTagMask   = 1<<logosStackTagBits - 1
	logosStackTagShift  = 64 - logosStackSlotBits - logosStackTagBits
	logosStackEMAShift  = 3
	logosStackUnitCap   = 1 << 20
	logosStackStateMax  = logosStackUnitCap<<logosStackEMAShift + 1<<logosStackEMAShift - 1
	logosStackHashMul   = 0x9E3779B97F4A7C15
)

const _ = uint(logosStackStateMask - logosStackStateMax)

var logosStackTable [logosStackSlots]atomic.Uint64

func logosStackKey(pc uintptr) (home uint32, tag uint64) {
	h := uint64(pc) * logosStackHashMul
	home = uint32(h >> (64 - logosStackSlotBits))
	tag = h >> logosStackTagShift & logosStackTagMask
	return home, tag
}

func logosStackUnits(size uintptr) uint64 {
	return max(1, min(uint64(size/fixedStack), logosStackUnitCap))
}

func logosStackPack(tag, state uint64) uint64 {
	return tag<<logosStackStateBits | state
}

func logosStackBlend(state, units uint64) uint64 {
	return state - state>>logosStackEMAShift + units
}

func logosStackLookup(pc uintptr) (state uint64, ok bool) {
	home, tag := logosStackKey(pc)
	for i := range uint32(logosStackWindow) {
		v := logosStackTable[(home+i)&(logosStackSlots-1)].Load()
		if v == 0 {
			return 0, false
		}
		if v>>logosStackStateBits == tag {
			return v & logosStackStateMask, true
		}
	}
	return 0, false
}

func logosStackRecord(pc, size uintptr) {
	home, tag := logosStackKey(pc)
	units := logosStackUnits(size)
	victim := &logosStackTable[home]
	var victimWord uint64
	victimState := ^uint64(0)
	for i := range uint32(logosStackWindow) {
		slot := &logosStackTable[(home+i)&(logosStackSlots-1)]
		v := slot.Load()
		if v == 0 {
			slot.CompareAndSwap(0, logosStackPack(tag, units<<logosStackEMAShift))
			return
		}
		if v>>logosStackStateBits == tag {
			if next := logosStackPack(tag, logosStackBlend(v&logosStackStateMask, units)); next != v {
				slot.CompareAndSwap(v, next)
			}
			return
		}
		if state := v & logosStackStateMask; state < victimState {
			victim, victimWord, victimState = slot, v, state
		}
	}
	victim.CompareAndSwap(victimWord, logosStackPack(tag, units<<logosStackEMAShift))
}

func logosStackPick(units uint64, limit uintptr) uint32 {
	last := units - 1
	size := uint64(1) << sys.Len64(last) * fixedStack
	capped := min(uint64(limit), logosStackUnitCap*fixedStack)
	ceiling := uint64(1) << sys.Len64(capped) >> 1
	return uint32(max(fixedStack, min(size, ceiling)))
}

func logosStackStart(pc uintptr) (size uint32, ok bool) {
	state, ok := logosStackLookup(pc)
	if !ok {
		return 0, false
	}
	return logosStackPick(state>>logosStackEMAShift, maxstacksize), true
}

func logosStackNoteExit(gp *g) {
	if !goexperiment.LogosStack {
		return
	}
	logosStackRecord(gp.gopc, gp.stack.hi-gp.stack.lo)
}

func logosStackFit(gp *g, pc uintptr) {
	if !goexperiment.LogosStack {
		return
	}
	size, ok := logosStackStart(pc)
	if !ok || uintptr(size) == gp.stack.hi-gp.stack.lo {
		return
	}
	systemstack(func() { logosRestack(gp, size) })
}

//go:systemstack
func logosRestack(gp *g, n uint32) {
	stackfree(gp.stack)
	if valgrindenabled {
		valgrindDeregisterStack(gp.valgrindStackID)
	}
	gp.stack = stackalloc(n)
	if valgrindenabled {
		gp.valgrindStackID = valgrindRegisterStack(unsafe.Pointer(gp.stack.lo), unsafe.Pointer(gp.stack.hi))
	}
	gp.stackguard0 = gp.stack.lo + stackGuard
}
