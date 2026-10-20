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

// logosStackStateFits fails the build if the largest state the average
// can reach, 8 times the sample cap plus 7, does not fit the state field.
const _ = uint(logosStackStateMask - logosStackStateMax)

// logosStackTable is the per-call-site initial goroutine stack size
// history of GOEXPERIMENT=logosstack (M17-S2). The runtime keeps, for
// every go statement, a moving average of the final stack sizes of the
// goroutines that statement created, and starts the next goroutine from
// that statement with a stack of that size. A statement without history
// starts with the global size that gcComputeStartingStackSize maintains.
//
// The history is a fixed table, so recording an exit and choosing a start
// size never allocate and the memory stays bounded however many distinct
// go statements run. Each word packs a hash tag of the statement's pc in
// the high logosStackTagBits bits with the state in the low
// logosStackStateBits bits, so a lookup, an update, and a replacement are
// each one atomic word operation and no reader sees a tag paired with
// another entry's average. The state is 2^logosStackEMAShift times the
// moving average of the final stack sizes in units of fixedStack, so the
// update state - state>>logosStackEMAShift + sample is the weight-1/8
// average with no division. An empty slot is zero and a stored state is
// never below 2^logosStackEMAShift.
//
// A new statement takes the first empty slot of its probe window, and a
// full window gives up the entry with the smallest state: large entries
// are the ones that change a start size, so they outlive the small ones.
// A compare-and-swap that loses a race drops its sample, which a moving
// average tolerates.
//
// The average is of final sizes, so a goroutine that starts at the
// learned size and never grows reports the learned size again. The
// average falls only when goroutines of a statement end with smaller
// stacks than the average, which happens when the collector shrinks the
// stacks of goroutines that outlive a cycle.
var logosStackTable [logosStackSlots]atomic.Uint64

// logosStackKey splits the go statement pc into the probe window's first
// slot and the tag that tells statements of one window apart. The slot
// and the tag are the top 50 bits of pc times a Fibonacci multiplier, a
// bijection of the 64-bit pc, so two pcs share both only if their products
// differ by less than 2^14. For pcs less than 2^32 apart, which covers
// every pc of one Go binary, the products differ by at least 5.0e7
// (best approximations of the multiplier over 2^64), so no two go
// statements of a binary share an entry.
func logosStackKey(pc uintptr) (home uint32, tag uint64) {
	h := uint64(pc) * logosStackHashMul
	home = uint32(h >> (64 - logosStackSlotBits))
	tag = h >> logosStackTagShift & logosStackTagMask
	return home, tag
}

// logosStackUnits converts a stack size to units of fixedStack, at least
// one and at most logosStackUnitCap.
func logosStackUnits(size uintptr) uint64 {
	return max(1, min(uint64(size/fixedStack), logosStackUnitCap))
}

// logosStackPack joins a tag and a state into a table word.
func logosStackPack(tag, state uint64) uint64 {
	return tag<<logosStackStateBits | state
}

// logosStackBlend folds one sample, in units of fixedStack, into state.
func logosStackBlend(state, units uint64) uint64 {
	return state - state>>logosStackEMAShift + units
}

// logosStackLookup returns the state recorded for pc. The probe stops at
// the first empty slot because slots never become empty again.
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

// logosStackRecord folds a goroutine's final stack size into the history
// of the go statement at pc.
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

// logosStackPick rounds a moving average of units of fixedStack up to a
// power of two and bounds it by the largest power of two within limit
// bytes, but never below fixedStack and never above the largest size the
// state field can express.
func logosStackPick(units uint64, limit uintptr) uint32 {
	last := units - 1
	size := uint64(1) << sys.Len64(last) * fixedStack
	capped := min(uint64(limit), logosStackUnitCap*fixedStack)
	ceiling := uint64(1) << sys.Len64(capped) >> 1
	return uint32(max(fixedStack, min(size, ceiling)))
}

// logosStackStart returns the stack size for goroutines created at pc and
// whether pc has history.
func logosStackStart(pc uintptr) (size uint32, ok bool) {
	state, ok := logosStackLookup(pc)
	if !ok {
		return 0, false
	}
	return logosStackPick(state>>logosStackEMAShift, maxstacksize), true
}

// logosStackNoteExit records the final stack size of the goroutine gp
// that goexit0 is about to destroy.
func logosStackNoteExit(gp *g) {
	if !goexperiment.LogosStack {
		return
	}
	logosStackRecord(gp.gopc, gp.stack.hi-gp.stack.lo)
}

// logosStackFit gives the fresh goroutine gp, whose stack is the global
// starting size, the stack size recorded for the go statement at pc.
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

// logosRestack replaces the stack of the dead goroutine gp with one of n
// bytes the way gfput and gfget free and allocate it.
//
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
