// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || loong64

// This provides common support for architectures that use extended register
// state in asynchronous preemption.
//
// While asynchronous preemption stores general-purpose (GP) registers on the
// preempted goroutine's own stack, extended register state can be used to save
// non-GP state off the stack. In particular, this is meant for large vector
// register files. Currently, we assume this contains only scalar data, though
// we could change this constraint by conservatively scanning this memory.
//
// For an architecture to support extended register state, it must provide a Go
// definition of an xRegState type for storing the state, and its asyncPreempt
// implementation must write this register state to p.xRegs.scratch.

package runtime

import (
	"internal/runtime/sys"
	"unsafe"
)

// xRegState is long-lived extended register state. It is allocated off-heap and
// manually managed.
type xRegState struct {
	_    sys.NotInHeap // Allocated from xRegAlloc
	regs xRegs
}

// xRegPerG stores extended register state while a goroutine is asynchronously
// preempted. This is nil otherwise, so we can reuse a (likely small) pool of
// xRegState objects.
type xRegPerG struct {
	state *xRegState
}

type xRegPerP struct {
	// scratch temporary per-P space where [asyncPreempt] saves the register
	// state before entering Go. It's quickly copied to per-G state.
	scratch xRegs

	// cache is a 1-element allocation cache of extended register state used by
	// asynchronous preemption. On entry to preemption, this is used as a simple
	// allocation cache. On exit from preemption, the G's xRegState is always
	// stored here where it can be restored, and later either freed or reused
	// for another preemption. On exit, this serves the dual purpose of
	// delay-freeing the allocated xRegState until after we've definitely
	// restored it.
	cache *xRegState
}

// xRegAlloc allocates xRegState objects.
var xRegAlloc struct {
	lock  mutex
	alloc fixalloc
}

func xRegInitAlloc() {
	lockInit(&xRegAlloc.lock, lockRankXRegAlloc)
	xRegAlloc.alloc.init(unsafe.Sizeof(xRegState{}), nil, nil, &memstats.other_sys)
}

// xRegSave saves the extended register state on this P to gp.
//
// This must run on the system stack because it assumes the P won't change.
//
//go:systemstack
func xRegSave(gp *g) {
	if gp.xRegs.state != nil {
		// Double preempt?
		throw("gp.xRegState.p != nil on async preempt")
	}

	// Get the place to save the register state.
	var dest *xRegState
	pp := gp.m.p.ptr()
	if pp.xRegs.cache != nil {
		// Use the cached allocation.
		dest = pp.xRegs.cache
		pp.xRegs.cache = nil
	} else {
		// Allocate a new save block.
		lock(&xRegAlloc.lock)
		dest = (*xRegState)(xRegAlloc.alloc.alloc())
		unlock(&xRegAlloc.lock)
	}

	// Copy state saved in the scratchpad to dest.
	//
	// If we ever need to save less state (e.g., avoid saving vector registers
	// that aren't in use), we could have multiple allocation pools for
	// different size states and copy only the registers we need.
	dest.regs = pp.xRegs.scratch

	// Save on the G.
	gp.xRegs.state = dest
}

// xRegRestore prepares the extended register state on gp to be restored.
//
// It moves the state to gp.m.p.xRegs.cache where [asyncPreempt] expects to find
// it. This means nothing else may use the cache between this call and the
// return to asyncPreempt. This is not quite symmetric with [xRegSave], which
// uses gp.m.p.xRegs.scratch. By using cache instead, we save a block copy.
//
// This is called with asyncPreempt on the stack and thus must not grow the
// stack.
//
//go:nosplit
func xRegRestore(gp *g) {
	if gp.xRegs.state == nil {
		throw("gp.xRegState.p == nil on return from async preempt")
	}
	// If the P has a block cached on it, free that so we can replace it.
	pp := gp.m.p.ptr()
	if pp.xRegs.cache != nil {
		// Don't grow the G stack.
		systemstack(func() {
			pp.xRegs.free()
		})
	}
	pp.xRegs.cache = gp.xRegs.state
	gp.xRegs.state = nil
}

func (xRegs *xRegPerP) free() {
	if xRegs.cache != nil {
		lock(&xRegAlloc.lock)
		xRegAlloc.alloc.free(unsafe.Pointer(xRegs.cache))
		xRegs.cache = nil
		unlock(&xRegAlloc.lock)
	}
}
