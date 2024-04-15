// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
)

// ----------------------------------------------------------------------------
// Loop Invariant Code Motion
//
// The main idea behind LICM is to move loop invariant values outside of the loop
// so that they are only executed once, instead of being repeatedly executed with
// each iteration of the loop. In the context of LICM, if a loop invariant can be
// speculatively executed, then it can be freely hoisted to the loop entry.
// However, if it cannot be speculatively executed, there is still a chance that
// it can be hoisted outside the loop under a few prerequisites:
//
//  #1 Instruction is guaranteed to execute unconditionally
//  #2 Instruction does not access memory locations that may alias with other
//    memory operations inside the loop
//
// For #1, this is guaranteed by loop rotation, where the loop is guaranteed to
// execute at least once after rotation. But that's not the whole story. If the
// instruction is guarded by a conditional expression (e.g., loading from a memory
// address usually guarded by an IsInBound check), in this case, we try to hoist
// it only if the loop invariant dominates all loop exits, which implies that it
// will be executed unconditionally as soon as it enters the loop.
// For #2, we always pessimistically assume that they are must-aliases and stop
// optimizing if we saw both load and store

func logInvariant(val *Value, src *Block, dest *Block) {
	hoistType := "Simple"
	if isHoistable(val) {
		hoistType = "Complex"
	}
	if dest.Func.pass.debug > 2 {
		fmt.Printf("Hoist%s %v from %v to %v in %v\n",
			hoistType, val.LongString(), src, dest, dest.Func.Name)
	}
}

func moveTo(val *Value, block *Block) {
	for valIdx, v := range val.Block.Values {
		if val != v {
			continue
		}
		val.moveTo(block, valIdx)
		break
	}
}

func isMemoryDef(val *Value) bool {
	switch val.Op {
	case OpStore, OpMove, OpZero, OpStoreWB, OpMoveWB, OpZeroWB,
		OpPanicBounds, OpPanicExtend,
		OpPubBarrier,
		OpVarDef, OpVarLive, OpKeepAlive:
		return true
	}
	return false
}

// alwaysExecute checks if Value is guaranteed to execute during loop iterations
// Otherwise, it should not be hoisted. The most common cases are invariants
// guarded by a conditional expression.
// TODO: If we can prove that Value can speculative execute nevertheless, e.g.
// Load from non-null pointer, this is not really necessary
func alwaysExecute(sdom SparseTree, loop *loop, val *Value) bool {
	block := val.Block
	// Because loop header can always jump to the loop exit, all blocks
	// inside the loop are never post-dominated by any loop exit.
	// Therefore, we need to first apply loop rotation to eliminate the path
	// from the loop header to the loop exit.
	for _, exit := range loop.exits {
		if exit == loop.exit {
			if !sdom.IsAncestorEq(block, loop.latch) {
				return false
			}
			continue
		}
		if !sdom.IsAncestorEq(block, exit) {
			return false
		}
	}
	return true
}

func isHoistable(val *Value) bool {
	// The protagonist of the whole story
	switch val.Op {
	case OpLoad, OpStore, OpNilCheck, OpGetG, OpVarDef, OpConvert:
		return true
	}
	return false
}

type hoister struct {
	fn      *Func
	sdom    SparseTree
	ln      *loopnest
	hoisted map[*Value]bool
}

func (h *hoister) hoist(block *Block, val *Value) {
	if arg := val.MemoryArg(); arg != nil {
		// If val produces memory, all its uses should be replaced with incoming
		// memory input of val
		if isMemoryDef(val) {
			mem := arg
			for _, b := range h.fn.Blocks {
				b.replaceUses(val, mem)
			}
		}
	}

	srcBlock := val.Block
	moveTo(val, block)
	logInvariant(val, srcBlock, block)
	h.hoisted[val] = true
}

// tryHoist hoists profitable loop invariant to block that dominates the entire
// loop. Value is considered as loop invariant if all its inputs are defined
// outside the loop or all its inputs are loop invariants. Since loop invariant
// will immediately moved to dominator block of loop, the first rule actually
// already implies the second rule
func (h *hoister) tryHoist(loop *loop, invariants loopInvariants, val *Value) bool {
	// Value is already hoisted
	if hoisted, exist := h.hoisted[val]; exist {
		return hoisted
	}
	// Value is type of Phi, we can not hoist it now
	if val.Op == OpPhi {
		h.hoisted[val] = false
		return false
	}

	// Try to hoist arguments of value first, they are guaranteed to be loop
	// invariants but not necessarily hoistable
	h.hoisted[val] = false
	for _, arg := range val.Args {
		if arg.Type.IsMemory() {
			if !isMemoryDef(arg) {
				continue
			}
		}
		if _, isInvariant := invariants[arg]; isInvariant {
			if !h.tryHoist(loop, invariants, arg) {
				return false
			}
		} else {
			// Value is not loop invariant, it must dominate the loop header
			// or type of memory, simply check it
			if arg.Op != OpUnknown && arg.Op != OpInvalid &&
				!arg.Type.IsMemory() &&
				!h.sdom.IsAncestorEq(arg.Block, loop.header) {
				h.fn.Fatalf("arg %v must define outside loop", arg)
			}
		}
	}

	// This catches most common case, e.g. arithmetic, bit operation, etc.
	if !isAccessMemory(val) {
		assert(val.MemoryArg() == nil, "sanity check")
		h.hoist(loop.land, val)
		return true
	}

	// Instructions are selected ones?
	if isHoistable(val) {
		assert(loop.IsRotatedForm(), "loop must be rotated")

		// Instructions are guaranteed to execute unconditionally?
		if !alwaysExecute(h.sdom, loop, val) {
			if h.fn.pass.debug > 1 {
				fmt.Printf("LICM failure: %v not always execute\n", val.LongString())
			}
			return false
		}

		h.hoist(loop.land, val)
		return true
	}

	if h.fn.pass.debug > 1 {
		fmt.Printf("LICM failure: %v is not hoistable\n", val.LongString())
	}
	return false
}

// Hoisting memory def to loop land may break memory state of loop header, this
// should be fixed after CFG transformation done
func (h *hoister) fixMemoryState(loop *loop, startMem, endMem []*Value) {
	// No instruction hoisted? Do nothing them
	if len(h.hoisted) == 0 {
		return
	}

	// Find last memory def in loop entry, which in turns become last memory
	// or loop guard, this implies that loop guard can not contain memory def
	lastMem := endMem[loop.entry.ID]
	for _, val := range loop.guard.Values {
		if isMemoryDef(val) {
			h.fn.Fatalf("Loop guard %v contains memory def %v", loop.guard, val)
		}
	}

	// Find last memory def in loop land
	oldLastMem := lastMem
	for _, val := range loop.land.Values {
		if arg := val.MemoryArg(); arg != nil {
			val.SetArg(len(val.Args)-1, lastMem)
		}
		if isMemoryDef(val) {
			lastMem = val
		}
	}

	// If loop land has new memory def, memory state of loop header should be
	// updated as well
	if oldLastMem != lastMem {
		headerMem := startMem[loop.header.ID]
		if headerMem == nil {
			h.fn.Fatalf("Canot find start memory of loop header %v", loop.header)
		}
		if headerMem.Op == OpPhi {
			landIdx := -1
			for idx, pred := range loop.header.Preds {
				if pred.b == loop.land {
					landIdx = idx
					break
				}
			}
			headerMem.SetArg(landIdx, lastMem)
		} else {
			loop.header.replaceUses(headerMem, lastMem)
		}
	}
}

type loopInvariants map[*Value]bool

func stableKeys(li loopInvariants) []*Value {
	keys := make([]*Value, 0)
	for k, _ := range li {
		keys = append(keys, k)
	}
	sort.SliceStable(keys, func(i, j int) bool {
		return keys[i].ID < keys[j].ID
	})
	return keys
}

// findInviant finds all loop invariants within the loop
func (loop *loop) findInvariant(ln *loopnest) loopInvariants {
	loopValues := make(map[*Value]bool)
	invariants := make(map[*Value]bool)
	loopBlocks := ln.findLoopBlocks(loop)

	// First, collect all def inside loop
	hasLoad, hasStore := false, false
	for _, block := range loopBlocks {
		for _, value := range block.Values {
			if value.Op == OpLoad {
				hasLoad = true
			} else if value.Op == OpStore {
				hasStore = true
			} else if value.Op.IsCall() {
				if ln.f.pass.debug > 1 {
					fmt.Printf("LICM failure: find call %v\n", value.LongString())
				}
				return nil
			}
			loopValues[value] = true
		}
	}

	// See if loop contains both Load and Store and pessimistically assume that
	// they are must-aliases and stop optimizing
	// TODO: We can do better here by using type-based alias analysis in
	// some cases
	if hasLoad && hasStore {
		if ln.f.pass.debug > 1 {
			fmt.Printf("LICM failure: %v has both load and store\n", loop)
		}
		return nil
	}

	changed := true
	for changed {
		numInvar := len(invariants)
		for val, _ := range loopValues {
			// If basic block is located in a nested loop rather than directly in
			// the current loop, it will not be processed.
			if ln.b2l[val.Block.ID] != loop {
				continue
			}
			isInvariant := true
			for _, use := range val.Args {
				if use.Type.IsMemory() {
					// Discard last memory value
					continue
				}
				if _, exist := invariants[use]; exist {
					continue
				}
				if _, exist := loopValues[use]; exist {
					isInvariant = false
					break
				}
			}
			if isInvariant {
				invariants[val] = true
			}
		}
		changed = (len(invariants) != numInvar)
	}

	return invariants
}

// licm stands for Loop Invariant Code Motion, it hoists expressions that computes
// the same value outside loop
func licm(fn *Func) {
	loopnest := fn.loopnest()
	if loopnest.hasIrreducible {
		return
	}
	if len(loopnest.loops) == 0 {
		return
	}

	loopnest.assembleChildren()
	loopnest.findExits()
	lcssa := make(map[*loop]bool, 0)

	// Transform all loops to loop closed form
	for _, loop := range loopnest.loops {
		lcssa[loop] = fn.BuildLoopClosedForm(loopnest, loop)
	}

	h := &hoister{
		fn:      fn,
		ln:      loopnest,
		hoisted: make(map[*Value]bool),
	}
	// Remember initial memory subgraph before LICM
	startMem, endMem := memState(fn)
	for _, loop := range loopnest.loops {
		// See if loop is in form of LCSSA
		if wellFormed := lcssa[loop]; !wellFormed {
			continue
		}

		// Rotate the loop to ensures that loop executes at least once
		if !fn.RotateLoop(loop) {
			continue
		}

		// Find loop invariants within the loop
		invariants := loop.findInvariant(loopnest)
		if invariants == nil || len(invariants) == 0 {
			continue
		}

		// Create a home for hoistable Values after rotation
		if !loop.CreateLoopLand(fn) {
			fn.Fatalf("Can not create loop land for %v", loop.LongString())
		}

		// All prerequisites are satisfied, try to hoist loop invariants
		h.sdom = fn.Sdom()
		for _, val := range stableKeys(invariants) {
			h.tryHoist(loop, invariants, val)
		}

		// Fix broken memory state given that CFG no longer changes
		h.fixMemoryState(loop, startMem, endMem)
	}
}
