// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
)

// ----------------------------------------------------------------------------
// Loop Rotation
//
// Loop rotation transforms while/for loop to do-while style loop. The original
// natural loop is in form of below IR
//
//	 loop entry
//	     │
//	     │  ┌───loop latch
//	     ▼  ▼       ▲
//	loop header     │
//	     │  │       │
//	     │  └──►loop body
//	     ▼
//	 loop exit
//
// We move the conditional test from loop header to loop latch, incoming backedge
// argument of conditional test should be updated as well otherwise we would lose
// one update. Also note that any other uses of moved values should be updated
// because moved Values now live in loop latch and may no longer dominates their
// uses. At this point, loop latch determines whether loop continues or exits
// based on rotated test.
//
//	loop entry
//	    │
//	    │
//	    ▼
//	loop header◄──┐
//	    │         │
//	    │         │
//	    ▼         │
//	loop body     │
//	    │         │
//	    │         │
//	    ▼         │
//	loop latch────┘
//	    │
//	    │
//	    ▼
//	loop exit
//
// Now loop header and loop body are executed unconditionally, this may changes
// program semantics while original program executes them only if test is okay.
// A so-called loop guard is inserted to ensure loop is executed at least once.
//
//	   loop entry
//	       │
//	       │
//	       ▼
//	┌──loop guard
//	│      │
//	│      │
//	│      ▼
//	│  loop header◄──┐
//	│      │         │
//	│      │         │
//	│      ▼         │
//	│  loop body     │
//	│      │         │
//	│      │         │
//	│      ▼         │
//	│  loop latch────┘
//	│      │
//	│      │
//	│      ▼
//	└─► loop exit
//
// Loop header no longer dominates entire loop, loop guard dominates it instead.
// If Values defined in the loop were used outside loop, all these uses should be
// replaced by a new Phi node at loop exit which merges control flow from loop
// header and loop guard. Based on Loop Closed SSA Form, these Phis have already
// been created. All we need to do is simply reset their operands to accurately
// reflect the fact that loop exit is a merge point now.
//
// One of the main purposes of Loop Rotation is to assist other optimizations
// such as LICM. They may require that the rotated loop has a proper while safe
// block to place new Values, an optional loop land block is hereby created to
// give these optimizations a chance to keep them from being homeless.
//
//	   loop entry
//	       │
//	       │
//	       ▼
//	┌──loop guard
//	│      │
//	│      │
//	│      ▼
//	|  loop land  <= safe land to place Values
//	│      │
//	│      │
//	│      ▼
//	│  loop header◄──┐
//	│      │         │
//	│      │         │
//	│      ▼         │
//	│  loop body     │
//	│      │         │
//	│      │         │
//	│      ▼         │
//	│  loop latch────┘
//	│      │
//	│      │
//	│      ▼
//	└─► loop exit
//
// The detailed loop rotation algorithm is summarized as following steps
//
//  1. Transform the loop to Loop Closed SSA Form
//     * All uses of loop defined Values will be replaced by uses of proxy phis
//
//  2. Check whether loop can apply loop rotate
//     * Loop must be a natural loop and have a single exit and so on..
//
//  3. Rotate loop conditional test and rewire loop edges
//     * Rewire loop header to loop body unconditionally.
//     * Rewire loop latch to header and exit based on new conditional test.
//     * Create new loop guard block and rewire loop entry to loop guard.
//     * Clone conditional test from loop header to loop guard.
//     * Rewire loop guard to original loop header and loop exit
//
//  4. Reconcile broken data dependency after CFG transformation
//     * Move conditional test from loop header to loop latch
//     * Update uses of moved Values because these defs no longer dominates uses
//       after they were moved to loop latch
//     * Add corresponding argument for phis at loop exits since new edge from
//       loop guard to loop exit had been created
//     * Update proxy phi to use the loop phi's incoming argument which comes
//       from loop latch since loop latch may terminate the loop now

// checkLoopForm checks if loop is well formed and returns failure reason if not
func (loop *loop) checkLoopForm(fn *Func, sdom SparseTree) string {
	loopHeader := loop.header
	// Check if loop header is well formed block
	if len(loopHeader.Preds) != 2 || len(loopHeader.Succs) != 2 ||
		loopHeader.Kind != BlockIf {
		return "bad loop header"
	}

	// Check if loop exit nears the loop header
	fn.loopnest().findExits() // initialize loop exits
	e1, e2 := loopHeader.Succs[1].b, loopHeader.Succs[0].b
	found := false
	for _, exit := range loop.exits {
		if exit == e1 {
			loop.exit = e1
			loop.body = loopHeader.Succs[0].b
			found = true
			break
		} else if exit == e2 {
			loop.exit = e2
			loop.body = loopHeader.Succs[1].b
			found = true
			break
		}
	}
	if !found {
		return "far loop exit beyond header"
	}

	loop.latch = loopHeader.Preds[1].b

	// Check if loop header dominates all loop exits
	if len(loop.exits) != 1 {
		for _, exit := range loop.exits {
			if exit == loop.exit {
				continue
			}
			// Loop header may not dominate all loop exist, given up for these
			// exotic guys
			if !sdom.IsAncestorEq(loopHeader, exit) {
				return "loop exit is not dominated by header"
			}
		}
	}

	// Check loop conditional test is "trivial"
	for _, ctrl := range loop.header.ControlValues() {
		if !loop.isTrivial(sdom, ctrl, true) {
			return "non trivial loop cond"
		}
	}

	// Check if all loop uses are "trivial"
	for ipred, pred := range loop.exit.Preds {
		if pred.b == loop.header {
			for _, val := range loop.exit.Values {
				// TODO: Relax or remove this restriction
				if val.Op == OpPhi {
					if arg := val.Args[ipred]; arg.Block == loop.header {
						if !loop.isTrivial(sdom, arg, false) {
							return "use non trivial loop def outside loop"
						}
					}
				} else if val.Block == loop.header {
					if !loop.isTrivial(sdom, val, false) {
						return "use non trivial loop def outside loop"
					}
				}
			}
			break
		}
	}
	return ""
}

// A loop def is "trivial" if, starting from the value, it is looked up along its
// argument until it encounters the loop phi defined in the loop header, no
// intractable values are encountered in the process, or the lookup depth does
// not exceed the MaxDepth. We need this restriction because all the values in
// the chain from the loop phi to the trivial loop def could be cloned into other
// block, and cloning without careful scrutiny would lead to code bloat and extra
// performance penalty.
const (
	InitDepth = 0
	MaxDepth  = 5
)

type loopTrivialVal struct {
	cloning  bool
	valBlock *Block
	touched  map[*Value]*Value
	visited  map[*Value]bool
}

func (t *loopTrivialVal) clone(val *Value, dest *Block, depth int) *Value {
	// If seeing Phi or value that lives different from source block? They must
	// not part of trivial loop def chain, do nothing
	if val.Op == OpPhi || val.Block != t.valBlock {
		return val
	}

	// If val is already cloned? Use cloned value instead.
	if c, exist := t.touched[val]; exist {
		return c
	}

	// Clone val and its arguments recursively
	clone := dest.Func.newValueNoBlock(val.Op, val.Type, val.Pos)
	clone.AuxInt = val.AuxInt
	clone.Aux = val.Aux
	args := make([]*Value, len(val.Args))
	for i := 0; i < len(val.Args); i++ {
		args[i] = t.clone(val.Args[i], dest, depth+1)
	}
	clone.AddArgs(args...)
	dest.placeValue(clone)
	t.touched[val] = clone // cache cloned value after cloning its arguments
	return clone
}

func (t *loopTrivialVal) move(val *Value, dest *Block, depth int) {
	if val.Op == OpPhi || val.Block != t.valBlock {
		return
	}
	for _, arg := range val.Args {
		t.move(arg, dest, depth+1)
	}
	moveTo(val, dest)
}

func (t *loopTrivialVal) update(val *Value, loop *loop, loopPhiIdx, depth int) {
	// It's a Phi or value that lives different from source block? It must not
	// part of trivial loop def chain, do nothing
	if val.Op == OpPhi || val.Block != t.valBlock {
		return
	}
	if _, hasCycle := t.visited[val]; hasCycle {
		// Just skip it to avoid infinite recursion
		return
	}
	t.visited[val] = true
	for iarg, arg := range val.Args {
		// If arg of val is a Phi which lives in loop header?
		if arg.Op == OpPhi && arg.Block == loop.header {
			// If expected incoming argument of arg is not visited, this implies
			// that it may comes from loop latch, this is the most common case,
			// update val to use incoming argument instead of arg. Otherwise,
			// there is a cyclic dependency, see below for more details.
			newUse := arg.Args[loopPhiIdx]
			if _, livesInHeader := t.touched[newUse]; !livesInHeader {
				// In original while/for loop, a critical edge is inserted at the
				// end of each iteration, Phi values are updated. All subsequent
				// uses of Phi rely on updated values. However, when converted
				// to a do-while loop, Phi nodes may be used at the end of each
				// iteration before they are updated. Therefore, we need to
				// replace all subsequent uses of Phi with use of Phi parameter.
				// This way, it is equivalent to using updated values of Phi
				// values. Here is a simple example:
				//
				// Normal case, if v2 uses v1 phi, and the backedge operand v4
				// of v1 phi is located in the loop latch block, we only need to
				// modify the usage of v1 by v2 to the usage of v4. This prevents
				// loss of updates, and the dominance relationship will not be
				// broken even after v2 is moved to the loop latch.
				//
				// Before:
				//  loop header:
				//  v1 = phi(0, v4)
				//  v2 = v1 + 1
				//  If v2 < 3 -> loop body, loop exit
				//
				//  loop latch:
				//  v4 = const 512
				//
				// After:
				//  loop header:
				//  v1 = phi(0, v4)
				//
				//  loop latch:
				//  v4 = const 512
				//  v2 = v4 + 1
				//  If v2 < 3 -> loop header, loop exit
				val.SetArg(iarg, newUse)
				// After updating uses of val, we may create yet another cyclic
				// dependency, i.e.
				//
				//  loop header:
				//  v1 = phi(0, v4)
				//  v2 = v1 + 1
				//  If v2 < 3 -> loop body, loop exit
				//
				//  loop latch:
				//  v4 = v2 + 1
				//
				// After updating iarg of val to newUse, it becomes
				//
				//  loop header:
				//  v1 = phi(0, v4)
				//
				//  loop latch:
				//  v2 = v4 + 1   ;;; cyclic dependency
				//  v4 = v2 + 1
				//  If v2 < 3 -> loop header, loop exit
				//
				// This is similiar to below case, and it would be properly handled
				// by updateMovedUses. For now, we just skip it to avoid infinite
				// recursion.
			} else {
				// If there is a value v1 in the loop header that is used to define
				// a v2 phi in the same basic block, and this v2 phi is used in
				// turn to use the value v1, there is a cyclic dependency, i.e.
				//
				//  loop header:
				//  v1 = phi(0, v2)   ;;; cyclic dependency
				//  v2 = v1 + 1
				//  If v2 < 3 -> loop body, loop exit
				//
				// In this case, we need to first convert the v1 phi into its
				// normal form, where its back edge parameter uses the value defined
				// in the loop latch.
				//
				//  loop header:
				//  v1 = phi(0, v3)
				//  v2 = v1 + 1
				//  If v2 < 3 -> loop body, loop exit
				//
				//  loop latch:
				//  v3 = Copy v2
				//
				// After this, the strange v1 phi is treated in the same way as
				// other phis. After moving the conditional test to the loop latch,
				// the relevant parameters will also be updated, i.e., v2 will
				// use v3 instead of v1 phi:
				//
				//  loop header:
				//  v1 = phi(0, v3)
				//
				//  loop latch:
				//  v3 = Copy v2
				//  v2 = v3 + 1
				//  If v2 < 3 -> loop header, loop exit
				//
				// Finally, since v3 is use of v2, after moving v2 to the loop
				// latch, updateMovedUses will update these uses and insert a
				// new v4 Phi.
				//
				//  loop header:
				//  v1 = phi(0, v3)
				//  v4 = phi(v2', v2)    ;;; v2' lives in loop guard
				//
				//  loop latch:
				//  v3 = Copy v4
				//  v2 = v3 + 1
				//  If v2 < 3 -> loop header, loop exit

				// Copy from cyclic dependency value and place it to loop latch
				fn := arg.Block.Func
				copy := fn.newValueNoBlock(OpCopy, arg.Type, arg.Pos)
				if t.cloning {
					// If we are cloning, we need to be very careful when updating
					// the clonee, not the clone, otherwise, it can lead to another
					// disastrous circular dependencies, e.g.
					//
					//  loop header:
					//  v1 = phi(0, v3)
					//
					//  loop latch:
					//  v3 = Copy v2
					//  v2 = v3 + 1
					//  If v2 < 3 -> loop header, loop exit
					//
					//  critical block(between loop latch and loop exit):
					//  v3' = Copy v2    ;;; copy from v2 instead of v2'
					//  v2' = v3' + 1
					for clonee, clone := range t.touched {
						if clone == val {
							copy.SetArgs1(clonee)
							break
						}
					}
					if len(copy.Args) == 0 {
						fn.Fatalf("can not found clone from clonee")
					}
				} else {
					copy.SetArgs1(newUse)
				}
				loop.latch.placeValue(copy)
				// Replace incoming argument of loop phi to copied value
				arg.SetArg(loopPhiIdx, copy)
				// Update val to use copied value as usual
				val.SetArg(iarg, copy)

				if fn.pass.debug > 1 {
					fmt.Printf("== Insert %v during updating %v\n", copy, val)
				}
			}
		} else {
			t.update(arg, loop, loopPhiIdx, depth+1)
		}
	}
}

func (t *loopTrivialVal) valid(sdom SparseTree, val *Value, allowSideEffect bool, depth int) bool {
	if depth >= MaxDepth {
		return false
	}

	if sdom.isAncestor(val.Block, t.valBlock) {
		return true
	}

	if val.Op == OpPhi {
		if val.Block == t.valBlock {
			return true
		}
		return false
	}

	if !allowSideEffect {
		if val.Op != OpLoad && isAccessMemory(val) {
			return false
		}
	}

	for _, arg := range val.Args {
		if !t.valid(sdom, arg, allowSideEffect, depth+1) {
			return false
		}
	}
	return true
}

// isTrivial checks if val is "trivial" and returns true if it is, otherwise false.
func (loop *loop) isTrivial(sdom SparseTree, val *Value, allowSideEffect bool) bool {
	t := &loopTrivialVal{
		valBlock: loop.header,
	}
	return t.valid(sdom, val, allowSideEffect, InitDepth)
}

// cloneTrivial clones val to destination block and updates its uses accordingly
func (loop *loop) cloneTrivial(val *Value, dest *Block, loopPhiIdx int) (*Value, map[*Value]*Value) {
	t := &loopTrivialVal{
		cloning:  true,
		valBlock: val.Block,
		touched:  make(map[*Value]*Value),
		visited:  make(map[*Value]bool),
	}
	clone := t.clone(val, dest, InitDepth)
	t.valBlock = dest
	t.update(clone, loop, loopPhiIdx, InitDepth)
	return clone, t.touched
}

// moveTrivial moves val to destination block and updates its uses accordingly
func (loop *loop) moveTrivial(val *Value, dest *Block, cloned map[*Value]*Value, loopPhiIdx int) {
	t := &loopTrivialVal{
		cloning:  false,
		valBlock: val.Block,
		visited:  make(map[*Value]bool),
	}
	t.move(val, dest, InitDepth)
	t.valBlock = dest
	t.touched = cloned
	t.update(val, loop, loopPhiIdx, InitDepth)
}

// moveCond moves conditional test from loop header to loop latch
func (loop *loop) moveCond(cond *Value, cloned map[*Value]*Value) {
	if cond.Block != loop.header {
		// More rare, ctrl Value is not live in loop header, do nothing
		return
	}

	if cond.Op == OpPhi {
		// Rare case, Phi is used as conditional test, use its incoming argument
		//     If (Phi v1 v2) -> loop body, loop exit
		// =>  If v1          -> loop header, loop exit
		cond = cond.Args[LoopLatch2HeaderPredIdx]
		loop.latch.SetControl(cond)
		return
	}

	// Normal case, update as usual
	//    If (Less v1 Phi(v2 v3)) -> loop body, loop exit
	// => If (Less v1 v2)         -> loop header, loop exit
	loop.moveTrivial(cond, loop.latch, cloned, LoopLatch2HeaderPredIdx)
}

// cloneCond clones conditional test from loop header to loop guard
func (loop *loop) cloneCond(cond *Value) (*Value, map[*Value]*Value) {
	if cond.Block != loop.header {
		// Dont clone if ctrl Value is not live in loop header
		return cond, nil
	}

	if cond.Op == OpPhi {
		// Use incoming argument of Phi as conditional test directly
		guardCond := cond.Args[LoopGuard2HeaderPredIdx]
		return guardCond, nil
	}

	// Clone as usual
	return loop.cloneTrivial(cond, loop.guard, LoopGuard2HeaderPredIdx)
}

const (
	LoopGuard2HeaderPredIdx = 0
	LoopLatch2HeaderPredIdx = 1
)

// rewireLoopHeader rewires loop header to loop body unconditionally
func (loop *loop) rewireLoopHeader() {
	loopHeader := loop.header
	loopHeader.Reset(BlockPlain)

	// loopHeader -> loopBody(0)
	loopHeader.Succs = loopHeader.Succs[:1]
	loopHeader.Succs[0] = Edge{loop.body, 0}
	assert(len(loop.body.Preds) == 1, "why not otherwise")
	loop.body.Preds[0] = Edge{loopHeader, 0}
}

// rewireLoopLatch rewires loop latch to loop header and loop exit
func (loop *loop) rewireLoopLatch(ctrl *Value, exitIdx int) {
	loopExit := loop.exit
	loopLatch := loop.latch
	loopHeader := loop.header
	loopLatch.resetWithControl(BlockIf, ctrl)
	loopLatch.Likely = loopHeader.Likely
	loopLatch.Pos = ctrl.Pos
	loopHeader.Likely = BranchUnknown

	var idx = -1
	for i := 0; i < len(loopExit.Preds); i++ {
		if loopExit.Preds[i].b == loop.header {
			idx = i
			break
		}
	}
	if exitIdx == 1 {
		// loopLatch -> loopHeader(0), loopExit(1)
		loopLatch.Succs = append(loopLatch.Succs, Edge{loopExit, idx})
	} else {
		// loopLatch -> loopExit(0), loopHeader(1)
		loopLatch.Succs = append([]Edge{{loopExit, idx}}, loopLatch.Succs[:]...)
	}
	// loopExit <- loopLatch, ...
	loopExit.Preds[idx] = Edge{loopLatch, exitIdx}
	// loopHeader <- loopLatch, ...
	for i := 0; i < len(loopHeader.Preds); i++ {
		if loopHeader.Preds[i].b == loopLatch {
			idx = i
			break
		}
	}
	loopHeader.Preds[idx] = Edge{loopLatch, 1 - exitIdx}
}

// rewireLoopGuard rewires loop guard to loop header and loop exit
func (loop *loop) rewireLoopGuard(guardCond *Value, exitIdx int) {
	assert(len(loop.guard.Preds) == 1, "already setup")
	loopHeader := loop.header
	loopGuard := loop.guard
	loopGuard.Pos = loopHeader.Pos
	loopGuard.Likely = loopHeader.Likely // respect header's branch predication
	loopGuard.SetControl(guardCond)

	var idx = -1
	assert(len(loopHeader.Preds) == 2, "sanity check")
	for i := 0; i < len(loopHeader.Preds); i++ {
		if loopHeader.Preds[i].b != loop.latch {
			idx = i
			break
		}
	}

	loopExit := loop.exit
	numExitPred := len(loopExit.Preds)
	if exitIdx == 1 {
		// loopGuard -> loopHeader(0), loopExit(1)
		loopGuard.Succs = append(loopGuard.Succs, Edge{loopHeader, idx})
		loopGuard.Succs = append(loopGuard.Succs, Edge{loopExit, numExitPred})
		loopExit.Preds = append(loopExit.Preds, Edge{loopGuard, 1})
		loopHeader.Preds[idx] = Edge{loopGuard, 0}
	} else {
		// loopGuard -> loopExit(0), loopHeader(1)
		loopGuard.Succs = append(loopGuard.Succs, Edge{loopExit, numExitPred})
		loopGuard.Succs = append(loopGuard.Succs, Edge{loopHeader, idx})
		loopExit.Preds = append(loopExit.Preds, Edge{loopGuard, 0})
		loopHeader.Preds[idx] = Edge{loopGuard, 1}
	}
}

// rewireLoopEntry rewires loop entry to loop guard
func (loop *loop) rewireLoopEntry(loopGuard *Block) {
	assert(len(loop.header.Preds) == 2, "sanity check")

	// Find loop entry from predecessor of loop header
	for _, pred := range loop.header.Preds {
		if pred.b != loop.latch {
			loop.entry = pred.b
			break
		}
	}
	assert(loop.entry != nil, "missing loop entry")

	// If loop entry is plain block, simply add edge from loop entry to guard
	loopEntry := loop.entry
	if len(loopEntry.Succs) == 1 {
		// loopEntry(0) -> loopGuard
		loopEntry.Succs = loopEntry.Succs[:0]
		loopEntry.AddEdgeTo(loopGuard)
	} else {
		// Rewire corresponding successor of loop entry to loop guard (This could
		// be constructed in artificial IR test, but does it really happen?...)
		var idx = -1
		for isucc, succ := range loopEntry.Succs {
			if succ.b == loop.header {
				idx = isucc
				break
			}
		}
		// loopEntry(idx) -> loopGuard, ...
		loopEntry.Succs[idx] = Edge{loopGuard, 0}
		loopGuard.Preds = append(loopGuard.Preds, Edge{loopEntry, idx})
	}
}

// insertBetween inserts an empty block in the middle of start and end block.
// If such block already exists, it will be returned instead.
func insertBetween(fn *Func, start, end *Block) *Block {
	for _, succ := range start.Succs {
		if succ.b == end {
			break
		} else if len(succ.b.Succs) == 1 && succ.b.Succs[0].b == end {
			return succ.b
		}
	}
	empty := fn.NewBlock(BlockPlain)
	empty.Preds = make([]Edge, 1, 1)
	empty.Succs = make([]Edge, 1, 1)
	start.ReplaceSucc(end, empty, 0)
	end.ReplacePred(start, empty, 0)
	return empty
}

func (loop *loop) findLoopGuardIndex() int {
	if loop.header.Preds[0].b == loop.latch {
		return 1
	}
	return 0
}

func (loop *loop) findLoopBackedgeIndex() int {
	return 1 - loop.findLoopGuardIndex()
}

// Loop header no longer dominates loop exit, a new edge from loop guard to loop
// exit is created, this is not reflected in proxy phis in loop exits, i.e. these
// proxy phis miss one argument that comes from loop guard, we need to reconcile
// the divergence
//
//	                              loop guard
//	                                   |
//	loop exit               loop exit  /
//	    |          =>            |    /
//	v1=phi(v1)              v1=phi(v1 v1') <= add missing g2e argument v1'
//
// Since LCSSA ensures that all loop uses are closed, i.e. any out-of-loop uses
// are replaced by proxy phis in loop exit, we only need to add missing argument
// v1' to v1 proxy phi
func (loop *loop) addG2EArg(fn *Func, sdom SparseTree) {
	var holder *Block
	for _, val := range loop.exit.Values {
		// Not even a phi?
		if val.Op != OpPhi {
			continue
		}
		// Num of args already satisfies the num of predecessors of loop exit?
		if len(val.Args) == len(loop.exit.Preds) {
			continue
		}
		if len(val.Args)+1 != len(loop.exit.Preds) {
			fn.Fatalf("Only miss one g2e arg")
		}
		assert(val.Block == loop.exit, "sanity check")

		// If arguments of the phi is not matched with predecessors of loop exit,
		// then add corresponding g2e argument to reflect the new edge from loop
		// guard to loop exit
		var g2eArg *Value // loop guard to loop exit
		for iarg, arg := range val.Args {
			exitPred := val.Block.Preds[iarg].b
			// If this predecessor is either loop header or inserted block?
			if exitPred == loop.latch || exitPred == holder {
				if sdom.isAncestor(arg.Block, loop.header) {
					// arg lives in block that dominates loop header, it could
					// be used as g2eArg directly
					g2eArg = arg
				} else if arg.Block == loop.header {
					// arg lives in loop header, find its counterpart from loop
					// guard or create a new one if not exist
					guardIdx := loop.findLoopGuardIndex()

					// It's a phi? Simply use its incoming argument that comes
					// from loop guard as g2eArg
					if arg.Op == OpPhi {
						g2eArg = arg.Args[guardIdx]
					} else {
						// Otherwise, split critical edge from loop guard to exit
						// and clone arg into new block, it becomes new g2eArg
						holder = insertBetween(fn, loop.guard, loop.exit)
						guardArg, _ := loop.cloneTrivial(arg, holder, guardIdx)
						g2eArg = guardArg
					}
				}
			}
		}

		// Add g2e argument for phi to reconcile the divergence between the num
		// of block predecessors and the num of phi arguments
		if g2eArg == nil {
			fn.Fatalf("Can not create new g2e arg for %v", val.LongString())
		}
		newArgs := make([]*Value, len(loop.exit.Preds))
		copy(newArgs, val.Args)
		newArgs[len(newArgs)-1] = g2eArg
		oldVal := val.LongString()
		val.resetArgs()
		val.AddArgs(newArgs...)
		if fn.pass.debug > 1 {
			fmt.Printf("== Add g2e argument %v to %v(%v)\n",
				g2eArg, val.LongString(), oldVal)
		}
	}
}

func (loop *loop) findGuardArg(fn *Func, val *Value) *Value {
	assert(val.Block == loop.header, "mirror comes from loop header")
	guardIdx := loop.findLoopGuardIndex()

	// It's a phi? Simply use its incoming argument that comes from loop guard
	// as counterpart
	if val.Op == OpPhi {
		return val.Args[guardIdx]
	}

	// Otherwise, split critical edge from loop guard to loop exit and
	// clone arg into the new block, this is the new counterpart
	holder := insertBetween(fn, loop.guard, loop.exit)
	guardArg, _ := loop.cloneTrivial(val, holder, guardIdx)
	return guardArg
}

func (loop *loop) findBackedgeArg(fn *Func, val *Value, start, end *Block) *Value {
	assert(val.Block == loop.header, "mirror comes from loop header")
	backedgeIdx := loop.findLoopBackedgeIndex()

	// It's a phi? Simply use its incoming argument that comes from loop latch
	// as counterpart
	if val.Op == OpPhi {
		return val.Args[backedgeIdx]
	}

	// Otherwise, split edge from start to end and clone arg into the new block,
	// this is the new counterpart
	holder := insertBetween(fn, start, end)
	backedgeArg, _ := loop.cloneTrivial(val, holder, backedgeIdx)
	return backedgeArg
}

// Loop latch now terminates the loop. If proxy phi uses the loop phi that lives
// in loop header, it should be replaced by using the loop phi's incoming argument
// which comes from loop latch instead, this avoids losing one update.
//
//	Before:
//	 loop header:
//	 v1 = phi(0, v4)
//
//	 loop latch:
//	 v4 = v1 + 1
//
//	 loop exit
//	 v3 = phi(v1, ...)
//
//	After:
//	 loop header:
//	 v1 = phi(0, v4)
//
//	 loop latch:
//	 v4 = v1 + 1
//
//	 loop exit
//	 v3 = phi(v4, ...)  ;; use v4 instead of v1
func (loop *loop) updateLoopUse(fn *Func) {
	fn.invalidateCFG()
	sdom := fn.Sdom()

	for _, loopExit := range loop.exits {
		// The loop exit is still dominated by loop header?
		if sdom.isAncestor(loop.header, loopExit) {
			continue
		}
		// Loop header no longer dominates this loop exit, find the corresponding
		// incoming argument and update it for every phi in exit block
		for _, val := range loopExit.Values {
			if val.Op != OpPhi {
				continue
			}

			sdom := fn.Sdom()
			loopExit := val.Block
			for iarg, arg := range val.Args {
				// Only arg lives in the loop header is of interest
				if arg.Block != loop.header {
					continue
				}
				// See if corresponding predecessor was not dominated by loop
				// header, if so, use corresponding argument to avoid losing one
				exitPred := loopExit.Preds[iarg].b
				if !sdom.isAncestor(loop.header, exitPred) {
					newArg := loop.findGuardArg(fn, arg)
					val.SetArg(iarg, newArg)
					if fn.pass.debug > 1 {
						fmt.Printf("== Update guard arg %v\n", val.LongString())
					}
					continue
				}

				// If the predecessor of loop exit was dominated by loop latch,
				// use corresponding argument to avoid losing one update
				if sdom.IsAncestorEq(loop.latch, exitPred) {
					newArg := loop.findBackedgeArg(fn, arg, exitPred, loopExit)
					val.SetArg(iarg, newArg)
					if fn.pass.debug > 1 {
						fmt.Printf("== Update backedge arg %v\n", val.LongString())
					}
					continue
				}
			}
		}
	}
}

// If the loop conditional test is "trivial", we will move the chain of this
// conditional test values to the loop latch, after that, they may not dominate
// the in-loop uses anymore:
//
//	loop header
//	v1 = phi(0, ...)
//	v2 = v1 + 1
//	If v2 < 3 ...
//
//	loop body:
//	v4 = v2 - 1
//
// So we need to create a new phi v5 at the loop header to merge the control flow
// from the loop guard to the loop header and the loop latch to the loop header
// and use this phi to replace the in-loop use v4. e.g.
//
//	loop header:
//	v1 = phi(0, ...)
//	v5 = phi(v2', v2)     ;;; v2' lives in loop guard
//
//	loop body:
//	v4 = v5 - 1
//
//	loop latch:
//	v2 = v1 + 1
//	If v2 < 3 ...
func (loop *loop) updateMovedUses(fn *Func, cloned map[*Value]*Value) {
	// Find all moved values and sort them in order to ensure determinism
	moved := make([]*Value, 0)
	for key, _ := range cloned {
		moved = append(moved, key)
	}
	sort.SliceStable(moved, func(i, j int) bool {
		return moved[i].ID < moved[j].ID
	})

	// One def may have multiple uses, all of these uses should be replaced by
	// the same def replacement
	replacement := make(map[*Value]*Value)
	// For each of moved value, find its uses inside loop
	defUses := buildDefUses(fn, moved)
	for _, def := range moved {
		uses := defUses[def]
		if def.Uses == 1 {
			assert(uses[0].useBlock() == loop.latch, "used by another moved val")
			continue
		}
		// For each use of def, if it is not one of the moved values or loop phi
		// in loop header, replace it with inserted Phi
		for _, use := range uses {
			// Used by other moved value or by loop phi in header? Skip them as
			// they are not needed to update
			if use.val != nil {
				if _, exist := cloned[use.val]; exist {
					continue
				}
				if use.val.Op == OpPhi && use.val.Block == loop.header {
					continue
				}
			} else {
				if _, exist := cloned[use.block.ControlValues()[0]]; exist {
					continue
				}
			}
			// Since LCSSA ensures that all uses of loop defined values are in
			// loop we can safely do replacement then
			// TODO: Add verification here to check if it does lives inside loop

			// Create phi at loop header, merge control flow from loop guard and
			// loop latch, and replace use with such phi. If phi already exists,
			// use it instead of creating a new one.
			var newUse *Value
			if phi, exist := replacement[def]; exist {
				newUse = phi
			} else {
				phi := fn.newValueNoBlock(OpPhi, def.Type, def.Pos)
				// Merge control flow from loop guard and loop latch
				arg1 := cloned[def]
				arg2 := def
				if arg1.Block != loop.guard {
					fn.Fatalf("arg1 must be live in loop guard")
				}
				if arg2.Block != loop.latch {
					fn.Fatalf("arg2 must be live in loop latch")
				}
				phi.AddArg2(arg1, arg2)
				loop.header.placeValue(phi)
				replacement[def] = phi
				newUse = phi
			}
			if fn.pass.debug > 1 {
				fmt.Printf("== Update moved use %v %v\n", use, newUse.LongString())
			}
			use.replaceUse(newUse)
		}
	}
}

// verifyRotatedForm verifies if given loop is rotated form
func (loop *loop) verifyRotatedForm(fn *Func) {
	if len(loop.header.Succs) != 1 || len(loop.exit.Preds) < 2 ||
		len(loop.latch.Succs) != 2 || len(loop.guard.Succs) != 2 {
		fn.Fatalf("Bad loop %v after rotation", loop.LongString())
	}
}

// IsRotatedForm returns true if loop is rotated
func (loop *loop) IsRotatedForm() bool {
	if loop.guard == nil {
		return false
	}
	return true
}

// CreateLoopLand creates a land block between loop guard and loop header, it
// executes only if entering loop.
func (loop *loop) CreateLoopLand(fn *Func) bool {
	if !loop.IsRotatedForm() {
		return false
	}
	if loop.land != nil {
		return true
	}

	// loopGuard -> loopLand
	// loopLand -> loopHeader
	loop.land = insertBetween(fn, loop.guard, loop.header)

	return true
}

// RotateLoop rotates the original loop to become a do-while style loop, returns
// true if loop is rotated, false otherwise.
func (fn *Func) RotateLoop(loop *loop) bool {
	if loop.IsRotatedForm() {
		return true
	}

	// Check loop form and bail out if failure
	sdom := fn.Sdom()
	if msg := loop.checkLoopForm(fn, sdom); msg != "" {
		if fn.pass.debug > 0 {
			fmt.Printf("Exotic %v for rotation: %s %v\n", loop.LongString(), msg, fn.Name)
		}
		return false
	}

	exitIdx := 1 // which successor of loop header wires to loop exit
	if loop.header.Succs[0].b == loop.exit {
		exitIdx = 0
	}

	assert(len(loop.header.ControlValues()) == 1, "more than 1 ctrl value")
	cond := loop.header.Controls[0]

	// Rewire loop header to loop body unconditionally
	loop.rewireLoopHeader()

	// Rewire loop latch to header and exit based on new conditional test
	loop.rewireLoopLatch(cond, exitIdx)

	// Create loop guard block
	// TODO(yyang): Creation of loop guard can be skipped if original IR already
	// exists such form. e.g. if 0 < len(b) { for i := 0; i < len(b); i++ {...} }
	loopGuard := fn.NewBlock(BlockIf)
	loop.guard = loopGuard

	// Rewire entry to loop guard instead of original loop header
	loop.rewireLoopEntry(loopGuard)

	// Clone old conditional test and its arguments to control loop guard
	guardCond, cloned := loop.cloneCond(cond)

	// Rewire loop guard to original loop header and loop exit
	loop.rewireLoopGuard(guardCond, exitIdx)

	// CFG changes are all done here, then update data dependencies accordingly

	// Move conditional test from loop header to loop latch
	loop.moveCond(cond, cloned)

	// Update uses of moved Values because these defs no longer dominates uses
	// after they were moved to loop latch
	loop.updateMovedUses(fn, cloned)

	// Add corresponding argument for phis at loop exits since new edge from
	// loop guard to loop exit had been created
	loop.addG2EArg(fn, sdom)

	// Update proxy phi to use the loop phi's incoming argument which comes from
	// loop latch since loop latch may terminate the loop now
	loop.updateLoopUse(fn)

	// Gosh, loop is rotated
	loop.verifyRotatedForm(fn)

	if fn.pass.debug > 0 {
		fmt.Printf("%v rotated in %v\n", loop.LongString(), fn.Name)
	}
	fn.invalidateCFG()
	return true
}

func moveBlock(slice []*Block, from, to int) []*Block {
	if from < 0 || to < 0 || from >= len(slice) || to >= len(slice) {
		return slice
	}

	elem := slice[from]
	if from < to {
		copy(slice[from:], slice[from+1:to+1])
	} else {
		copy(slice[to+1:], slice[to:from])
	}

	slice[to] = elem
	return slice
}

// layoutLoop converts loops with a check-loop-condition-at-beginning
// to loops with a check-loop-condition-at-end by reordering blocks. no
// CFG changes here. This helps loops avoid extra unnecessary jumps.
//
//	 loop:
//	   CMPQ ...
//	   JGE exit
//	   ...
//	   JMP loop
//	 exit:
//
//	  JMP entry
//	loop:
//	  ...
//	entry:
//	  CMPQ ...
//	  JLT loop
func layoutLoop(f *Func) {
	loopnest := f.loopnest()
	if loopnest.hasIrreducible {
		return
	}
	if len(loopnest.loops) == 0 {
		return
	}

	for _, loop := range loopnest.loops {
		header := loop.header
		// If loop rotation is already applied, loop latch should be right after
		// all loop body blocks
		if header.Kind == BlockPlain && len(header.Succs) == 1 {
			continue
		}
		// Otherwise, place loop header right after all body blocks
		var latch *Block // b's in-loop predecessor
		for _, e := range header.Preds {
			if e.b.Kind != BlockPlain {
				continue
			}
			if loopnest.b2l[e.b.ID] != loop {
				continue
			}
			latch = e.b
		}
		if latch == nil || latch == header {
			continue
		}
		iheader, ilatch := 0, 0
		for ib, b := range f.Blocks {
			if b == header {
				iheader = ib
			} else if b == latch {
				ilatch = ib
			}
		}
		// Reordering the loop blocks from [header,body,latch] to [latch,body,header]
		f.Blocks = moveBlock(f.Blocks, iheader, ilatch)
	}
}
