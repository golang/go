// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "fmt"

const MaxLoopBlockSize = 8

func printInvariant(val *Value, block *Block, domBlock *Block) {
	fmt.Printf("== Hoist %v(%v) from b%v to b%v in %v\n",
		val.Op.String(), val.String(),
		block.ID, domBlock.ID, block.Func.Name)
	fmt.Printf("  %v\n", val.LongString())
}

func isCandidate(block *Block, val *Value) bool {
	if len(val.Args) == 0 {
		// not a profitable expression, e.g. constant
		return false
	}
	if block.Likely == BranchUnlikely {
		// all values are excluded as candidate when branch becomes unlikely to reach
		return false
	}
	return true
}

func isInsideLoop(loopBlocks []*Block, v *Value) bool {
	for _, block := range loopBlocks {
		for _, val := range block.Values {
			if val == v {
				return true
			}
		}
	}
	return false
}

// tryHoist hoists profitable loop invariant to block that dominates the entire loop.
// Value is considered as loop invariant if all its inputs are defined outside the loop
// or all its inputs are loop invariants. Since loop invariant will immediately moved
// to dominator block of loop, the first rule actually already implies the second rule
func tryHoist(loopnest *loopnest, loop *loop, loopBlocks []*Block) {
	for _, block := range loopBlocks {
		// if basic block is located in a nested loop rather than directly in the
		// current loop, it will not be processed.
		if loopnest.b2l[block.ID] != loop {
			continue
		}
		for i := 0; i < len(block.Values); i++ {
			var val *Value = block.Values[i]
			if !isCandidate(block, val) {
				continue
			}
			// value can hoist because it may causes observable side effects
			if hasSideEffect(val) {
				continue
			}
			// consider the following operation as pinned anyway
			switch val.Op {
			case OpInlMark,
				OpAtomicLoad8, OpAtomicLoad32, OpAtomicLoad64,
				OpAtomicLoadPtr, OpAtomicLoadAcq32, OpAtomicLoadAcq64:
				continue
			}
			// input def is inside loop, consider as variant
			isInvariant := true
			loopnest.assembleChildren()
			for _, arg := range val.Args {
				if isInsideLoop(loopBlocks, arg) {
					isInvariant = false
					break
				}
			}
			if isInvariant {
				for valIdx, v := range block.Values {
					if val != v {
						continue
					}
					domBlock := loopnest.sdom.Parent(loop.header)
					if block.Func.pass.debug >= 1 {
						printInvariant(val, block, domBlock)
					}
					val.moveTo(domBlock, valIdx)
					i--
					break
				}
			}
		}
	}
}

// hoistLoopInvariant hoists expressions that computes the same value
// while has no effect outside loop
func hoistLoopInvariant(f *Func) {
	loopnest := f.loopnest()
	if loopnest.hasIrreducible {
		return
	}
	if len(loopnest.loops) == 0 {
		return
	}
	for _, loop := range loopnest.loops {
		loopBlocks := loopnest.findLoopBlocks(loop)
		if len(loopBlocks) >= MaxLoopBlockSize {
			continue
		}

		// check if it's too complicated for such optmization
		tooComplicated := false
	Out:
		for _, block := range loopBlocks {
			for _, val := range block.Values {
				if val.Op.IsCall() || val.Op.HasSideEffects() {
					tooComplicated = true
					break Out
				}
				switch val.Op {
				case OpLoad, OpStore:
					tooComplicated = true
					break Out
				}
			}
		}
		// try to hoist loop invariant outside the loop
		if !tooComplicated {
			tryHoist(loopnest, loop, loopBlocks)
		}
	}
}
