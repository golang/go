// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
)

// ----------------------------------------------------------------------------
// Sparse Conditional Constant Propagation
//
// Described in
// Mark N. Wegman, F. Kenneth Zadeck: Constant Propagation with Conditional Branches. TOPLAS 1991.
//
// This algorithm uses three level lattice for SSA value
//
//      Top        undefined
//     / | \
// .. 1  2  3 ..   constant
//     \ | /
//     Bottom      not constant
//
// It starts with optimistically assuming that all SSA values are initially Top
// and then propagates constant facts only along reachable control flow paths.
// Due to certain basic blocks being never accessed, some inputs of phi become
// Top, we use the meet(args...) for phi to compute its lattice.
//
// 	  Top ∩ any = any
// 	  Bottom ∩ any = Bottom
// 	  ConstantA ∩ ConstantA = ConstantA
// 	  ConstantA ∩ ConstantB = Bottom
//
// Each lattice value is lowered most twice(Top to Constant, Constant to Bottom)
// due to lattice depth, resulting in a fast convergence speed of the algorithm.
// In this way, sccp can discover optimization opportunities that cannot be found
// by just combining constant folding and constant propagation and dead code
// elimination separately.

type edge struct {
	start *Block
	dest  *Block
}

// Three level lattice holds compile time knowledge about SSA value
const (
	top      int8 = iota // undefined
	constant             // constant
	bottom               // not a constant
)

type lattice struct {
	tag int8   // lattice type
	val *Value // constant value
}

type worklist struct {
	f            *Func               // the target function to be optmized out
	edges        []edge              // propagate constant facts through these control flow edges
	uses         []*Value            // uses for re-visiting because lattice of def is changed
	visited      map[edge]bool       // visited edges
	latticeCells map[*Value]lattice  // constant lattices
	defUse       map[*Value][]*Value // def-use chains for some values
	defBlock     map[*Value][]*Block // use blocks of def
}

func possibleConst(val *Value) bool {
	switch val.Op {
	case OpConst64, OpConst32, OpConst16, OpConst8,
		OpConstBool, OpConst32F, OpConst64F:
		fallthrough
	case OpCopy:
		fallthrough
	case OpPhi:
		fallthrough
	case
		OpNeg8, OpNeg16, OpNeg32, OpNeg64, OpNeg32F, OpNeg64F,
		OpCom8, OpCom16, OpCom32, OpCom64,
		OpFloor, OpCeil, OpTrunc, OpRoundToEven,
		OpNot:
		fallthrough
	case
		OpAdd64, OpAdd32, OpAdd16, OpAdd8,
		OpAdd32F, OpAdd64F,
		OpSub64, OpSub32, OpSub16, OpSub8,
		OpSub32F, OpSub64F,
		OpMul64, OpMul32, OpMul16, OpMul8,
		OpMul32F, OpMul64F,
		OpDiv32F, OpDiv64F,
		OpDiv8, OpDiv16, OpDiv32, OpDiv64,
		OpDiv8u, OpDiv16u, OpDiv32u, OpDiv64u,
		OpMod8, OpMod16, OpMod32, OpMod64,
		OpMod8u, OpMod16u, OpMod32u, OpMod64u,
		OpEq64, OpEq32, OpEq16, OpEq8,
		OpEq32F, OpEq64F,
		OpLess64, OpLess32, OpLess16, OpLess8,
		OpLess64U, OpLess32U, OpLess16U, OpLess8U,
		OpLess32F, OpLess64F,
		OpLeq64, OpLeq32, OpLeq16, OpLeq8,
		OpLeq64U, OpLeq32U, OpLeq16U, OpLeq8U,
		OpLeq32F, OpLeq64F,
		OpEqB, OpNeqB,
		OpLsh64x64, OpRsh64x64, OpRsh64Ux64, OpLsh32x64,
		OpRsh32x64, OpRsh32Ux64, OpLsh16x64, OpRsh16x64,
		OpRsh16Ux64, OpLsh8x64, OpRsh8x64, OpRsh8Ux64,
		OpIsInBounds, OpIsSliceInBounds,
		OpAnd8, OpAnd16, OpAnd32, OpAnd64,
		OpOr8, OpOr16, OpOr32, OpOr64,
		OpXor8, OpXor16, OpXor32, OpXor64:
		return true
	default:
		return false
	}
}

func (t *worklist) getLatticeCell(val *Value) lattice {
	if !possibleConst(val) {
		// they are always worst
		return lattice{bottom, nil}
	}
	lt, exist := t.latticeCells[val]
	if !exist {
		return lattice{top, nil} // optimistically for un-visited value
	}
	return lt
}

func isConst(val *Value) bool {
	switch val.Op {
	case OpConst64, OpConst32, OpConst16, OpConst8,
		OpConstBool, OpConst32F, OpConst64F:
		return true
	default:
		return false
	}
}

func isDivByZero(op Op, divisor *Value) bool {
	switch op {
	case OpDiv32F, OpDiv64F,
		OpDiv8, OpDiv16, OpDiv32, OpDiv64,
		OpMod8, OpMod16, OpMod32, OpMod64,
		OpDiv8u, OpDiv16u, OpDiv32u, OpDiv64u,
		OpMod8u, OpMod16u, OpMod32u, OpMod64u:
		if divisor.AuxInt == 0 {
			return true
		}
	}
	return false
}

// buildDefUses builds def-use chain for some values early, because once lattice of value
// is changed, all uses would be added into re-visit uses worklist, we rely heavily on
// the def-use chain in subsequent propagation.
func (t *worklist) buildDefUses() {
	for _, block := range t.f.Blocks {
		for _, val := range block.Values {
			for _, arg := range val.Args {
				if possibleConst(arg) {
					// for every value, find their uses
					if _, exist := t.defUse[arg]; !exist {
						t.defUse[arg] = make([]*Value, 0)
					}
					t.defUse[arg] = append(t.defUse[arg], val)
				}
			}
		}
		for _, ctl := range block.ControlValues() {
			// for every control values, find their use blocks
			if _, exist := t.defBlock[ctl]; !exist {
				t.defBlock[ctl] = make([]*Block, 0)
			}
			t.defBlock[ctl] = append(t.defBlock[ctl], block)
		}
	}
	// verify def-use chains
	for key, value := range t.defUse {
		if int(key.Uses) != (len(value) + len(t.defBlock[key])) {
			fmt.Printf("%v\n", t.f.String())
			t.f.Fatalf("def-use chain of %v is problematic: expect %v but got%v\n",
				key.LongString(), key.Uses, value)
		}
	}
}

// addUses finds all uses of value and appends them into work list for further process
func (t *worklist) addUses(val *Value) {
	for _, use := range t.defUse[val] {
		if val == use {
			// Phi may refer to itself as uses, ignore them
			continue
		}
		t.uses = append(t.uses, use)
	}
	for _, block := range t.defBlock[val] {
		t.propagate(block)
	}
}

func meet(arr []lattice) lattice {
	var lt = lattice{top, nil}
	for _, t := range arr {
		if t.tag == bottom {
			return lattice{bottom, nil}
		} else if t.tag == constant {
			if lt.tag == top {
				lt.tag = constant
				lt.val = t.val
			} else {
				if lt.val != t.val {
					return lattice{bottom, nil}
				}
			}
		} else {
			// Top ∩ any = any
		}
	}
	return lt
}

func (t *worklist) visitPhi(val *Value) {
	var argLattice = make([]lattice, 0)
	for i := 0; i < len(val.Args); i++ {
		var edge = edge{val.Block.Preds[i].b, val.Block}
		// If incoming edge for phi is not visited, assume top optimistically. According to rules
		// of meet:
		// 		Top ∩ any = any
		// Top participates in meet() but does not affect the result, so here we will ignore Top
		// and only take Bottom and Constant lattices into consideration.
		if _, exist := t.visited[edge]; exist {
			argLattice = append(argLattice, t.getLatticeCell(val.Args[i]))
		} else {
			// ignore Top intentionally
		}
	}
	// meet all of phi arguments
	t.latticeCells[val] = meet(argLattice)
}

func computeConstValue(f *Func, val *Value, args ...*Value) (*Value, bool) {
	// In general, we need to perform constant evaluation based on two constant lattices:
	//
	//  var res = lattice{constant, nil}
	// 	switch op {
	// 	case OpAdd16 :
	// 		res.val = newConst(argLt1.val.AuxInt16() + argLt2.val.AuxInt16())
	// 	case OpAdd32:
	// 		res.val = newConst(argLt1.val.AuxInt32() + argLt2.val.AuxInt32())
	// 		...
	// 	}
	//
	// However, this wouold create a huge switch for all opcodes that can be evaluted during
	// compile time, it's fragile and error prone. We did a trick by reusing the existing rules
	// in generic rules for compile-time evaluation. But generic rules rewrite original value,
	// this behavior is undesired, because the lattice of values may change multiple times, once
	// it was rewritten, we lose the opportunity to change it permanently, which can lead to
	// errors. For example, We cannot change its value immediately after visiting Phi, because
	// some of its input edges may still not be visited at this moment.
	var constValue = f.newValue(val.Op, val.Type, f.Entry, val.Pos)
	constValue.AddArgs(args...)
	var matched = rewriteValuegeneric(constValue)
	if matched {
		if !isConst(constValue) {
			// If we are able to match the above selected opcodes in generic rules
			// the rewrited value must be a constant value
			f.Fatalf("%v must be a constant value, missing or matched unexpected generic rule?",
				val.LongString())
		}
	}
	return constValue, matched
}

func (t *worklist) visitValue(val *Value) {
	var oldLt = t.getLatticeCell(val)
	defer func() {
		// re-visit all uses of value if its lattice is changed
		var newLt = t.getLatticeCell(val)
		if newLt != oldLt {
			t.addUses(val)
		}
	}()

	var worstLt = lattice{bottom, nil}
	switch val.Op {
	// they are constant values, aren't they?
	case OpConst64, OpConst32, OpConst16, OpConst8,
		OpConstBool, OpConst32F, OpConst64F: //TODO: support ConstNil ConstString etc
		t.latticeCells[val] = lattice{constant, val}
	// lattice value of copy(x) actually means lattice value of (x)
	case OpCopy:
		t.latticeCells[val] = t.getLatticeCell(val.Args[0])
	// phi should be processed specially
	case OpPhi:
		t.visitPhi(val)
	// fold 1-input operations:
	case
		// negate
		OpNeg8, OpNeg16, OpNeg32, OpNeg64, OpNeg32F, OpNeg64F,
		OpCom8, OpCom16, OpCom32, OpCom64,
		// math
		OpFloor, OpCeil, OpTrunc, OpRoundToEven,
		// not
		OpNot:
		var lt1 = t.getLatticeCell(val.Args[0])
		if lt1.tag != constant {
			t.latticeCells[val] = worstLt
			return
		}

		// here we take a shortcut by reusing generic rules to fold constants
		var constValue, matched = computeConstValue(t.f, val, lt1.val)
		if matched {
			t.latticeCells[val] = lattice{constant, constValue}
		} else {
			t.latticeCells[val] = worstLt
		}
	// fold 2-input operations
	case
		// add
		OpAdd64, OpAdd32, OpAdd16, OpAdd8,
		OpAdd32F, OpAdd64F,
		// sub
		OpSub64, OpSub32, OpSub16, OpSub8,
		OpSub32F, OpSub64F,
		// mul
		OpMul64, OpMul32, OpMul16, OpMul8,
		OpMul32F, OpMul64F,
		// div
		OpDiv32F, OpDiv64F,
		OpDiv8, OpDiv16, OpDiv32, OpDiv64,
		OpDiv8u, OpDiv16u, OpDiv32u, OpDiv64u, //TODO: support div128u
		// mod
		OpMod8, OpMod16, OpMod32, OpMod64,
		OpMod8u, OpMod16u, OpMod32u, OpMod64u,
		// compare
		OpEq64, OpEq32, OpEq16, OpEq8,
		OpEq32F, OpEq64F,
		OpLess64, OpLess32, OpLess16, OpLess8,
		OpLess64U, OpLess32U, OpLess16U, OpLess8U,
		OpLess32F, OpLess64F,
		OpLeq64, OpLeq32, OpLeq16, OpLeq8,
		OpLeq64U, OpLeq32U, OpLeq16U, OpLeq8U,
		OpLeq32F, OpLeq64F,
		OpEqB, OpNeqB,
		// shift
		OpLsh64x64, OpRsh64x64, OpRsh64Ux64, OpLsh32x64,
		OpRsh32x64, OpRsh32Ux64, OpLsh16x64, OpRsh16x64,
		OpRsh16Ux64, OpLsh8x64, OpRsh8x64, OpRsh8Ux64,
		// inbound safety check
		OpIsInBounds, OpIsSliceInBounds,
		// bit
		OpAnd8, OpAnd16, OpAnd32, OpAnd64,
		OpOr8, OpOr16, OpOr32, OpOr64,
		OpXor8, OpXor16, OpXor32, OpXor64:
		var lt1 = t.getLatticeCell(val.Args[0])
		var lt2 = t.getLatticeCell(val.Args[1])
		if lt1.tag != constant || lt2.tag != constant || isDivByZero(val.Op, lt2.val) {
			t.latticeCells[val] = worstLt
			return
		}

		// here we take a shortcut by reusing generic rules to fold constants
		var constValue, matched = computeConstValue(t.f, val, lt1.val, lt2.val)
		if matched {
			t.latticeCells[val] = lattice{constant, constValue}
		} else {
			t.latticeCells[val] = worstLt
		}
	default:
		// Any other type of value cannot be a constant, they are always worst(Bottom)
	}
}

// propagate propagates constants facts through CFG. If the block has single successor,
// add the successor anyway. If the block has multiple successors, only add the branch
// destination corresponding to lattice value of condition value.
func (t *worklist) propagate(block *Block) {
	switch block.Kind {
	case BlockExit, BlockRet, BlockRetJmp, BlockInvalid:
		// control flow ends, do nothing then
		break
	case BlockDefer:
		// we know nothing about control flow, add all branch destinations
		for _, succ := range block.Succs {
			t.edges = append(t.edges, edge{block, succ.b})
		}
	case BlockFirst:
		fallthrough // always takes the first branch
	case BlockPlain:
		t.edges = append(t.edges, edge{block, block.Succs[0].b})
	case BlockIf, BlockJumpTable:
		var cond = block.ControlValues()[0]
		var condLattice = t.getLatticeCell(cond)
		if condLattice.tag == bottom {
			// we know nothing about control flow, add all branch destinations
			for _, succ := range block.Succs {
				t.edges = append(t.edges, edge{block, succ.b})
			}
		} else if condLattice.tag == constant {
			// add branchIdx destinations depends on its condition
			var branchIdx int64
			if block.Kind == BlockIf {
				branchIdx = 1 - condLattice.val.AuxInt
			} else {
				branchIdx = condLattice.val.AuxInt
			}
			t.edges = append(t.edges, edge{block, block.Succs[branchIdx].b})
		} else {
			// condition value is not visited yet, don't propagate it now
		}
	default:
		t.f.Fatalf("All kind of block should be processed above.")
	}
}

// replaceConst will replace non-constant values that have been proven by sccp to be
// constants. If value controls blocks, this will rewire corresponding block successors
// according to constant condition test.
func (t *worklist) replaceConst() (int, int) {
	var constCnt, rewireCnt = 0, 0
	for val, lt := range t.latticeCells {
		if lt.tag == constant {
			// replace constant immediately
			if !isConst(val) {
				if t.f.pass.debug > 0 {
					fmt.Printf("Replace %v with %v\n", val.LongString(), lt.val.LongString())
				}
				val.reset(lt.val.Op)
				val.AuxInt = lt.val.AuxInt
				constCnt++
			}
			// rewire corresponding successors according to constant value
			var ctrlBlock = t.defBlock[val]
			for _, block := range ctrlBlock {
				switch block.Kind {
				case BlockIf:
					// Jump directly to successor block
					block.removeEdge(int(lt.val.AuxInt))
					block.Kind = BlockPlain
					block.Likely = BranchUnknown
					block.ResetControls()
					rewireCnt++
					if t.f.pass.debug > 0 {
						fmt.Printf("Rewire %v successors\n", block)
					}
				case BlockJumpTable:
					// TODO: optimize jump table
				}
			}
		}
	}
	return constCnt, rewireCnt
}

func sccp(f *Func) {
	var t worklist
	t.f = f
	t.edges = make([]edge, 0)
	t.visited = make(map[edge]bool)
	t.edges = append(t.edges, edge{f.Entry, f.Entry})
	t.defUse = make(map[*Value][]*Value)
	t.defBlock = make(map[*Value][]*Block)
	t.latticeCells = make(map[*Value]lattice)
	t.buildDefUses()

	var visitedBlock = f.newSparseSet(f.NumBlocks())
	defer f.retSparseSet(visitedBlock)

	// pick up either an edge or SSA value from worklilst, process it
	for {
		if len(t.edges) > 0 {
			var e = t.edges[0]
			t.edges = t.edges[1:]
			if _, exist := t.visited[e]; !exist {
				var dest = e.dest
				var destVisited = visitedBlock.contains(dest.ID)

				// mark edge as visited
				t.visited[e] = true
				visitedBlock.add(dest.ID)
				for _, val := range dest.Values {
					if val.Op == OpPhi || !destVisited {
						t.visitValue(val)
					}
				}
				// propagates constants facts through CFG, taking condition test into account
				if !destVisited {
					t.propagate(dest)
				}
			}
			continue
		}
		if len(t.uses) > 0 {
			var use = t.uses[0]
			t.uses = t.uses[1:]
			t.visitValue(use)
			continue
		}
		break
	}

	// apply optimizations based on discovered constants
	var constCnt, rewireCnt = t.replaceConst()
	if f.pass.debug > 0 {
		if constCnt > 0 || rewireCnt > 0 {
			fmt.Printf("Phase SCCP for %v : %v constants, %v dce\n", f.Name, constCnt, rewireCnt)
		}
	}
}
