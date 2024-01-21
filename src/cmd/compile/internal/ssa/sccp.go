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
// Mark N. Wegman, F. Kenneth Zadeck: Constant Propagation with Conditional Branches.
// TOPLAS 1991.
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
// Since some basic blocks are not visited yet, corresponding inputs of phi become
// Top, we use the meet(phi) to compute its lattice.
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
	f            *Func               // the target function to be optimized out
	edges        []Edge              // propagate constant facts through edges
	uses         []*Value            // re-visiting set
	visited      map[Edge]bool       // visited edges
	latticeCells map[*Value]lattice  // constant lattices
	defUse       map[*Value][]*Value // def-use chains for some values
	defBlock     map[*Value][]*Block // use blocks of def
	visitedBlock []bool              // visited block
}

// sccp stands for sparse conditional constant propagation, it propagates constants
// through CFG conditionally and applies constant folding, constant replacement and
// dead code elimination all together.
func sccp(f *Func) {
	var t worklist
	t.f = f
	t.edges = make([]Edge, 0)
	t.visited = make(map[Edge]bool)
	t.edges = append(t.edges, Edge{f.Entry, 0})
	t.defUse = make(map[*Value][]*Value)
	t.defBlock = make(map[*Value][]*Block)
	t.latticeCells = make(map[*Value]lattice)
	t.visitedBlock = f.Cache.allocBoolSlice(f.NumBlocks())
	defer f.Cache.freeBoolSlice(t.visitedBlock)

	// build it early since we rely heavily on the def-use chain later
	t.buildDefUses()

	// pick up either an edge or SSA value from worklist, process it
	for {
		if len(t.edges) > 0 {
			edge := t.edges[0]
			t.edges = t.edges[1:]
			if _, exist := t.visited[edge]; !exist {
				dest := edge.b
				destVisited := t.visitedBlock[dest.ID]

				// mark edge as visited
				t.visited[edge] = true
				t.visitedBlock[dest.ID] = true
				for _, val := range dest.Values {
					if val.Op == OpPhi || !destVisited {
						t.visitValue(val)
					}
				}
				// propagates constants facts through CFG, taking condition test
				// into account
				if !destVisited {
					t.propagate(dest)
				}
			}
			continue
		}
		if len(t.uses) > 0 {
			use := t.uses[0]
			t.uses = t.uses[1:]
			t.visitValue(use)
			continue
		}
		break
	}

	// apply optimizations based on discovered constants
	constCnt, rewireCnt := t.replaceConst()
	if f.pass.debug > 0 {
		if constCnt > 0 || rewireCnt > 0 {
			fmt.Printf("Phase SCCP for %v : %v constants, %v dce\n", f.Name, constCnt, rewireCnt)
		}
	}
}

func equals(a, b lattice) bool {
	if a == b {
		// fast path
		return true
	}
	if a.tag != b.tag {
		return false
	}
	if a.tag == constant {
		// The same content of const value may be different, we should
		// compare with auxInt instead
		v1 := a.val
		v2 := b.val
		if v1.Op == v2.Op && v1.AuxInt == v2.AuxInt {
			return true
		} else {
			return false
		}
	}
	return true
}

// possibleConst checks if Value can be folded to const. For those Values that can
// never become constants(e.g. StaticCall), we don't make futile efforts.
func possibleConst(val *Value) bool {
	if isConst(val) {
		return true
	}
	switch val.Op {
	case OpCopy:
		return true
	case OpPhi:
		return true
	case
		// negate
		OpNeg8, OpNeg16, OpNeg32, OpNeg64, OpNeg32F, OpNeg64F,
		OpCom8, OpCom16, OpCom32, OpCom64,
		// math
		OpFloor, OpCeil, OpTrunc, OpRoundToEven, OpSqrt,
		// conversion
		OpTrunc16to8, OpTrunc32to8, OpTrunc32to16, OpTrunc64to8,
		OpTrunc64to16, OpTrunc64to32, OpCvt32to32F, OpCvt32to64F,
		OpCvt64to32F, OpCvt64to64F, OpCvt32Fto32, OpCvt32Fto64,
		OpCvt64Fto32, OpCvt64Fto64, OpCvt32Fto64F, OpCvt64Fto32F,
		OpCvtBoolToUint8,
		OpZeroExt8to16, OpZeroExt8to32, OpZeroExt8to64, OpZeroExt16to32,
		OpZeroExt16to64, OpZeroExt32to64, OpSignExt8to16, OpSignExt8to32,
		OpSignExt8to64, OpSignExt16to32, OpSignExt16to64, OpSignExt32to64,
		// bit
		OpCtz8, OpCtz16, OpCtz32, OpCtz64,
		// mask
		OpSlicemask,
		// safety check
		OpIsNonNil,
		// not
		OpNot:
		return true
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
		OpDiv8u, OpDiv16u, OpDiv32u, OpDiv64u,
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
		// safety check
		OpIsInBounds, OpIsSliceInBounds,
		// bit
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

// buildDefUses builds def-use chain for some values early, because once the
// lattice of a value is changed, we need to update lattices of use. But we don't
// need all uses of it, only uses that can become constants would be added into
// re-visit worklist since no matter how many times they are revisited, uses which
// can't become constants lattice remains unchanged, i.e. Bottom.
func (t *worklist) buildDefUses() {
	for _, block := range t.f.Blocks {
		for _, val := range block.Values {
			for _, arg := range val.Args {
				// find its uses, only uses that can become constants take into account
				if possibleConst(arg) && possibleConst(val) {
					if _, exist := t.defUse[arg]; !exist {
						t.defUse[arg] = make([]*Value, 0, arg.Uses)
					}
					t.defUse[arg] = append(t.defUse[arg], val)
				}
			}
		}
		for _, ctl := range block.ControlValues() {
			// for control values that can become constants, find their use blocks
			if possibleConst(ctl) {
				t.defBlock[ctl] = append(t.defBlock[ctl], block)
			}
		}
	}
}

// addUses finds all uses of value and appends them into work list for further process
func (t *worklist) addUses(val *Value) {
	for _, use := range t.defUse[val] {
		if val == use {
			// Phi may refer to itself as uses, ignore them to avoid re-visiting phi
			// for performance reason
			continue
		}
		t.uses = append(t.uses, use)
	}
	for _, block := range t.defBlock[val] {
		if t.visitedBlock[block.ID] {
			t.propagate(block)
		}
	}
}

// meet meets all of phi arguments and computes result lattice
func (t *worklist) meet(val *Value) lattice {
	optimisticLt := lattice{top, nil}
	for i := 0; i < len(val.Args); i++ {
		edge := Edge{val.Block, i}
		// If incoming edge for phi is not visited, assume top optimistically.
		// According to rules of meet:
		// 		Top ∩ any = any
		// Top participates in meet() but does not affect the result, so here
		// we will ignore Top and only take other lattices into consideration.
		if _, exist := t.visited[edge]; exist {
			lt := t.getLatticeCell(val.Args[i])
			if lt.tag == constant {
				if optimisticLt.tag == top {
					optimisticLt = lt
				} else {
					if !equals(optimisticLt, lt) {
						// ConstantA ∩ ConstantB = Bottom
						return lattice{bottom, nil}
					}
				}
			} else if lt.tag == bottom {
				// Bottom ∩ any = Bottom
				return lattice{bottom, nil}
			} else {
				// Top ∩ any = any
			}
		} else {
			// Top ∩ any = any
		}
	}

	// ConstantA ∩ ConstantA = ConstantA or Top ∩ any = any
	return optimisticLt
}

func computeLattice(f *Func, val *Value, args ...*Value) lattice {
	// In general, we need to perform constant evaluation based on constant args:
	//
	//  res := lattice{constant, nil}
	// 	switch op {
	// 	case OpAdd16:
	//		res.val = newConst(argLt1.val.AuxInt16() + argLt2.val.AuxInt16())
	// 	case OpAdd32:
	// 		res.val = newConst(argLt1.val.AuxInt32() + argLt2.val.AuxInt32())
	//	case OpDiv8:
	//		if !isDivideByZero(argLt2.val.AuxInt8()) {
	//			res.val = newConst(argLt1.val.AuxInt8() / argLt2.val.AuxInt8())
	//		}
	//  ...
	// 	}
	//
	// However, this would create a huge switch for all opcodes that can be
	// evaluated during compile time. Moreover, some operations can be evaluated
	// only if its arguments satisfy additional conditions(e.g. divide by zero).
	// It's fragile and error prone. We did a trick by reusing the existing rules
	// in generic rules for compile-time evaluation. But generic rules rewrite
	// original value, this behavior is undesired, because the lattice of values
	// may change multiple times, once it was rewritten, we lose the opportunity
	// to change it permanently, which can lead to errors. For example, We cannot
	// change its value immediately after visiting Phi, because some of its input
	// edges may still not be visited at this moment.
	constValue := f.newValue(val.Op, val.Type, f.Entry, val.Pos)
	constValue.AddArgs(args...)
	matched := rewriteValuegeneric(constValue)
	if matched {
		if isConst(constValue) {
			return lattice{constant, constValue}
		}
	}
	// Either we can not match generic rules for given value or it does not
	// satisfy additional constraints(e.g. divide by zero), in these cases, clean
	// up temporary value immediately in case they are not dominated by their args.
	constValue.reset(OpInvalid)
	return lattice{bottom, nil}
}

func (t *worklist) visitValue(val *Value) {
	if !possibleConst(val) {
		// fast fail for always worst Values, i.e. there is no lowering happen
		// on them, their lattices must be initially worse Bottom.
		return
	}

	oldLt := t.getLatticeCell(val)
	defer func() {
		// re-visit all uses of value if its lattice is changed
		newLt := t.getLatticeCell(val)
		if !equals(newLt, oldLt) {
			if int8(oldLt.tag) > int8(newLt.tag) {
				t.f.Fatalf("Must lower lattice\n")
			}
			t.addUses(val)
		}
	}()

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
		t.latticeCells[val] = t.meet(val)
	// fold 1-input operations:
	case
		// negate
		OpNeg8, OpNeg16, OpNeg32, OpNeg64, OpNeg32F, OpNeg64F,
		OpCom8, OpCom16, OpCom32, OpCom64,
		// math
		OpFloor, OpCeil, OpTrunc, OpRoundToEven, OpSqrt,
		// conversion
		OpTrunc16to8, OpTrunc32to8, OpTrunc32to16, OpTrunc64to8,
		OpTrunc64to16, OpTrunc64to32, OpCvt32to32F, OpCvt32to64F,
		OpCvt64to32F, OpCvt64to64F, OpCvt32Fto32, OpCvt32Fto64,
		OpCvt64Fto32, OpCvt64Fto64, OpCvt32Fto64F, OpCvt64Fto32F,
		OpCvtBoolToUint8,
		OpZeroExt8to16, OpZeroExt8to32, OpZeroExt8to64, OpZeroExt16to32,
		OpZeroExt16to64, OpZeroExt32to64, OpSignExt8to16, OpSignExt8to32,
		OpSignExt8to64, OpSignExt16to32, OpSignExt16to64, OpSignExt32to64,
		// bit
		OpCtz8, OpCtz16, OpCtz32, OpCtz64,
		// mask
		OpSlicemask,
		// safety check
		OpIsNonNil,
		// not
		OpNot:
		lt1 := t.getLatticeCell(val.Args[0])

		if lt1.tag == constant {
			// here we take a shortcut by reusing generic rules to fold constants
			t.latticeCells[val] = computeLattice(t.f, val, lt1.val)
		} else {
			t.latticeCells[val] = lattice{lt1.tag, nil}
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
		// safety check
		OpIsInBounds, OpIsSliceInBounds,
		// bit
		OpAnd8, OpAnd16, OpAnd32, OpAnd64,
		OpOr8, OpOr16, OpOr32, OpOr64,
		OpXor8, OpXor16, OpXor32, OpXor64:
		lt1 := t.getLatticeCell(val.Args[0])
		lt2 := t.getLatticeCell(val.Args[1])

		if lt1.tag == constant && lt2.tag == constant {
			// here we take a shortcut by reusing generic rules to fold constants
			t.latticeCells[val] = computeLattice(t.f, val, lt1.val, lt2.val)
		} else {
			if lt1.tag == bottom || lt2.tag == bottom {
				t.latticeCells[val] = lattice{bottom, nil}
			} else {
				t.latticeCells[val] = lattice{top, nil}
			}
		}
	default:
		// Any other type of value cannot be a constant, they are always worst(Bottom)
	}
}

// propagate propagates constants facts through CFG. If the block has single successor,
// add the successor anyway. If the block has multiple successors, only add the
// branch destination corresponding to lattice value of condition value.
func (t *worklist) propagate(block *Block) {
	switch block.Kind {
	case BlockExit, BlockRet, BlockRetJmp, BlockInvalid:
		// control flow ends, do nothing then
		break
	case BlockDefer:
		// we know nothing about control flow, add all branch destinations
		t.edges = append(t.edges, block.Succs...)
	case BlockFirst:
		fallthrough // always takes the first branch
	case BlockPlain:
		t.edges = append(t.edges, block.Succs[0])
	case BlockIf, BlockJumpTable:
		cond := block.ControlValues()[0]
		condLattice := t.getLatticeCell(cond)
		if condLattice.tag == bottom {
			// we know nothing about control flow, add all branch destinations
			t.edges = append(t.edges, block.Succs...)
		} else if condLattice.tag == constant {
			// add branchIdx destinations depends on its condition
			var branchIdx int64
			if block.Kind == BlockIf {
				branchIdx = 1 - condLattice.val.AuxInt
			} else {
				branchIdx = condLattice.val.AuxInt
			}
			t.edges = append(t.edges, block.Succs[branchIdx])
		} else {
			// condition value is not visited yet, don't propagate it now
		}
	default:
		t.f.Fatalf("All kind of block should be processed above.")
	}
}

// rewireSuccessor rewires corresponding successors according to constant value
// discovered by previous analysis. As the result, some successors become unreachable
// and thus can be removed in further deadcode phase
func rewireSuccessor(block *Block, constVal *Value) bool {
	switch block.Kind {
	case BlockIf:
		block.removeEdge(int(constVal.AuxInt))
		block.Kind = BlockPlain
		block.Likely = BranchUnknown
		block.ResetControls()
		return true
	case BlockJumpTable:
		// Remove everything but the known taken branch.
		idx := int(constVal.AuxInt)
		if idx < 0 || idx >= len(block.Succs) {
			// This can only happen in unreachable code,
			// as an invariant of jump tables is that their
			// input index is in range.
			// See issue 64826.
			return false
		}
		block.swapSuccessorsByIdx(0, idx)
		for len(block.Succs) > 1 {
			block.removeEdge(1)
		}
		block.Kind = BlockPlain
		block.Likely = BranchUnknown
		block.ResetControls()
		return true
	default:
		return false
	}
}

// replaceConst will replace non-constant values that have been proven by sccp
// to be constants.
func (t *worklist) replaceConst() (int, int) {
	constCnt, rewireCnt := 0, 0
	for val, lt := range t.latticeCells {
		if lt.tag == constant {
			if !isConst(val) {
				if t.f.pass.debug > 0 {
					fmt.Printf("Replace %v with %v\n", val.LongString(), lt.val.LongString())
				}
				val.reset(lt.val.Op)
				val.AuxInt = lt.val.AuxInt
				constCnt++
			}
			// If const value controls this block, rewires successors according to its value
			ctrlBlock := t.defBlock[val]
			for _, block := range ctrlBlock {
				if rewireSuccessor(block, lt.val) {
					rewireCnt++
					if t.f.pass.debug > 0 {
						fmt.Printf("Rewire %v %v successors\n", block.Kind, block)
					}
				}
			}
		}
	}
	return constCnt, rewireCnt
}
