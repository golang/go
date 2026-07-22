// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	blockpkg "cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
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
	tag int8           // lattice type
	val *ssacore.Value // constant value
}

type worklist struct {
	f            *ssacore.Func                       // the target function to be optimized out
	edges        []ssacore.Edge                      // propagate constant facts through edges
	inUses       *ssacore.SparseSet                  // IDs already in uses, for duplicate check
	uses         []*ssacore.Value                    // re-visiting set
	visited      map[ssacore.Edge]bool               // visited edges
	latticeCells map[*ssacore.Value]lattice          // constant lattices
	defUse       map[*ssacore.Value][]*ssacore.Value // def-use chains for some values
	defBlock     map[*ssacore.Value][]*ssacore.Block // use blocks of def
	visitedBlock []bool                              // visited block
}

// sccp stands for sparse conditional constant propagation, it propagates constants
// through CFG conditionally and applies constant folding, constant replacement and
// dead code elimination all together.
func sccp(f *ssacore.Func) {
	var t worklist
	t.f = f
	t.edges = make([]ssacore.Edge, 0)
	t.visited = make(map[ssacore.Edge]bool)
	t.edges = append(t.edges, ssacore.Edge{B: f.Entry, I: 0})
	t.defUse = make(map[*ssacore.Value][]*ssacore.Value)
	t.defBlock = make(map[*ssacore.Value][]*ssacore.Block)
	t.latticeCells = make(map[*ssacore.Value]lattice)
	t.visitedBlock = f.Cache.AllocBoolSlice(f.NumBlocks())
	t.inUses = f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(t.inUses)
	defer f.Cache.FreeBoolSlice(t.visitedBlock)

	// build it early since we rely heavily on the def-use chain later
	t.buildDefUses()

	// pick up either an edge or SSA value from worklist, process it
	for {
		if len(t.edges) > 0 {
			edge := t.edges[0]
			t.edges = t.edges[1:]
			if _, exist := t.visited[edge]; !exist {
				dest := edge.B
				destVisited := t.visitedBlock[dest.ID]

				// mark edge as visited
				t.visited[edge] = true
				t.visitedBlock[dest.ID] = true
				for _, val := range dest.Values {
					if val.Op == ssaop.OpPhi || !destVisited {
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
			t.inUses.Remove(use.ID)
			t.visitValue(use)
			continue
		}
		break
	}

	// apply optimizations based on discovered constants
	constCnt, rewireCnt := t.replaceConst()
	if f.Pass.Debug > 0 {
		if constCnt > 0 || rewireCnt > 0 {
			f.Warnl(f.Entry.Pos, "Phase SCCP for %v : %v constants, %v dce", f.Name, constCnt, rewireCnt)
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
func possibleConst(val *ssacore.Value) bool {
	if isConst(val) {
		return true
	}
	switch val.Op {
	case ssaop.OpCopy:
		return true
	case ssaop.OpPhi:
		return true
	case
		// negate
		ssaop.OpNeg8, ssaop.OpNeg16, ssaop.OpNeg32, ssaop.OpNeg64, ssaop.OpNeg32F, ssaop.OpNeg64F,
		ssaop.OpCom8, ssaop.OpCom16, ssaop.OpCom32, ssaop.OpCom64,
		// math
		ssaop.OpFloor, ssaop.OpCeil, ssaop.OpTrunc, ssaop.OpRoundToEven, ssaop.OpSqrt,
		// conversion
		ssaop.OpTrunc16to8, ssaop.OpTrunc32to8, ssaop.OpTrunc32to16, ssaop.OpTrunc64to8,
		ssaop.OpTrunc64to16, ssaop.OpTrunc64to32, ssaop.OpCvt32to32F, ssaop.OpCvt32to64F,
		ssaop.OpCvt64to32F, ssaop.OpCvt64to64F, ssaop.OpCvt32Fto32, ssaop.OpCvt32Fto64,
		ssaop.OpCvt64Fto32, ssaop.OpCvt64Fto64, ssaop.OpCvt32Fto64F, ssaop.OpCvt64Fto32F,
		ssaop.OpCvtBoolToUint8,
		ssaop.OpZeroExt8to16, ssaop.OpZeroExt8to32, ssaop.OpZeroExt8to64, ssaop.OpZeroExt16to32,
		ssaop.OpZeroExt16to64, ssaop.OpZeroExt32to64, ssaop.OpSignExt8to16, ssaop.OpSignExt8to32,
		ssaop.OpSignExt8to64, ssaop.OpSignExt16to32, ssaop.OpSignExt16to64, ssaop.OpSignExt32to64,
		// bit
		ssaop.OpCtz8, ssaop.OpCtz16, ssaop.OpCtz32, ssaop.OpCtz64,
		// mask
		ssaop.OpSlicemask,
		// safety check
		ssaop.OpIsNonNil,
		// not
		ssaop.OpNot:
		return true
	case
		// add
		ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8,
		ssaop.OpAdd32F, ssaop.OpAdd64F,
		// sub
		ssaop.OpSub64, ssaop.OpSub32, ssaop.OpSub16, ssaop.OpSub8,
		ssaop.OpSub32F, ssaop.OpSub64F,
		// mul
		ssaop.OpMul64, ssaop.OpMul32, ssaop.OpMul16, ssaop.OpMul8,
		ssaop.OpMul32F, ssaop.OpMul64F,
		// div
		ssaop.OpDiv32F, ssaop.OpDiv64F,
		ssaop.OpDiv8, ssaop.OpDiv16, ssaop.OpDiv32, ssaop.OpDiv64,
		ssaop.OpDiv8u, ssaop.OpDiv16u, ssaop.OpDiv32u, ssaop.OpDiv64u,
		ssaop.OpMod8, ssaop.OpMod16, ssaop.OpMod32, ssaop.OpMod64,
		ssaop.OpMod8u, ssaop.OpMod16u, ssaop.OpMod32u, ssaop.OpMod64u,
		// compare
		ssaop.OpEq64, ssaop.OpEq32, ssaop.OpEq16, ssaop.OpEq8,
		ssaop.OpEq32F, ssaop.OpEq64F,
		ssaop.OpLess64, ssaop.OpLess32, ssaop.OpLess16, ssaop.OpLess8,
		ssaop.OpLess64U, ssaop.OpLess32U, ssaop.OpLess16U, ssaop.OpLess8U,
		ssaop.OpLess32F, ssaop.OpLess64F,
		ssaop.OpLeq64, ssaop.OpLeq32, ssaop.OpLeq16, ssaop.OpLeq8,
		ssaop.OpLeq64U, ssaop.OpLeq32U, ssaop.OpLeq16U, ssaop.OpLeq8U,
		ssaop.OpLeq32F, ssaop.OpLeq64F,
		ssaop.OpEqB, ssaop.OpNeqB,
		// shift
		ssaop.OpLsh64x64, ssaop.OpRsh64x64, ssaop.OpRsh64Ux64, ssaop.OpLsh32x64,
		ssaop.OpRsh32x64, ssaop.OpRsh32Ux64, ssaop.OpLsh16x64, ssaop.OpRsh16x64,
		ssaop.OpRsh16Ux64, ssaop.OpLsh8x64, ssaop.OpRsh8x64, ssaop.OpRsh8Ux64,
		// safety check
		ssaop.OpIsInBounds, ssaop.OpIsSliceInBounds,
		// bit
		ssaop.OpAnd8, ssaop.OpAnd16, ssaop.OpAnd32, ssaop.OpAnd64,
		ssaop.OpOr8, ssaop.OpOr16, ssaop.OpOr32, ssaop.OpOr64,
		ssaop.OpXor8, ssaop.OpXor16, ssaop.OpXor32, ssaop.OpXor64:
		return true
	default:
		return false
	}
}

func (t *worklist) getLatticeCell(val *ssacore.Value) lattice {
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

func isConst(val *ssacore.Value) bool {
	switch val.Op {
	case ssaop.OpConst64, ssaop.OpConst32, ssaop.OpConst16, ssaop.OpConst8,
		ssaop.OpConstBool, ssaop.OpConst32F, ssaop.OpConst64F:
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
					// Phi may refer to itself as uses, avoid duplicate visits
					if arg == val {
						continue
					}
					if _, exist := t.defUse[arg]; !exist {
						t.defUse[arg] = make([]*ssacore.Value, 0, arg.Uses)
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
func (t *worklist) addUses(val *ssacore.Value) {
	for _, use := range t.defUse[val] {
		// Provenly not a constant, ignore
		useLt := t.getLatticeCell(use)
		if useLt.tag == bottom {
			continue
		}
		// Avoid duplicate visits
		if !t.inUses.Contains(use.ID) {
			t.inUses.Add(use.ID)
			t.uses = append(t.uses, use)
		}
	}
	for _, block := range t.defBlock[val] {
		if t.visitedBlock[block.ID] {
			t.propagate(block)
		}
	}
}

// meet meets all of phi arguments and computes result lattice
func (t *worklist) meet(val *ssacore.Value) lattice {
	optimisticLt := lattice{top, nil}
	for i := 0; i < len(val.Args); i++ {
		edge := ssacore.Edge{B: val.Block, I: i}
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

func computeLattice(f *ssacore.Func, val *ssacore.Value, args ...*ssacore.Value) lattice {
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
	// It's fragile and error-prone. We did a trick by reusing the existing rules
	// in generic rules for compile-time evaluation. But generic rules rewrite
	// original value, this behavior is undesired, because the lattice of values
	// may change multiple times, once it was rewritten, we lose the opportunity
	// to change it permanently, which can lead to errors. For example, We cannot
	// change its value immediately after visiting Phi, because some of its input
	// edges may still not be visited at this moment.
	constValue := f.NewValue(val.Op, val.Type, f.Entry, val.Pos)
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
	constValue.Reset(ssaop.OpInvalid)
	return lattice{bottom, nil}
}

func (t *worklist) visitValue(val *ssacore.Value) {
	// Impossible to be a constant, fast fail
	if !possibleConst(val) {
		return
	}

	// Provenly not a constant, fast fail
	oldLt := t.getLatticeCell(val)
	if oldLt.tag == bottom {
		return
	}

	// Re-visit all uses of value if its lattice is changed
	defer func() {
		newLt := t.getLatticeCell(val)
		if !equals(newLt, oldLt) {
			if oldLt.tag > newLt.tag {
				t.f.Fatalf("Must lower lattice\n")
			}
			t.addUses(val)
		}
	}()

	switch val.Op {
	// they are constant values, aren't they?
	case ssaop.OpConst64, ssaop.OpConst32, ssaop.OpConst16, ssaop.OpConst8,
		ssaop.OpConstBool, ssaop.OpConst32F, ssaop.OpConst64F: //TODO: support ConstNil ConstString etc
		t.latticeCells[val] = lattice{constant, val}
	// lattice value of copy(x) actually means lattice value of (x)
	case ssaop.OpCopy:
		t.latticeCells[val] = t.getLatticeCell(val.Args[0])
	// phi should be processed specially
	case ssaop.OpPhi:
		t.latticeCells[val] = t.meet(val)
	// fold 1-input operations:
	case
		// negate
		ssaop.OpNeg8, ssaop.OpNeg16, ssaop.OpNeg32, ssaop.OpNeg64, ssaop.OpNeg32F, ssaop.OpNeg64F,
		ssaop.OpCom8, ssaop.OpCom16, ssaop.OpCom32, ssaop.OpCom64,
		// math
		ssaop.OpFloor, ssaop.OpCeil, ssaop.OpTrunc, ssaop.OpRoundToEven, ssaop.OpSqrt,
		// conversion
		ssaop.OpTrunc16to8, ssaop.OpTrunc32to8, ssaop.OpTrunc32to16, ssaop.OpTrunc64to8,
		ssaop.OpTrunc64to16, ssaop.OpTrunc64to32, ssaop.OpCvt32to32F, ssaop.OpCvt32to64F,
		ssaop.OpCvt64to32F, ssaop.OpCvt64to64F, ssaop.OpCvt32Fto32, ssaop.OpCvt32Fto64,
		ssaop.OpCvt64Fto32, ssaop.OpCvt64Fto64, ssaop.OpCvt32Fto64F, ssaop.OpCvt64Fto32F,
		ssaop.OpCvtBoolToUint8,
		ssaop.OpZeroExt8to16, ssaop.OpZeroExt8to32, ssaop.OpZeroExt8to64, ssaop.OpZeroExt16to32,
		ssaop.OpZeroExt16to64, ssaop.OpZeroExt32to64, ssaop.OpSignExt8to16, ssaop.OpSignExt8to32,
		ssaop.OpSignExt8to64, ssaop.OpSignExt16to32, ssaop.OpSignExt16to64, ssaop.OpSignExt32to64,
		// bit
		ssaop.OpCtz8, ssaop.OpCtz16, ssaop.OpCtz32, ssaop.OpCtz64,
		// mask
		ssaop.OpSlicemask,
		// safety check
		ssaop.OpIsNonNil,
		// not
		ssaop.OpNot:
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
		ssaop.OpAdd64, ssaop.OpAdd32, ssaop.OpAdd16, ssaop.OpAdd8,
		ssaop.OpAdd32F, ssaop.OpAdd64F,
		// sub
		ssaop.OpSub64, ssaop.OpSub32, ssaop.OpSub16, ssaop.OpSub8,
		ssaop.OpSub32F, ssaop.OpSub64F,
		// mul
		ssaop.OpMul64, ssaop.OpMul32, ssaop.OpMul16, ssaop.OpMul8,
		ssaop.OpMul32F, ssaop.OpMul64F,
		// div
		ssaop.OpDiv32F, ssaop.OpDiv64F,
		ssaop.OpDiv8, ssaop.OpDiv16, ssaop.OpDiv32, ssaop.OpDiv64,
		ssaop.OpDiv8u, ssaop.OpDiv16u, ssaop.OpDiv32u, ssaop.OpDiv64u, //TODO: support div128u
		// mod
		ssaop.OpMod8, ssaop.OpMod16, ssaop.OpMod32, ssaop.OpMod64,
		ssaop.OpMod8u, ssaop.OpMod16u, ssaop.OpMod32u, ssaop.OpMod64u,
		// compare
		ssaop.OpEq64, ssaop.OpEq32, ssaop.OpEq16, ssaop.OpEq8,
		ssaop.OpEq32F, ssaop.OpEq64F,
		ssaop.OpLess64, ssaop.OpLess32, ssaop.OpLess16, ssaop.OpLess8,
		ssaop.OpLess64U, ssaop.OpLess32U, ssaop.OpLess16U, ssaop.OpLess8U,
		ssaop.OpLess32F, ssaop.OpLess64F,
		ssaop.OpLeq64, ssaop.OpLeq32, ssaop.OpLeq16, ssaop.OpLeq8,
		ssaop.OpLeq64U, ssaop.OpLeq32U, ssaop.OpLeq16U, ssaop.OpLeq8U,
		ssaop.OpLeq32F, ssaop.OpLeq64F,
		ssaop.OpEqB, ssaop.OpNeqB,
		// shift
		ssaop.OpLsh64x64, ssaop.OpRsh64x64, ssaop.OpRsh64Ux64, ssaop.OpLsh32x64,
		ssaop.OpRsh32x64, ssaop.OpRsh32Ux64, ssaop.OpLsh16x64, ssaop.OpRsh16x64,
		ssaop.OpRsh16Ux64, ssaop.OpLsh8x64, ssaop.OpRsh8x64, ssaop.OpRsh8Ux64,
		// safety check
		ssaop.OpIsInBounds, ssaop.OpIsSliceInBounds,
		// bit
		ssaop.OpAnd8, ssaop.OpAnd16, ssaop.OpAnd32, ssaop.OpAnd64,
		ssaop.OpOr8, ssaop.OpOr16, ssaop.OpOr32, ssaop.OpOr64,
		ssaop.OpXor8, ssaop.OpXor16, ssaop.OpXor32, ssaop.OpXor64:
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
func (t *worklist) propagate(block *ssacore.Block) {
	switch block.Kind {
	case blockpkg.BlockExit, blockpkg.BlockRet, blockpkg.BlockRetJmp, blockpkg.BlockInvalid:
		// control flow ends, do nothing then
		break
	case blockpkg.BlockDefer:
		// we know nothing about control flow, add all branch destinations
		t.edges = append(t.edges, block.Succs...)
	case blockpkg.BlockFirst:
		fallthrough // always takes the first branch
	case blockpkg.BlockPlain:
		t.edges = append(t.edges, block.Succs[0])
	case blockpkg.BlockIf, blockpkg.BlockJumpTable:
		cond := block.ControlValues()[0]
		condLattice := t.getLatticeCell(cond)
		if condLattice.tag == bottom {
			// we know nothing about control flow, add all branch destinations
			t.edges = append(t.edges, block.Succs...)
		} else if condLattice.tag == constant {
			// add branchIdx destinations depends on its condition
			var branchIdx int64
			if block.Kind == blockpkg.BlockIf {
				branchIdx = 1 - condLattice.val.AuxInt
			} else {
				branchIdx = condLattice.val.AuxInt
				if branchIdx < 0 || branchIdx >= int64(len(block.Succs)) {
					// unreachable code, do nothing then
					break
				}
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
func rewireSuccessor(block *ssacore.Block, constVal *ssacore.Value) bool {
	switch block.Kind {
	case blockpkg.BlockIf:
		block.RemoveEdge(int(constVal.AuxInt))
		block.Kind = blockpkg.BlockPlain
		block.Likely = ssacore.BranchUnknown
		block.ResetControls()
		return true
	case blockpkg.BlockJumpTable:
		// Remove everything but the known taken branch.
		idx := int(constVal.AuxInt)
		if idx < 0 || idx >= len(block.Succs) {
			// This can only happen in unreachable code,
			// as an invariant of jump tables is that their
			// input index is in range.
			// See issue 64826.
			return false
		}
		block.SwapSuccessorsByIdx(0, idx)
		for len(block.Succs) > 1 {
			block.RemoveEdge(1)
		}
		block.Kind = blockpkg.BlockPlain
		block.Likely = ssacore.BranchUnknown
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
				if t.f.Pass.Debug > 0 {
					t.f.Warnl(val.Pos, "Replace %v with %v", val.LongString(), lt.val.LongString())
				}
				val.Reset(lt.val.Op)
				val.AuxInt = lt.val.AuxInt
				constCnt++
			}
			// If const value controls this block, rewires successors according to its value
			ctrlBlock := t.defBlock[val]
			for _, block := range ctrlBlock {
				if rewireSuccessor(block, lt.val) {
					rewireCnt++
					if t.f.Pass.Debug > 0 {
						t.f.Warnl(block.Pos, "Rewire %v %v successors", block.Kind, block)
					}
				}
			}
		}
	}
	return constCnt, rewireCnt
}
