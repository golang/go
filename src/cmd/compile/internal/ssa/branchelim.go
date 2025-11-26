// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "cmd/internal/src"

// branchelim tries to eliminate branches by
// generating CondSelect instructions.
//
// Search for basic blocks that look like
//
//	bb0            bb0
//	 | \          /   \
//	 | bb1  or  bb1   bb2    <- trivial if/else blocks
//	 | /          \   /
//	bb2            bb3
//
// where the intermediate blocks are mostly empty (with no side-effects);
// rewrite Phis in the postdominator as CondSelects.
func branchelim(f *Func) {
	// FIXME: add support for lowering CondSelects on more architectures
	if !f.Config.haveCondSelect {
		return
	}

	// Find all the values used in computing the address of any load.
	// Typically these values have operations like AddPtr, Lsh64x64, etc.
	loadAddr := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(loadAddr)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpLoad, OpAtomicLoad8, OpAtomicLoad32, OpAtomicLoad64, OpAtomicLoadPtr, OpAtomicLoadAcq32, OpAtomicLoadAcq64:
				loadAddr.add(v.Args[0].ID)
			case OpMove:
				loadAddr.add(v.Args[1].ID)
			}
		}
	}
	po := f.postorder()
	for {
		n := loadAddr.size()
		for _, b := range po {
			for i := len(b.Values) - 1; i >= 0; i-- {
				v := b.Values[i]
				if !loadAddr.contains(v.ID) {
					continue
				}
				for _, a := range v.Args {
					if a.Type.IsInteger() || a.Type.IsPtr() || a.Type.IsUnsafePtr() {
						loadAddr.add(a.ID)
					}
				}
			}
		}
		if loadAddr.size() == n {
			break
		}
	}

	change := true
	for change {
		change = false
		for _, b := range f.Blocks {
			change = elimIf(f, loadAddr, b) || elimIfElse(f, loadAddr, b) || change
		}
	}
}

func canCondSelect(v *Value, arch string, loadAddr *sparseSet) bool {
	if loadAddr != nil && // prove calls this on some multiplies and doesn't take care of loadAddrs
		loadAddr.contains(v.ID) {
		// The result of the soon-to-be conditional move is used to compute a load address.
		// We want to avoid generating a conditional move in this case
		// because the load address would now be data-dependent on the condition.
		// Previously it would only be control-dependent on the condition, which is faster
		// if the branch predicts well (or possibly even if it doesn't, if the load will
		// be an expensive cache miss).
		// See issue #26306.
		return false
	}
	if arch == "loong64" {
		// We should not generate conditional moves if neither of the arguments is constant zero,
		// because it requires three instructions (OR, MASKEQZ, MASKNEZ) and will increase the
		// register pressure.
		if !(v.Args[0].isGenericIntConst() && v.Args[0].AuxInt == 0) &&
			!(v.Args[1].isGenericIntConst() && v.Args[1].AuxInt == 0) {
			return false
		}
	}
	// For now, stick to simple scalars that fit in registers
	switch {
	case v.Type.Size() > v.Block.Func.Config.RegSize:
		return false
	case v.Type.IsPtrShaped():
		return true
	case v.Type.IsInteger():
		if arch == "amd64" && v.Type.Size() < 2 {
			// amd64 doesn't support CMOV with byte registers
			return false
		}
		return true
	default:
		return false
	}
}

// elimIf converts the one-way branch starting at dom in f to a conditional move if possible.
// loadAddr is a set of values which are used to compute the address of a load.
// Those values are exempt from CMOV generation.
func elimIf(f *Func, loadAddr *sparseSet, dom *Block) bool {
	// See if dom is an If with one arm that
	// is trivial and succeeded by the other
	// successor of dom.
	if dom.Kind != BlockIf || dom.Likely != BranchUnknown {
		return false
	}
	var simple, post *Block
	for i := range dom.Succs {
		bb, other := dom.Succs[i].Block(), dom.Succs[i^1].Block()
		if isLeafPlain(bb) && bb.Succs[0].Block() == other {
			simple = bb
			post = other
			break
		}
	}
	if simple == nil || len(post.Preds) != 2 || post == dom {
		return false
	}

	// We've found our diamond CFG of blocks.
	// Now decide if fusing 'simple' into dom+post
	// looks profitable.

	// Check that there are Phis, and that all of them
	// can be safely rewritten to CondSelect.
	hasphis := false
	for _, v := range post.Values {
		if v.Op == OpPhi {
			hasphis = true
			if !canCondSelect(v, f.Config.arch, loadAddr) {
				return false
			}
		}
	}
	if !hasphis {
		return false
	}

	// Pick some upper bound for the number of instructions
	// we'd be willing to execute just to generate a dead
	// argument to CondSelect. In the worst case, this is
	// the number of useless instructions executed.
	const maxfuseinsts = 2

	if len(simple.Values) > maxfuseinsts || !canSpeculativelyExecute(simple) {
		return false
	}

	if !shouldElimIf(simple, post, dom, f.Config.arch) {
		return false
	}

	// Replace Phi instructions in b with CondSelect instructions
	swap := (post.Preds[0].Block() == dom) != (dom.Succs[0].Block() == post)
	for _, v := range post.Values {
		if v.Op != OpPhi {
			continue
		}
		v.Op = OpCondSelect
		if swap {
			v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
		}
		v.AddArg(dom.Controls[0])
	}

	// Put all of the instructions into 'dom'
	// and update the CFG appropriately.
	dom.Kind = post.Kind
	dom.CopyControls(post)
	dom.Aux = post.Aux
	dom.Succs = append(dom.Succs[:0], post.Succs...)
	for i := range dom.Succs {
		e := dom.Succs[i]
		e.b.Preds[e.i].b = dom
	}

	// Try really hard to preserve statement marks attached to blocks.
	simplePos := simple.Pos
	postPos := post.Pos
	simpleStmt := simplePos.IsStmt() == src.PosIsStmt
	postStmt := postPos.IsStmt() == src.PosIsStmt

	for _, v := range simple.Values {
		v.Block = dom
	}
	for _, v := range post.Values {
		v.Block = dom
	}

	// findBlockPos determines if b contains a stmt-marked value
	// that has the same line number as the Pos for b itself.
	// (i.e. is the position on b actually redundant?)
	findBlockPos := func(b *Block) bool {
		pos := b.Pos
		for _, v := range b.Values {
			// See if there is a stmt-marked value already that matches simple.Pos (and perhaps post.Pos)
			if pos.SameFileAndLine(v.Pos) && v.Pos.IsStmt() == src.PosIsStmt {
				return true
			}
		}
		return false
	}
	if simpleStmt {
		simpleStmt = !findBlockPos(simple)
		if !simpleStmt && simplePos.SameFileAndLine(postPos) {
			postStmt = false
		}

	}
	if postStmt {
		postStmt = !findBlockPos(post)
	}

	// If simpleStmt and/or postStmt are still true, then try harder
	// to find the corresponding statement marks new homes.

	// setBlockPos determines if b contains a can-be-statement value
	// that has the same line number as the Pos for b itself, and
	// puts a statement mark on it, and returns whether it succeeded
	// in this operation.
	setBlockPos := func(b *Block) bool {
		pos := b.Pos
		for _, v := range b.Values {
			if pos.SameFileAndLine(v.Pos) && !isPoorStatementOp(v.Op) {
				v.Pos = v.Pos.WithIsStmt()
				return true
			}
		}
		return false
	}
	// If necessary and possible, add a mark to a value in simple
	if simpleStmt {
		if setBlockPos(simple) && simplePos.SameFileAndLine(postPos) {
			postStmt = false
		}
	}
	// If necessary and possible, add a mark to a value in post
	if postStmt {
		postStmt = !setBlockPos(post)
	}

	// Before giving up (this was added because it helps), try the end of "dom", and if that is not available,
	// try the values in the successor block if it is uncomplicated.
	if postStmt {
		if dom.Pos.IsStmt() != src.PosIsStmt {
			dom.Pos = postPos
		} else {
			// Try the successor block
			if len(dom.Succs) == 1 && len(dom.Succs[0].Block().Preds) == 1 {
				succ := dom.Succs[0].Block()
				for _, v := range succ.Values {
					if isPoorStatementOp(v.Op) {
						continue
					}
					if postPos.SameFileAndLine(v.Pos) {
						v.Pos = v.Pos.WithIsStmt()
					}
					postStmt = false
					break
				}
				// If postStmt still true, tag the block itself if possible
				if postStmt && succ.Pos.IsStmt() != src.PosIsStmt {
					succ.Pos = postPos
				}
			}
		}
	}

	dom.Values = append(dom.Values, simple.Values...)
	dom.Values = append(dom.Values, post.Values...)

	// Trash 'post' and 'simple'
	clobberBlock(post)
	clobberBlock(simple)

	f.invalidateCFG()
	return true
}

// is this a BlockPlain with one predecessor?
func isLeafPlain(b *Block) bool {
	return b.Kind == BlockPlain && len(b.Preds) == 1
}

func clobberBlock(b *Block) {
	b.Values = nil
	b.Preds = nil
	b.Succs = nil
	b.Aux = nil
	b.ResetControls()
	b.Likely = BranchUnknown
	b.Kind = BlockInvalid
}

// elimIfElse converts the two-way branch starting at dom in f to a conditional move if possible.
// loadAddr is a set of values which are used to compute the address of a load.
// Those values are exempt from CMOV generation.
func elimIfElse(f *Func, loadAddr *sparseSet, b *Block) bool {
	// See if 'b' ends in an if/else: it should
	// have two successors, both of which are BlockPlain
	// and succeeded by the same block.
	if b.Kind != BlockIf || b.Likely != BranchUnknown {
		return false
	}
	yes, no := b.Succs[0].Block(), b.Succs[1].Block()
	if !isLeafPlain(yes) || len(yes.Values) > 1 || !canSpeculativelyExecute(yes) {
		return false
	}
	if !isLeafPlain(no) || len(no.Values) > 1 || !canSpeculativelyExecute(no) {
		return false
	}
	if b.Succs[0].Block().Succs[0].Block() != b.Succs[1].Block().Succs[0].Block() {
		return false
	}
	// block that postdominates the if/else
	post := b.Succs[0].Block().Succs[0].Block()
	if len(post.Preds) != 2 || post == b {
		return false
	}
	hasphis := false
	for _, v := range post.Values {
		if v.Op == OpPhi {
			hasphis = true
			if !canCondSelect(v, f.Config.arch, loadAddr) {
				return false
			}
		}
	}
	if !hasphis {
		return false
	}

	// Don't generate CondSelects if branch is cheaper.
	if !shouldElimIfElse(no, yes, post, b.Controls[0], f.Config.arch) {
		return false
	}

	// now we're committed: rewrite each Phi as a CondSelect
	swap := post.Preds[0].Block() != b.Succs[0].Block()
	for _, v := range post.Values {
		if v.Op != OpPhi {
			continue
		}
		v.Op = OpCondSelect
		if swap {
			v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
		}
		v.AddArg(b.Controls[0])
	}

	// Move the contents of all of these
	// blocks into 'b' and update CFG edges accordingly
	b.Kind = post.Kind
	b.CopyControls(post)
	b.Aux = post.Aux
	b.Succs = append(b.Succs[:0], post.Succs...)
	for i := range b.Succs {
		e := b.Succs[i]
		e.b.Preds[e.i].b = b
	}
	for i := range post.Values {
		post.Values[i].Block = b
	}
	for i := range yes.Values {
		yes.Values[i].Block = b
	}
	for i := range no.Values {
		no.Values[i].Block = b
	}
	b.Values = append(b.Values, yes.Values...)
	b.Values = append(b.Values, no.Values...)
	b.Values = append(b.Values, post.Values...)

	// trash post, yes, and no
	clobberBlock(yes)
	clobberBlock(no)
	clobberBlock(post)

	f.invalidateCFG()
	return true
}

// shouldElimIf reports whether estimated cost of eliminating a one-way branch
// is lower than threshold. This is similar to shouldElimIfElse but for elimIf.
func shouldElimIf(simple, post, dom *Block, arch string) bool {
	switch arch {
	default:
		return true
	case "riscv64":
		// Use zicond when: zero/const select (1 inst), conditional arithmetic (2 inst),
		// multiple phis, or unpredictable branches. Otherwise prefer branches (2 inst vs 4 inst).
		phi := 0
		hasConditionalArithmetic := false
		hasZeroOrConstSelect := false

		for _, v := range post.Values {
			if v.Op == OpPhi {
				phi++
				if isZeroOrConstSelectForElimIf(v, simple, post) {
					hasZeroOrConstSelect = true
				}
				if isConditionalArithmeticCandidateForElimIf(v, simple, post) {
					hasConditionalArithmetic = true
				}
			}
		}
		if hasZeroOrConstSelect || hasConditionalArithmetic {
			return true
		}
		if phi == 1 {
			if isLikelyUnpredictableBranch(dom.Controls[0]) {
				return true
			}
			if isInequalityOp(dom.Controls[0]) {
				return false
			}
			if len(simple.Values) == 1 {
				op := simple.Values[0].Op
				if op == OpNeg64 || op == OpNeg32 || op == OpCom64 || op == OpCom32 ||
					op == OpAdd64 || op == OpAdd32 || op == OpSub64 || op == OpSub32 {
					return true
				}
			}
			return false
		}
		return phi >= 2
	}
	// For other architectures, default case handles eliminating branches
}

// shouldElimIfElse reports whether estimated cost of eliminating branch
// is lower than threshold.
func shouldElimIfElse(no, yes, post *Block, cond *Value, arch string) bool {
	switch arch {
	default:
		return true
	case "amd64":
		const maxcost = 2
		phi := 0
		other := 0
		for _, v := range post.Values {
			if v.Op == OpPhi {
				// Each phi results in CondSelect, which lowers into CMOV,
				// CMOV has latency >1 on most CPUs.
				phi++
			}
			for _, x := range v.Args {
				if x.Block == no || x.Block == yes {
					other++
				}
			}
		}
		cost := phi * 1
		if phi > 1 {
			// If we have more than 1 phi and some values in post have args
			// in yes or no blocks, we may have to recalculate condition, because
			// those args may clobber flags. For now assume that all operations clobber flags.
			cost += other * 1
		}
		return cost < maxcost
	case "riscv64":
		// Use zicond when: zero/const select (1 inst), conditional arithmetic (2 inst),
		// multiple phis, or unpredictable branches. Otherwise prefer branches (2 inst vs 4 inst).
		phi := 0
		hasConditionalArithmetic := false
		hasZeroOrConstSelect := false
		for _, v := range post.Values {
			if v.Op == OpPhi {
				phi++
				if isZeroOrConstSelect(v, no, yes) {
					hasZeroOrConstSelect = true
				}
				if isConditionalArithmeticCandidate(v, no, yes) {
					hasConditionalArithmetic = true
				}
			}
		}
		if hasZeroOrConstSelect || hasConditionalArithmetic {
			return true
		}
		if phi == 1 {
			if isLikelyUnpredictableBranch(cond) {
				return true
			}
			if isInequalityOp(cond) {
				return false
			}
			hasSimpleOp := false
			for _, block := range []*Block{yes, no} {
				if len(block.Values) == 1 {
					op := block.Values[0].Op
					if op == OpNeg64 || op == OpNeg32 || op == OpCom64 || op == OpCom32 ||
						op == OpAdd64 || op == OpAdd32 || op == OpSub64 || op == OpSub32 ||
						op == OpOr64 || op == OpOr32 || op == OpXor64 || op == OpXor32 {
						hasSimpleOp = true
						break
					}
				}
			}
			if hasSimpleOp {
				return true
			}
			return false
		}
		return phi >= 2
	}
}

// isConditionalArithmeticCandidateForElimIf checks if a phi in elimIf represents
// conditional arithmetic (e.g., if cond { a += b } else { a = a }).
func isConditionalArithmeticCandidateForElimIf(v *Value, simple, post *Block) bool {
	if len(v.Args) != 2 {
		return false
	}
	// Check if one arg is an arithmetic operation from simple block
	// and the other arg is the base value (first operand of the arithmetic op)
	for i, arg := range v.Args {
		if arg.Block == simple && isArithmeticOp(arg.Op) {
			// Check if the arithmetic operation has at least 2 operands
			if len(arg.Args) >= 2 {
				baseValue := arg.Args[0] // First operand is typically the base value
				// Check if the other phi arg matches the base value
				otherArg := v.Args[1-i]
				if baseValue == otherArg {
					return true
				}
			}
		}
	}
	return false
}

// isConditionalArithmeticCandidate checks if a phi represents conditional arithmetic
// (e.g., if cond { a += b } else { a = a }). Optimized to 2 instructions by RISCV64.rules.
func isConditionalArithmeticCandidate(v *Value, no, yes *Block) bool {
	if len(v.Args) != 2 {
		return false
	}
	// Check if one arg is an arithmetic operation from no/yes blocks
	// and the other arg is the base value (first operand of the arithmetic op)
	for i, arg := range v.Args {
		if (arg.Block == no || arg.Block == yes) && isArithmeticOp(arg.Op) {
			// Check if the arithmetic operation has at least 2 operands
			if len(arg.Args) >= 2 {
				baseValue := arg.Args[0] // First operand is typically the base value
				// Check if the other phi arg matches the base value
				otherArg := v.Args[1-i]
				if baseValue == otherArg {
					return true
				}
			}
		}
	}
	return false
}

// isArithmeticOp checks if an operation can be optimized by conditional arithmetic rules.
func isArithmeticOp(op Op) bool {
	switch op {
	case OpAdd64, OpAdd32, OpAdd16, OpAdd8,
		OpSub64, OpSub32, OpSub16, OpSub8,
		OpOr64, OpOr32, OpOr16, OpOr8,
		OpXor64, OpXor32, OpXor16, OpXor8,
		OpAdd32F, OpAdd64F,
		OpSub32F, OpSub64F:
		return true
	}
	return false
}

func isZeroOrConstSelect(v *Value, no, yes *Block) bool {
	if len(v.Args) != 2 {
		return false
	}
	for i, arg := range v.Args {
		otherArg := v.Args[1-i]
		if arg.isGenericIntConst() && arg.AuxInt == 0 {
			if otherArg.isGenericIntConst() && otherArg.AuxInt != 0 {
				return true
			}
			if otherArg.Block == no || otherArg.Block == yes {
				return true
			}
		}
		if otherArg.isGenericIntConst() && otherArg.AuxInt == 0 {
			if arg.isGenericIntConst() && arg.AuxInt != 0 {
				return true
			}
			if arg.Block == no || arg.Block == yes {
				return true
			}
		}
	}
	return false
}

func isZeroOrConstSelectForElimIf(v *Value, simple, post *Block) bool {
	if len(v.Args) != 2 || len(post.Preds) != 2 {
		return false
	}
	var simpleArgIdx int = -1
	for i, pred := range post.Preds {
		if pred.Block() == simple {
			simpleArgIdx = i
			break
		}
	}
	if simpleArgIdx == -1 {
		return false
	}
	simpleArg := v.Args[simpleArgIdx]
	otherArgIdx := 1 - simpleArgIdx
	otherArg := v.Args[otherArgIdx]

	if otherArg.isGenericIntConst() && otherArg.AuxInt == 0 {
		if simpleArg.isGenericIntConst() && simpleArg.AuxInt != 0 {
			return true
		}
		if simpleArg.Block == simple {
			return true
		}
	}

	if simpleArg.isGenericIntConst() && simpleArg.AuxInt == 0 {
		if otherArg.isGenericIntConst() && otherArg.AuxInt != 0 {
			return true
		}
		if otherArg.Block != simple {
			return true
		}
	}

	return false
}

// isInequalityOp checks if a branch condition is an inequality comparison (<, >, <=, >=).
// Inequalities use efficient branch instructions (2 inst) vs zicond (4 inst).
// Note: Does not include ==/!= which may benefit from zicond when unpredictable.
func isInequalityOp(cond *Value) bool {
	if cond == nil {
		return false
	}
	switch cond.Op {
	case OpLess64, OpLess32, OpLess16, OpLess8,
		OpLess64U, OpLess32U, OpLess16U, OpLess8U,
		OpLeq64, OpLeq32, OpLeq16, OpLeq8,
		OpLeq64U, OpLeq32U, OpLeq16U, OpLeq8U:
		return true
	}
	switch cond.Op {
	case OpRISCV64SLT, OpRISCV64SLTU, OpRISCV64SLTI, OpRISCV64SLTIU:
		return true
	}
	return false
}

// isLikelyUnpredictableBranch checks if a branch condition is likely unpredictable
// (XOR, bit manipulation, hash-like). Unpredictable branches benefit from zicond
func isLikelyUnpredictableBranch(cond *Value) bool {
	if cond == nil {
		return false
	}

	var checkUnpredictable func(*Value, int) bool
	checkUnpredictable = func(v *Value, depth int) bool {
		if v == nil || depth > 5 {
			return false
		}

		switch v.Op {
		case OpXor64, OpXor32, OpXor16, OpXor8:
			return true

		case OpAnd64, OpAnd32, OpAnd16, OpAnd8:
			for _, arg := range v.Args {
				if arg.Op == OpConst64 || arg.Op == OpConst32 || arg.Op == OpConst16 || arg.Op == OpConst8 {
					mask := arg.AuxInt
					if mask == 1 || mask == 3 || mask == 7 || mask == 15 {
						for _, otherArg := range v.Args {
							if otherArg != arg && checkUnpredictable(otherArg, depth+1) {
								return true
							}
						}
					}
				}
			}
			for _, arg := range v.Args {
				if checkUnpredictable(arg, depth+1) {
					return true
				}
			}

		case OpNeq64, OpNeq32, OpNeq16, OpNeq8,
			OpEq64, OpEq32, OpEq16, OpEq8:
			for _, arg := range v.Args {
				if checkUnpredictable(arg, depth+1) {
					return true
				}
			}

		case OpRsh64Ux64, OpRsh32Ux64, OpRsh16Ux64, OpRsh8Ux64,
			OpRsh64Ux32, OpRsh32Ux32, OpRsh16Ux32, OpRsh8Ux32:
			return depth > 0

		case OpOr64, OpOr32, OpOr16, OpOr8:
			for _, arg := range v.Args {
				if checkUnpredictable(arg, depth+1) {
					return true
				}
			}
		}

		return false
	}

	return checkUnpredictable(cond, 0)
}

// canSpeculativelyExecute reports whether every value in the block can
// be evaluated without causing any observable side effects (memory
// accesses, panics and so on) except for execution time changes. It
// also ensures that the block does not contain any phis which we can't
// speculatively execute.
// Warning: this function cannot currently detect values that represent
// instructions the execution of which need to be guarded with CPU
// hardware feature checks. See issue #34950.
func canSpeculativelyExecute(b *Block) bool {
	// don't fuse memory ops, Phi ops, divides (can panic),
	// or anything else with side-effects
	for _, v := range b.Values {
		if v.Op == OpPhi || isDivMod(v.Op) || isPtrArithmetic(v.Op) ||
			v.Type.IsMemory() || opcodeTable[v.Op].hasSideEffects {
			return false
		}

		// Allow inlining markers to be speculatively executed
		// even though they have a memory argument.
		// See issue #74915.
		if v.Op != OpInlMark && v.MemoryArg() != nil {
			return false
		}
	}
	return true
}

func isDivMod(op Op) bool {
	switch op {
	case OpDiv8, OpDiv8u, OpDiv16, OpDiv16u,
		OpDiv32, OpDiv32u, OpDiv64, OpDiv64u, OpDiv128u,
		OpDiv32F, OpDiv64F,
		OpMod8, OpMod8u, OpMod16, OpMod16u,
		OpMod32, OpMod32u, OpMod64, OpMod64u:
		return true
	default:
		return false
	}
}

func isPtrArithmetic(op Op) bool {
	// Pointer arithmetic can't be speculatively executed because the result
	// may be an invalid pointer (if, for example, the condition is that the
	// base pointer is not nil). See issue 56990.
	switch op {
	case OpOffPtr, OpAddPtr, OpSubPtr:
		return true
	default:
		return false
	}
}
