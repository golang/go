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

	// For riscv64, check if eliminating this branch is profitable.
	// Simple conditional selects (single phi, not conditional arithmetic)
	// should use traditional branches instead of zicond (4+ instructions).
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
		// For riscv64, zicond has different cost characteristics:
		// - Simple conditional select: 4 instructions (SLT + CZEROEQZ + CZERONEZ + OR)
		//   vs 2 instructions (BLT + MOV) for predictable branches
		// - Conditional arithmetic: 2 instructions (optimized by rules in RISCV64.rules)
		//   vs 2-3 instructions for branches
		//
		// Strategy: Only use zicond when:
		// 1. Zero/const select patterns (optimized to 1 instruction) - always beneficial
		// 2. Conditional arithmetic operations (optimized to 2 instructions) - always beneficial
		// 3. Multiple phis (amortize overhead of 4 instructions)
		// 4. Unpredictable branches (misprediction penalty > zicond overhead)
		phi := 0
		hasConditionalArithmetic := false
		hasZeroOrConstSelect := false

		// For elimIf, the structure is:
		// - dom: if block
		// - simple: if branch block (one arm)
		// - post: merge block (the other successor of dom, also the fallthrough)
		// The Phi nodes in post have two args:
		// - One from simple block (if branch)
		// - One from dom or the fallthrough path (else branch)
		// We need to find which arg comes from which path.
		// Since post is the merge point and simple->post, the fallthrough is the other path.
		// For Phi nodes, we check if one arg is from simple and the other is the base value.

		for _, v := range post.Values {
			if v.Op == OpPhi {
				phi++
				// Check if this phi can be optimized to a single zicond instruction.
				// Pattern: if cond { x = value } else { x = 0 } → CZEROEQZ
				// Pattern: if cond { x = 0 } else { x = value } → CZERONEZ
				if isZeroOrConstSelectForElimIf(v, simple, post) {
					hasZeroOrConstSelect = true
				}
				// Check if this phi can be optimized by conditional arithmetic rules.
				// These rules (e.g., cmoveAddZero) reduce zicond to 2 instructions,
				// making it always beneficial compared to branches.
				if isConditionalArithmeticCandidateForElimIf(v, simple, post) {
					hasConditionalArithmetic = true
				}
			}
		}
		// Zero/const select patterns are always beneficial (1 instruction after optimization)
		if hasZeroOrConstSelect {
			return true
		}
		// Conditional arithmetic operations are always beneficial (2 instructions after optimization)
		if hasConditionalArithmetic {
			return true
		}
		// For simple conditional select (single phi):
		// - Traditional branch: 2 instructions (BLT + MOV) with high prediction success
		// - Zicond: 3-4 instructions (after optimization, can be 3: CZEROEQZ + CZERONEZ + OR)
		// However, zicond avoids branch misprediction penalty (17 cycles), which can be
		// significant even for seemingly predictable branches in tight loops.
		// For now, we allow zicond for single phi when:
		// 1. Branch is likely unpredictable (XOR, bit manipulation) - including unpredictable ==/!=
		// 2. The operation in simple block is simple (arithmetic operations like NEG, NOT, INC)
		//    These can benefit from zicond even if not strictly "unpredictable"
		// 3. NOT an inequality operation (<, >, <=, >=) - these use efficient branch instructions
		if phi == 1 {
			// First check if the branch condition is likely unpredictable.
			// This includes unpredictable equality/inequality operations (e.g., (x & 1) != (y & 1))
			// which should use zicond to avoid misprediction penalty.
			if isLikelyUnpredictableBranch(dom.Controls[0]) {
				return true
			}
			// For inequality operations (<, >, <=, >=), traditional branches (BGEZ, BLTU, etc.)
			// are more efficient (2 instructions) than zicond (4 instructions).
			// Only use zicond for inequalities if they are in optimizable patterns
			// (which are already handled above).
			if isInequalityOp(dom.Controls[0]) {
				return false
			}
			// Check if simple block contains simple arithmetic operations that benefit from zicond
			// even for predictable branches (e.g., NEG, NOT, INC operations)
			if len(simple.Values) == 1 {
				op := simple.Values[0].Op
				// Simple arithmetic operations that are cheap to speculatively execute
				if op == OpNeg64 || op == OpNeg32 || op == OpCom64 || op == OpCom32 ||
					op == OpAdd64 || op == OpAdd32 || op == OpSub64 || op == OpSub32 {
					// Allow zicond for these simple operations, as they avoid branch misprediction
					// and the instruction count difference (3 vs 2) is small
					return true
				}
			}
			// Conservative: don't use zicond for single simple conditional select
			// with predictable branches and complex operations
			return false
		}
		// Multiple phis: cost is amortized
		// Each phi adds 2 extra instructions, but we avoid multiple branches
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
		// For riscv64, zicond has different cost characteristics:
		// - Simple conditional select: 4 instructions (SLT + CZEROEQZ + CZERONEZ + OR)
		//   vs 2 instructions (BLT + MOV) for predictable branches
		// - Conditional arithmetic: 2 instructions (optimized by rules in RISCV64.rules)
		//   vs 2-3 instructions for branches
		//
		// Strategy: Only use zicond when:
		// 1. Zero/const select patterns (optimized to 1 instruction) - always beneficial
		// 2. Conditional arithmetic operations (optimized to 2 instructions) - always beneficial
		// 3. Multiple phis (amortize overhead of 4 instructions)
		// 4. Unpredictable branches (misprediction penalty > zicond overhead)
		phi := 0
		hasConditionalArithmetic := false
		hasZeroOrConstSelect := false
		for _, v := range post.Values {
			if v.Op == OpPhi {
				phi++
				// Check if this phi can be optimized to a single zicond instruction.
				// Pattern: if cond { x = value } else { x = 0 } → CZEROEQZ
				// Pattern: if cond { x = 0 } else { x = value } → CZERONEZ
				if isZeroOrConstSelect(v, no, yes) {
					hasZeroOrConstSelect = true
				}
				// Check if this phi can be optimized by conditional arithmetic rules.
				// These rules (e.g., cmoveAddZero) reduce zicond to 2 instructions,
				// making it always beneficial compared to branches.
				if isConditionalArithmeticCandidate(v, no, yes) {
					hasConditionalArithmetic = true
				}
			}
		}
		// Zero/const select patterns are always beneficial (1 instruction after optimization)
		if hasZeroOrConstSelect {
			return true
		}
		// Conditional arithmetic operations are always beneficial (2 instructions after optimization)
		if hasConditionalArithmetic {
			return true
		}
		// For simple conditional select (single phi):
		// - Traditional branch: 2 instructions (BLT + MOV) with high prediction success
		// - Zicond: 3-4 instructions (after optimization, can be 3: CZEROEQZ + CZERONEZ + OR)
		// However, zicond avoids branch misprediction penalty (17 cycles), which can be
		// significant even for seemingly predictable branches in tight loops.
		// For now, we allow zicond for single phi when:
		// 1. Branch is likely unpredictable (XOR, bit manipulation) - including unpredictable ==/!=
		// 2. The operations in yes/no blocks are simple (arithmetic operations)
		//    These can benefit from zicond even if not strictly "unpredictable"
		// 3. NOT an inequality operation (<, >, <=, >=) - these use efficient branch instructions
		if phi == 1 {
			// First check if the branch condition is likely unpredictable.
			// This includes unpredictable equality/inequality operations (e.g., (x & 1) != (y & 1))
			// which should use zicond to avoid misprediction penalty.
			if isLikelyUnpredictableBranch(cond) {
				return true
			}
			// For inequality operations (<, >, <=, >=), traditional branches (BGEZ, BLTU, etc.)
			// are more efficient (2 instructions) than zicond (4 instructions).
			// Only use zicond for inequalities if they are in optimizable patterns
			// (which are already handled above).
			if isInequalityOp(cond) {
				return false
			}
			// Check if yes/no blocks contain simple arithmetic operations that benefit from zicond
			// even for predictable branches (e.g., NEG, NOT, INC, ADD operations)
			hasSimpleOp := false
			for _, block := range []*Block{yes, no} {
				if len(block.Values) == 1 {
					op := block.Values[0].Op
					// Simple arithmetic operations that are cheap to speculatively execute
					if op == OpNeg64 || op == OpNeg32 || op == OpCom64 || op == OpCom32 ||
						op == OpAdd64 || op == OpAdd32 || op == OpSub64 || op == OpSub32 ||
						op == OpOr64 || op == OpOr32 || op == OpXor64 || op == OpXor32 {
						hasSimpleOp = true
						break
					}
				}
			}
			if hasSimpleOp {
				// Allow zicond for these simple operations, as they avoid branch misprediction
				// and the instruction count difference (3 vs 2) is small
				return true
			}
			// Conservative: don't use zicond for single simple conditional select
			// with predictable branches and complex operations
			return false
		}
		// Multiple phis: cost is amortized across multiple conditional selects.
		// Each phi adds 2 extra instructions, but we avoid multiple branches
		// and potential misprediction penalties.
		return phi >= 2
	}
}

// isConditionalArithmeticCandidateForElimIf checks if a phi in elimIf can be optimized
// by conditional arithmetic rules. Similar to isConditionalArithmeticCandidate but
// for the elimIf case where one branch is 'simple' and the other is fallthrough.
//
// Pattern: if cond { result = base op value } else { result = base }
// Example: if cond == 0 { a += b } else { a = a }
// In SSA: result = Phi [base, base op value] or Phi [base op value, base]
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

// isConditionalArithmeticCandidate checks if a phi can be optimized
// by conditional arithmetic rules in RISCV64.rules (e.g., cmoveAddZero).
// These rules reduce zicond from 4 instructions to 2 instructions,
// making it always beneficial compared to branches.
//
// Pattern: if cond { result = base op value } else { result = base }
// Example: if cond == 0 { a += b } else { a = a }
// In SSA: result = Phi [base, base op value] or Phi [base op value, base]
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

// isArithmeticOp checks if an operation is an arithmetic operation
// that can be optimized by conditional arithmetic rules.
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

// isZeroOrConstSelect checks if a phi represents a pattern that can be
// optimized to a single zicond instruction (CZEROEQZ or CZERONEZ).
//
// Patterns detected:
// 1. if cond { x = value } else { x = 0 } → CZEROEQZ value cond
// 2. if cond { x = 0 } else { x = value } → CZERONEZ value cond
// 3. if cond { x = value } else { x = const } → can optimize if const is 0
// 4. if cond { x = const } else { x = value } → can optimize if const is 0
// 5. if cond { x = -1 } else { x = 0 } → CZEROEQZ -1 cond (special case)
//
// These patterns are optimized by RISCV64.rules:
// - Rule 870: (CZERO(EQ|NE)Z (MOVDconst [0]) _) => (MOVDconst [0])
// - New rules: OR (CZEROEQZ x cond) (CZERONEZ (MOVDconst [0]) cond) => CZEROEQZ x cond
func isZeroOrConstSelect(v *Value, no, yes *Block) bool {
	if len(v.Args) != 2 {
		return false
	}
	// Check if one arg is zero constant and the other is a constant or from no/yes blocks
	for i, arg := range v.Args {
		otherArg := v.Args[1-i]
		// Pattern 1: one arg is zero constant, other arg is non-zero constant or from branch block
		if arg.isGenericIntConst() && arg.AuxInt == 0 {
			// The other arg can be:
			// 1. A non-zero constant (from entry block or elsewhere)
			if otherArg.isGenericIntConst() && otherArg.AuxInt != 0 {
				return true
			}
			// 2. A value from one of the branch blocks
			if otherArg.Block == no || otherArg.Block == yes {
				return true
			}
		}
		// Pattern 2: other arg is zero, this arg is non-zero constant or from branch
		if otherArg.isGenericIntConst() && otherArg.AuxInt == 0 {
			// This arg can be:
			// 1. A non-zero constant (from entry block or elsewhere)
			if arg.isGenericIntConst() && arg.AuxInt != 0 {
				return true
			}
			// 2. A value from one of the branch blocks
			if arg.Block == no || arg.Block == yes {
				return true
			}
		}
	}
	return false
}

// isZeroOrConstSelectForElimIf checks if a phi in elimIf represents a pattern
// that can be optimized to a single zicond instruction.
// Similar to isZeroOrConstSelect but for elimIf structure.
//
// For elimIf, the pattern is:
// - simple block: contains one value (e.g., x = 182)
// - post block: contains Phi [value_from_simple, value_from_dom/entry]
// - Phi args correspond to post.Preds order: Args[i] comes from post.Preds[i].Block()
// We check if one arg is zero constant and the other comes from simple block.
func isZeroOrConstSelectForElimIf(v *Value, simple, post *Block) bool {
	if len(v.Args) != 2 || len(post.Preds) != 2 {
		return false
	}
	// Find which pred is simple block
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
	// Check if the arg from simple block is a non-zero value (constant or from simple block)
	// and the other arg is zero constant
	simpleArg := v.Args[simpleArgIdx]
	otherArgIdx := 1 - simpleArgIdx
	otherArg := v.Args[otherArgIdx]

	// Pattern 1: simple arg is non-zero (constant or value from simple block),
	//            other arg is zero constant
	// Example: if c { x = 182 } else { x = 0 } → CZEROEQZ
	if otherArg.isGenericIntConst() && otherArg.AuxInt == 0 {
		// simpleArg can be a constant (non-zero) or a value from simple block
		if simpleArg.isGenericIntConst() && simpleArg.AuxInt != 0 {
			return true
		}
		// Or simpleArg is a value defined in simple block
		if simpleArg.Block == simple {
			return true
		}
	}

	// Pattern 2: simple arg is zero constant, other arg is non-zero
	// Example: if c { x = 0 } else { x = value } → CZERONEZ
	if simpleArg.isGenericIntConst() && simpleArg.AuxInt == 0 {
		// otherArg should be a non-zero constant or value from other path
		if otherArg.isGenericIntConst() && otherArg.AuxInt != 0 {
			return true
		}
		// Or otherArg is a value from the fallthrough path (not from simple block)
		// This includes values from dom block or entry block
		if otherArg.Block != simple {
			return true
		}
	}

	return false
}

// isInequalityOp checks if a branch condition is an inequality comparison operation
// (less than, greater than, less or equal, greater or equal, but NOT equal/not equal).
// For RISC-V, inequality operations can be directly used in branch instructions
// (BGEZ, BLTU, etc.) which require only 2 instructions, while zicond requires 4 instructions.
// Therefore, we should prefer traditional branches for inequality operations.
// Note: This function does NOT include equality/inequality operations (==, !=) as they
// may benefit from zicond when unpredictable.
func isInequalityOp(cond *Value) bool {
	if cond == nil {
		return false
	}
	// Check if the condition is an inequality comparison operation (only <, >, <=, >=)
	switch cond.Op {
	case OpLess64, OpLess32, OpLess16, OpLess8,
		OpLess64U, OpLess32U, OpLess16U, OpLess8U,
		OpLeq64, OpLeq32, OpLeq16, OpLeq8,
		OpLeq64U, OpLeq32U, OpLeq16U, OpLeq8U:
		return true
	}
	// Also check for RISC-V specific inequality operations that may be generated
	// from generic comparisons (SLT, SLTU, SLTI, SLTIU)
	switch cond.Op {
	case OpRISCV64SLT, OpRISCV64SLTU, OpRISCV64SLTI, OpRISCV64SLTIU:
		return true
	}
	return false
}

// isLikelyUnpredictableBranch checks if a branch condition is likely to be
// unpredictable. Unpredictable branches benefit from zicond even for single
// phi nodes because misprediction penalty (17 cycles) is much higher than
// zicond overhead (2 extra instructions).
//
// Patterns that indicate unpredictable branches:
// - XOR operations: (x ^ y) & 1, (x ^ y) != 0
// - Bit manipulation: (x & mask) != (y & mask)
// - Hash-like computations: complex bit operations
// - Data-dependent conditions that don't follow simple patterns
func isLikelyUnpredictableBranch(cond *Value) bool {
	if cond == nil {
		return false
	}

	// Helper function to recursively check for unpredictable patterns
	var checkUnpredictable func(*Value, int) bool
	checkUnpredictable = func(v *Value, depth int) bool {
		if v == nil || depth > 5 {
			// Limit recursion depth to avoid infinite loops
			return false
		}

		switch v.Op {
		// XOR operations are typically unpredictable
		case OpXor64, OpXor32, OpXor16, OpXor8:
			return true

		// AND with small mask (especially & 1) combined with comparisons
		// often indicates bit-level unpredictability
		case OpAnd64, OpAnd32, OpAnd16, OpAnd8:
			// Check if one operand is a small constant mask (like 1)
			for _, arg := range v.Args {
				if arg.Op == OpConst64 || arg.Op == OpConst32 || arg.Op == OpConst16 || arg.Op == OpConst8 {
					mask := arg.AuxInt
					// Small masks (especially 1) often indicate bit-level checks
					// These are typically unpredictable when combined with XOR or comparisons
					if mask == 1 || mask == 3 || mask == 7 || mask == 15 {
						// Check if the other operand involves XOR or is from a data-dependent source
						for _, otherArg := range v.Args {
							if otherArg != arg && checkUnpredictable(otherArg, depth+1) {
								return true
							}
						}
					}
				}
			}
			// Recursively check arguments
			for _, arg := range v.Args {
				if checkUnpredictable(arg, depth+1) {
					return true
				}
			}

		// Comparisons involving XOR or AND results are often unpredictable
		case OpNeq64, OpNeq32, OpNeq16, OpNeq8,
			OpEq64, OpEq32, OpEq16, OpEq8:
			// Check if comparison involves XOR or bit manipulation
			for _, arg := range v.Args {
				if checkUnpredictable(arg, depth+1) {
					return true
				}
			}

		// Right shifts (especially >> 1, >> 2) combined with XOR
		// often indicate hash-like computations
		case OpRsh64Ux64, OpRsh32Ux64, OpRsh16Ux64, OpRsh8Ux64,
			OpRsh64Ux32, OpRsh32Ux32, OpRsh16Ux32, OpRsh8Ux32:
			// If combined with XOR in parent, it's likely unpredictable
			return depth > 0

		// OR operations with XOR or bit manipulation
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
