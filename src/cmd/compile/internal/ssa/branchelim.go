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
// bb0            bb0
//  | \          /   \
//  | bb1  or  bb1   bb2    <- trivial if/else blocks
//  | /          \   /
// bb2            bb3
//
// where the intermediate blocks are mostly empty (with no side-effects);
// rewrite Phis in the postdominator as CondSelects.
func branchelim(f *Func) {
	// FIXME: add support for lowering CondSelects on more architectures
	switch f.Config.arch {
	case "arm64", "amd64", "wasm":
		// implemented
	default:
		return
	}

	// Find all the values used in computing the address of any load.
	// Typically these values have operations like AddPtr, Lsh64x64, etc.
	loadAddr := f.newSparseSet(f.NumValues())
	defer f.retSparseSet(loadAddr)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpLoad, OpAtomicLoad8, OpAtomicLoad32, OpAtomicLoad64, OpAtomicLoadPtr, OpAtomicLoadAcq32:
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
	if loadAddr.contains(v.ID) {
		// The result of the soon-to-be conditional move is used to compute a load address.
		// We want to avoid generating a conditional move in this case
		// because the load address would now be data-dependent on the condition.
		// Previously it would only be control-dependent on the condition, which is faster
		// if the branch predicts well (or possibly even if it doesn't, if the load will
		// be an expensive cache miss).
		// See issue #26306.
		return false
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
	if !shouldElimIfElse(no, yes, post, f.Config.arch) {
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

// shouldElimIfElse reports whether estimated cost of eliminating branch
// is lower than threshold.
func shouldElimIfElse(no, yes, post *Block, arch string) bool {
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
	}
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
		if v.Op == OpPhi || isDivMod(v.Op) || v.Type.IsMemory() ||
			v.MemoryArg() != nil || opcodeTable[v.Op].hasSideEffects {
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
