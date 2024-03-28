// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
)

// ----------------------------------------------------------------------------
// Loop Closed SSA Form
//
// loop closed SSA form is a special form of SSA form, which is used to simplify
// loop optimization. It ensures that all values defined inside the loop are only
// used within loop. The transformation looks up loop uses outside the loop and
// inserts the appropriate "proxy phi" at the loop exit, after which the outside
// of the loop does not use the loop def directly but the proxy phi.
//
//	 loop header:                         loop header:
//	 v3 = Phi(0, v4)                      v3 = Phi(0, v4)
//	 If cond->loop latch,loop exit        If cond->loop latch,loop exit
//
//	 loop latch:                          loop latch:
//	 v4 = Add(v3, 1)                =>    v4 = Add(v3, 1)
//	 Plain->loop header                   Plain->loop header
//
//	 loop exit:                           loop exit:
//	 v5 = Add(5, v3)                      v6 = Phi(v3)  <= Proxy Phi
//	 Ret v18                              v5 = Add(5, v6)
//	                                      Ret v18
//
// Previously, v5 used v3 directly, where v5 is in the loop exit which is outside
// the loop. After LCSSA transformation, v5 uses v6, which in turn uses v3. Here,
// v6 is the proxy phi. In the context of LCSSA, we can consider the use block of
// v6 to be the loop header rather than the loop exit. This way, all values defined
// in the loop are loop "closed", i.e. only used within the loop.
//
// Any further changes to the loop definition only need to update the proxy phi,
// rather than iterating through all its uses and handling properties such as
// dominance relationships carefully, which is error prone and hard to maintain.

// Def-Use utilities
type user struct {
	def   *Value // the definition
	val   *Value // used by value
	block *Block // used by block's ctrl value
	idx   int    // in which arg index of user is def located
}

type defUses map[*Value][]*user

func (u *user) String() string {
	if u.val != nil {
		return fmt.Sprintf("{%v:%v}", u.def, u.val)
	} else {
		return fmt.Sprintf("{%v:%v}", u.def, u.block)
	}
}

// useBlock returns the block where the def is used
func (u *user) useBlock() *Block {
	if u.val != nil {
		return u.val.Block
	} else {
		return u.block
	}
}

// replaceUse replaces the use of def with new use at given index
func (u *user) replaceUse(newUse *Value) {
	if val := u.val; val != nil {
		idx := u.idx
		assert(val.Args[idx] == u.def, "sanity check")
		val.SetArg(idx, newUse)
	} else if block := u.block; block != nil {
		idx := u.idx
		assert(block.ControlValues()[idx] == u.def, "sanity check")
		block.ReplaceControl(idx, newUse)
	} else {
		panic("def is neither used by value nor by block ctrl")
	}
}

// buildDefUses builds def-use map for given defs Values
func buildDefUses(fn *Func, defs []*Value) defUses {
	defUses := make(defUses, 0)
	for _, def := range defs {
		if _, exist := defUses[def]; !exist {
			// Many duplicate definitions, avoid redundant memory allocations
			defUses[def] = make([]*user, 0, def.Uses)
		}
	}
	for _, block := range fn.Blocks {
		for _, val := range block.Values {
			for iarg, arg := range val.Args {
				if _, exist := defUses[arg]; exist {
					defUses[arg] = append(defUses[arg], &user{arg, val, nil, iarg})
				}
			}
		}
		for ictrl, ctrl := range block.ControlValues() {
			if _, exist := defUses[ctrl]; exist {
				defUses[ctrl] = append(defUses[ctrl], &user{ctrl, nil, block, ictrl})
			}
		}
	}
	return defUses
}

// stableDefs returns the defs in stable order for deterministic compilation
func stableDefs(defUses defUses) []*Value {
	keys := make([]*Value, 0)
	for k := range defUses {
		keys = append(keys, k)
	}
	sort.SliceStable(keys, func(i, j int) bool {
		return keys[i].ID < keys[j].ID
	})

	return keys
}

type lcssa struct {
	fn    *Func
	mphis []*Value          // inserted memory proxy phi
	e2phi map[*Block]*Value // exit block to proxy phi mapping
}

// findUseBlock returns the block where the def is used. If the use is type of Phi,
// then the use block is the corresponding incoming block. Note that this is ONLY
// valid in context of LCSSA.
func findUseBlock(u *user) *Block {
	var ub *Block
	if val := u.val; val != nil {
		if val.Op == OpPhi {
			ipred := u.idx
			ub = val.Block.Preds[ipred].b
		} else {
			ub = val.Block
		}
	} else {
		ub = u.block
	}
	assert(ub != nil, "no use block")
	return ub
}

// containsBlock returns true if the block is part of the loop or part of the
// inner loop
func (ln *loopnest) containsBlock(loop *loop, block *Block) bool {
	assert(ln.initializedChildren, "initialize loopnest children first")

	// Block is part of current loop?
	if ln.b2l[block.ID] == loop {
		return true
	}
	// Block is part of inner loop?
	for _, child := range loop.children {
		if ln.containsBlock(child, block) {
			return true
		}
	}
	return false
}

// allocateProxyPhi allocates a proxy phi at specific loop exit
func (lc *lcssa) allocateProxyPhi(exit *Block, loopDef ...*Value) *Value {
	assert(len(loopDef) > 0, "must have at least one loop def")
	if phival, exist := lc.e2phi[exit]; exist {
		return phival
	}

	phi := lc.fn.newValueNoBlock(OpPhi, loopDef[0].Type, loopDef[0].Pos)
	if len(loopDef) == 1 {
		phiArgs := make([]*Value, len(exit.Preds))
		for idx := range exit.Preds {
			phiArgs[idx] = loopDef[0]
		}
		phi.AddArgs(phiArgs...)
	} else {
		phi.AddArgs(loopDef...)
	}

	exit.placeValue(phi)
	lc.e2phi[exit] = phi
	if phi.Type.IsMemory() {
		lc.mphis = append(lc.mphis, phi)
	}
	return phi
}

func (lc *lcssa) fixProxyPhiMem(fn *Func) {
	if len(lc.mphis) == 0 {
		// No mem proxy phi to fix
		return
	}
	lastMem := computeLastMem(fn)
	for _, phi := range lc.mphis {
		assert(phi.Type.IsMemory(), "must be memory phi")

		for iarg, arg := range phi.Args {
			mem := lastMem[phi.Block.Preds[iarg].b.ID]
			if mem != arg && mem != nil {
				if mem.Args[0] != arg {
					fn.Fatalf("must use old memory")
				}
				oldPhiStr := phi.LongString()
				phi.SetArg(iarg, mem)
				if fn.pass.debug > 1 {
					fmt.Printf("== Fix memory proxy phi %v to %v\n",
						oldPhiStr, phi.LongString())
				}
			}
		}
	}
}

// placeProxyPhi places the proxy phi at loop exits to make sure all uses of a
// loop defined value are dominated by the proxy phi
func (lc *lcssa) placeProxyPhi(ln *loopnest, loop *loop, defs []*Value) bool {
	defUses := buildDefUses(ln.f, defs)

	use2exits := make(map[*user][]*Block, 0)
	loopDefs := stableDefs(defUses)
	for _, loopDef := range loopDefs {
		for _, use := range defUses[loopDef] {
			useBlock := findUseBlock(use)
			// It's an in-loop use?
			if ln.b2l[useBlock.ID] == loop {
				continue
			}

			// Loop def does not dominate use? Possibly dead block
			if !ln.sdom.IsAncestorEq(loopDef.Block, useBlock) {
				continue
			}

			// Possibly a dead block, ignore it
			if len(useBlock.Preds) == 0 {
				assert(useBlock.Kind == BlockInvalid, "why not otherwise")
				continue
			}

			// Only loop use that is not part of current loop takes into account.
			if useBlock != loopDef.Block && !ln.containsBlock(loop, useBlock) {
				// Simple case, try to find a loop exit that dominates the use
				// block and place the proxy phi at this loop exit, this is the
				// most common case
				var domExit *Block
				for _, exit := range loop.exits {
					if ln.sdom.IsAncestorEq(exit, useBlock) {
						domExit = exit
						break
					}
				}
				if domExit != nil {
					use2exits[use] = append(use2exits[use], domExit)
					continue
				}
				// Harder case, loop use block is not dominated by a single loop
				// exit, instead it has many predecessors and all of them are
				// dominated by different loop exits, we are probably reaching to
				// it from all of these predecessors. In this case, we need to
				// place the proxy phi at all loop exits and merge them at loop
				// use block by yet another proxy phi
				domExits := make([]*Block, 0, len(useBlock.Preds))
				for _, pred := range useBlock.Preds {
					found := false
					for _, e := range loop.exits {
						if ln.sdom.IsAncestorEq(e, pred.b) {
							domExits = append(domExits, e)
							found = true
							break
						}
					}
					if !found {
						break
					}
				}
				if cap(domExits) == len(domExits) {
					use2exits[use] = domExits
					continue
				}

				// Worst case, loop use block is not dominated by any of loop exits
				// we start from all loop exits(including inner loop exits) though
				// dominance frontier and see if we can reach to the use block,
				// if so, we place the proxy phi at the loop exit that is closest
				// to the use block. This is rare, but it does happen, give up
				// for now as it's hard to handle.
				// TODO(yyang): Correctly handle this case
				if ln.f.pass.debug > 1 {
					fmt.Printf("== Can not process use %v in %v\n", use, loop)
				}
				return false
			}
		}
	}

	// For every use of loop def, place the proxy phi at proper exit block
	// and replace such use with the proxy phi, this is the core of LCSSA,
	// since proxy phi is "inside the loop" in context of LCSSA, now all uses
	// of loop def are loop closed, e.g. lives in the loop.
	for _, loopDef := range loopDefs {
		uses := defUses[loopDef]
		if len(uses) == 0 {
			continue
		}
		// multiple uses shares the same proxy phi if they live in same exit block
		// also note that only users of the same loop def could share proxy phi
		lc.e2phi = make(map[*Block]*Value, 0)
		for _, use := range uses {
			useBlock := findUseBlock(use)
			exits := use2exits[use]
			if len(exits) == 1 {
				domExit := exits[0]
				// Replace all uses of loop def with new proxy phi
				lcphi := lc.allocateProxyPhi(domExit, loopDef)
				if ln.f.pass.debug > 1 {
					fmt.Printf("== Replace use %v with proxy phi %v\n",
						use, lcphi.LongString())
				}
				use.replaceUse(lcphi)
			} else if len(exits) > 1 {
				// Place proxy phi at all dominator loop exits
				phis := make([]*Value, 0, len(exits))
				for _, exit := range exits {
					lcphi := lc.allocateProxyPhi(exit, loopDef)
					phis = append(phis, lcphi)
				}
				// Merge them at loop use block by yet another proxy phi
				lcphi := lc.allocateProxyPhi(useBlock, phis...)
				use.replaceUse(lcphi)
				if ln.f.pass.debug > 1 {
					fmt.Printf("== Replace use %v with proxy phi %v\n",
						use, lcphi.LongString())
				}
			}
		}
	}

	// Since we may have placed memory proxy phi at some loop exits, which
	// use loop def and produce new memory. If this block is a predecessor
	// of another loop exit, we need to use memory proxy phi instead of loop
	// def as a parameter of new proxy phi.
	lc.fixProxyPhiMem(ln.f)

	return true
}

// BuildLoopClosedForm builds loop closed SSA form upon original loop, this is
// the cornerstone of other loop optimizations such as LICM, loop unswitching
// and empty loop elimination.
func (fn *Func) BuildLoopClosedForm(ln *loopnest, loop *loop) bool {
	assert(ln.initializedExits && ln.initializedChildren, "must be initialized")
	if len(loop.exits) == 0 {
		return true
	}

	sdom := ln.sdom // lcssa does not wire up CFG, reusing sdom is okay
	domBlocks := make([]*Block, 0)
	blocks := make([]*Block, 0)
	blocks = append(blocks, loop.exits...)

	// Outside the loop we can only use values defined in the blocks of arbitrary
	// loop exit dominators, so first collect these blocks and treat the Values
	// in them as loop def
	for len(blocks) > 0 {
		block := blocks[0]
		blocks = blocks[1:]
		if block == loop.header {
			continue
		}
		idom := sdom.Parent(block)
		if ln.b2l[idom.ID] != loop {
			continue
		}

		domBlocks = append(domBlocks, idom)
		blocks = append(blocks, idom)
	}

	// Look for out-of-loop users of these loop defs
	defs := make([]*Value, 0)
	for _, block := range domBlocks {
		for _, val := range block.Values {
			if val.Uses == 0 {
				continue
			}
			defs = append(defs, val)
		}
	}

	// For every use of loop def, place the proxy phi at the proper block
	lc := &lcssa{
		fn:    fn,
		mphis: make([]*Value, 0, len(defs)),
		e2phi: nil,
	}
	return lc.placeProxyPhi(ln, loop, defs)
}
