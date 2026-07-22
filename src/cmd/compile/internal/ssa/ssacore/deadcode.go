// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/src"
)

// LiveValues returns the live values in f and a list of values that are eligible
// to be statements in reversed data flow order.
// The second result is used to help conserve statement boundaries for debugging.
// reachable is a map from block ID to whether the block is reachable.
// The caller should call f.Cache.freeBoolSlice(live) and f.Cache.freeValueSlice(liveOrderStmts).
// when they are done with the return values.
func LiveValues(f *Func, reachable []bool) (live []bool, liveOrderStmts []*Value) {
	live = f.Cache.AllocBoolSlice(f.NumValues())
	liveOrderStmts = f.Cache.AllocValueSlice(f.NumValues())[:0]

	// After regalloc, consider all values to be live.
	// See the comment at the top of regalloc.go and in deadcode for details.
	if f.RegAlloc != nil {
		for i := range live {
			live[i] = true
		}
		return
	}

	// Record all the inline indexes we need
	var liveInlIdx map[int]bool
	pt := f.Config.Ctxt.PosTable
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			i := pt.Pos(v.Pos).Base().InliningIndex()
			if i < 0 {
				continue
			}
			if liveInlIdx == nil {
				liveInlIdx = map[int]bool{}
			}
			liveInlIdx[i] = true
		}
		i := pt.Pos(b.Pos).Base().InliningIndex()
		if i < 0 {
			continue
		}
		if liveInlIdx == nil {
			liveInlIdx = map[int]bool{}
		}
		liveInlIdx[i] = true
	}

	// Find all live values
	q := f.Cache.AllocValueSlice(f.NumValues())[:0]
	defer f.Cache.FreeValueSlice(q)

	// Starting set: all control values of reachable blocks are live.
	// Calls are live (because callee can observe the memory state).
	for _, b := range f.Blocks {
		if !reachable[b.ID] {
			continue
		}
		for _, v := range b.ControlValues() {
			if !live[v.ID] {
				live[v.ID] = true
				q = append(q, v)
				if v.Pos.IsStmt() != src.PosNotStmt {
					liveOrderStmts = append(liveOrderStmts, v)
				}
			}
		}
		for _, v := range b.Values {
			if (ssaop.OpcodeTable[v.Op].Call || ssaop.OpcodeTable[v.Op].HasSideEffects || ssaop.OpcodeTable[v.Op].NilCheck) && !live[v.ID] {
				live[v.ID] = true
				q = append(q, v)
				if v.Pos.IsStmt() != src.PosNotStmt {
					liveOrderStmts = append(liveOrderStmts, v)
				}
			}
			if v.Op == ssaop.OpInlMark {
				if !liveInlIdx[int(v.AuxInt)] {
					// We don't need marks for bodies that
					// have been completely optimized away.
					// TODO: save marks only for bodies which
					// have a faulting instruction or a call?
					continue
				}
				live[v.ID] = true
				q = append(q, v)
				if v.Pos.IsStmt() != src.PosNotStmt {
					liveOrderStmts = append(liveOrderStmts, v)
				}
			}
		}
	}

	// Compute transitive closure of live values.
	for len(q) > 0 {
		// pop a reachable value
		v := q[len(q)-1]
		q[len(q)-1] = nil
		q = q[:len(q)-1]
		for i, x := range v.Args {
			if v.Op == ssaop.OpPhi && !reachable[v.Block.Preds[i].B.ID] {
				continue
			}
			if !live[x.ID] {
				live[x.ID] = true
				q = append(q, x) // push
				if x.Pos.IsStmt() != src.PosNotStmt {
					liveOrderStmts = append(liveOrderStmts, x)
				}
			}
		}
	}

	return
}

// ReachableBlocks returns the reachable blocks in f.
func ReachableBlocks(f *Func) []bool {
	reachable := make([]bool, f.NumBlocks())
	reachable[f.Entry.ID] = true
	p := make([]*Block, 0, 64) // stack-like worklist
	p = append(p, f.Entry)
	for len(p) > 0 {
		// Pop a reachable block
		b := p[len(p)-1]
		p = p[:len(p)-1]
		// Mark successors as reachable
		s := b.Succs
		if b.Kind == block.BlockFirst {
			s = s[:1]
		}
		for _, e := range s {
			c := e.B
			if int(c.ID) >= len(reachable) {
				f.Fatalf("block %s >= f.NumBlocks()=%d?", c, len(reachable))
			}
			if !reachable[c.ID] {
				reachable[c.ID] = true
				p = append(p, c) // push
			}
		}
	}
	return reachable
}

// findlive returns the reachable blocks and live values in f.
// The caller should call f.Cache.freeBoolSlice(live) when it is done with it.
func findlive(f *Func) (reachable []bool, live []bool) {
	reachable = ReachableBlocks(f)
	var order []*Value
	live, order = LiveValues(f, reachable)
	f.Cache.FreeValueSlice(order)
	return
}

// RemoveEdge removes the i'th outgoing edge from b (and
// the corresponding incoming edge from b.Succs[i].b).
// Note that this potentially reorders successors of b, so it
// must be used very carefully.
func (b *Block) RemoveEdge(i int) {
	e := b.Succs[i]
	c := e.B
	j := e.I

	// Adjust b.Succs
	b.RemoveSucc(i)

	// Adjust c.Preds
	c.RemovePred(j)

	// Remove phi args from c's phis.
	for _, v := range c.Values {
		if v.Op != ssaop.OpPhi {
			continue
		}
		c.RemovePhiArg(v, j)
		// Note: this is trickier than it looks. Replacing
		// a Phi with a Copy can in general cause problems because
		// Phi and Copy don't have exactly the same semantics.
		// Phi arguments always come from a predecessor block,
		// whereas copies don't. This matters in loops like:
		// 1: x = (Phi y)
		//    y = (Add x 1)
		//    goto 1
		// If we replace Phi->Copy, we get
		// 1: x = (Copy y)
		//    y = (Add x 1)
		//    goto 1
		// (Phi y) refers to the *previous* value of y, whereas
		// (Copy y) refers to the *current* value of y.
		// The modified code has a cycle and the scheduler
		// will barf on it.
		//
		// Fortunately, this situation can only happen for dead
		// code loops. We know the code we're working with is
		// not dead, so we're ok.
		// Proof: If we have a potential bad cycle, we have a
		// situation like this:
		//   x = (Phi z)
		//   y = (op1 x ...)
		//   z = (op2 y ...)
		// Where opX are not Phi ops. But such a situation
		// implies a cycle in the dominator graph. In the
		// example, x.Block dominates y.Block, y.Block dominates
		// z.Block, and z.Block dominates x.Block (treating
		// "dominates" as reflexive).  Cycles in the dominator
		// graph can only happen in an unreachable cycle.
	}
}
