// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// findlive returns the reachable blocks and live values in f.
func findlive(f *Func) (reachable []bool, live []bool) {
	reachable = reachableBlocks(f)
	live = liveValues(f, reachable)
	return
}

// reachableBlocks returns the reachable blocks in f.
func reachableBlocks(f *Func) []bool {
	reachable := make([]bool, f.NumBlocks())
	reachable[f.Entry.ID] = true
	p := []*Block{f.Entry} // stack-like worklist
	for len(p) > 0 {
		// Pop a reachable block
		b := p[len(p)-1]
		p = p[:len(p)-1]
		// Mark successors as reachable
		s := b.Succs
		if b.Kind == BlockFirst {
			s = s[:1]
		}
		for _, c := range s {
			if !reachable[c.ID] {
				reachable[c.ID] = true
				p = append(p, c) // push
			}
		}
	}
	return reachable
}

// liveValues returns the live values in f.
// reachable is a map from block ID to whether the block is reachable.
func liveValues(f *Func, reachable []bool) []bool {
	live := make([]bool, f.NumValues())

	// After regalloc, consider all values to be live.
	// See the comment at the top of regalloc.go and in deadcode for details.
	if f.RegAlloc != nil {
		for i := range live {
			live[i] = true
		}
		return live
	}

	// Find all live values
	var q []*Value // stack-like worklist of unscanned values

	// Starting set: all control values of reachable blocks are live.
	for _, b := range f.Blocks {
		if !reachable[b.ID] {
			continue
		}
		if v := b.Control; v != nil && !live[v.ID] {
			live[v.ID] = true
			q = append(q, v)
		}
	}

	// Compute transitive closure of live values.
	for len(q) > 0 {
		// pop a reachable value
		v := q[len(q)-1]
		q = q[:len(q)-1]
		for i, x := range v.Args {
			if v.Op == OpPhi && !reachable[v.Block.Preds[i].ID] {
				continue
			}
			if !live[x.ID] {
				live[x.ID] = true
				q = append(q, x) // push
			}
		}
	}

	return live
}

// deadcode removes dead code from f.
func deadcode(f *Func) {
	// deadcode after regalloc is forbidden for now.  Regalloc
	// doesn't quite generate legal SSA which will lead to some
	// required moves being eliminated.  See the comment at the
	// top of regalloc.go for details.
	if f.RegAlloc != nil {
		f.Fatalf("deadcode after regalloc")
	}

	// Find reachable blocks.
	reachable := reachableBlocks(f)

	// Get rid of edges from dead to live code.
	for _, b := range f.Blocks {
		if reachable[b.ID] {
			continue
		}
		for _, c := range b.Succs {
			if reachable[c.ID] {
				c.removePred(b)
			}
		}
	}

	// Get rid of dead edges from live code.
	for _, b := range f.Blocks {
		if !reachable[b.ID] {
			continue
		}
		if b.Kind != BlockFirst {
			continue
		}
		c := b.Succs[1]
		b.Succs[1] = nil
		b.Succs = b.Succs[:1]
		b.Kind = BlockPlain
		b.Likely = BranchUnknown

		if reachable[c.ID] {
			// Note: c must be reachable through some other edge.
			c.removePred(b)
		}
	}

	// Splice out any copies introduced during dead block removal.
	copyelim(f)

	// Find live values.
	live := liveValues(f, reachable)

	// Remove dead & duplicate entries from namedValues map.
	s := newSparseSet(f.NumValues())
	i := 0
	for _, name := range f.Names {
		j := 0
		s.clear()
		values := f.NamedValues[name]
		for _, v := range values {
			if live[v.ID] && !s.contains(v.ID) {
				values[j] = v
				j++
				s.add(v.ID)
			}
		}
		if j == 0 {
			delete(f.NamedValues, name)
		} else {
			f.Names[i] = name
			i++
			for k := len(values) - 1; k >= j; k-- {
				values[k] = nil
			}
			f.NamedValues[name] = values[:j]
		}
	}
	for k := len(f.Names) - 1; k >= i; k-- {
		f.Names[k] = LocalSlot{}
	}
	f.Names = f.Names[:i]

	// Remove dead values from blocks' value list.  Return dead
	// values to the allocator.
	for _, b := range f.Blocks {
		i := 0
		for _, v := range b.Values {
			if live[v.ID] {
				b.Values[i] = v
				i++
			} else {
				f.freeValue(v)
			}
		}
		// aid GC
		tail := b.Values[i:]
		for j := range tail {
			tail[j] = nil
		}
		b.Values = b.Values[:i]
	}

	// Remove unreachable blocks.  Return dead block ids to allocator.
	i = 0
	for _, b := range f.Blocks {
		if reachable[b.ID] {
			f.Blocks[i] = b
			i++
		} else {
			if len(b.Values) > 0 {
				b.Fatalf("live values in unreachable block %v: %v", b, b.Values)
			}
			b.Preds = nil
			b.Succs = nil
			b.Control = nil
			b.Kind = BlockDead
			f.freeBlock(b)
		}
	}
	// zero remainder to help GC
	tail := f.Blocks[i:]
	for j := range tail {
		tail[j] = nil
	}
	f.Blocks = f.Blocks[:i]

	// TODO: renumber Blocks and Values densely?
	// TODO: save dead Values and Blocks for reuse?  Or should we just let GC handle it?
}

// removePred removes the predecessor p from b's predecessor list.
func (b *Block) removePred(p *Block) {
	var i int
	found := false
	for j, q := range b.Preds {
		if q == p {
			i = j
			found = true
			break
		}
	}
	// TODO: the above loop could make the deadcode pass take quadratic time
	if !found {
		b.Fatalf("can't find predecessor %v of %v\n", p, b)
	}

	n := len(b.Preds) - 1
	b.Preds[i] = b.Preds[n]
	b.Preds[n] = nil // aid GC
	b.Preds = b.Preds[:n]

	// rewrite phi ops to match the new predecessor list
	for _, v := range b.Values {
		if v.Op != OpPhi {
			continue
		}
		v.Args[i] = v.Args[n]
		v.Args[n] = nil // aid GC
		v.Args = v.Args[:n]
		if n == 1 {
			v.Op = OpCopy
			// Note: this is trickier than it looks.  Replacing
			// a Phi with a Copy can in general cause problems because
			// Phi and Copy don't have exactly the same semantics.
			// Phi arguments always come from a predecessor block,
			// whereas copies don't.  This matters in loops like:
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
			// code loops.  We know the code we're working with is
			// not dead, so we're ok.
			// Proof: If we have a potential bad cycle, we have a
			// situation like this:
			//   x = (Phi z)
			//   y = (op1 x ...)
			//   z = (op2 y ...)
			// Where opX are not Phi ops.  But such a situation
			// implies a cycle in the dominator graph.  In the
			// example, x.Block dominates y.Block, y.Block dominates
			// z.Block, and z.Block dominates x.Block (treating
			// "dominates" as reflexive).  Cycles in the dominator
			// graph can only happen in an unreachable cycle.
		}
	}
}
