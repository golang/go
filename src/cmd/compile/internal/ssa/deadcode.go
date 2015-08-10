// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// findlive returns the reachable blocks and live values in f.
func findlive(f *Func) (reachable []bool, live []bool) {
	// Find all reachable basic blocks.
	reachable = make([]bool, f.NumBlocks())
	reachable[f.Entry.ID] = true
	p := []*Block{f.Entry} // stack-like worklist
	for len(p) > 0 {
		// Pop a reachable block
		b := p[len(p)-1]
		p = p[:len(p)-1]
		// Mark successors as reachable
		for _, c := range b.Succs {
			if !reachable[c.ID] {
				reachable[c.ID] = true
				p = append(p, c) // push
			}
		}
	}

	// Find all live values
	live = make([]bool, f.NumValues()) // flag to set for each live value
	var q []*Value                     // stack-like worklist of unscanned values

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

	return reachable, live
}

// deadcode removes dead code from f.
func deadcode(f *Func) {
	reachable, live := findlive(f)

	// Remove dead values from blocks' value list.  Return dead
	// value ids to the allocator.
	for _, b := range f.Blocks {
		i := 0
		for _, v := range b.Values {
			if live[v.ID] {
				b.Values[i] = v
				i++
			} else {
				f.vid.put(v.ID)
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
	i := 0
	for _, b := range f.Blocks {
		if reachable[b.ID] {
			f.Blocks[i] = b
			i++
		} else {
			if len(b.Values) > 0 {
				b.Fatalf("live values in unreachable block %v: %v", b, b.Values)
			}
			s := b.Succs
			b.Succs = nil
			for _, c := range s {
				f.removePredecessor(b, c)
			}
			f.bid.put(b.ID)
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

// There was an edge b->c.  c has been removed from b's successors.
// Fix up c to handle that fact.
func (f *Func) removePredecessor(b, c *Block) {
	work := [][2]*Block{{b, c}}

	for len(work) > 0 {
		b, c := work[0][0], work[0][1]
		work = work[1:]

		// Find index of b in c's predecessor list
		// TODO: This could conceivably cause O(n^2) work.  Imagine a very
		// wide phi in (for example) the return block.  If we determine that
		// lots of panics won't happen, we remove each edge at a cost of O(n) each.
		var i int
		found := false
		for j, p := range c.Preds {
			if p == b {
				i = j
				found = true
				break
			}
		}
		if !found {
			f.Fatalf("can't find predecessor %v of %v\n", b, c)
		}

		n := len(c.Preds) - 1
		c.Preds[i] = c.Preds[n]
		c.Preds[n] = nil // aid GC
		c.Preds = c.Preds[:n]

		// rewrite phi ops to match the new predecessor list
		for _, v := range c.Values {
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
				// code loops.  So although the value graph is transiently
				// bad, we'll throw away the bad part by the end of
				// the next deadcode phase.
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
		if n == 0 {
			// c is now dead--recycle its values
			for _, v := range c.Values {
				f.vid.put(v.ID)
			}
			c.Values = nil
			// Also kill any successors of c now, to spare later processing.
			for _, succ := range c.Succs {
				work = append(work, [2]*Block{c, succ})
			}
			c.Succs = nil
			c.Kind = BlockDead
			c.Control = nil
		}
	}
}
