// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "log"

// deadcode removes dead code from f.
func deadcode(f *Func) {

	// Find all reachable basic blocks.
	reachable := make([]bool, f.NumBlocks())
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
	live := make([]bool, f.NumValues()) // flag to set for each live value
	var q []*Value                      // stack-like worklist of unscanned values

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
		for _, x := range v.Args {
			if !live[x.ID] {
				live[x.ID] = true
				q = append(q, x) // push
			}
		}
	}

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
				panic("live value in unreachable block")
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

// There was an edge b->c.  It has been removed from b's successors.
// Fix up c to handle that fact.
func removePredecessor(b, c *Block) {
	n := len(c.Preds) - 1
	if n == 0 {
		// c is now dead - don't bother working on it
		if c.Preds[0] != b {
			log.Panicf("%s.Preds[0]==%s, want %s", c, c.Preds[0], b)
		}
		return
	}

	// find index of b in c's predecessor list
	var i int
	for j, p := range c.Preds {
		if p == b {
			i = j
			break
		}
	}

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
		}
	}
}
