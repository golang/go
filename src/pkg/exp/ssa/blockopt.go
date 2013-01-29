package ssa

// Simple block optimisations to simplify the control flow graph.

// TODO(adonovan): instead of creating several "unreachable" blocks
// per function in the Builder, reuse a single one (e.g. at Blocks[1])
// to reduce garbage.

import (
	"fmt"
	"os"
)

// If true, perform sanity checking and show progress at each
// successive iteration of optimizeBlocks.  Very verbose.
const debugBlockOpt = false

func hasPhi(b *BasicBlock) bool {
	_, ok := b.Instrs[0].(*Phi)
	return ok
}

// prune attempts to prune block b if it is unreachable (i.e. has no
// predecessors other than itself), disconnecting it from the CFG.
// The result is true if the optimisation was applied.  i is the block
// index within the function.
//
func prune(f *Function, i int, b *BasicBlock) bool {
	if i == 0 {
		return false // don't prune entry block
	}
	if len(b.Preds) == 0 || len(b.Preds) == 1 && b.Preds[0] == b {
		// Disconnect it from its successors.
		for _, c := range b.Succs {
			c.removePred(b)
		}
		if debugBlockOpt {
			fmt.Fprintln(os.Stderr, "prune", b.Name)
		}

		// Delete b.
		f.Blocks[i] = nil
		return true
	}
	return false
}

// jumpThreading attempts to apply simple jump-threading to block b,
// in which a->b->c become a->c if b is just a Jump.
// The result is true if the optimisation was applied.
// i is the block index within the function.
//
func jumpThreading(f *Function, i int, b *BasicBlock) bool {
	if i == 0 {
		return false // don't apply to entry block
	}
	if b.Instrs == nil {
		fmt.Println("empty block ", b.Name)
		return false
	}
	if _, ok := b.Instrs[0].(*Jump); !ok {
		return false // not just a jump
	}
	c := b.Succs[0]
	if c == b {
		return false // don't apply to degenerate jump-to-self.
	}
	if hasPhi(c) {
		return false // not sound without more effort
	}
	for j, a := range b.Preds {
		a.replaceSucc(b, c)

		// If a now has two edges to c, replace its degenerate If by Jump.
		if len(a.Succs) == 2 && a.Succs[0] == c && a.Succs[1] == c {
			jump := new(Jump)
			jump.SetBlock(a)
			a.Instrs[len(a.Instrs)-1] = jump
			a.Succs = a.Succs[:1]
			c.removePred(b)
		} else {
			if j == 0 {
				c.replacePred(b, a)
			} else {
				c.Preds = append(c.Preds, a)
			}
		}

		if debugBlockOpt {
			fmt.Fprintln(os.Stderr, "jumpThreading", a.Name, b.Name, c.Name)
		}
	}
	f.Blocks[i] = nil
	return true
}

// fuseBlocks attempts to apply the block fusion optimisation to block
// a, in which a->b becomes ab if len(a.Succs)==len(b.Preds)==1.
// The result is true if the optimisation was applied.
//
func fuseBlocks(f *Function, a *BasicBlock) bool {
	if len(a.Succs) != 1 {
		return false
	}
	b := a.Succs[0]
	if len(b.Preds) != 1 {
		return false
	}
	// Eliminate jump at end of A, then copy all of B across.
	a.Instrs = append(a.Instrs[:len(a.Instrs)-1], b.Instrs...)
	for _, instr := range b.Instrs {
		instr.SetBlock(a)
	}

	// A inherits B's successors
	a.Succs = append(a.succs2[:0], b.Succs...)

	// Fix up Preds links of all successors of B.
	for _, c := range b.Succs {
		c.replacePred(b, a)
	}

	if debugBlockOpt {
		fmt.Fprintln(os.Stderr, "fuseBlocks", a.Name, b.Name)
	}

	// Make b unreachable.  Subsequent pruning will reclaim it.
	b.Preds = nil
	return true
}

// optimizeBlocks() performs some simple block optimizations on a
// completed function: dead block elimination, block fusion, jump
// threading.
//
func optimizeBlocks(f *Function) {
	// Loop until no further progress.
	changed := true
	for changed {
		changed = false

		if debugBlockOpt {
			f.DumpTo(os.Stderr)
			MustSanityCheck(f, nil)
		}

		for i, b := range f.Blocks {
			// f.Blocks will temporarily contain nils to indicate
			// deleted blocks; we remove them at the end.
			if b == nil {
				continue
			}

			// Prune unreachable blocks (including all empty blocks).
			if prune(f, i, b) {
				changed = true
				continue // (b was pruned)
			}

			// Fuse blocks.  b->c becomes bc.
			if fuseBlocks(f, b) {
				changed = true
			}

			// a->b->c becomes a->c if b contains only a Jump.
			if jumpThreading(f, i, b) {
				changed = true
				continue // (b was disconnected)
			}
		}
	}

	// Eliminate nils from Blocks.
	j := 0
	for _, b := range f.Blocks {
		if b != nil {
			f.Blocks[j] = b
			j++
		}
	}
	// Nil out b.Blocks[j:] to aid GC.
	for i := j; i < len(f.Blocks); i++ {
		f.Blocks[i] = nil
	}
	f.Blocks = f.Blocks[:j]
}
