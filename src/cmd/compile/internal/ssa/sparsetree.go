// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"strings"
)

type SparseTreeNode struct {
	child   *Block
	sibling *Block
	parent  *Block

	// Every block has 6 numbers associated with it:
	// entry-1, entry, entry+1, exit-1, and exit, exit+1.
	// entry and exit are conceptually the top of the block (phi functions)
	// entry+1 and exit-1 are conceptually the bottom of the block (ordinary defs)
	// entry-1 and exit+1 are conceptually "just before" the block (conditions flowing in)
	//
	// This simplifies life if we wish to query information about x
	// when x is both an input to and output of a block.
	entry, exit int32
}

func (s *SparseTreeNode) String() string {
	return fmt.Sprintf("[%d,%d]", s.entry, s.exit)
}

func (s *SparseTreeNode) Entry() int32 {
	return s.entry
}

func (s *SparseTreeNode) Exit() int32 {
	return s.exit
}

const (
	// When used to lookup up definitions in a sparse tree,
	// these adjustments to a block's entry (+adjust) and
	// exit (-adjust) numbers allow a distinction to be made
	// between assignments (typically branch-dependent
	// conditionals) occurring "before" the block (e.g., as inputs
	// to the block and its phi functions), "within" the block,
	// and "after" the block.
	AdjustBefore = -1 // defined before phi
	AdjustWithin = 0  // defined by phi
	AdjustAfter  = 1  // defined within block
)

// A SparseTree is a tree of Blocks.
// It allows rapid ancestor queries,
// such as whether one block dominates another.
type SparseTree []SparseTreeNode

// newSparseTree creates a SparseTree from a block-to-parent map (array indexed by Block.ID)
func newSparseTree(f *Func, parentOf []*Block) SparseTree {
	t := make(SparseTree, f.NumBlocks())
	for _, b := range f.Blocks {
		n := &t[b.ID]
		if p := parentOf[b.ID]; p != nil {
			n.parent = p
			n.sibling = t[p.ID].child
			t[p.ID].child = b
		}
	}
	t.numberBlock(f.Entry, 1)
	return t
}

// newSparseOrderedTree creates a SparseTree from a block-to-parent map (array indexed by Block.ID)
// children will appear in the reverse of their order in reverseOrder
// in particular, if reverseOrder is a dfs-reversePostOrder, then the root-to-children
// walk of the tree will yield a pre-order.
func newSparseOrderedTree(f *Func, parentOf, reverseOrder []*Block) SparseTree {
	t := make(SparseTree, f.NumBlocks())
	for _, b := range reverseOrder {
		n := &t[b.ID]
		if p := parentOf[b.ID]; p != nil {
			n.parent = p
			n.sibling = t[p.ID].child
			t[p.ID].child = b
		}
	}
	t.numberBlock(f.Entry, 1)
	return t
}

// treestructure provides a string description of the dominator
// tree and flow structure of block b and all blocks that it
// dominates.
func (t SparseTree) treestructure(b *Block) string {
	return t.treestructure1(b, 0)
}
func (t SparseTree) treestructure1(b *Block, i int) string {
	s := "\n" + strings.Repeat("\t", i) + b.String() + "->["
	for i, e := range b.Succs {
		if i > 0 {
			s = s + ","
		}
		s = s + e.b.String()
	}
	s += "]"
	if c0 := t[b.ID].child; c0 != nil {
		s += "("
		for c := c0; c != nil; c = t[c.ID].sibling {
			if c != c0 {
				s += " "
			}
			s += t.treestructure1(c, i+1)
		}
		s += ")"
	}
	return s
}

// numberBlock assigns entry and exit numbers for b and b's
// children in an in-order walk from a gappy sequence, where n
// is the first number not yet assigned or reserved. N should
// be larger than zero. For each entry and exit number, the
// values one larger and smaller are reserved to indicate
// "strictly above" and "strictly below". numberBlock returns
// the smallest number not yet assigned or reserved (i.e., the
// exit number of the last block visited, plus two, because
// last.exit+1 is a reserved value.)
//
// examples:
//
// single node tree Root, call with n=1
//         entry=2 Root exit=5; returns 7
//
// two node tree, Root->Child, call with n=1
//         entry=2 Root exit=11; returns 13
//         entry=5 Child exit=8
//
// three node tree, Root->(Left, Right), call with n=1
//         entry=2 Root exit=17; returns 19
// entry=5 Left exit=8;  entry=11 Right exit=14
//
// This is the in-order sequence of assigned and reserved numbers
// for the last example:
//   root     left     left      right       right       root
//  1 2e 3 | 4 5e 6 | 7 8x 9 | 10 11e 12 | 13 14x 15 | 16 17x 18

func (t SparseTree) numberBlock(b *Block, n int32) int32 {
	// reserve n for entry-1, assign n+1 to entry
	n++
	t[b.ID].entry = n
	// reserve n+1 for entry+1, n+2 is next free number
	n += 2
	for c := t[b.ID].child; c != nil; c = t[c.ID].sibling {
		n = t.numberBlock(c, n) // preserves n = next free number
	}
	// reserve n for exit-1, assign n+1 to exit
	n++
	t[b.ID].exit = n
	// reserve n+1 for exit+1, n+2 is next free number, returned.
	return n + 2
}

// Sibling returns a sibling of x in the dominator tree (i.e.,
// a node with the same immediate dominator) or nil if there
// are no remaining siblings in the arbitrary but repeatable
// order chosen. Because the Child-Sibling order is used
// to assign entry and exit numbers in the treewalk, those
// numbers are also consistent with this order (i.e.,
// Sibling(x) has entry number larger than x's exit number).
func (t SparseTree) Sibling(x *Block) *Block {
	return t[x.ID].sibling
}

// Child returns a child of x in the dominator tree, or
// nil if there are none. The choice of first child is
// arbitrary but repeatable.
func (t SparseTree) Child(x *Block) *Block {
	return t[x.ID].child
}

// isAncestorEq reports whether x is an ancestor of or equal to y.
func (t SparseTree) isAncestorEq(x, y *Block) bool {
	if x == y {
		return true
	}
	xx := &t[x.ID]
	yy := &t[y.ID]
	return xx.entry <= yy.entry && yy.exit <= xx.exit
}

// isAncestor reports whether x is a strict ancestor of y.
func (t SparseTree) isAncestor(x, y *Block) bool {
	if x == y {
		return false
	}
	xx := &t[x.ID]
	yy := &t[y.ID]
	return xx.entry < yy.entry && yy.exit < xx.exit
}

// domorder returns a value for dominator-oriented sorting.
// Block domination does not provide a total ordering,
// but domorder two has useful properties.
// (1) If domorder(x) > domorder(y) then x does not dominate y.
// (2) If domorder(x) < domorder(y) and domorder(y) < domorder(z) and x does not dominate y,
//     then x does not dominate z.
// Property (1) means that blocks sorted by domorder always have a maximal dominant block first.
// Property (2) allows searches for dominated blocks to exit early.
func (t SparseTree) domorder(x *Block) int32 {
	// Here is an argument that entry(x) provides the properties documented above.
	//
	// Entry and exit values are assigned in a depth-first dominator tree walk.
	// For all blocks x and y, one of the following holds:
	//
	// (x-dom-y) x dominates y => entry(x) < entry(y) < exit(y) < exit(x)
	// (y-dom-x) y dominates x => entry(y) < entry(x) < exit(x) < exit(y)
	// (x-then-y) neither x nor y dominates the other and x walked before y => entry(x) < exit(x) < entry(y) < exit(y)
	// (y-then-x) neither x nor y dominates the other and y walked before y => entry(y) < exit(y) < entry(x) < exit(x)
	//
	// entry(x) > entry(y) eliminates case x-dom-y. This provides property (1) above.
	//
	// For property (2), assume entry(x) < entry(y) and entry(y) < entry(z) and x does not dominate y.
	// entry(x) < entry(y) allows cases x-dom-y and x-then-y.
	// But by supposition, x does not dominate y. So we have x-then-y.
	//
	// For contractidion, assume x dominates z.
	// Then entry(x) < entry(z) < exit(z) < exit(x).
	// But we know x-then-y, so entry(x) < exit(x) < entry(y) < exit(y).
	// Combining those, entry(x) < entry(z) < exit(z) < exit(x) < entry(y) < exit(y).
	// By supposition, entry(y) < entry(z), which allows cases y-dom-z and y-then-z.
	// y-dom-z requires entry(y) < entry(z), but we have entry(z) < entry(y).
	// y-then-z requires exit(y) < entry(z), but we have entry(z) < exit(y).
	// We have a contradiction, so x does not dominate z, as required.
	return t[x.ID].entry
}
