// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// This file defines algorithms related to dominance.

// Dominator tree construction ----------------------------------------
//
// We use the algorithm described in Lengauer & Tarjan. 1979.  A fast
// algorithm for finding dominators in a flowgraph.
// http://doi.acm.org/10.1145/357062.357071
//
// We also apply the optimizations to SLT described in Georgiadis et
// al, Finding Dominators in Practice, JGAA 2006,
// http://jgaa.info/accepted/2006/GeorgiadisTarjanWerneck2006.10.1.pdf
// to avoid the need for buckets of size > 1.

import (
	"bytes"
	"fmt"
	"math/big"
	"os"
	"sort"
)

// Idom returns the block that immediately dominates b:
// its parent in the dominator tree, if any.
// Neither the entry node (b.Index==0) nor recover node
// (b==b.Parent().Recover()) have a parent.
//
func (b *BasicBlock) Idom() *BasicBlock { return b.dom.idom }

// Dominees returns the list of blocks that b immediately dominates:
// its children in the dominator tree.
//
func (b *BasicBlock) Dominees() []*BasicBlock { return b.dom.children }

// Dominates reports whether b dominates c.
func (b *BasicBlock) Dominates(c *BasicBlock) bool {
	return b.dom.pre <= c.dom.pre && c.dom.post <= b.dom.post
}

type byDomPreorder []*BasicBlock

func (a byDomPreorder) Len() int           { return len(a) }
func (a byDomPreorder) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byDomPreorder) Less(i, j int) bool { return a[i].dom.pre < a[j].dom.pre }

// DomPreorder returns a new slice containing the blocks of f in
// dominator tree preorder.
//
func (f *Function) DomPreorder() []*BasicBlock {
	n := len(f.Blocks)
	order := make(byDomPreorder, n, n)
	copy(order, f.Blocks)
	sort.Sort(order)
	return order
}

// domInfo contains a BasicBlock's dominance information.
type domInfo struct {
	idom      *BasicBlock   // immediate dominator (parent in domtree)
	children  []*BasicBlock // nodes immediately dominated by this one
	pre, post int32         // pre- and post-order numbering within domtree
}

// ltState holds the working state for Lengauer-Tarjan algorithm
// (during which domInfo.pre is repurposed for CFG DFS preorder number).
type ltState struct {
	// Each slice is indexed by b.Index.
	sdom     []*BasicBlock // b's semidominator
	parent   []*BasicBlock // b's parent in DFS traversal of CFG
	ancestor []*BasicBlock // b's ancestor with least sdom
}

// dfs implements the depth-first search part of the LT algorithm.
func (lt *ltState) dfs(v *BasicBlock, i int32, preorder []*BasicBlock) int32 {
	preorder[i] = v
	v.dom.pre = i // For now: DFS preorder of spanning tree of CFG
	i++
	lt.sdom[v.Index] = v
	lt.link(nil, v)
	for _, w := range v.Succs {
		if lt.sdom[w.Index] == nil {
			lt.parent[w.Index] = v
			i = lt.dfs(w, i, preorder)
		}
	}
	return i
}

// eval implements the EVAL part of the LT algorithm.
func (lt *ltState) eval(v *BasicBlock) *BasicBlock {
	// TODO(adonovan): opt: do path compression per simple LT.
	u := v
	for ; lt.ancestor[v.Index] != nil; v = lt.ancestor[v.Index] {
		if lt.sdom[v.Index].dom.pre < lt.sdom[u.Index].dom.pre {
			u = v
		}
	}
	return u
}

// link implements the LINK part of the LT algorithm.
func (lt *ltState) link(v, w *BasicBlock) {
	lt.ancestor[w.Index] = v
}

// buildDomTree computes the dominator tree of f using the LT algorithm.
// Precondition: all blocks are reachable (e.g. optimizeBlocks has been run).
//
func buildDomTree(f *Function) {
	// The step numbers refer to the original LT paper; the
	// reordering is due to Georgiadis.

	// Clear any previous domInfo.
	for _, b := range f.Blocks {
		b.dom = domInfo{}
	}

	n := len(f.Blocks)
	// Allocate space for 5 contiguous [n]*BasicBlock arrays:
	// sdom, parent, ancestor, preorder, buckets.
	space := make([]*BasicBlock, 5*n, 5*n)
	lt := ltState{
		sdom:     space[0:n],
		parent:   space[n : 2*n],
		ancestor: space[2*n : 3*n],
	}

	// Step 1.  Number vertices by depth-first preorder.
	preorder := space[3*n : 4*n]
	root := f.Blocks[0]
	prenum := lt.dfs(root, 0, preorder)
	recover := f.Recover
	if recover != nil {
		lt.dfs(recover, prenum, preorder)
	}

	buckets := space[4*n : 5*n]
	copy(buckets, preorder)

	// In reverse preorder...
	for i := int32(n) - 1; i > 0; i-- {
		w := preorder[i]

		// Step 3. Implicitly define the immediate dominator of each node.
		for v := buckets[i]; v != w; v = buckets[v.dom.pre] {
			u := lt.eval(v)
			if lt.sdom[u.Index].dom.pre < i {
				v.dom.idom = u
			} else {
				v.dom.idom = w
			}
		}

		// Step 2. Compute the semidominators of all nodes.
		lt.sdom[w.Index] = lt.parent[w.Index]
		for _, v := range w.Preds {
			u := lt.eval(v)
			if lt.sdom[u.Index].dom.pre < lt.sdom[w.Index].dom.pre {
				lt.sdom[w.Index] = lt.sdom[u.Index]
			}
		}

		lt.link(lt.parent[w.Index], w)

		if lt.parent[w.Index] == lt.sdom[w.Index] {
			w.dom.idom = lt.parent[w.Index]
		} else {
			buckets[i] = buckets[lt.sdom[w.Index].dom.pre]
			buckets[lt.sdom[w.Index].dom.pre] = w
		}
	}

	// The final 'Step 3' is now outside the loop.
	for v := buckets[0]; v != root; v = buckets[v.dom.pre] {
		v.dom.idom = root
	}

	// Step 4. Explicitly define the immediate dominator of each
	// node, in preorder.
	for _, w := range preorder[1:] {
		if w == root || w == recover {
			w.dom.idom = nil
		} else {
			if w.dom.idom != lt.sdom[w.Index] {
				w.dom.idom = w.dom.idom.dom.idom
			}
			// Calculate Children relation as inverse of Idom.
			w.dom.idom.dom.children = append(w.dom.idom.dom.children, w)
		}
	}

	pre, post := numberDomTree(root, 0, 0)
	if recover != nil {
		numberDomTree(recover, pre, post)
	}

	// printDomTreeDot(os.Stderr, f)        // debugging
	// printDomTreeText(os.Stderr, root, 0) // debugging

	if f.Prog.mode&SanityCheckFunctions != 0 {
		sanityCheckDomTree(f)
	}
}

// numberDomTree sets the pre- and post-order numbers of a depth-first
// traversal of the dominator tree rooted at v.  These are used to
// answer dominance queries in constant time.
//
func numberDomTree(v *BasicBlock, pre, post int32) (int32, int32) {
	v.dom.pre = pre
	pre++
	for _, child := range v.dom.children {
		pre, post = numberDomTree(child, pre, post)
	}
	v.dom.post = post
	post++
	return pre, post
}

// Testing utilities ----------------------------------------

// sanityCheckDomTree checks the correctness of the dominator tree
// computed by the LT algorithm by comparing against the dominance
// relation computed by a naive Kildall-style forward dataflow
// analysis (Algorithm 10.16 from the "Dragon" book).
//
func sanityCheckDomTree(f *Function) {
	n := len(f.Blocks)

	// D[i] is the set of blocks that dominate f.Blocks[i],
	// represented as a bit-set of block indices.
	D := make([]big.Int, n)

	one := big.NewInt(1)

	// all is the set of all blocks; constant.
	var all big.Int
	all.Set(one).Lsh(&all, uint(n)).Sub(&all, one)

	// Initialization.
	for i, b := range f.Blocks {
		if i == 0 || b == f.Recover {
			// A root is dominated only by itself.
			D[i].SetBit(&D[0], 0, 1)
		} else {
			// All other blocks are (initially) dominated
			// by every block.
			D[i].Set(&all)
		}
	}

	// Iteration until fixed point.
	for changed := true; changed; {
		changed = false
		for i, b := range f.Blocks {
			if i == 0 || b == f.Recover {
				continue
			}
			// Compute intersection across predecessors.
			var x big.Int
			x.Set(&all)
			for _, pred := range b.Preds {
				x.And(&x, &D[pred.Index])
			}
			x.SetBit(&x, i, 1) // a block always dominates itself.
			if D[i].Cmp(&x) != 0 {
				D[i].Set(&x)
				changed = true
			}
		}
	}

	// Check the entire relation.  O(n^2).
	// The Recover block (if any) must be treated specially so we skip it.
	ok := true
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			b, c := f.Blocks[i], f.Blocks[j]
			if c == f.Recover {
				continue
			}
			actual := b.Dominates(c)
			expected := D[j].Bit(i) == 1
			if actual != expected {
				fmt.Fprintf(os.Stderr, "dominates(%s, %s)==%t, want %t\n", b, c, actual, expected)
				ok = false
			}
		}
	}

	preorder := f.DomPreorder()
	for _, b := range f.Blocks {
		if got := preorder[b.dom.pre]; got != b {
			fmt.Fprintf(os.Stderr, "preorder[%d]==%s, want %s\n", b.dom.pre, got, b)
			ok = false
		}
	}

	if !ok {
		panic("sanityCheckDomTree failed for " + f.String())
	}

}

// Printing functions ----------------------------------------

// printDomTree prints the dominator tree as text, using indentation.
func printDomTreeText(buf *bytes.Buffer, v *BasicBlock, indent int) {
	fmt.Fprintf(buf, "%*s%s\n", 4*indent, "", v)
	for _, child := range v.dom.children {
		printDomTreeText(buf, child, indent+1)
	}
}

// printDomTreeDot prints the dominator tree of f in AT&T GraphViz
// (.dot) format.
func printDomTreeDot(buf *bytes.Buffer, f *Function) {
	fmt.Fprintln(buf, "//", f)
	fmt.Fprintln(buf, "digraph domtree {")
	for i, b := range f.Blocks {
		v := b.dom
		fmt.Fprintf(buf, "\tn%d [label=\"%s (%d, %d)\",shape=\"rectangle\"];\n", v.pre, b, v.pre, v.post)
		// TODO(adonovan): improve appearance of edges
		// belonging to both dominator tree and CFG.

		// Dominator tree edge.
		if i != 0 {
			fmt.Fprintf(buf, "\tn%d -> n%d [style=\"solid\",weight=100];\n", v.idom.dom.pre, v.pre)
		}
		// CFG edges.
		for _, pred := range b.Preds {
			fmt.Fprintf(buf, "\tn%d -> n%d [style=\"dotted\",weight=0];\n", pred.dom.pre, v.pre)
		}
	}
	fmt.Fprintln(buf, "}")
}
