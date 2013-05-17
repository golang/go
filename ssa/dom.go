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
	"fmt"
	"io"
	"math/big"
	"os"
)

// domNode represents a node in the dominator tree.
//
// TODO(adonovan): export this, when ready.
type domNode struct {
	Block     *BasicBlock // the basic block; n.Block.dom == n
	Idom      *domNode    // immediate dominator (parent in dominator tree)
	Children  []*domNode  // nodes dominated by this one
	Level     int         // level number of node within tree; zero for root
	Pre, Post int         // pre- and post-order numbering within dominator tree

	// Working state for Lengauer-Tarjan algorithm
	// (during which Pre is repurposed for CFG DFS preorder number).
	// TODO(adonovan): opt: measure allocating these as temps.
	semi     *domNode // semidominator
	parent   *domNode // parent in DFS traversal of CFG
	ancestor *domNode // ancestor with least sdom
}

// ltDfs implements the depth-first search part of the LT algorithm.
func ltDfs(v *domNode, i int, preorder []*domNode) int {
	preorder[i] = v
	v.Pre = i // For now: DFS preorder of spanning tree of CFG
	i++
	v.semi = v
	v.ancestor = nil
	for _, succ := range v.Block.Succs {
		if w := succ.dom; w.semi == nil {
			w.parent = v
			i = ltDfs(w, i, preorder)
		}
	}
	return i
}

// ltEval implements the EVAL part of the LT algorithm.
func ltEval(v *domNode) *domNode {
	// TODO(adonovan): opt: do path compression per simple LT.
	u := v
	for ; v.ancestor != nil; v = v.ancestor {
		if v.semi.Pre < u.semi.Pre {
			u = v
		}
	}
	return u
}

// ltLink implements the LINK part of the LT algorithm.
func ltLink(v, w *domNode) {
	w.ancestor = v
}

// buildDomTree computes the dominator tree of f using the LT algorithm.
// Precondition: all blocks are reachable (e.g. optimizeBlocks has been run).
//
func buildDomTree(f *Function) {
	// The step numbers refer to the original LT paper; the
	// reodering is due to Georgiadis.

	// Initialize domNode nodes.
	for _, b := range f.Blocks {
		dom := b.dom
		if dom == nil {
			dom = &domNode{Block: b}
			b.dom = dom
		} else {
			dom.Block = b // reuse
		}
	}

	// Step 1.  Number vertices by depth-first preorder.
	n := len(f.Blocks)
	preorder := make([]*domNode, n)
	root := f.Blocks[0].dom
	ltDfs(root, 0, preorder)

	buckets := make([]*domNode, n)
	copy(buckets, preorder)

	// In reverse preorder...
	for i := n - 1; i > 0; i-- {
		w := preorder[i]

		// Step 3. Implicitly define the immediate dominator of each node.
		for v := buckets[i]; v != w; v = buckets[v.Pre] {
			u := ltEval(v)
			if u.semi.Pre < i {
				v.Idom = u
			} else {
				v.Idom = w
			}
		}

		// Step 2. Compute the semidominators of all nodes.
		w.semi = w.parent
		for _, pred := range w.Block.Preds {
			v := pred.dom
			u := ltEval(v)
			if u.semi.Pre < w.semi.Pre {
				w.semi = u.semi
			}
		}

		ltLink(w.parent, w)

		if w.parent == w.semi {
			w.Idom = w.parent
		} else {
			buckets[i] = buckets[w.semi.Pre]
			buckets[w.semi.Pre] = w
		}
	}

	// The final 'Step 3' is now outside the loop.
	for v := buckets[0]; v != root; v = buckets[v.Pre] {
		v.Idom = root
	}

	// Step 4. Explicitly define the immediate dominator of each
	// node, in preorder.
	for _, w := range preorder[1:] {
		if w == root {
			w.Idom = nil
		} else {
			if w.Idom != w.semi {
				w.Idom = w.Idom.Idom
			}
			// Calculate Children relation as inverse of Idom.
			w.Idom.Children = append(w.Idom.Children, w)
		}

		// Clear working state.
		w.semi = nil
		w.parent = nil
		w.ancestor = nil
	}

	numberDomTree(root, 0, 0, 0)

	// printDomTreeDot(os.Stderr, f)        // debugging
	// printDomTreeText(os.Stderr, root, 0) // debugging

	if f.Prog.mode&SanityCheckFunctions != 0 {
		sanityCheckDomTree(f)
	}
}

// numberDomTree sets the pre- and post-order numbers of a depth-first
// traversal of the dominator tree rooted at v.  These are used to
// answer dominance queries in constant time.  Also, it sets the level
// numbers (zero for the root) used for frontier computation.
//
func numberDomTree(v *domNode, pre, post, level int) (int, int) {
	v.Level = level
	level++
	v.Pre = pre
	pre++
	for _, child := range v.Children {
		pre, post = numberDomTree(child, pre, post, level)
	}
	v.Post = post
	post++
	return pre, post
}

// dominates returns true if b dominates c.
// Requires that dominance information is up-to-date.
//
func dominates(b, c *BasicBlock) bool {
	return b.dom.Pre <= c.dom.Pre && c.dom.Post <= b.dom.Post
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
	for i := range f.Blocks {
		if i == 0 {
			// The root is dominated only by itself.
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
			if i == 0 {
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
	ok := true
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			b, c := f.Blocks[i], f.Blocks[j]
			actual := dominates(b, c)
			expected := D[j].Bit(i) == 1
			if actual != expected {
				fmt.Fprintf(os.Stderr, "dominates(%s, %s)==%t, want %t\n", b, c, actual, expected)
				ok = false
			}
		}
	}
	if !ok {
		panic("sanityCheckDomTree failed for " + f.FullName())
	}
}

// Printing functions ----------------------------------------

// printDomTree prints the dominator tree as text, using indentation.
func printDomTreeText(w io.Writer, v *domNode, indent int) {
	fmt.Fprintf(w, "%*s%s\n", 4*indent, "", v.Block)
	for _, child := range v.Children {
		printDomTreeText(w, child, indent+1)
	}
}

// printDomTreeDot prints the dominator tree of f in AT&T GraphViz
// (.dot) format.
func printDomTreeDot(w io.Writer, f *Function) {
	fmt.Fprintln(w, "//", f.FullName())
	fmt.Fprintln(w, "digraph domtree {")
	for i, b := range f.Blocks {
		v := b.dom
		fmt.Fprintf(w, "\tn%d [label=\"%s (%d, %d)\",shape=\"rectangle\"];\n", v.Pre, b, v.Pre, v.Post)
		// TODO(adonovan): improve appearance of edges
		// belonging to both dominator tree and CFG.

		// Dominator tree edge.
		if i != 0 {
			fmt.Fprintf(w, "\tn%d -> n%d [style=\"solid\",weight=100];\n", v.Idom.Pre, v.Pre)
		}
		// CFG edges.
		for _, pred := range b.Preds {
			fmt.Fprintf(w, "\tn%d -> n%d [style=\"dotted\",weight=0];\n", pred.dom.Pre, v.Pre)
		}
	}
	fmt.Fprintln(w, "}")
}
