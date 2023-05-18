// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// ilp pass (Instruction Level Parallelism) balances trees of commutative computation
// to help CPU pipeline instructions more efficiently. It only works block by block
// so that it doesn't end up pulling loop invariant expressions into tight loops
func ilp(f *Func) {
	visited := f.newSparseSet(f.NumValues())

	for _, b := range f.Postorder() {
		for i := len(b.Values) - 1; i >= 0; i-- {
			val := b.Values[i]
			rebalance(val, visited)
		}
	}
}

// balanceExprTree repurposes all nodes and leaves into a well-balanced expression tree.
// It doesn't truly balance the tree in the sense of a BST, rather it
// prioritizes pairing up innermost (rightmost) expressions and their results and only
// pairing results of outermost (leftmost) expressions up with them when no other nice pairing exists
func balanceExprTree(nodes, leaves []*Value) {
	// reset all arguments of nodes to reuse them
	for _, n := range nodes {
		n.reset(n.Op)
	}

	// we bfs'ed through the nodes in reverse topological order
	// (expression dominated by all others to expression dominated by none of the others),
	// we want to rebuild the tree reverse topological order
	for i, j := 0, len(nodes)-1; i <= j; i, j = i+1, j-1 {
		nodes[i], nodes[j] = nodes[j], nodes[i]
	}

	// rebuild expression trees from the bottom up, prioritizing
	// right grouping.
	// if the number of leaves is not even, skip the first leaf
	// and add it to be paired up later
	i := 0
	subTrees := leaves
	for len(subTrees) != 1 {
		nextSubTrees := make([]*Value, 0, (len(subTrees)+1)/2)

		start := len(subTrees) % 2
		if start != 0 {
			nextSubTrees = append(nextSubTrees, subTrees[0])
		}

		for j := start; j < len(subTrees)-1; j += 2 {
			nodes[i].AddArgs(subTrees[j], subTrees[j+1])
			nextSubTrees = append(nextSubTrees, nodes[i])
			i++
		}

		subTrees = nextSubTrees
	}
}

// rebalance balances associative computation to better help CPU instruction pipelining (#49331)
//
// a + b + c + d compiles to to v1:(a + v2:(b + v3:(c + d)) which is an unbalanced expression tree
// Which is suboptimal since it requires the CPU to compute v3 before fetching it use its result in
// v2, and v2 before its use in v1
//
// This optimization rebalances this expression tree to look like (a + b) + (c + d) ,
// which removes such dependencies and frees up the CPU pipeline.
func rebalance(v *Value, visited *sparseSet) {
	// We cannot apply this optimization to non-commutative operations.
	// We also exclude 3+ arg ops because there are 0 opportunities in the std lib,
	// and the benefit for maintenance cost is not currently worth it.
	// Try and save time by not revisiting nodes
	if visited.contains(v.ID) || !opcodeTable[v.Op].commutative || len(v.Args) > 2{
		return
	}

	// The smallest possible rebalanceable binary expression has 3 nodes and 4 leaves,
	// so preallocate the lists to save time if it is not rebalanceable
	leaves := make([]*Value, 0, 4)
	nodes := make([]*Value, 0, 3)

	// Do a bfs on v to keep a nice reverse topological order
	haystack := []*Value{v}
	for i := 0; i < len(haystack); i++ {
		needle := haystack[i]
		// if we are searching a value, it must be a node so add it to our node list
		nodes = append(nodes, needle)

		// Only visit nodes. Leafs may be rebalancable for a different op type
		visited.add(v.ID)

		for _, a := range needle.Args {
			// If the ops aren't the same, have more than one use, or not in the same BB it must be a leaf.
			if a.Op != v.Op || a.Uses != 1 || a.Block != v.Block  {
				leaves = append(leaves, a)
				continue
			}

			// nodes in the tree now hold the invariants that:
			// - they are of a common associative operation as the rest of the tree
			// - they have only a single use
			// - they are in the same basic block
			haystack = append(haystack, a)
		}
	}

	// we need at least args^2 leaves for this expression to be rebalanceable,
	if len(leaves) < 4 {
		return
	}

	balanceExprTree(nodes, leaves)
}
