// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// ilp pass (Instruction Level Parallelism) balances trees of commutative computation
// to help CPU pipeline instructions more efficiently. It only works block by block
// so that it doesn't end up pulling loop invariant expressions into tight loops
func ilp(f *Func) {
	for _, b := range f.Blocks {
		for _, v := range findRoots(b) {
			rebalance(v)
		}
	}
}

// isILPOp only returns true if the operation is commutative
// and associative, which for our case would be only commutative integer ops
func isILPOp(o Op) bool {
	// if the op isn't commutative it won't be useable for ilp
	if !opcodeTable[o].commutative {
		return false
	}

	// filter out float ops because they are not associative
	switch o {
	case OpAdd32F, OpAdd64F, OpMul32F, OpMul64F:
		return false
	default:
		return true
	}
}

// findRoots looks for the root of a rebalanceable expressions.
// It does this by building a poor man's def-use chain, counting up the uses of
// a given expression as an argument within the block. 
// Any roots of rebalanceable expressions should have a count of zero meaning
// they aren't used as arguments in rebalanceable expressions.
func findRoots(b *Block) []*Value {
	uses := make(map[*Value]int)
	candidates := make([]*Value, 0)
	roots := make([]*Value, 0)
	
	for _, v := range b.Values {
		if !isILPOp(v.Op) {
			continue
		}
		
		// could be a possible root, add it as a candidate to remove later
		candidates = append(candidates, v)
		
		// mark the arguments of the expression as being used
		// if they are also a rebalanceable op (making them a non-root node)
		if isILPOp(v.Args[0].Op) && v.Op == v.Args[0].Op {
			uses[v.Args[0]]++
		}

		if isILPOp(v.Args[1].Op) && v.Op == v.Args[1].Op {
			uses[v.Args[1]]++
		}
	}
	
	for _, c := range candidates {
		if uses[c] == 0 {
			roots = append(roots, c)
		}
	}
	
	return roots
}

// balanceExprTree repurposes all nodes and leaves into a well-balanced expression tree.
// It doesn't truly balance the tree in the sense of a BST, rather it
// prioritizes pairing up innermost (rightmost) expressions and their results and only
// pairing results of outermost (leftmost) expressions up with them when no other nice pairing exists
// TODO(ryan-berger): implement Huffman Tree-Height Reduction instead?
func balanceExprTree(nodes, leaves []*Value) {
	// reset all arguments of nodes to reuse them
	for _, n := range nodes {
		n.reset(n.Op)
	}

	// rebuild expression trees from the bottom up, prioritizing right grouping.
	// if the number of leaves is not even, skip the first leaf
	// and add it to be paired up later
	subTrees := leaves
	i := len(nodes)-1
	for len(subTrees) != 1 {
		nextSubTrees := make([]*Value, 0, (len(subTrees)+1)/2)

		start := len(subTrees) % 2
		if start != 0 {
			nextSubTrees = append(nextSubTrees, subTrees[0])
		}

		// pair leaves using the last nodes and move towards nodes[0] which is the root
		for j := start; j < len(subTrees)-1; j += 2 {
			nodes[i].AddArgs(subTrees[j], subTrees[j+1])
			nextSubTrees = append(nextSubTrees, nodes[i])
			i--
		}

		subTrees = nextSubTrees
	}
}

// rebalance balances associative computation to better help CPU instruction pipelining (#49331)
//
// a + b + c + d compiles to v1:(a + v2:(b + v3:(c + d)) which is an unbalanced expression tree.
// It is suboptimal since it requires the CPU to compute v3 before fetching it use its result in
// v2, and v2 before its use in v1
//
// This optimization rebalances this expression tree to look like v1:(v2:(a + b) + v3:(c + d)),
// which removes such dependencies and frees up the CPU pipeline.
func rebalance(v *Value) {
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

	// we need at least 3 nodes (root, two children) and len(args)^2 leaves
	// for this expression to be rebalanceable
	if len(nodes) < 3 || len(leaves) < 4 {
		return
	}

	balanceExprTree(nodes, leaves)
}
