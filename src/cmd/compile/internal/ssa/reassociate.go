// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// balanceExprTree repurposes all nodes and leaves into a well-balanced expression tree.
// It doesn't truly balance the tree in the sense of a BST, rather it 
// prioritizes pairing up innermost (rightmost) expressions and their results and only
// pairing results of outermost (leftmost) expressions up with them when no other nice pairing exists 
func balanceExprTree(v *Value, visited map[*Value]bool, nodes, leaves []*Value) {
	// reset all arguments of nodes to help rebalancing
	for i, n := range nodes {
		n.reset(n.Op)

		// sometimes nodes in the tree are in different blocks
		// so pull them in into a common block (v's block)
		// to make sure nodes don't end up dominating their leaves
		if v.Block != n.Block {
			copied := n.copyInto(v.Block)
			n.Op = OpInvalid
			visited[n] = true // "revisit" the copied node
			nodes[i] = copied
		}
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
		
		start := len(subTrees)%2
		if start != 0 {
			nextSubTrees = append(nextSubTrees, subTrees[0])
		}
		
		for j := start; j < len(subTrees)-1; j+=2 {
			nodes[i].AddArg2(subTrees[j], subTrees[j+1])
			nextSubTrees = append(nextSubTrees, nodes[i])
			i++
		}
		
		subTrees = nextSubTrees
	}
}

func isOr(op Op) bool {
	switch op {
	case OpOr8, OpOr16, OpOr32, OpOr64:
		return true
	default:
		return false
	}
}

// probablyMemcombine helps find a pattern of leaves that form
// a load that can be widened which looks like:
//
//	(l | l << 8 | l << 18 | l << 24)
//
// which cannot be rebalanced or else it won't fire load widening rewrite rules
func probablyMemcombine(op Op, leaves []*Value) bool {
	if !isOr(op) {
		return false
	}

	lshCount := 0
	for _, l := range leaves {
		switch l.Op {
		case OpLsh8x8, OpLsh8x16, OpLsh8x32, OpLsh8x64,
			OpLsh16x8, OpLsh16x16, OpLsh16x32, OpLsh16x64,
			OpLsh32x8, OpLsh32x16, OpLsh32x32, OpLsh32x64,
			OpLsh64x8, OpLsh64x16, OpLsh64x32, OpLsh64x64:
			lshCount++
		}
	}

	// there are a few algorithms in the std lib expressed as two 32 bit loads
	// which can get turned into a 64 bit load
	// conservatively estimate that if there are more shifts than not then it is
	// some sort of load waiting to be widened
	return lshCount > len(leaves)/2
}

// rebalance balances associative computation to better help CPU instruction pipelining (#49331)
// and groups constants together catch more constant folding opportunities.
//
// a + b + c + d compiles to to (a + (b + (c + d)) which is an unbalanced expression tree
// Which is suboptimal since it requires the CPU to compute v3 before fetching it use its result in
// v2, and v2 before its use in v1
//
// This optimization rebalances this expression tree to look like (a + b) + (c + d) ,
// which removes such dependencies and frees up the CPU pipeline.
//
// The above optimization is a good starting point for other sorts of operations such as
// turning a + a + a => 3*a, cancelling pairs a + (-a), collecting up common factors TODO(ryan-berger)
func rebalance(v *Value, visited map[*Value]bool) {
	// We cannot apply this optimization to non-commutative operations,
	// values that have more than one use, or non-binary ops (would need more log math).
	// Try and save time by not revisiting nodes
	if visited[v] || !(v.Uses == 1 && opcodeTable[v.Op].commutative) || len(v.Args) > 2 {
		return
	}

	// The smallest possible rebalanceable expression has 3 nodes and 4 leaves,
	// so preallocate the lists to save time if it is not rebalanceable
	leaves := make([]*Value, 0, 4)
	nodes := make([]*Value, 0, 3)

	// Do a bfs on v to keep a nice reverse topological order
	haystack := []*Value{v}
	for len(haystack) != 0 {
		nextHaystack := make([]*Value, 0, len(v.Args)*len(haystack))
		for _, needle := range haystack {
			// if we are searching a value, it must be a node so add it to our node list
			nodes = append(nodes, needle)

			// Only visit nodes. Leafs may be rebalancable for a different op type
			visited[needle] = true

			for _, a := range needle.Args {
				// If the ops aren't the same or have more than one use it must be a leaf.
				if a.Op != v.Op || a.Uses != 1 {
					leaves = append(leaves, a)
					continue
				}

				// nodes in the tree now hold the invariants that:
				// - they are of a common associative operation as the rest of the tree
				// - they have only a single use (this invariant could be removed with further analysis TODO(ryan-berger)
				nextHaystack = append(nextHaystack, a)
			}
		}
		haystack = nextHaystack
	}

	// we need at least 4 leaves for this expression to be rebalanceable,
	// and we can't balance a potential load widening (see memcombine)
	if len(leaves) < 4 || probablyMemcombine(v.Op, leaves) {
		return
	}

	balanceExprTree(v, visited, nodes, leaves)
}

// reassociate balances trees of commutative computation
// to better group expressions to expose easy optimizations in
// cse, cancelling/counting/factoring expressions, etc.
func reassociate(f *Func) {
	visited := make(map[*Value]bool)

	for _, b := range f.Postorder() {
		for i := len(b.Values) - 1; i >= 0; i-- {
			val := b.Values[i]
			rebalance(val, visited)
		}
	}

	for k := range visited {
		delete(visited, k)
	}
}
