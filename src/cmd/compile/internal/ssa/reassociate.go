// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"sort"
)

// balanceExprTree repurposes all nodes and leafs into a
// balanced expression tree
func balanceExprTree(v *Value, visited map[*Value]bool, nodes, leafs []*Value) {
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

	// push all leafs which are constants as far off to the
	// right as possible to give the constant folder more opportunities
	sort.Slice(leafs, func(i, j int) bool {
		switch leafs[j].Op {
		case OpConst8, OpConst16, OpConst32, OpConst64:
			return false
		default:
			return true
		}
	})

	// build tree in reverse topological order
	for i := 0; i < len(nodes); i++ {
		if len(leafs) < 2 { // we need at least two leafs per node, balance went very wrong
			panic(fmt.Sprint("leafs needs to be >= 2, got", len(leafs)))
		}

		// Take two leaves out and attach them to a node,
		// use the node as a new leaf in the "next layer" of the tree
		nodes[i].AddArg2(leafs[0], leafs[1])
		leafs = append(leafs[2:], nodes[i])
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
// which cannot be rebalanced or else it won't fire rewrite rules
func probablyMemcombine(op Op, leafs []*Value) bool {
	if !isOr(op) {
		return false
	}

	lshCount := 0
	for _, l := range leafs {
		switch l.Op {
		case OpLsh8x8, OpLsh8x16, OpLsh8x32, OpLsh8x64,
			OpLsh16x8, OpLsh16x16, OpLsh16x32, OpLsh16x64,
			OpLsh32x8, OpLsh32x16, OpLsh32x32, OpLsh32x64,
			OpLsh64x8, OpLsh64x16, OpLsh64x32, OpLsh64x64:
			lshCount++
		}
	}

	return lshCount == len(leafs)-1
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

	// The smallest possible rebalanceable expression has 3 nodes and 4 leafs,
	// so preallocate the lists to save time if it is not rebalanceable
	leafs := make([]*Value, 0, 4)
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
					leafs = append(leafs, a)
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

	// we need at least 4 leafs for this expression to be rebalanceable,
	// and we can't balance a potential load widening (memcombine)
	if len(leafs) < 4 || probablyMemcombine(v.Op, leafs) {
		return
	}

	balanceExprTree(v, visited, nodes, leafs)
}

// reassociate balances trees of commutative computation
// to better group expressions for better constant folding,
// cse, etc.
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
