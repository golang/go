// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This benchmark, taken from the shootout, tests garbage collector
// performance by generating and discarding large binary trees.

package go1

import "testing"

type binaryNode struct {
	item        int
	left, right *binaryNode
}

func bottomUpTree(item, depth int) *binaryNode {
	if depth <= 0 {
		return &binaryNode{item: item}
	}
	return &binaryNode{item, bottomUpTree(2*item-1, depth-1), bottomUpTree(2*item, depth-1)}
}

func (n *binaryNode) itemCheck() int {
	if n.left == nil {
		return n.item
	}
	return n.item + n.left.itemCheck() - n.right.itemCheck()
}

const minDepth = 4

func binarytree(n int) {
	maxDepth := n
	if minDepth+2 > n {
		maxDepth = minDepth + 2
	}
	stretchDepth := maxDepth + 1

	check := bottomUpTree(0, stretchDepth).itemCheck()
	//fmt.Printf("stretch tree of depth %d\t check: %d\n", stretchDepth, check)

	longLivedTree := bottomUpTree(0, maxDepth)

	for depth := minDepth; depth <= maxDepth; depth += 2 {
		iterations := 1 << uint(maxDepth-depth+minDepth)
		check = 0

		for i := 1; i <= iterations; i++ {
			check += bottomUpTree(i, depth).itemCheck()
			check += bottomUpTree(-i, depth).itemCheck()
		}
		//fmt.Printf("%d\t trees of depth %d\t check: %d\n", iterations*2, depth, check)
	}
	longLivedTree.itemCheck()
	//fmt.Printf("long lived tree of depth %d\t check: %d\n", maxDepth, longLivedTree.itemCheck())
}

func BenchmarkBinaryTree17(b *testing.B) {
	for i := 0; i < b.N; i++ {
		binarytree(17)
	}
}
