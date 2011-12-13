/*
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
    notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

    * Neither the name of "The Computer Language Benchmarks Game" nor the
    name of "The Computer Language Shootout Benchmarks" nor the names of
    its contributors may be used to endorse or promote products derived
    from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
*/

/* The Computer Language Benchmarks Game
 * http://shootout.alioth.debian.org/
 *
 * contributed by The Go Authors.
 * based on C program by Kevin Carson
 */

package main

import (
	"flag"
	"fmt"
)

var n = flag.Int("n", 15, "depth")

type Node struct {
	item        int
	left, right *Node
}

type Arena struct {
	head *Node
}

var arena Arena

func (n *Node) free() {
	if n.left != nil {
		n.left.free()
	}
	if n.right != nil {
		n.right.free()
	}
	n.left = arena.head
	arena.head = n
}

func (a *Arena) New(item int, left, right *Node) *Node {
	if a.head == nil {
		nodes := make([]Node, 3<<uint(*n))
		for i := 0; i < len(nodes)-1; i++ {
			nodes[i].left = &nodes[i+1]
		}
		a.head = &nodes[0]
	}
	n := a.head
	a.head = a.head.left
	n.item = item
	n.left = left
	n.right = right
	return n
}

func bottomUpTree(item, depth int) *Node {
	if depth <= 0 {
		return arena.New(item, nil, nil)
	}
	return arena.New(item, bottomUpTree(2*item-1, depth-1), bottomUpTree(2*item, depth-1))
}

func (n *Node) itemCheck() int {
	if n.left == nil {
		return n.item
	}
	return n.item + n.left.itemCheck() - n.right.itemCheck()
}

const minDepth = 4

func main() {
	flag.Parse()

	maxDepth := *n
	if minDepth+2 > *n {
		maxDepth = minDepth + 2
	}
	stretchDepth := maxDepth + 1

	check := bottomUpTree(0, stretchDepth).itemCheck()
	fmt.Printf("stretch tree of depth %d\t check: %d\n", stretchDepth, check)

	longLivedTree := bottomUpTree(0, maxDepth)

	for depth := minDepth; depth <= maxDepth; depth += 2 {
		iterations := 1 << uint(maxDepth-depth+minDepth)
		check = 0

		for i := 1; i <= iterations; i++ {
			t := bottomUpTree(i, depth)
			check += t.itemCheck()
			t.free()
			t = bottomUpTree(-i, depth)
			check += t.itemCheck()
			t.free()
		}
		fmt.Printf("%d\t trees of depth %d\t check: %d\n", iterations*2, depth, check)
	}
	fmt.Printf("long lived tree of depth %d\t check: %d\n", maxDepth, longLivedTree.itemCheck())
}
