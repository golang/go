// asmcheck -O=3

package main

import (
	"fmt"
	"slices"
)

// M13 row #134 (SPEC Sec 1.5, T-105, T-302), Go twin of recursive_stack.lgo.
// At -O=3 every variable-size make and every append onto a non-escaping
// slice that lands in a function frame takes a 4096-byte stack buffer
// (K = 4096/sizeof(elem) elements). Helpers inlined into a recursive
// function add their buffers to that function's frame once per recursion
// level, so a 10,000-level recursion needs gigabytes of stack (the
// encoding/asn1 parseField overflow at 1 GB). Inside a recursive call-graph
// cycle the buffer therefore stays at the upstream 32 bytes, and a
// non-recursive function keeps the full tier buffer. The helpers call
// slices.Contains so that the inliner instantiates a generic function
// while the cycle marks of this package are live.

type node struct {
	kids []*node
	data []byte
}

func sumA(b []byte) int {
	buf := make([]byte, 0, len(b))
	for _, c := range b {
		buf = append(buf, c)
	}
	if slices.Contains(buf, 0) {
		return -1
	}
	return len(buf)
}

func sumB(b []byte) int {
	buf := make([]byte, 0, len(b)+1)
	for _, c := range b {
		buf = append(buf, c+1)
	}
	if slices.Contains(buf, 1) {
		return -1
	}
	return len(buf)
}

func sumC(b []byte) int {
	buf := make([]byte, 0, len(b)+2)
	for _, c := range b {
		buf = append(buf, c+2)
	}
	if slices.Contains(buf, 2) {
		return -1
	}
	return len(buf)
}

// own holds its own variable-size make; helpers are not involved.
//
// amd64:"TEXT .*, [$][0-9]{1,3}-"
// arm64:"TEXT .*, [$][0-9]{1,3}-"
func own(n *node) int {
	buf := make([]byte, 0, len(n.data)+8)
	buf = append(buf, n.data...)
	s := len(buf)
	for _, k := range n.kids {
		s += own(k)
	}
	return s
}

// total inlines three helpers (six buffers) into a recursive frame.
//
// amd64:"TEXT .*, [$][0-9]{1,3}-"
// arm64:"TEXT .*, [$][0-9]{1,3}-"
func total(n *node) int {
	if n == nil {
		return 0
	}
	s := sumA(n.data) + sumB(n.data) + sumC(n.data)
	for _, k := range n.kids {
		s += total(k)
	}
	return s
}

// ping and pong form a two-function cycle; the helper lands in one of them.
//
// amd64:"TEXT .*, [$][0-9]{1,3}-"
// arm64:"TEXT .*, [$][0-9]{1,3}-"
func ping(n *node) int {
	if n == nil {
		return 0
	}
	return sumA(n.data) + sumB(n.data) + pong(n.kids)
}

// amd64:"TEXT .*, [$][0-9]{1,3}-"
// arm64:"TEXT .*, [$][0-9]{1,3}-"
func pong(kids []*node) int {
	s := 0
	for _, k := range kids {
		s += ping(k) + sumC(k.data)
	}
	return s
}

// flat is not recursive, so it keeps the full tier buffers: the same three
// helpers take six 4096-byte buffers.
//
// amd64:"TEXT .*, [$][0-9]{5,}-"
// arm64:"TEXT .*, [$][0-9]{5,}-"
func flat(n *node) int {
	if n == nil {
		return 0
	}
	return sumA(n.data) + sumB(n.data) + sumC(n.data)
}

func main() {
	n := &node{data: []byte("abc"), kids: []*node{{data: []byte("de")}}}
	fmt.Println(total(n), flat(n), own(n), ping(n))
}
