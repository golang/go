// asmcheck -O=3

package main

import (
	"fmt"
	"slices"
)

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
