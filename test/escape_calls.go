// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for function parameters.

// In this test almost everything is BAD except the simplest cases
// where input directly flows to output.

package foo

func f(buf []byte) []byte { // ERROR "leaking param: buf to result ~r0 level=0$"
	return buf
}

func g(*byte) string

func h(e int) {
	var x [32]byte // ERROR "moved to heap: x$"
	g(&f(x[:])[0])
}

type Node struct {
	s           string
	left, right *Node
}

func walk(np **Node) int { // ERROR "leaking param content: np"
	n := *np
	w := len(n.s)
	if n == nil {
		return 0
	}
	wl := walk(&n.left)
	wr := walk(&n.right)
	if wl < wr {
		n.left, n.right = n.right, n.left // ERROR "ignoring self-assignment"
		wl, wr = wr, wl
	}
	*np = n
	return w + wl + wr
}

// Test for bug where func var f used prototype's escape analysis results.
func prototype(xyz []string) {} // ERROR "xyz does not escape"
func bar() {
	var got [][]string
	f := prototype
	f = func(ss []string) { got = append(got, ss) } // ERROR "leaking param: ss" "func literal does not escape" "append escapes to heap"
	s := "string"
	f([]string{s}) // ERROR "\[\]string{...} escapes to heap"
}

func strmin(a, b, c string) string { // ERROR "leaking param: a to result ~r0 level=0" "leaking param: b to result ~r0 level=0" "leaking param: c to result ~r0 level=0"
	return min(a, b, c)
}
func strmax(a, b, c string) string { // ERROR "leaking param: a to result ~r0 level=0" "leaking param: b to result ~r0 level=0" "leaking param: c to result ~r0 level=0"
	return max(a, b, c)
}
