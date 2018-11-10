// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for function parameters.

// In this test almost everything is BAD except the simplest cases
// where input directly flows to output.

package foo

func f(buf []byte) []byte { // ERROR "leaking param: buf to result ~r1 level=0$"
	return buf
}

func g(*byte) string

func h(e int) {
	var x [32]byte // ERROR "moved to heap: x$"
	g(&f(x[:])[0]) // ERROR "&f\(x\[:\]\)\[0\] escapes to heap$" "x escapes to heap$"
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
	wl := walk(&n.left)  // ERROR "walk &n.left does not escape"
	wr := walk(&n.right) // ERROR "walk &n.right does not escape"
	if wl < wr {
		n.left, n.right = n.right, n.left
		wl, wr = wr, wl
	}
	*np = n
	return w + wl + wr
}

// Test for bug where func var f used prototype's escape analysis results.
func prototype(xyz []string) {} // ERROR "prototype xyz does not escape"
func bar() {
	var got [][]string
	f := prototype
	f = func(ss []string) { got = append(got, ss) } // ERROR "leaking param: ss" "func literal does not escape"
	s := "string"
	f([]string{s}) // ERROR "\[\]string literal escapes to heap"
}
