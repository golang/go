// errorcheck -0 -l -m=2

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure that -m=2's escape analysis diagnostics don't
// go into an infinite loop when handling negative dereference
// cycles. The critical thing being tested here is that compilation
// succeeds ("errorcheck -0"), not any particular diagnostic output,
// hence the very lax ERROR patterns below.

package p

type Node struct {
	Orig *Node
}

var sink *Node

func f1() {
	var n Node // ERROR "."
	n.Orig = &n

	m := n // ERROR "."
	sink = &m
}

func f2() {
	var n1, n2 Node // ERROR "."
	n1.Orig = &n2
	n2 = n1

	m := n2 // ERROR "."
	sink = &m
}

func f3() {
	var n1, n2 Node // ERROR "."
	n1.Orig = &n1
	n1.Orig = &n2

	sink = n1.Orig.Orig
}
