// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 26495: gccgo produces incorrect order of evaluation
// for expressions involving &&, || subexpressions.

package main

var i int

func checkorder(order int) {
	if i != order {
		panic("FAIL: wrong evaluation order")
	}
	i++
}

func A() bool              { checkorder(1); return true }
func B() bool              { checkorder(2); return true }
func C() bool              { checkorder(5); return false }
func D() bool              { panic("FAIL: D should not be called") }
func E() int               { checkorder(3); return 0 }
func F() int               { checkorder(0); return 0 }
func G(bool) int           { checkorder(9); return 0 }
func H(int, bool, int) int { checkorder(7); return 0 }
func I(int) bool           { checkorder(8); return true }
func J() int               { checkorder(4); return 0 }
func K() int               { checkorder(6); return 0 }
func L() int               { checkorder(10); return 0 }

func main() {
	_ = F() + G(A() && B() && I(E()+H(J(), C() && D(), K()))) + L()
}
