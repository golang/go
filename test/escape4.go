// errorcheck -0 -m

//go:build !goexperiment.newinliner

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that the escape analysis is working.
// Compiles but does not run.  Inlining is enabled.

package foo

var p *int

func alloc(x int) *int { // ERROR "can inline alloc" "moved to heap: x"
	return &x
}

var f func()

func f1() {
	p = alloc(2) // ERROR "inlining call to alloc" "moved to heap: x"

	// Escape analysis used to miss inlined code in closures.

	func() { // ERROR "can inline f1.func1"
		p = alloc(3) // ERROR "inlining call to alloc"
	}() // ERROR "inlining call to f1.func1" "inlining call to alloc" "moved to heap: x"

	f = func() { // ERROR "func literal escapes to heap" "can inline f1.func2"
		p = alloc(3) // ERROR "inlining call to alloc" "moved to heap: x"
	}
	f()
}

func f2() {} // ERROR "can inline f2"

// No inline for recover; panic now allowed to inline.
func f3() { panic(1) } // ERROR "can inline f3" "1 escapes to heap"
func f4() { recover() }

func f5() *byte { // ERROR "can inline f5"
	type T struct {
		x [1]byte
	}
	t := new(T) // ERROR "new.T. escapes to heap"
	return &t.x[0]
}

func f6() *byte { // ERROR "can inline f6"
	type T struct {
		x struct {
			y byte
		}
	}
	t := new(T) // ERROR "new.T. escapes to heap"
	return &t.x.y
}
