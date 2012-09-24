// errorcheck -0 -m

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that the escape analysis is working.
// Compiles but does not run.  Inlining is enabled.

package foo

var p *int

func alloc(x int) *int { // ERROR "can inline alloc" "moved to heap: x"
	return &x // ERROR "&x escapes to heap"
}

var f func()

func f1() {
	p = alloc(2) // ERROR "inlining call to alloc" "&x escapes to heap" "moved to heap: x"

	// Escape analysis used to miss inlined code in closures.

	func() { // ERROR "func literal does not escape"
		p = alloc(3) // ERROR "inlining call to alloc" "&x escapes to heap" "moved to heap: x"
	}()

	f = func() { // ERROR "func literal escapes to heap"
		p = alloc(3) // ERROR "inlining call to alloc" "&x escapes to heap" "moved to heap: x"
	}
	f()
}

func f2() {} // ERROR "can inline f2"

// No inline for panic, recover.
func f3() { panic(1) }
func f4() { recover() }

func f5() *byte {
	type T struct {
		x [1]byte
	}
	t := new(T) // ERROR "new.T. escapes to heap"
	return &t.x[0] // ERROR "&t.x.0. escapes to heap"
}

func f6() *byte {
	type T struct {
		x struct {
			y byte
		}
	}
	t := new(T) // ERROR "new.T. escapes to heap"
	return &t.x.y // ERROR "&t.x.y escapes to heap"
}
