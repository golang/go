// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for closure arguments.

package escape

var sink interface{}

func ClosureCallArgs0() {
	x := 0
	func(p *int) { // ERROR "p does not escape" "func literal does not escape"
		*p = 1
	}(&x)
}

func ClosureCallArgs1() {
	x := 0
	for {
		func(p *int) { // ERROR "p does not escape" "func literal does not escape"
			*p = 1
		}(&x)
	}
}

func ClosureCallArgs2() {
	for {
		x := 0
		func(p *int) { // ERROR "p does not escape" "func literal does not escape"
			*p = 1
		}(&x)
	}
}

func ClosureCallArgs3() {
	x := 0         // ERROR "moved to heap: x"
	func(p *int) { // ERROR "leaking param: p" "func literal does not escape"
		sink = p
	}(&x)
}

func ClosureCallArgs4() {
	x := 0
	_ = func(p *int) *int { // ERROR "leaking param: p to result ~r0" "func literal does not escape"
		return p
	}(&x)
}

func ClosureCallArgs5() {
	x := 0 // ERROR "moved to heap: x"
	// TODO(mdempsky): We get "leaking param: p" here because the new escape analysis pass
	// can tell that p flows directly to sink, but it's a little weird. Re-evaluate.
	sink = func(p *int) *int { // ERROR "leaking param: p" "func literal does not escape"
		return p
	}(&x)
}

func ClosureCallArgs6() {
	x := 0         // ERROR "moved to heap: x"
	func(p *int) { // ERROR "moved to heap: p" "func literal does not escape"
		sink = &p
	}(&x)
}

func ClosureCallArgs7() {
	var pp *int
	for {
		x := 0         // ERROR "moved to heap: x"
		func(p *int) { // ERROR "leaking param: p" "func literal does not escape"
			pp = p
		}(&x)
	}
	_ = pp
}

func ClosureCallArgs8() {
	x := 0
	defer func(p *int) { // ERROR "p does not escape" "func literal does not escape"
		*p = 1
	}(&x)
}

func ClosureCallArgs9() {
	// BAD: x should not leak
	x := 0 // ERROR "moved to heap: x"
	for {
		defer func(p *int) { // ERROR "func literal escapes to heap" "p does not escape"
			*p = 1
		}(&x)
	}
}

func ClosureCallArgs10() {
	for {
		x := 0               // ERROR "moved to heap: x"
		defer func(p *int) { // ERROR "func literal escapes to heap" "p does not escape"
			*p = 1
		}(&x)
	}
}

func ClosureCallArgs11() {
	x := 0               // ERROR "moved to heap: x"
	defer func(p *int) { // ERROR "leaking param: p" "func literal does not escape"
		sink = p
	}(&x)
}

func ClosureCallArgs12() {
	x := 0
	defer func(p *int) *int { // ERROR "leaking param: p to result ~r0" "func literal does not escape"
		return p
	}(&x)
}

func ClosureCallArgs13() {
	x := 0               // ERROR "moved to heap: x"
	defer func(p *int) { // ERROR "moved to heap: p" "func literal does not escape"
		sink = &p
	}(&x)
}

func ClosureCallArgs14() {
	x := 0
	p := &x
	_ = func(p **int) *int { // ERROR "leaking param: p to result ~r0 level=1" "func literal does not escape"
		return *p
	}(&p)
}

func ClosureCallArgs15() {
	x := 0 // ERROR "moved to heap: x"
	p := &x
	sink = func(p **int) *int { // ERROR "leaking param: p to result ~r0 level=1" "func literal does not escape"
		return *p
	}(&p)
}

func ClosureLeak1(s string) string { // ERROR "s does not escape"
	t := s + "YYYY"         // ERROR "escapes to heap"
	return ClosureLeak1a(t) // ERROR "... argument does not escape"
}

// See #14409 -- returning part of captured var leaks it.
func ClosureLeak1a(a ...string) string { // ERROR "leaking param: a to result ~r0 level=1$"
	return func() string { // ERROR "func literal does not escape"
		return a[0]
	}()
}

func ClosureLeak2(s string) string { // ERROR "s does not escape"
	t := s + "YYYY"       // ERROR "escapes to heap"
	c := ClosureLeak2a(t) // ERROR "... argument does not escape"
	return c
}
func ClosureLeak2a(a ...string) string { // ERROR "leaking param content: a"
	return ClosureLeak2b(func() string { // ERROR "func literal does not escape"
		return a[0]
	})
}
func ClosureLeak2b(f func() string) string { // ERROR "f does not escape"
	return f()
}

func ClosureIndirect() {
	f := func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	f(new(int))          // ERROR "new\(int\) does not escape"

	g := f
	g(new(int)) // ERROR "new\(int\) does not escape"

	h := nopFunc
	h(new(int)) // ERROR "new\(int\) does not escape"
}

func nopFunc(p *int) {} // ERROR "p does not escape"

func ClosureIndirect2() {
	f := func(p *int) *int { return p } // ERROR "leaking param: p to result ~r0 level=0" "func literal does not escape"

	f(new(int)) // ERROR "new\(int\) does not escape"

	g := f
	g(new(int)) // ERROR "new\(int\) does not escape"

	h := nopFunc2
	h(new(int)) // ERROR "new\(int\) does not escape"
}

func nopFunc2(p *int) *int { return p } // ERROR "leaking param: p to result ~r0 level=0"

func ClosureIndirectDeclAssign() {
	var f func(p *int)
	f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	f(new(int))         // ERROR "new\(int\) does not escape"
}

func ClosureIndirectDeclAssign2() {
	var f func(p *int) *int
	f = func(p *int) *int { return p } // ERROR "leaking param: p to result ~r0 level=0" "func literal does not escape"
	f(new(int))                        // ERROR "new\(int\) does not escape"
}

func ClosureIndirectDeclAssign3() {
	var f func(p *int)
	f = func(p *int) { // ERROR "leaking param: p" "func literal does not escape"
		sink = p
	}
	f(new(int)) // ERROR "new\(int\) escapes to heap"
}

func ClosureIndirectDeclReassign() {
	var f func(p *int)
	f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	f = func(p *int) {  // ERROR "leaking param: p" "func literal does not escape"
		sink = p
	}
	f(new(int)) // ERROR "new\(int\) escapes to heap"
}

func ClosureIndirectDeclRecursive() {
	var visit func(p *int)
	visit = func(p *int) { // ERROR "p does not escape" "func literal does not escape"
		visit(p)
	}
	visit(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectNilInit() {
	var f func(p *int) = nil
	f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	f(new(int))         // ERROR "new\(int\) does not escape"
}

func ClosureIndirectTypedNilInit() {
	var f func(p *int) = (func(*int))(nil)
	f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	f(new(int))         // ERROR "new\(int\) does not escape"
}

func ClosureIndirectNestedAssign() {
	var f func(p *int)
	func() { // ERROR "func literal does not escape"
		f = func(p *int) {} // ERROR "p does not escape" "func literal escapes to heap"
	}()
	f(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectNestedAssignLeak() {
	var f func(p *int)
	func() { // ERROR "func literal does not escape"
		f = func(p *int) { // ERROR "leaking param: p" "func literal escapes to heap"
			sink = p
		}
	}()
	f(new(int)) // ERROR "new\(int\) escapes to heap"
}

func ClosureIndirectTypeSwitch() {
	foo := any(func(a *int) { sink = a }) // ERROR "func literal does not escape" "leaking param: a"
	switch foo := foo.(type) {
	case func(a *int):
		foo(new(int)) // ERROR "new\(int\) escapes to heap"
	}
}

func ClosureIndirectTypeSwitchReassign() {
	foo := any(func(a *int) { sink = a }) // ERROR "func literal does not escape" "leaking param: a"
	switch foo := foo.(type) {
	case func(a *int):
		foo = func(a *int) {} // ERROR "func literal does not escape" "a does not escape"
		foo(new(int))         // ERROR "new\(int\) escapes to heap"
	}
}

func ClosureIndirectNilReassign() {
	var f func(p *int)
	f = nil
	f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	f(new(int))         // ERROR "new\(int\) does not escape"
}

func ClosureIndirectMultiAssign(b bool) {
	var f func(p *int)
	if b {
		f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	} else {
		f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	}
	f(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectMultiAssignNamed(b bool) {
	var f func(*int)
	if b {
		f = nopFunc
	} else {
		f = nopFunc
	}
	f(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectMultiAssignResult(b bool) *int {
	var f func(p *int) *int
	if b {
		f = func(p *int) *int { return p } // ERROR "leaking param: p to result ~r0 level=0" "func literal does not escape"
	} else {
		f = func(p *int) *int { return p } // ERROR "leaking param: p to result ~r0 level=0" "func literal does not escape"
	}
	return f(new(int)) // ERROR "new\(int\) escapes to heap"
}

func ClosureIndirectMultiAssignSafe(b bool) int {
	var f func(p *int) int
	if b {
		f = func(p *int) int { return *p } // ERROR "p does not escape" "func literal does not escape"
	} else {
		f = func(p *int) int { return 42 } // ERROR "p does not escape" "func literal does not escape"
	}
	return f(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectTripleAssign(x int) {
	var f func(p *int)
	switch x {
	case 1:
		f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	case 2:
		f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	default:
		f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	}
	f(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectReassignInit(b bool) {
	f := func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	if b {
		f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	}
	f(new(int)) // ERROR "new\(int\) does not escape"
}

func ClosureIndirectNestedMultiAssign(b bool) {
	var f func(p *int)
	f = func(p *int) {} // ERROR "p does not escape" "func literal does not escape"
	func() {            // ERROR "func literal does not escape"
		f = func(p *int) {} // ERROR "p does not escape" "func literal escapes to heap"
	}()
	f(new(int)) // ERROR "new\(int\) does not escape"
}

type myFloat struct{ v float64 }

func (f *myFloat) add(p *myFloat) *myFloat { // ERROR "leaking param: f to result ~r0 level=0" "p does not escape"
	f.v += p.v
	return f
}

func (f *myFloat) sub(p *myFloat) *myFloat { // ERROR "leaking param: f to result ~r0 level=0" "p does not escape"
	f.v -= p.v
	return f
}

func ClosureIndirectMethodExpr(b bool) {
	var op func(*myFloat, *myFloat) *myFloat
	if b {
		op = (*myFloat).add
	} else {
		op = (*myFloat).sub
	}
	f := &myFloat{1.0} // ERROR "&myFloat{...} does not escape"
	g := &myFloat{2.0} // ERROR "&myFloat{...} does not escape"
	op(f, g)
}

func ClosureIndirectMethodExprMixed(b bool) {
	var op func(*myFloat, *myFloat) *myFloat
	if b {
		op = (*myFloat).add
	} else {
		op = func(f, g *myFloat) *myFloat { // ERROR "f does not escape" "g does not escape" "func literal does not escape"
			return nil
		}
	}
	f := &myFloat{1.0} // ERROR "&myFloat{...} does not escape"
	g := &myFloat{2.0} // ERROR "&myFloat{...} does not escape"
	op(f, g)
}
