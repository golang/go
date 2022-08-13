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
	sink = func(p **int) *int { // ERROR "leaking param content: p" "func literal does not escape"
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
