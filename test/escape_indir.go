// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis when assigning to indirections.

package escape

var sink interface{}

type ConstPtr struct {
	p *int
	c ConstPtr2
	x **ConstPtr
}

type ConstPtr2 struct {
	p *int
	i int
}

func constptr0() {
	i := 0           // ERROR "moved to heap: i"
	x := &ConstPtr{} // ERROR "&ConstPtr literal does not escape"
	// BAD: i should not escape here
	x.p = &i // ERROR "&i escapes to heap"
	_ = x
}

func constptr01() *ConstPtr {
	i := 0           // ERROR "moved to heap: i"
	x := &ConstPtr{} // ERROR "&ConstPtr literal escapes to heap"
	x.p = &i         // ERROR "&i escapes to heap"
	return x
}

func constptr02() ConstPtr {
	i := 0           // ERROR "moved to heap: i"
	x := &ConstPtr{} // ERROR "&ConstPtr literal does not escape"
	x.p = &i         // ERROR "&i escapes to heap"
	return *x
}

func constptr03() **ConstPtr {
	i := 0           // ERROR "moved to heap: i"
	x := &ConstPtr{} // ERROR "&ConstPtr literal escapes to heap" "moved to heap: x"
	x.p = &i         // ERROR "&i escapes to heap"
	return &x        // ERROR "&x escapes to heap"
}

func constptr1() {
	i := 0           // ERROR "moved to heap: i"
	x := &ConstPtr{} // ERROR "&ConstPtr literal escapes to heap"
	x.p = &i         // ERROR "&i escapes to heap"
	sink = x         // ERROR "x escapes to heap"
}

func constptr2() {
	i := 0           // ERROR "moved to heap: i"
	x := &ConstPtr{} // ERROR "&ConstPtr literal does not escape"
	x.p = &i         // ERROR "&i escapes to heap"
	sink = *x        // ERROR "\*x escapes to heap"
}

func constptr4() *ConstPtr {
	p := new(ConstPtr) // ERROR "new\(ConstPtr\) escapes to heap"
	*p = *&ConstPtr{}  // ERROR "&ConstPtr literal does not escape"
	return p
}

func constptr5() *ConstPtr {
	p := new(ConstPtr) // ERROR "new\(ConstPtr\) escapes to heap"
	p1 := &ConstPtr{}  // ERROR "&ConstPtr literal does not escape"
	*p = *p1
	return p
}

// BAD: p should not escape here
func constptr6(p *ConstPtr) { // ERROR "leaking param content: p"
	p1 := &ConstPtr{} // ERROR "&ConstPtr literal does not escape"
	*p1 = *p
	_ = p1
}

func constptr7() **ConstPtr {
	p := new(ConstPtr) // ERROR "new\(ConstPtr\) escapes to heap" "moved to heap: p"
	var tmp ConstPtr2
	p1 := &tmp // ERROR "&tmp does not escape"
	p.c = *p1
	return &p // ERROR "&p escapes to heap"
}

func constptr8() *ConstPtr {
	p := new(ConstPtr) // ERROR "new\(ConstPtr\) escapes to heap"
	var tmp ConstPtr2
	p.c = *&tmp // ERROR "&tmp does not escape"
	return p
}

func constptr9() ConstPtr {
	p := new(ConstPtr) // ERROR "new\(ConstPtr\) does not escape"
	var p1 ConstPtr2
	i := 0    // ERROR "moved to heap: i"
	p1.p = &i // ERROR "&i escapes to heap"
	p.c = p1
	return *p
}

func constptr10() ConstPtr {
	x := &ConstPtr{} // ERROR "moved to heap: x" "&ConstPtr literal escapes to heap"
	i := 0           // ERROR "moved to heap: i"
	var p *ConstPtr
	p = &ConstPtr{p: &i, x: &x} // ERROR "&i escapes to heap" "&x escapes to heap" "&ConstPtr literal does not escape"
	var pp **ConstPtr
	pp = &p // ERROR "&p does not escape"
	return **pp
}

func constptr11() *ConstPtr {
	i := 0             // ERROR "moved to heap: i"
	p := new(ConstPtr) // ERROR "new\(ConstPtr\) escapes to heap"
	p1 := &ConstPtr{}  // ERROR "&ConstPtr literal does not escape"
	p1.p = &i          // ERROR "&i escapes to heap"
	*p = *p1
	return p
}

func foo(p **int) { // ERROR "foo p does not escape"
	i := 0 // ERROR "moved to heap: i"
	y := p
	*y = &i // ERROR "&i escapes to heap"
}

func foo1(p *int) { // ERROR "p does not escape"
	i := 0  // ERROR "moved to heap: i"
	y := &p // ERROR "&p does not escape"
	*y = &i // ERROR "&i escapes to heap"
}

func foo2() {
	type Z struct {
		f **int
	}
	x := new(int) // ERROR "moved to heap: x" "new\(int\) escapes to heap"
	sink = &x     // ERROR "&x escapes to heap"
	var z Z
	z.f = &x // ERROR "&x does not escape"
	p := z.f
	i := 0  // ERROR "moved to heap: i"
	*p = &i // ERROR "&i escapes to heap"
}

var global *byte

func f() {
	var x byte    // ERROR "moved to heap: x"
	global = &*&x // ERROR "&\(\*\(&x\)\) escapes to heap" "&x escapes to heap"
}
