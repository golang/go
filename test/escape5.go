// errorcheck -0 -m -l

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that the escape analysis is working.
// Compiles but does not run.  Inlining is disabled.

package foo

func noleak(p *int) int { // ERROR "p does not escape"
	return *p
}

func leaktoret(p *int) *int { // ERROR "leaking param: p to result"
	return p
}

func leaktoret2(p *int) (*int, *int) { // ERROR "leaking param: p to result .anon1" "leaking param: p to result .anon2"
	return p, p
}

func leaktoret22(p, q *int) (*int, *int) { // ERROR "leaking param: p to result .anon2" "leaking param: q to result .anon3"
	return p, q
}

func leaktoret22b(p, q *int) (*int, *int) { // ERROR "leaking param: p to result .anon3" "leaking param: q to result .anon2"
	return leaktoret22(q, p)
}

func leaktoret22c(p, q *int) (*int, *int) { // ERROR "leaking param: p to result .anon3" "leaking param: q to result .anon2"
	r, s := leaktoret22(q, p)
	return r, s
}

func leaktoret22d(p, q *int) (r, s *int) { // ERROR "leaking param: p to result s" "leaking param: q to result r"
	r, s = leaktoret22(q, p)
	return
}

func leaktoret22e(p, q *int) (r, s *int) { // ERROR "leaking param: p to result s" "leaking param: q to result r"
	r, s = leaktoret22(q, p)
	return r, s
}

func leaktoret22f(p, q *int) (r, s *int) { // ERROR "leaking param: p to result s" "leaking param: q to result r"
	rr, ss := leaktoret22(q, p)
	return rr, ss
}

var gp *int

func leaktosink(p *int) *int { // ERROR "leaking param: p"
	gp = p
	return p
}

func f1() {
	var x int
	p := noleak(&x) // ERROR "&x does not escape"
	_ = p
}

func f2() {
	var x int
	p := leaktoret(&x) // ERROR "&x does not escape"
	_ = p
}

func f3() {
	var x int          // ERROR "moved to heap: x"
	p := leaktoret(&x) // ERROR "&x escapes to heap"
	gp = p
}

func f4() {
	var x int              // ERROR "moved to heap: x"
	p, q := leaktoret2(&x) // ERROR "&x escapes to heap"
	gp = p
	gp = q
}

func f5() {
	var x int
	leaktoret22(leaktoret2(&x)) // ERROR "&x does not escape"
}

func f6() {
	var x int                               // ERROR "moved to heap: x"
	px1, px2 := leaktoret22(leaktoret2(&x)) // ERROR "&x escapes to heap"
	gp = px1
	_ = px2
}

type T struct{ x int }

func (t *T) Foo(u int) (*T, bool) { // ERROR "leaking param: t to result"
	t.x += u
	return t, true
}

func f7() *T {
	r, _ := new(T).Foo(42) // ERROR "new.T. escapes to heap"
	return r
}

func leakrecursive1(p, q *int) (*int, *int) { // ERROR "leaking param: p" "leaking param: q"
	return leakrecursive2(q, p)
}

func leakrecursive2(p, q *int) (*int, *int) { // ERROR "leaking param: p" "leaking param: q"
	if *p > *q {
		return leakrecursive1(q, p)
	}
	// without this, leakrecursive? are safe for p and q, b/c in fact their graph does not have leaking edges.
	return p, q
}


var global interface{}

type T1 struct {
	X *int
}

type T2 struct {
	Y *T1
}

func f8(p *T1) (k T2) { // ERROR "leaking param: p to result k" "leaking param: p"
	if p == nil {
		k = T2{}
		return
	}

	global = p // should make p leak always
	return T2{p}
}

func f9() {
	var j T1 // ERROR "moved to heap: j"
	f8(&j) // ERROR "&j escapes to heap"
}

func f10() {
	// These don't escape but are too big for the stack
	var x [1<<30]byte // ERROR "moved to heap: x"
	var y = make([]byte, 1<<30) // ERROR "does not escape"
	_ = x[0] + y[0]
}
