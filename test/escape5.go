// errorcheck -0 -m -l

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test, using compiler diagnostic flags, that the escape analysis is working.
// Compiles but does not run.  Inlining is disabled.

package foo

import (
	"runtime"
	"unsafe"
)

func noleak(p *int) int { // ERROR "p does not escape"
	return *p
}

func leaktoret(p *int) *int { // ERROR "leaking param: p to result"
	return p
}

func leaktoret2(p *int) (*int, *int) { // ERROR "leaking param: p to result ~r0" "leaking param: p to result ~r1"
	return p, p
}

func leaktoret22(p, q *int) (*int, *int) { // ERROR "leaking param: p to result ~r0" "leaking param: q to result ~r1"
	return p, q
}

func leaktoret22b(p, q *int) (*int, *int) { // ERROR "leaking param: p to result ~r1" "leaking param: q to result ~r0"
	return leaktoret22(q, p)
}

func leaktoret22c(p, q *int) (*int, *int) { // ERROR "leaking param: p to result ~r1" "leaking param: q to result ~r0"
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
	p := noleak(&x)
	_ = p
}

func f2() {
	var x int
	p := leaktoret(&x)
	_ = p
}

func f3() {
	var x int // ERROR "moved to heap: x"
	p := leaktoret(&x)
	gp = p
}

func f4() {
	var x int // ERROR "moved to heap: x"
	p, q := leaktoret2(&x)
	gp = p
	gp = q
}

func f5() {
	var x int
	leaktoret22(leaktoret2(&x))
}

func f6() {
	var x int // ERROR "moved to heap: x"
	px1, px2 := leaktoret22(leaktoret2(&x))
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

func f8(p *T1) (k T2) { // ERROR "leaking param: p$"
	if p == nil {
		k = T2{}
		return
	}

	// should make p leak always
	global = p
	return T2{p}
}

func f9() {
	var j T1 // ERROR "moved to heap: j"
	f8(&j)
}

func f10() {
	// These don't escape but are too big for the stack
	var x [1 << 30]byte         // ERROR "moved to heap: x"
	var y = make([]byte, 1<<30) // ERROR "make\(\[\]byte, 1 << 30\) escapes to heap"
	_ = x[0] + y[0]
}

// Test for issue 19687 (passing to unnamed parameters does not escape).
func f11(**int) {
}
func f12(_ **int) {
}
func f13() {
	var x *int
	f11(&x)
	f12(&x)
	runtime.KeepAlive(&x)
}

// Test for issue 24305 (passing to unnamed receivers does not escape).
type U int

func (*U) M()   {}
func (_ *U) N() {}

func fbad24305a() {
	var u U
	u.M()
	u.N()
}

func fbad24305b() {
	var u U
	(*U).M(&u)
	(*U).N(&u)
}

// Issue 24730: taking address in a loop causes unnecessary escape
type T24730 struct {
	x [64]byte
}

func (t *T24730) g() { // ERROR "t does not escape"
	y := t.x[:]
	for i := range t.x[:] {
		y = t.x[:]
		y[i] = 1
	}

	var z *byte
	for i := range t.x[:] {
		z = &t.x[i]
		*z = 2
	}
}

// Issue 15730: copy causes unnecessary escape

var sink []byte
var sink2 []int
var sink3 []*int

func f15730a(args ...interface{}) { // ERROR "args does not escape"
	for _, arg := range args {
		switch a := arg.(type) {
		case string:
			copy(sink, a)
		}
	}
}

func f15730b(args ...interface{}) { // ERROR "args does not escape"
	for _, arg := range args {
		switch a := arg.(type) {
		case []int:
			copy(sink2, a)
		}
	}
}

func f15730c(args ...interface{}) { // ERROR "leaking param content: args"
	for _, arg := range args {
		switch a := arg.(type) {
		case []*int:
			// copy pointerful data should cause escape
			copy(sink3, a)
		}
	}
}

// Issue 29000: unnamed parameter is not handled correctly

var sink4 interface{}
var alwaysFalse = false

func f29000(_ int, x interface{}) { // ERROR "leaking param: x"
	sink4 = x
	if alwaysFalse {
		g29000()
	}
}

func g29000() {
	x := 1
	f29000(2, x) // ERROR "x escapes to heap"
}

// Issue 28369: taking an address of a parameter and converting it into a uintptr causes an
// unnecessary escape.

var sink28369 uintptr

func f28369(n int) int {
	if n == 0 {
		sink28369 = uintptr(unsafe.Pointer(&n))
		return n
	}

	return 1 + f28369(n-1)
}

// Issue 44614: parameters that flow to a heap-allocated result
// parameter must be recorded as a heap-flow rather than a
// result-flow.

// N.B., must match "leaking param: p",
// but *not* "leaking param: p to result r level=0".
func f(p *int) (r *int) { // ERROR "leaking param: p$" "moved to heap: r"
	sink4 = &r
	return p
}
