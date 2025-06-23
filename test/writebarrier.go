// errorcheck -0 -l -d=wb

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test where write barriers are and are not emitted.

package p

import "unsafe"

func f(x **byte, y *byte) {
	*x = y // no barrier (dead store)

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f1(x *[]byte, y []byte) {
	*x = y // no barrier (dead store)

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f1a(x *[]byte, y *[]byte) {
	*x = *y // ERROR "write barrier"

	z := *y // no barrier
	*x = z  // ERROR "write barrier"
}

func f2(x *interface{}, y interface{}) {
	*x = y // no barrier (dead store)

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f2a(x *interface{}, y *interface{}) {
	*x = *y // no barrier (dead store)

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f3(x *string, y string) {
	*x = y // no barrier (dead store)

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f3a(x *string, y *string) {
	*x = *y // ERROR "write barrier"

	z := *y // no barrier
	*x = z  // ERROR "write barrier"
}

func f4(x *[2]string, y [2]string) {
	*x = y // ERROR "write barrier"

	z := y // no barrier
	*x = z // ERROR "write barrier"
}

func f4a(x *[2]string, y *[2]string) {
	*x = *y // ERROR "write barrier"

	z := *y // no barrier
	*x = z  // ERROR "write barrier"
}

type T struct {
	X *int
	Y int
	M map[int]int
}

func f5(t, u *T) {
	t.X = &u.Y // ERROR "write barrier"
}

func f6(t *T) {
	t.M = map[int]int{1: 2} // ERROR "write barrier"
}

func f7(x, y *int) []*int {
	var z [3]*int
	i := 0
	z[i] = x // ERROR "write barrier"
	i++
	z[i] = y // ERROR "write barrier"
	i++
	return z[:i]
}

func f9(x *interface{}, v *byte) {
	*x = v // ERROR "write barrier"
}

func f10(x *byte, f func(interface{})) {
	f(x)
}

func f11(x *unsafe.Pointer, y unsafe.Pointer) {
	*x = unsafe.Pointer(uintptr(y) + 1) // ERROR "write barrier"
}

func f12(x []*int, y *int) []*int {
	// write barrier for storing y in x's underlying array
	x = append(x, y) // ERROR "write barrier"
	return x
}

func f12a(x []int, y int) []int {
	// y not a pointer, so no write barriers in this function
	x = append(x, y)
	return x
}

func f13(x []int, y *[]int) {
	*y = append(x, 1) // ERROR "write barrier"
}

func f14(y *[]int) {
	*y = append(*y, 1) // ERROR "write barrier"
}

type T1 struct {
	X *int
}

func f15(x []T1, y T1) []T1 {
	return append(x, y) // ERROR "write barrier"
}

type T8 struct {
	X [8]*int
}

func f16(x []T8, y T8) []T8 {
	return append(x, y) // ERROR "write barrier"
}

func t1(i interface{}) **int {
	// From issue 14306, make sure we have write barriers in a type switch
	// where the assigned variable escapes.
	switch x := i.(type) {
	case *int: // ERROR "write barrier"
		return &x
	}
	switch y := i.(type) {
	case **int: // no write barrier here
		return y
	}
	return nil
}

type T17 struct {
	f func(*T17)
}

func f17(x *T17) {
	// Originally from golang.org/issue/13901, but the hybrid
	// barrier requires both to have barriers.
	x.f = f17                      // ERROR "write barrier"
	x.f = func(y *T17) { *y = *x } // ERROR "write barrier"
}

type T18 struct {
	a []int
	s string
}

func f18(p *T18, x *[]int) {
	p.a = p.a[:5]    // no barrier
	*x = (*x)[0:5]   // no barrier
	p.a = p.a[3:5]   // ERROR "write barrier"
	p.a = p.a[1:2:3] // ERROR "write barrier"
	p.s = p.s[8:9]   // ERROR "write barrier"
	*x = (*x)[3:5]   // ERROR "write barrier"
}

func f19(x, y *int, i int) int {
	// Constructing a temporary slice on the stack should not
	// require any write barriers. See issue 14263.
	a := []*int{x, y} // no barrier
	return *a[i]
}

func f20(x, y *int, i int) []*int {
	// ... but if that temporary slice escapes, then the
	// write barriers are necessary.
	a := []*int{x, y} // ERROR "write barrier"
	return a
}

var x21 *int
var y21 struct {
	x *int
}
var z21 int

// f21x: Global -> heap pointer updates must have write barriers.
func f21a(x *int) {
	x21 = x   // ERROR "write barrier"
	y21.x = x // ERROR "write barrier"
}

func f21b(x *int) {
	x21 = &z21   // ERROR "write barrier"
	y21.x = &z21 // ERROR "write barrier"
}

func f21c(x *int) {
	y21 = struct{ x *int }{x} // ERROR "write barrier"
}

func f22(x *int) (y *int) {
	// pointer write on stack should have no write barrier.
	// this is a case that the frontend failed to eliminate.
	p := &y
	*p = x // no barrier
	return
}

type T23 struct {
	p *int
	a int
}

var t23 T23
var i23 int

// f23x: zeroing global needs write barrier for the hybrid barrier.
func f23a() {
	t23 = T23{} // ERROR "write barrier"
}

func f23b() {
	// also test partial assignments
	t23 = T23{a: 1} // ERROR "write barrier"
}

func f23c() {
	t23 = T23{} // no barrier (dead store)
	// also test partial assignments
	t23 = T23{p: &i23} // ERROR "write barrier"
}

var g int

func f24() **int {
	p := new(*int)
	*p = &g // no write barrier here
	return p
}
func f25() []string {
	return []string{"abc", "def", "ghi"} // no write barrier here
}

type T26 struct {
	a, b, c int
	d, e, f *int
}

var g26 int

func f26(p *int) *T26 { // see issue 29573
	return &T26{
		a: 5,
		b: 6,
		c: 7,
		d: &g26, // no write barrier: global ptr
		e: nil,  // no write barrier: nil ptr
		f: p,    // ERROR "write barrier"
	}
}

func f27(p *int) []interface{} {
	return []interface{}{
		nil,         // no write barrier: zeroed memory, nil ptr
		(*T26)(nil), // no write barrier: zeroed memory, type ptr & nil ptr
		&g26,        // no write barrier: zeroed memory, type ptr & global ptr
		7,           // no write barrier: zeroed memory, type ptr & global ptr
		p,           // ERROR "write barrier"
	}
}

var g28 [256]uint64

func f28() []interface{} {
	return []interface{}{
		false,      // no write barrier
		true,       // no write barrier
		0,          // no write barrier
		1,          // no write barrier
		uint8(127), // no write barrier
		int8(-4),   // no write barrier
		&g28[5],    // no write barrier
	}
}
