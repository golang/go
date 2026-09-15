// errorcheck -0 -d=ssa/decompose_addr/debug=1

//go:build goexperiment.simd

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for variables that can be decomposed by the decomposeAddr pass.

package main

import "unsafe"

type point struct {
	x, y int
}

func simpleStruct(c bool) int {
	p := point{10, 20} // ERROR "promoted p"
	ptr := &p
	if c {
		ptr.x += 1
	} else {
		ptr.y += 1
	}
	return ptr.x + ptr.y
}

type inner struct {
	a, b int
}

type outer struct {
	in inner
	c  int
}

func nestedStruct(c bool) int {
	o := outer{inner{1, 2}, 3} // ERROR "promoted o"
	ptr := &o
	if c {
		ptr.in.a += 10
	} else {
		ptr.in.b += 20
	}
	return ptr.in.a + ptr.in.b + ptr.c
}

func array(c bool) int {
	var a [3]int // ERROR "promoted a"
	pa := &a
	pa[0] = 10
	pa[1] = 20
	pa[2] = 30
	if c {
		pa[1] += 5
	}
	return pa[0] + pa[1] + pa[2]
}

func zeroStruct(c bool) int {
	var p point // ERROR "promoted p"
	ptr := &p
	if c {
		ptr.x = 10
	} else {
		ptr.y = 20
	}
	return ptr.x + ptr.y
}

func moveStruct(x, y int, c bool) int {
	p1 := point{x, y} // ERROR "promoted p1"
	var p2 point      // ERROR "promoted p2"
	ptr1 := &p1
	ptr2 := &p2
	if c {
		*ptr2 = *ptr1
	} else {
		*ptr2 = point{30, 40}
	}
	return ptr2.x + ptr2.y + ptr1.x
}

type arrayStruct struct {
	a [2]int
	b int
}

func arrayStructAddr(c bool) int {
	s := arrayStruct{[2]int{1, 2}, 3} // ERROR "promoted s"
	ps := &s
	if c {
		ps.a[0] += 10
	} else {
		ps.b += 20
	}
	return ps.a[0] + ps.a[1] + ps.b
}

func partialZero() int {
	type T struct {
		a int
		b [2]int
		c int
	}

	t := T{a: 9, c: 9} // ERROR "promoted t"
	p := &t.b          // OffPtr
	*p = [2]int{}      // Zero
	return t.a
}

func partialMove() int {
	type T struct {
		a int
		b [2]int
		c int
	}

	b2 := [2]int{1, 2} // ERROR "promoted b2"

	t := T{a: 9, c: 9} // ERROR "promoted t"
	p := &t.b          // OffPtr
	*p = b2            // Move
	return t.a
}

type bigStruct struct {
	a, b, c, d, e int
	f             [30]int
	g, h, i, j, k int
}

func partialActiveBigStruct(branch bool) {
	// branch is used to fool opt.
	var x, y bigStruct // ERROR "promoted x"
	xp := &x
	yp := &y
	if branch {
		xp.a = 1
		xp.b = 2
	} else {
		xp.a = 4
		xp.b = 5
	}
	xp.f = [30]int{3, 4}
	xp.c = xp.a + xp.b // 2 leaves of x are active
	if branch {
		yp.c = 1
		yp.d = 2
		yp.e = 3
		yp.f[5] = 4
		yp.g = 5
	} else {
		yp.c = 4
		yp.d = 5
		yp.e = 6
		yp.f[5] = 7
		yp.g = 8
	}
	yp.h = yp.c + yp.d + yp.e + yp.f[5] + yp.g // 5 leaves of y are used
}

// x looks at the Move first and would rule y out (y's leaves do not line
// up with x's), but x then gives up on itself because of the 2-byte read
// that load forwarding created on it. Giving up must also take back the
// verdict on y: y is then free to break the Move down by its own leaves.
func unsafeAliasMoveOrder(c bool) int {
	var x [4]byte
	x[0], x[1], x[2], x[3] = 1, 2, 3, 4
	var y [2]uint16 // ERROR "promoted y"
	y = *(*[2]uint16)(unsafe.Pointer(&x))
	if c {
		y[0]++
	}
	return int(y[0]) + int(y[1])
}

type paddedA struct {
	a             uint8
	b, c, d, e, f uint32
}

type paddedB struct {
	x, y          uint8
	b, c, d, e, f uint32
}

// a looks at the Move first. Every leaf of a lines up with a leaf of b,
// but b has a leaf (y) where a only has padding, so a breakdown of the
// Move by a's leaves would never write b.y. The Move must cover exactly
// the same leaves on both sides for both to be decomposed, so b is ruled
// out and stays in memory.
func unsafeAliasMovePadding(v uint8, c bool) (uint8, uint32) {
	var a paddedA // ERROR "promoted a"
	a.a = v
	a.b, a.c = 1, 2
	var b paddedB
	b.x, b.y = 7, 9
	b = *(*paddedB)(unsafe.Pointer(&a))
	b.b++
	if c {
		b.c++
	}
	return b.y, b.b + b.c
}

func main() {}
