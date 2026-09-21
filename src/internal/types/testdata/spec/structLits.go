// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type A struct {
	a int
	B
	C
	*D
}

type B struct {
	b int
	C
}

type C struct {
	c int
	D
}

type D struct {
	x, y, z int
}

var (
	_ = &A{}
	_ = &A{a: 0}
	_ = A{A /* ERROR "unknown field A in struct literal of type A, but does have a" */ : 0}
	_ = A{X /* ERROR "unknown field X in struct literal of type A, but does have x" */ : 0}
	_ = A{a: 0, b: 0, c: 0}
	_ = A{x /* ERROR "pointer indirection" */ : 0}
	_ = B{b: 0, c: 0, x: 0, y: 0, z: 0}
	_ = B{C: C{}, x /* ERROR "cannot specify promoted field x and enclosing embedded field C" */ : 0}
	_ = A{b: 0, B /* ERROR "cannot specify embedded field B and enclosed promoted field b" */ : B{}}
	_ = A{B: B{}, b /* ERROR "cannot specify promoted field b and enclosing embedded field B" */ : 0}
)
