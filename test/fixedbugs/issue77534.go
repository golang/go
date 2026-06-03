// compile

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 77534: compiler crash when >4 fields, but only one nonempty pointer field.

package p

type T struct {
	a, b, c, d struct{}
	e          *byte
}

func f1(p *any, t T) {
	*p = t
}

func f2(p *any, t *T) {
	*p = *t
}

func f3(p, x, y *T, b bool) {
	var z T
	if b {
		z = *x
	} else {
		z = *y
	}
	*p = z
}

func f4(i any) T {
	return i.(T)
}

type Inner struct {
	a struct{}
	p *byte
}

type Outer struct {
	inner Inner
}

func f5(o1, o2 Outer, c bool) Outer {
	var i any
	if c {
		i = o1
	} else {
		i = o2
	}
	return i.(Outer)
}
