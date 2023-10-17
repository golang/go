//gofmt -r=(x)->x

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Rewriting of parenthesized expressions (x) -> x
// must not drop parentheses if that would lead to
// wrong association of the operands.
// Was issue 1847.

package main

// From example 1 of issue 1847.
func _() {
	var t = (&T{1000}).Id()
}

// From example 2 of issue 1847.
func _() {
       fmt.Println((*xpp).a)
}

// Some more test cases.
func _() {
	_ = (-x).f
	_ = (*x).f
	_ = (&x).f
	_ = (!x).f
	_ = (-x.f)
	_ = (*x.f)
	_ = (&x.f)
	_ = (!x.f)
	(-x).f()
	(*x).f()
	(&x).f()
	(!x).f()
	_ = (-x.f())
	_ = (*x.f())
	_ = (&x.f())
	_ = (!x.f())

	_ = ((-x)).f
	_ = ((*x)).f
	_ = ((&x)).f
	_ = ((!x)).f
	_ = ((-x.f))
	_ = ((*x.f))
	_ = ((&x.f))
	_ = ((!x.f))
	((-x)).f()
	((*x)).f()
	((&x)).f()
	((!x)).f()
	_ = ((-x.f()))
	_ = ((*x.f()))
	_ = ((&x.f()))
	_ = ((!x.f()))

	_ = -(x).f
	_ = *(x).f
	_ = &(x).f
	_ = !(x).f
	_ = -x.f
	_ = *x.f
	_ = &x.f
	_ = !x.f
	_ = -(x).f()
	_ = *(x).f()
	_ = &(x).f()
	_ = !(x).f()
	_ = -x.f()
	_ = *x.f()
	_ = &x.f()
	_ = !x.f()
}
