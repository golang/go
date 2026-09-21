// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type E struct {
	e int
}

func (E) m() {}

type S struct {
	E
	x int
}

func (S) n() {}

func _() {
	_ = S.X // ERROR "S.X undefined (type S has no field or method X, but does have field x)"
	_ = S /* ERROR "operand for field selector E must be value of type S" */ .E
	_ = S /* ERROR "operand for field selector x must be value of type S" */ .x
	_ = S /* ERROR "operand for field selector e must be value of type S" */ .e
	_ = S.m
	_ = S.n

	var s S
	_ = s.X // ERROR "s.X undefined (type S has no field or method X, but does have field x)"
	_ = s.E
	_ = s.x
	_ = s.e
	_ = s.m
	_ = s.n
}
