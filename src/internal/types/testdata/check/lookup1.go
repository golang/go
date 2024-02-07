// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lookup

import "math/big" // provides big.Float struct with unexported fields and methods

func _() {
	var s struct {
		x, aBc int
	}
	_ = s.x
	_ = s /* ERROR "invalid operation: cannot call non-function s.x (variable of type int)" */ .x()
	_ = s.X // ERROR "s.X undefined (type struct{x int; aBc int} has no field or method X, but does have field x)"
	_ = s.X /* ERROR "s.X undefined (type struct{x int; aBc int} has no field or method X, but does have field x)" */ ()

	_ = s.aBc
	_ = s.abc // ERROR "s.abc undefined (type struct{x int; aBc int} has no field or method abc, but does have field aBc)"
	_ = s.ABC // ERROR "s.ABC undefined (type struct{x int; aBc int} has no field or method ABC, but does have field aBc)"
}

func _() {
	type S struct {
		x int
	}
	var s S
	_ = s.x
	_ = s /* ERROR "invalid operation: cannot call non-function s.x (variable of type int)" */ .x()
	_ = s.X // ERROR "s.X undefined (type S has no field or method X, but does have field x)"
	_ = s.X /* ERROR "s.X undefined (type S has no field or method X, but does have field x)" */ ()
}

type S struct {
	x int
}

func (S) m()   {}
func (S) aBc() {}

func _() {
	var s S
	_ = s.m
	s.m()
	_ = s.M // ERROR "s.M undefined (type S has no field or method M, but does have method m)"
	s.M /* ERROR "s.M undefined (type S has no field or method M, but does have method m)" */ ()

	_ = s.aBc
	_ = s.abc // ERROR "s.abc undefined (type S has no field or method abc, but does have method aBc)"
	_ = s.ABC // ERROR "s.ABC undefined (type S has no field or method ABC, but does have method aBc)"
}

func _() {
	type P *S
	var s P
	_ = s.m // ERROR "s.m undefined (type P has no field or method m)"
	_ = s.M // ERROR "s.M undefined (type P has no field or method M)"
	_ = s.x
	_ = s.X // ERROR "s.X undefined (type P has no field or method X, but does have field x)"
}

func _() {
	var x big.Float
	_ = x.neg // ERROR "x.neg undefined (type big.Float has no field or method neg, but does have method Neg)"
	_ = x.nEg // ERROR "x.nEg undefined (type big.Float has no field or method nEg)"
	_ = x.Neg
	_ = x.NEg // ERROR "x.NEg undefined (type big.Float has no field or method NEg, but does have method Neg)"

	_ = x.form // ERROR "x.form undefined (cannot refer to unexported field form)"
	_ = x.fOrm // ERROR "x.fOrm undefined (type big.Float has no field or method fOrm)"
	_ = x.Form // ERROR "x.Form undefined (type big.Float has no field or method Form, but does have unexported field form)"
	_ = x.FOrm // ERROR "x.FOrm undefined (type big.Float has no field or method FOrm)"
}
