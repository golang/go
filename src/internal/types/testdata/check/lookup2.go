// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

import (
	"go/ast"
	"math/big"
)

// case           sel     pkg   have   message (examples for general lookup)
// ---------------------------------------------------------------------------------------------------------
// ok             x.Foo   ==    Foo
// misspelled     x.Foo   ==    FoO    type X has no field or method Foo, but does have field FoO
// misspelled     x.Foo   ==    foo    type X has no field or method Foo, but does have field foo
// misspelled     x.Foo   ==    foO    type X has no field or method Foo, but does have field foO
//
// misspelled     x.foo   ==    Foo    type X has no field or method foo, but does have field Foo
// misspelled     x.foo   ==    FoO    type X has no field or method foo, but does have field FoO
// ok             x.foo   ==    foo
// misspelled     x.foo   ==    foO    type X has no field or method foo, but does have field foO
//
// ok             x.Foo   !=    Foo
// misspelled     x.Foo   !=    FoO    type X has no field or method Foo, but does have field FoO
// unexported     x.Foo   !=    foo    type X has no field or method Foo, but does have unexported field foo
// missing        x.Foo   !=    foO    type X has no field or method Foo
//
// misspelled     x.foo   !=    Foo    type X has no field or method foo, but does have field Foo
// missing        x.foo   !=    FoO    type X has no field or method foo
// inaccessible   x.foo   !=    foo    cannot refer to unexported field foo
// missing        x.foo   !=    foO    type X has no field or method foo

type S struct {
	Foo1 int
	FoO2 int
	foo3 int
	foO4 int
}

func _() {
	var x S
	_ = x.Foo1 // OK
	_ = x.Foo2 // ERROR "x.Foo2 undefined (type S has no field or method Foo2, but does have field FoO2)"
	_ = x.Foo3 // ERROR "x.Foo3 undefined (type S has no field or method Foo3, but does have field foo3)"
	_ = x.Foo4 // ERROR "x.Foo4 undefined (type S has no field or method Foo4, but does have field foO4)"

	_ = x.foo1 // ERROR "x.foo1 undefined (type S has no field or method foo1, but does have field Foo1)"
	_ = x.foo2 // ERROR "x.foo2 undefined (type S has no field or method foo2, but does have field FoO2)"
	_ = x.foo3 // OK
	_ = x.foo4 // ERROR "x.foo4 undefined (type S has no field or method foo4, but does have field foO4)"
}

func _() {
	_ = S{Foo1: 0} // OK
	_ = S{Foo2 /* ERROR "unknown field Foo2 in struct literal of type S, but does have FoO2" */ : 0}
	_ = S{Foo3 /* ERROR "unknown field Foo3 in struct literal of type S, but does have foo3" */ : 0}
	_ = S{Foo4 /* ERROR "unknown field Foo4 in struct literal of type S, but does have foO4" */ : 0}

	_ = S{foo1 /* ERROR "unknown field foo1 in struct literal of type S, but does have Foo1" */ : 0}
	_ = S{foo2 /* ERROR "unknown field foo2 in struct literal of type S, but does have FoO2" */ : 0}
	_ = S{foo3: 0} // OK
	_ = S{foo4 /* ERROR "unknown field foo4 in struct literal of type S, but does have foO4" */ : 0}
}

// The following tests follow the same pattern as above but operate on an imported type instead of S.
// Currently our testing framework doesn't make it easy to define an imported package for testing, so
// instead we use the big.Float and ast.File types as they provide a suitable mix of exported and un-
// exported fields and methods.

func _() {
	var x *big.Float
	_ = x.Neg  // OK
	_ = x.NeG  // ERROR "x.NeG undefined (type *big.Float has no field or method NeG, but does have method Neg)"
	_ = x.Form // ERROR "x.Form undefined (type *big.Float has no field or method Form, but does have unexported field form)"
	_ = x.ForM // ERROR "x.ForM undefined (type *big.Float has no field or method ForM)"

	_ = x.abs  // ERROR "x.abs undefined (type *big.Float has no field or method abs, but does have method Abs)"
	_ = x.abS  // ERROR "x.abS undefined (type *big.Float has no field or method abS)"
	_ = x.form // ERROR "x.form undefined (cannot refer to unexported field form)"
	_ = x.forM // ERROR "x.forM undefined (type *big.Float has no field or method forM)"
}

func _() {
	_ = ast.File{Name: nil} // OK
	_ = ast.File{NamE /* ERROR "unknown field NamE in struct literal of type ast.File, but does have Name" */ : nil}
	_ = big.Float{Form /* ERROR "unknown field Form in struct literal of type big.Float, but does have unexported form" */ : 0}
	_ = big.Float{ForM /* ERROR "unknown field ForM in struct literal of type big.Float" */ : 0}

	_ = ast.File{name /* ERROR "unknown field name in struct literal of type ast.File, but does have Name" */ : nil}
	_ = ast.File{namE /* ERROR "unknown field namE in struct literal of type ast.File" */ : nil}
	_ = big.Float{form /* ERROR "cannot refer to unexported field form in struct literal of type big.Float" */ : 0}
	_ = big.Float{forM /* ERROR "unknown field forM in struct literal of type big.Float" */ : 0}
}
