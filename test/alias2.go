// errorcheck

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test basic restrictions on type aliases.

package p

import (
	"reflect"
	. "reflect"
)

type T0 struct{}

// Valid type alias declarations.

type _ = T0
type _ = int
type _ = struct{}
type _ = reflect.Value
type _ = Value

type (
	A0 = T0
	A1 = int
	A2 = struct{}
	A3 = reflect.Value
	A4 = Value
	A5 = Value

	N0 A0
)

// Methods can be declared on the original named type and the alias.
func (T0) m1()  {} // GCCGO_ERROR "previous"
func (*T0) m1() {} // ERROR "method redeclared: T0\.m1|T0\.m1 already declared|redefinition of .m1."
func (A0) m1()  {} // ERROR "T0\.m1 already declared|redefinition of .m1."
func (A0) m1()  {} // ERROR "T0\.m1 already declared|redefinition of .m1."
func (A0) m2()  {}

// Type aliases and the original type name can be used interchangeably.
var _ A0 = T0{}
var _ T0 = A0{}

// But aliases and original types cannot be used with new types based on them.
var _ N0 = T0{} // ERROR "cannot use T0{} \(value of struct type T0\) as N0 value in variable declaration"
var _ N0 = A0{} // ERROR "cannot use A0{} \(value of struct type A0\) as N0 value in variable declaration"

var _ A5 = Value{}

var _ interface {
	m1()
	m2()
} = T0{}

var _ interface {
	m1()
	m2()
} = A0{}

func _() {
	type _ = T0
	type _ = int
	type _ = struct{}
	type _ = reflect.Value
	type _ = Value

	type (
		A0 = T0
		A1 = int
		A2 = struct{}
		A3 = reflect.Value
		A4 = Value
		A5 Value

		N0 A0
	)

	var _ A0 = T0{}
	var _ T0 = A0{}

	var _ N0 = T0{} // ERROR "cannot use T0{} \(value of struct type T0\) as N0 value in variable declaration"
	var _ N0 = A0{} // ERROR "cannot use A0{} \(value of struct type A0\) as N0 value in variable declaration"

	var _ A5 = Value{} // ERROR "cannot use Value{} \(value of struct type reflect\.Value\) as A5 value in variable declaration"
}

// Invalid type alias declarations.

type _ = reflect.ValueOf // ERROR "reflect.ValueOf .*is not a type|expected type"

func (A1) m() {} // ERROR "cannot define new methods on non-local type|may not define methods on non-local type"
func (A2) m() {} // ERROR "invalid receiver type"
func (A3) m() {} // ERROR "cannot define new methods on non-local type|may not define methods on non-local type"
func (A4) m() {} // ERROR "cannot define new methods on non-local type|may not define methods on non-local type"

type B1 = struct{}

func (B1) m() {} // ERROR "invalid receiver type"

// TODO(gri) expand
