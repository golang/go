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
func (T0) m1() {}
func (A0) m1() {} // TODO(gri) this should be an error
func (A0) m2() {}

// Type aliases and the original type name can be used interchangeably.
var _ A0 = T0{}
var _ T0 = A0{}

// But aliases and original types cannot be used with new types based on them.
var _ N0 = T0{} // ERROR "cannot use T0 literal \(type T0\) as type N0 in assignment"
var _ N0 = A0{} // ERROR "cannot use T0 literal \(type T0\) as type N0 in assignment"

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

	var _ N0 = T0{} // ERROR "cannot use T0 literal \(type T0\) as type N0 in assignment"
	var _ N0 = A0{} // ERROR "cannot use T0 literal \(type T0\) as type N0 in assignment"

	var _ A5 = Value{} // ERROR "cannot use reflect\.Value literal \(type reflect.Value\) as type A5 in assignment"
}

// Invalid type alias declarations.

type _ = reflect.ValueOf // ERROR "reflect.ValueOf is not a type"

func (A1) m() {} // ERROR "cannot define new methods on non-local type int"

type B1 = struct{}

func (B1) m() {} // ERROR "invalid receiver type"

// TODO(gri) expand
