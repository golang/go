// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conversions

import "unsafe"

// constant conversions

func _[T ~byte]() T { return 255 }
func _[T ~byte]() T { return 256 /* ERROR cannot use 256 .* as T value */ }

func _[T ~byte]() {
	const _ = T /* ERROR T\(0\) .* is not constant */ (0)
	var _ T = 255
	var _ T = 256 // ERROR cannot use 256 .* as T value
}

func _[T ~string]() T                { return T('a') }
func _[T ~int | ~string]() T         { return T('a') }
func _[T ~byte | ~int | ~string]() T { return T(256 /* ERROR cannot convert 256 .* to T */) }

// implicit conversions never convert to string
func _[T ~string]() {
	var _ string = 0 // ERROR cannot use .* as string value
	var _ T = 0      // ERROR cannot use .* as T value
}

// failing const conversions of constants to type parameters report a cause
func _[
	T1 any,
	T2 interface{ m() },
	T3 ~int | ~float64 | ~bool,
	T4 ~int | ~string,
]() {
	_ = T1(0 /* ERROR cannot convert 0 .* to T1\n\tT1 does not contain specific types */)
	_ = T2(1 /* ERROR cannot convert 1 .* to T2\n\tT2 does not contain specific types */)
	_ = T3(2 /* ERROR cannot convert 2 .* to T3\n\tcannot convert 2 .* to bool \(in T3\) */)
	_ = T4(3.14 /* ERROR cannot convert 3.14 .* to T4\n\tcannot convert 3.14 .* to int \(in T4\) */)
}

// "x is assignable to T"
// - tested via assignability tests

// "x's type and T have identical underlying types if tags are ignored"

func _[X ~int, T ~int](x X) T { return T(x) }
func _[X struct {
	f int "foo"
}, T struct {
	f int "bar"
}](x X) T {
	return T(x)
}

type Foo struct {
	f int "foo"
}
type Bar struct {
	f int "bar"
}
type Far struct{ f float64 }

func _[X Foo, T Bar](x X) T       { return T(x) }
func _[X Foo | Bar, T Bar](x X) T { return T(x) }
func _[X Foo, T Foo | Bar](x X) T { return T(x) }
func _[X Foo, T Far](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by Foo\) to T\n\tcannot convert Foo \(in X\) to Far \(in T\) */)
}

// "x's type and T are unnamed pointer types and their pointer base types
// have identical underlying types if tags are ignored"

func _[X ~*Foo, T ~*Bar](x X) T         { return T(x) }
func _[X ~*Foo | ~*Bar, T ~*Bar](x X) T { return T(x) }
func _[X ~*Foo, T ~*Foo | ~*Bar](x X) T { return T(x) }
func _[X ~*Foo, T ~*Far](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by ~\*Foo\) to T\n\tcannot convert \*Foo \(in X\) to \*Far \(in T\) */)
}

// Verify that the defined types in constraints are considered for the rule above.

type (
	B  int
	C  int
	X0 *B
	T0 *C
)

func _(x X0) T0           { return T0(x /* ERROR cannot convert */) } // non-generic reference
func _[X X0, T T0](x X) T { return T(x /* ERROR cannot convert */) }
func _[T T0](x X0) T      { return T(x /* ERROR cannot convert */) }
func _[X X0](x X) T0      { return T0(x /* ERROR cannot convert */) }

// "x's type and T are both integer or floating point types"

func _[X Integer, T Integer](x X) T  { return T(x) }
func _[X Unsigned, T Integer](x X) T { return T(x) }
func _[X Float, T Integer](x X) T    { return T(x) }

func _[X Integer, T Unsigned](x X) T  { return T(x) }
func _[X Unsigned, T Unsigned](x X) T { return T(x) }
func _[X Float, T Unsigned](x X) T    { return T(x) }

func _[X Integer, T Float](x X) T  { return T(x) }
func _[X Unsigned, T Float](x X) T { return T(x) }
func _[X Float, T Float](x X) T    { return T(x) }

func _[X, T Integer | Unsigned | Float](x X) T { return T(x) }
func _[X, T Integer | ~string](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by Integer \| ~string\) to T\n\tcannot convert string \(in X\) to int \(in T\) */)
}

// "x's type and T are both complex types"

func _[X, T Complex](x X) T { return T(x) }
func _[X, T Float | Complex](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by Float \| Complex\) to T\n\tcannot convert float32 \(in X\) to complex64 \(in T\) */)
}

// "x is an integer or a slice of bytes or runes and T is a string type"

type myInt int
type myString string

func _[T ~string](x int) T      { return T(x) }
func _[T ~string](x myInt) T    { return T(x) }
func _[X Integer](x X) string   { return string(x) }
func _[X Integer](x X) myString { return myString(x) }
func _[X Integer](x X) *string {
	return (*string)(x /* ERROR cannot convert x \(variable of type X constrained by Integer\) to \*string\n\tcannot convert int \(in X\) to \*string */)
}

func _[T ~string](x []byte) T                           { return T(x) }
func _[T ~string](x []rune) T                           { return T(x) }
func _[X ~[]byte, T ~string](x X) T                     { return T(x) }
func _[X ~[]rune, T ~string](x X) T                     { return T(x) }
func _[X Integer | ~[]byte | ~[]rune, T ~string](x X) T { return T(x) }
func _[X Integer | ~[]byte | ~[]rune, T ~*string](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by Integer \| ~\[\]byte \| ~\[\]rune\) to T\n\tcannot convert int \(in X\) to \*string \(in T\) */)
}

// "x is a string and T is a slice of bytes or runes"

func _[T ~[]byte](x string) T { return T(x) }
func _[T ~[]rune](x string) T { return T(x) }
func _[T ~[]rune](x *string) T {
	return T(x /* ERROR cannot convert x \(variable of type \*string\) to T\n\tcannot convert \*string to \[\]rune \(in T\) */)
}

func _[X ~string, T ~[]byte](x X) T           { return T(x) }
func _[X ~string, T ~[]rune](x X) T           { return T(x) }
func _[X ~string, T ~[]byte | ~[]rune](x X) T { return T(x) }
func _[X ~*string, T ~[]byte | ~[]rune](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by ~\*string\) to T\n\tcannot convert \*string \(in X\) to \[\]byte \(in T\) */)
}

// package unsafe:
// "any pointer or value of underlying type uintptr can be converted into a unsafe.Pointer"

type myUintptr uintptr

func _[X ~uintptr](x X) unsafe.Pointer  { return unsafe.Pointer(x) }
func _[T unsafe.Pointer](x myUintptr) T { return T(x) }
func _[T unsafe.Pointer](x int64) T {
	return T(x /* ERROR cannot convert x \(variable of type int64\) to T\n\tcannot convert int64 to unsafe\.Pointer \(in T\) */)
}

// "and vice versa"

func _[T ~uintptr](x unsafe.Pointer) T  { return T(x) }
func _[X unsafe.Pointer](x X) uintptr   { return uintptr(x) }
func _[X unsafe.Pointer](x X) myUintptr { return myUintptr(x) }
func _[X unsafe.Pointer](x X) int64 {
	return int64(x /* ERROR cannot convert x \(variable of type X constrained by unsafe\.Pointer\) to int64\n\tcannot convert unsafe\.Pointer \(in X\) to int64 */)
}

// "x is a slice, T is a pointer-to-array type,
// and the slice and array types have identical element types."

func _[X ~[]E, T ~*[10]E, E any](x X) T { return T(x) }
func _[X ~[]E, T ~[10]E, E any](x X) T {
	return T(x /* ERROR cannot convert x \(variable of type X constrained by ~\[\]E\) to T\n\tcannot convert \[\]E \(in X\) to \[10\]E \(in T\) */)
}

// ----------------------------------------------------------------------------
// The following declarations can be replaced by the exported types of the
// constraints package once all builders support importing interfaces with
// type constraints.

type Signed interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64
}

type Unsigned interface {
	~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

type Integer interface {
	Signed | Unsigned
}

type Float interface {
	~float32 | ~float64
}

type Complex interface {
	~complex64 | ~complex128
}
