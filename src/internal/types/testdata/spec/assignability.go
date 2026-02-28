// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package assignability

// See the end of this package for the declarations
// of the types and variables used in these tests.

// "x's type is identical to T"
func _[TP any](X TP) {
	b = b
	a = a
	l = l
	s = s
	p = p
	f = f
	i = i
	m = m
	c = c
	d = d

	B = B
	A = A
	L = L
	S = S
	P = P
	F = F
	I = I
	M = M
	C = C
	D = D
	X = X
}

// "x's type V and T have identical underlying types
// and at least one of V or T is not a named type."
// (here a named type is a type with a name)
func _[TP1, TP2 Interface](X1 TP1, X2 TP2) {
	b = B // ERRORx `cannot use B .* as (int|_Basic.*) value`
	a = A
	l = L
	s = S
	p = P
	f = F
	i = I
	m = M
	c = C
	d = D

	B = b // ERRORx `cannot use b .* as Basic value`
	A = a
	L = l
	S = s
	P = p
	F = f
	I = i
	M = m
	C = c
	D = d
	X1 = i  // ERRORx `cannot use i .* as TP1 value`
	X1 = X2 // ERRORx `cannot use X2 .* as TP1 value`
}

// "T is an interface type and x implements T and T is not a type parameter"
func _[TP Interface](X TP) {
	i = d // ERROR "missing method m"
	i = D
	i = X
	X = i // ERRORx `cannot use i .* as TP value`
}

// "x is a bidirectional channel value, T is a channel type, x's type V and T have identical element types, and at least one of V or T is not a named type"
// (here a named type is a type with a name)
type (
	_SendChan = chan<- int
	_RecvChan = <-chan int

	SendChan _SendChan
	RecvChan _RecvChan
)

func _[
	_CC ~_Chan,
	_SC ~_SendChan,
	_RC ~_RecvChan,

	CC Chan,
	SC SendChan,
	RC RecvChan,
]() {
	var (
		_ _SendChan = c
		_ _RecvChan = c
		_ _Chan     = c

		_ _SendChan = C
		_ _RecvChan = C
		_ _Chan     = C

		_ SendChan = c
		_ RecvChan = c
		_ Chan     = c

		_ SendChan = C // ERRORx `cannot use C .* as SendChan value`
		_ RecvChan = C // ERRORx `cannot use C .* as RecvChan value`
		_ Chan     = C
		_ Chan     = make /* ERRORx `cannot use make\(chan Basic\) .* as Chan value` */ (chan Basic)
	)

	var (
		_ _CC = C // ERRORx `cannot use C .* as _CC value`
		_ _SC = C // ERRORx `cannot use C .* as _SC value`
		_ _RC = C // ERRORx `cannot use C .* as _RC value`

		_ CC = _CC /* ERRORx `cannot use _CC\(nil\) .* as CC value` */ (nil)
		_ SC = _CC /* ERRORx `cannot use _CC\(nil\) .* as SC value` */ (nil)
		_ RC = _CC /* ERRORx `cannot use _CC\(nil\) .* as RC value` */ (nil)

		_ CC = C // ERRORx `cannot use C .* as CC value`
		_ SC = C // ERRORx `cannot use C .* as SC value`
		_ RC = C // ERRORx `cannot use C .* as RC value`
	)
}

// "x's type V is not a named type and T is a type parameter, and x is assignable to each specific type in T's type set."
func _[
	TP0 any,
	TP1 ~_Chan,
	TP2 ~chan int | ~chan byte,
]() {
	var (
		_ TP0 = c // ERRORx `cannot use c .* as TP0 value`
		_ TP0 = C // ERRORx `cannot use C .* as TP0 value`
		_ TP1 = c
		_ TP1 = C // ERRORx `cannot use C .* as TP1 value`
		_ TP2 = c // ERRORx `.* cannot assign (chan int|_Chan.*) to chan byte`
	)
}

// "x's type V is a type parameter and T is not a named type, and values x' of each specific type in V's type set are assignable to T."
func _[
	TP0 Interface,
	TP1 ~_Chan,
	TP2 ~chan int | ~chan byte,
](X0 TP0, X1 TP1, X2 TP2) {
	i = X0
	I = X0
	c = X1
	C = X1 // ERRORx `cannot use X1 .* as Chan value`
	c = X2 // ERRORx `.* cannot assign chan byte \(in TP2\) to (chan int|_Chan.*)`
}

// "x is the predeclared identifier nil and T is a pointer, function, slice, map, channel, or interface type"
func _[TP Interface](X TP) {
	b = nil // ERROR "cannot use nil"
	a = nil // ERROR "cannot use nil"
	l = nil
	s = nil // ERROR "cannot use nil"
	p = nil
	f = nil
	i = nil
	m = nil
	c = nil
	d = nil // ERROR "cannot use nil"

	B = nil // ERROR "cannot use nil"
	A = nil // ERROR "cannot use nil"
	L = nil
	S = nil // ERROR "cannot use nil"
	P = nil
	F = nil
	I = nil
	M = nil
	C = nil
	D = nil // ERROR "cannot use nil"
	X = nil // ERROR "cannot use nil"
}

// "x is an untyped constant representable by a value of type T"
func _[
	Int8 ~int8,
	Int16 ~int16,
	Int32 ~int32,
	Int64 ~int64,
	Int8_16 ~int8 | ~int16,
](
	i8 Int8,
	i16 Int16,
	i32 Int32,
	i64 Int64,
	i8_16 Int8_16,
) {
	b = 42
	b = 42.0
	// etc.

	i8 = -1 << 7
	i8 = 1<<7 - 1
	i16 = -1 << 15
	i16 = 1<<15 - 1
	i32 = -1 << 31
	i32 = 1<<31 - 1
	i64 = -1 << 63
	i64 = 1<<63 - 1

	i8_16 = -1 << 7
	i8_16 = 1<<7 - 1
	i8_16 = - /* ERRORx `cannot use .* as Int8_16` */ 1 << 15
	i8_16 = 1 /* ERRORx `cannot use .* as Int8_16` */ <<15 - 1
}

// proto-types for tests

type (
	_Basic     = int
	_Array     = [10]int
	_Slice     = []int
	_Struct    = struct{ f int }
	_Pointer   = *int
	_Func      = func(x int) string
	_Interface = interface{ m() int }
	_Map       = map[string]int
	_Chan      = chan int

	Basic     _Basic
	Array     _Array
	Slice     _Slice
	Struct    _Struct
	Pointer   _Pointer
	Func      _Func
	Interface _Interface
	Map       _Map
	Chan      _Chan
	Defined   _Struct
)

func (Defined) m() int

// proto-variables for tests

var (
	b _Basic
	a _Array
	l _Slice
	s _Struct
	p _Pointer
	f _Func
	i _Interface
	m _Map
	c _Chan
	d _Struct

	B Basic
	A Array
	L Slice
	S Struct
	P Pointer
	F Func
	I Interface
	M Map
	C Chan
	D Defined
)
