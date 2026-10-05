// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package untyped_lit

// Type definitions used across tests.

type (
	_Basic     = int
	_Array     = [10]int
	_Slice     = []int
	_Struct    = struct{ f int }
	_Pointer   = *int
	_PointerS  = *_Struct
	_Func      = func(int) string
	_Interface = interface{ m() int }
	_Map       = map[string]int
	_Chan      = chan int

	Basic     _Basic
	Array     _Array
	Slice     _Slice
	Struct    _Struct
	Pointer   _Pointer
	PointerS  _PointerS
	Func      _Func
	Interface _Interface
	Map       _Map
	Chan      _Chan

	GArray[E any]             [10]E
	GSlice[E any]             []E
	GStruct[E any]            struct{ f E }
	GMap[K comparable, V any] map[K]V
)

// ----------------------------------------------------------------------------
// Assignability across target types T:
// x is an untyped composite literal and T(x) is a valid composite literal of type T.

func _() {
	// alias / unnamed types
	var (
		_ _Basic     = { /* ERROR "invalid composite literal type _Basic" */ }
		_ _Array     = {}
		_ _Array     = {1, 2, 3}
		_ _Slice     = {}
		_ _Slice     = {1, 2, 3}
		_ _Struct    = {}
		_ _Struct    = {1}
		_ _Struct    = {f: 1}
		_ _Pointer   = { /* ERROR "invalid composite literal type _Pointer" */ }
		_ _PointerS  = {}
		_ _PointerS  = {1}
		_ _PointerS  = {f: 1}
		_ _Func      = { /* ERROR "invalid composite literal type _Func" */ }
		_ _Interface = { /* ERROR "invalid composite literal type _Interface" */ }
		_ any        = { /* ERROR "invalid composite literal type any" */ }
		_ _Map       = {}
		_ _Map       = {"a": 1}
		_ _Chan      = { /* ERROR "invalid composite literal type _Chan" */ }
	)

	// defined types
	var (
		_ Basic     = { /* ERROR "invalid composite literal type Basic" */ }
		_ Array     = {}
		_ Array     = {1, 2, 3}
		_ Slice     = {}
		_ Slice     = {1, 2, 3}
		_ Struct    = {}
		_ Struct    = {1}
		_ Struct    = {f: 1}
		_ Pointer   = { /* ERROR "invalid composite literal type Pointer" */ }
		_ PointerS  = {}
		_ PointerS  = {1}
		_ PointerS  = {f: 1}
		_ Func      = { /* ERROR "invalid composite literal type Func" */ }
		_ Interface = { /* ERROR "invalid composite literal type Interface" */ }
		_ Map       = {}
		_ Map       = {"a": 1}
		_ Chan      = { /* ERROR "invalid composite literal type Chan" */ }
	)

	// instantiated generic types
	var (
		_ GArray[int]       = {}
		_ GArray[int]       = {1, 2, 3}
		_ GSlice[int]       = {}
		_ GSlice[int]       = {1, 2, 3}
		_ GStruct[int]      = {}
		_ GStruct[int]      = {1}
		_ GStruct[int]      = {f: 1}
		_ GMap[string, int] = {}
		_ GMap[string, int] = {"a": 1}
	)
}

// Type parameter target types T.
func _[
	P0 any,
	PB ~int,
	PA ~[10]int,
	PA2 [10]int | Array,
	PA_bad [10]int | [20]int,
	PL ~[]int,
	PL2 []int | Slice,
	PLA_bad []int | [10]int,
	PS ~struct{ f int },
	PS2 struct{ f int } | Struct,
	PS_bad struct{ f int } | struct{ g int },
	PP ~*int,
	PPS ~*_Struct,
	PF ~func(int) string,
	PI Interface,
	PM ~map[string]int,
	PM2 map[string]int | Map,
	PC ~chan int,
]() {
	var (
		_ P0      = { /* ERROR "invalid composite literal type P0" */ }
		_ PB      = { /* ERROR "invalid composite literal type PB" */ }
		_ PA      = {}
		_ PA      = {1, 2, 3}
		_ PA2     = {0: 1, 9: 2}
		_ PA_bad  = { /* ERROR "invalid composite literal type PA_bad (no common underlying type)" */ }
		_ PL      = {}
		_ PL      = {1, 2, 3}
		_ PL2     = {0: 1, 2: 3}
		_ PLA_bad = { /* ERROR "invalid composite literal type PLA_bad (no common underlying type)" */ }
		_ PS      = {}
		_ PS      = {1}
		_ PS      = {f: 1}
		_ PS2     = {f: 1}
		_ PS_bad  = { /* ERROR "invalid composite literal type PS_bad (no common underlying type)" */ }
		_ PP      = { /* ERROR "invalid composite literal type PP" */ }
		_ PPS     = {}
		_ PPS     = {f: 1}
		_ PF      = { /* ERROR "invalid composite literal type PF" */ }
		_ PI      = { /* ERROR "invalid composite literal type PI" */ }
		_ PM      = {}
		_ PM      = {"a": 1}
		_ PM2     = {"a": 1}
		_ PC      = { /* ERROR "invalid composite literal type PC" */ }
	)
}

// ----------------------------------------------------------------------------
// Validity of literal elements for each composite type kind.

func _() {
	// struct literals (including promoted fields)
	type Inner struct {
		a, b int
	}
	type Outer struct {
		Inner
		x int
		s string
	}

	var (
		_ Outer = {}
		_ Outer = {{1, 2}, 3, "foo"}
		_ Outer = {Inner: {1, 2}, x: 3, s: "foo"}
		_ Outer = {a: 1, b: 2, x: 3, s: "foo"}
		_ Outer = {{1, 2}, 3} // ERROR "too few values"
		_ Outer = {{1, 2}, 3, "foo", 4 /* ERROR "too many values" */}
		_ Outer = {x: 3, "foo" /* ERROR "mixture of field:value and value elements" */}
		_ Outer = {{1, 2}, x /* ERROR "mixture of field:value and value elements" */ : 3, "foo"}
		_ Outer = {z /* ERROR "unknown field z" */ : 1}
		_ Outer = {x: 1, x /* ERROR "duplicate field name x" */ : 2}
		_ Outer = {Inner: {1, 2}, a /* ERROR "cannot specify promoted field a and enclosing embedded field Inner" */ : 1}
		_ Outer = {x: "foo" /* ERRORx `cannot use "foo" .* as int value` */}
	)

	// array literals
	var (
		_ [3]int = {}
		_ [3]int = {1, 2, 3}
		_ [3]int = {0: 1, 2: 3}
		_ [3]int = {1: 2, 3}
		_ [3]int = {1, 2, 3, 4 /* ERROR "index 3 is out of bounds" */}
		_ [3]int = {3 /* ERROR "index 3 out of bounds" */ : 1}
		_ [3]int = {- /* ERROR "index -1 (constant of type int) must not be negative" */ 1: 1}
		_ [3]int = {"a" /* ERRORx `cannot convert "a" .* to type int` */ : 1}
		_ [3]int = {0: 1, 0 /* ERROR "duplicate index 0" */ : 2}
		_ [3]int = {1: 1, 0: 2, 3 /* ERROR "duplicate index 1" */}
		_ [3]int = {"a" /* ERRORx `cannot use "a" .* as int value` */}
	)

	// slice literals
	var (
		_ []int = {}
		_ []int = {1, 2, 3}
		_ []int = {0: 1, 10: 2}
		_ []int = {- /* ERROR "index -1 (constant of type int) must not be negative" */ 1: 1}
		_ []int = {"a" /* ERRORx `cannot convert "a" .* to type int` */ : 1}
		_ []int = {0: 1, 0 /* ERROR "duplicate index 0" */ : 2}
		_ []int = {"a" /* ERRORx `cannot use "a" .* as int value` */}
	)

	// map literals
	k := "key"
	var (
		_ map[string]int = {}
		_ map[string]int = {"a": 1, "b": 2}
		_ map[string]int = {k: 1}
		_ map[string]int = {1 /* ERROR "missing key in map literal" */}
		_ map[string]int = {"a": 1, "a" /* ERROR `duplicate key "a"` */ : 2}
		_ map[string]int = {1 /* ERRORx `cannot use 1 .* as string value` */ : 1}
		_ map[string]int = {"a": "b" /* ERRORx `cannot use "b" .* as int value` */}
	)
}

// ----------------------------------------------------------------------------
// Contexts where untyped composite literals may appear.

// 1. Variable declarations and short variable declarations
func _() {
	var _ = { /* ERROR "missing type in composite literal" */ }
	var _ = { /* ERROR "missing type in composite literal" */ 1, 2}
	var _ = { /* ERROR "missing type in composite literal" */ f: 1}

	var s Struct = {1}
	var s1, l1 = Struct{1}, Slice{1, 2}
	var s2, s3 Struct = {1}, {f: 2}
	_, _, _, _ = s, s1, s2, s3
	_ = l1

	x := { /* ERROR "missing type in composite literal" */ }
	_ = x

	// Redeclared variable in short variable declaration already has a type.
	var r Struct
	r, y := {1}, 42
	_, _ = r, y
}

// 2. Constant declarations
func _() {
	const _ Struct /* ERROR "invalid constant type Struct" */ = {}
	const _ int = { /* ERROR "missing type in composite literal" */ }
	const _ = { /* ERROR "missing type in composite literal" */ }
}

// 3. Assignments
func _() {
	_ = { /* ERROR "missing type in composite literal" */ }

	var (
		s   Struct
		a   Array
		l   Slice
		m   Map
		p   PointerS
		box struct{ s Struct }
		arr [1]Struct
		ptr *Struct
		mp  map[string]Struct
	)

	s = {}
	s = {1}
	s = {f: 1}
	a = {1, 2, 3, 9: 10}
	l = {1, 2, 3}
	m = {"a": 1, "b": 2}
	p = {f: 1}

	// multiple assignment
	s, l, m = {1}, {1, 2}, {"a": 1}

	// assignment to addressable operands and map index
	box.s = {1}
	arr[0] = {f: 2}
	*ptr = {3}
	mp["k"] = {4}

	_, _, _, _, _, _, _ = s, a, l, m, p, box, arr
}

// 4. Return statements
func _() Struct {
	return {}
	return {1}
	return {f: 1}
}

func _() (Struct, Slice, Map) {
	return {1}, {1, 2, 3}, {"a": 1}
}

func _() (s Struct, a Array, l Slice, m Map) {
	return {f: 1}, {0: 1}, {1, 2}, {"a": 1}
}

// 5. Function and method calls
type Recv struct{ x int }

func (Recv) m(s Struct, rest ...Slice) {}

func _() {
	f := func(s Struct, a Array, l Slice, m Map) {}
	f({}, {}, {}, {})
	f({1}, {1, 2}, {3, 4}, {"a": 5})

	fv := func(x int, rest ...Struct) {}
	fv(0)
	fv(0, {})
	fv(0, {1}, {f: 2})

	var r Recv
	r.m({1}, {1, 2}, {3, 4})
	Recv.m({1}, {2}, {1, 2}, {3, 4})
}

func fg[P any](x P)               {}
func fgc[P ~struct{ f int }](x P) {}

func _() {
	fg[Struct]({})
	fg[Struct]({1})
	fg[Slice]({1, 2, 3})
	fg[Map]({"a": 1})

	fgc[Struct]({1})
	fgc({f: 1})
}

// 6. Nested composite literals
func _() {
	type Nested struct {
		s Struct
		a [2]Struct
		l []Struct
		m map[Struct]Slice
		p *Struct
	}

	// typed outer literal with untyped inner literals
	_ = Nested{
		s: {1},
		a: {{1}, {f: 2}},
		l: {{1}, {f: 2}},
		m: {{1}: {1, 2}, {f: 2}: {}},
		p: {f: 3},
	}
	_ = Nested{
		{1},
		{{1}, {f: 2}},
		{{1}, {f: 2}},
		{{1}: {1, 2}, {f: 2}: {}},
		{f: 3},
	}

	// untyped outer literal with untyped inner literals
	var _ Nested = {
		s: {1},
		a: {{1}, {f: 2}},
		l: {{1}, {f: 2}},
		m: {{1}: {1, 2}, {f: 2}: {}},
		p: {f: 3},
	}
	var _ Nested = {
		{1},
		{{1}, {f: 2}},
		{{1}, {f: 2}},
		{{1}: {1, 2}, {f: 2}: {}},
		{f: 3},
	}

	// [...]T array literal with untyped element literals
	_ = [...]Struct{{}, {1}, {f: 2}}
	_ = [...]*Struct{{}, {1}, {f: 2}}

	// generic function value inside untyped struct literal (reverse type inference)
	type FuncHolder struct {
		fn func(int)
	}
	var _ FuncHolder = {fg}
	var _ FuncHolder = {fn: fg}
}

// 7. Map index expressions
func _[M ~map[Struct]Slice](gm M) {
	var m map[Struct]Slice
	_ = m[{}]
	_ = m[{1}]
	_ = m[{f: 1}]
	v, ok := m[{f: 1}]
	_, _ = v, ok
	m[{1}] = {1, 2, 3}

	var counts map[Struct]int
	counts[{1}]++
	counts[{f: 2}] += 10

	_ = gm[{}]
	gm[{1}] = {1, 2, 3}
}

// 8. Channel send statements
func _[C ~chan Struct](gch C) {
	var ch chan Struct
	var sch chan<- Struct
	ch <- {}
	ch <- {1}
	sch <- {f: 1}
	gch <- {1}

	select {
	case ch <- {1}:
	case sch <- {f: 2}:
	default:
	}
}

// 9. Conversions T(x)
func _[P ~struct{ f int }]() {
	_ = _Array({1, 2, 3})
	_ = _Slice({1, 2, 3})
	_ = _Struct({1})
	_ = _Struct({f: 1})
	_ = _PointerS({f: 1})
	_ = _Map({"a": 1})

	_ = Array({1, 2, 3})
	_ = Slice({1, 2, 3})
	_ = Struct({1})
	_ = Struct({f: 1})
	_ = PointerS({f: 1})
	_ = Map({"a": 1})

	_ = ([3]int)({1, 2, 3})
	_ = ([]int)({1, 2, 3})
	_ = (struct{ f int })({f: 1})
	_ = (*struct{ f int })({f: 1})
	_ = (map[string]int)({"a": 1})

	_ = P({})
	_ = P({1})
	_ = P({f: 1})

	// selector, indexing, and slicing on converted untyped composite literals
	_ = Struct({f: 1}).f
	_ = Slice({1, 2, 3})[0]
	_ = Slice({1, 2, 3})[1:3]
	_ = Map({"a": 1})["a"]
}

// 10. Other expressions and statements (parentheses, unary &, variadic ..., comparisons, switch, built-ins)
func _() {
	var (
		s  Struct
		sl []Struct
		l  Slice
		m  map[Struct]int
	)

	// parenthesized untyped composite literals (type inference does not see through parentheses)
	var _ Struct = ({ /* ERROR "missing type in composite literal" */ 1})

	// address of untyped composite literal (target type does not propagate through &)
	var _ *Struct = &{ /* ERROR "missing type in composite literal" */ 1}

	// variadic call with ... (currently target type is element type rather than slice type)
	fv := func(x int, rest ...int) {}
	fv(0, { /* ERROR "invalid composite literal type int" */ 1, 2, 3}...)

	// comparisons
	// TODO(gri) these should work
	_ = s == { /* ERROR "missing type in composite literal" */ 1}
	_ = { /* ERROR "missing type in composite literal" */ 1} == s

	// switch cases
	// TODO(gri) these should work
	switch s {
	case { /* ERROR "missing type in composite literal" */ 1}:
	}

	// built-in functions where untyped composite literals are allowed
	// TODO(gri) there should work
	_ = append(sl, { /* ERROR "missing type in composite literal" */ 1})
	_ = copy(l, { /* ERROR "missing type in composite literal" */ 1, 2, 3})
	delete(m, { /* ERROR "missing type in composite literal" */ 1})

	// built-in functions where untyped composite literals are not allowed
	_ = new({ /* ERROR "missing type in composite literal" */ })
	_ = len({ /* ERROR "missing type in composite literal" */ })
	_ = cap({ /* ERROR "missing type in composite literal" */ })
	clear({ /* ERROR "missing type in composite literal" */ })
	panic({ /* ERROR "missing type in composite literal" */ })
	print({ /* ERROR "missing type in composite literal" */ })
	println({ /* ERROR "missing type in composite literal" */ })
}
