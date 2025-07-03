// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comparisons

type (
	B int // basic type representative
	A [10]func()
	L []byte
	S struct{ f []byte }
	P *S
	F func()
	I interface{}
	M map[string]int
	C chan int
)

var (
	b B
	a A
	l L
	s S
	p P
	f F
	i I
	m M
	c C
)

func _() {
	_ = nil == nil // ERROR "operator == not defined on untyped nil"
	_ = b == b
	_ = a /* ERROR "A cannot be compared" */ == a
	_ = l /* ERROR "slice can only be compared to nil" */ == l
	_ = s /* ERROR "struct containing []byte cannot be compared" */ == s
	_ = p == p
	_ = f /* ERROR "func can only be compared to nil" */ == f
	_ = i == i
	_ = m /* ERROR "map can only be compared to nil" */ == m
	_ = c == c

	_ = b == nil /* ERROR "mismatched types" */
	_ = a == nil /* ERROR "mismatched types" */
	_ = l == nil
	_ = s == nil /* ERROR "mismatched types" */
	_ = p == nil
	_ = f == nil
	_ = i == nil
	_ = m == nil
	_ = c == nil

	_ = nil /* ERROR "operator < not defined on untyped nil" */ < nil
	_ = b < b
	_ = a /* ERROR "operator < not defined on array" */ < a
	_ = l /* ERROR "operator < not defined on slice" */ < l
	_ = s /* ERROR "operator < not defined on struct" */ < s
	_ = p /* ERROR "operator < not defined on pointer" */ < p
	_ = f /* ERROR "operator < not defined on func" */ < f
	_ = i /* ERROR "operator < not defined on interface" */ < i
	_ = m /* ERROR "operator < not defined on map" */ < m
	_ = c /* ERROR "operator < not defined on chan" */ < c
}

func _[
	B int,
	A [10]func(),
	L []byte,
	S struct{ f []byte },
	P *S,
	F func(),
	I interface{},
	J comparable,
	M map[string]int,
	C chan int,
](
	b B,
	a A,
	l L,
	s S,
	p P,
	f F,
	i I,
	j J,
	m M,
	c C,
) {
	_ = b == b
	_ = a /* ERROR "incomparable types in type set" */ == a
	_ = l /* ERROR "incomparable types in type set" */ == l
	_ = s /* ERROR "incomparable types in type set" */ == s
	_ = p == p
	_ = f /* ERROR "incomparable types in type set" */ == f
	_ = i /* ERROR "incomparable types in type set" */ == i
	_ = j == j
	_ = m /* ERROR "incomparable types in type set" */ == m
	_ = c == c

	_ = b == nil /* ERROR "mismatched types" */
	_ = a == nil /* ERROR "mismatched types" */
	_ = l == nil
	_ = s == nil /* ERROR "mismatched types" */
	_ = p == nil
	_ = f == nil
	_ = i == nil /* ERROR "mismatched types" */
	_ = j == nil /* ERROR "mismatched types" */
	_ = m == nil
	_ = c == nil

	_ = b < b
	_ = a /* ERROR "type parameter A cannot use operator <" */ < a
	_ = l /* ERROR "type parameter L cannot use operator <" */ < l
	_ = s /* ERROR "type parameter S cannot use operator <" */ < s
	_ = p /* ERROR "type parameter P cannot use operator <" */ < p
	_ = f /* ERROR "type parameter F cannot use operator <" */ < f
	_ = i /* ERROR "type parameter I cannot use operator <" */ < i
	_ = j /* ERROR "type parameter J cannot use operator <" */ < j
	_ = m /* ERROR "type parameter M cannot use operator <" */ < m
	_ = c /* ERROR "type parameter C cannot use operator <" */ < c
}
