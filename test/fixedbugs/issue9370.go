// errorcheck

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that concrete/interface comparisons are
// typechecked correctly by the compiler.

package main

type I interface {
	Method()
}

type C int

func (C) Method() {}

type G func()

func (G) Method() {}

var (
	e interface{}
	i I
	c C
	n int
	f func()
	g G
)

var (
	_ = e == c
	_ = e != c
	_ = e >= c // ERROR "invalid operation.*not defined"
	_ = c == e
	_ = c != e
	_ = c >= e // ERROR "invalid operation.*not defined"

	_ = i == c
	_ = i != c
	_ = i >= c // ERROR "invalid operation.*not defined"
	_ = c == i
	_ = c != i
	_ = c >= i // ERROR "invalid operation.*not defined"

	_ = e == n
	_ = e != n
	_ = e >= n // ERROR "invalid operation.*not defined"
	_ = n == e
	_ = n != e
	_ = n >= e // ERROR "invalid operation.*not defined"

	// i and n are not assignable to each other
	_ = i == n // ERROR "invalid operation.*mismatched types"
	_ = i != n // ERROR "invalid operation.*mismatched types"
	_ = i >= n // ERROR "invalid operation.*mismatched types"
	_ = n == i // ERROR "invalid operation.*mismatched types"
	_ = n != i // ERROR "invalid operation.*mismatched types"
	_ = n >= i // ERROR "invalid operation.*mismatched types"

	_ = e == 1
	_ = e != 1
	_ = e >= 1 // ERROR "invalid operation.*not defined"
	_ = 1 == e
	_ = 1 != e
	_ = 1 >= e // ERROR "invalid operation.*not defined"

	_ = i == 1 // ERROR "invalid operation.*mismatched types"
	_ = i != 1 // ERROR "invalid operation.*mismatched types"
	_ = i >= 1 // ERROR "invalid operation.*mismatched types"
	_ = 1 == i // ERROR "invalid operation.*mismatched types"
	_ = 1 != i // ERROR "invalid operation.*mismatched types"
	_ = 1 >= i // ERROR "invalid operation.*mismatched types"

	_ = e == f // ERROR "invalid operation.*not defined"
	_ = e != f // ERROR "invalid operation.*not defined"
	_ = e >= f // ERROR "invalid operation.*not defined"
	_ = f == e // ERROR "invalid operation.*not defined"
	_ = f != e // ERROR "invalid operation.*not defined"
	_ = f >= e // ERROR "invalid operation.*not defined"

	_ = i == f // ERROR "invalid operation.*mismatched types"
	_ = i != f // ERROR "invalid operation.*mismatched types"
	_ = i >= f // ERROR "invalid operation.*mismatched types"
	_ = f == i // ERROR "invalid operation.*mismatched types"
	_ = f != i // ERROR "invalid operation.*mismatched types"
	_ = f >= i // ERROR "invalid operation.*mismatched types"

	_ = e == g // ERROR "invalid operation.*not defined"
	_ = e != g // ERROR "invalid operation.*not defined"
	_ = e >= g // ERROR "invalid operation.*not defined"
	_ = g == e // ERROR "invalid operation.*not defined"
	_ = g != e // ERROR "invalid operation.*not defined"
	_ = g >= e // ERROR "invalid operation.*not defined"

	_ = i == g // ERROR "invalid operation.*not defined"
	_ = i != g // ERROR "invalid operation.*not defined"
	_ = i >= g // ERROR "invalid operation.*not defined"
	_ = g == i // ERROR "invalid operation.*not defined"
	_ = g != i // ERROR "invalid operation.*not defined"
	_ = g >= i // ERROR "invalid operation.*not defined"

	_ = _ == e // ERROR "cannot use _ as value"
	_ = _ == i // ERROR "cannot use _ as value"
	_ = _ == c // ERROR "cannot use _ as value"
	_ = _ == n // ERROR "cannot use _ as value"
	_ = _ == f // ERROR "cannot use _ as value"
	_ = _ == g // ERROR "cannot use _ as value"

	_ = e == _ // ERROR "cannot use _ as value"
	_ = i == _ // ERROR "cannot use _ as value"
	_ = c == _ // ERROR "cannot use _ as value"
	_ = n == _ // ERROR "cannot use _ as value"
	_ = f == _ // ERROR "cannot use _ as value"
	_ = g == _ // ERROR "cannot use _ as value"

	_ = _ == _ // ERROR "cannot use _ as value"

	_ = e ^ c // ERROR "invalid operation.*mismatched types"
	_ = c ^ e // ERROR "invalid operation.*mismatched types"
	_ = 1 ^ e // ERROR "invalid operation.*mismatched types"
	_ = e ^ 1 // ERROR "invalid operation.*mismatched types"
	_ = 1 ^ c
	_ = c ^ 1
)
