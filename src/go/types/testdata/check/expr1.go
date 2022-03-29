// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// binary expressions

package expr1

type mybool bool

func _(x, y bool, z mybool) {
	x = x || y
	x = x || true
	x = x || false
	x = x && y
	x = x && true
	x = x && false

	z = z /* ERROR mismatched types */ || y
	z = z || true
	z = z || false
	z = z /* ERROR mismatched types */ && y
	z = z && true
	z = z && false
}

type myint int

func _(x, y int, z myint) {
	x = x + 1
	x = x + 1.0
	x = x + 1.1 // ERROR truncated to int
	x = x + y
	x = x - y
	x = x * y
	x = x / y
	x = x % y
	x = x << y
	x = x >> y

	z = z + 1
	z = z + 1.0
	z = z + 1.1 // ERROR truncated to int
	z = z /* ERROR mismatched types */ + y
	z = z /* ERROR mismatched types */ - y
	z = z /* ERROR mismatched types */ * y
	z = z /* ERROR mismatched types */ / y
	z = z /* ERROR mismatched types */ % y
	z = z << y
	z = z >> y
}

type myuint uint

func _(x, y uint, z myuint) {
	x = x + 1
	x = x + - /* ERROR overflows uint */ 1
	x = x + 1.0
	x = x + 1.1 // ERROR truncated to uint
	x = x + y
	x = x - y
	x = x * y
	x = x / y
	x = x % y
	x = x << y
	x = x >> y

	z = z + 1
	z = x + - /* ERROR overflows uint */ 1
	z = z + 1.0
	z = z + 1.1 // ERROR truncated to uint
	z = z /* ERROR mismatched types */ + y
	z = z /* ERROR mismatched types */ - y
	z = z /* ERROR mismatched types */ * y
	z = z /* ERROR mismatched types */ / y
	z = z /* ERROR mismatched types */ % y
	z = z << y
	z = z >> y
}

type myfloat64 float64

func _(x, y float64, z myfloat64) {
	x = x + 1
	x = x + -1
	x = x + 1.0
	x = x + 1.1
	x = x + y
	x = x - y
	x = x * y
	x = x / y
	x = x /* ERROR not defined */ % y
	x = x /* ERROR operand x .* must be integer */ << y
	x = x /* ERROR operand x .* must be integer */ >> y

	z = z + 1
	z = z + -1
	z = z + 1.0
	z = z + 1.1
	z = z /* ERROR mismatched types */ + y
	z = z /* ERROR mismatched types */ - y
	z = z /* ERROR mismatched types */ * y
	z = z /* ERROR mismatched types */ / y
	z = z /* ERROR mismatched types */ % y
	z = z /* ERROR operand z .* must be integer */ << y
	z = z /* ERROR operand z .* must be integer */ >> y
}

type mystring string

func _(x, y string, z mystring) {
	x = x + "foo"
	x = x /* ERROR not defined */ - "foo"
	x = x /* ERROR mismatched types string and untyped int */ + 1
	x = x + y
	x = x /* ERROR not defined */ - y
	x = x /* ERROR mismatched types string and untyped int */* 10
}

func f() (a, b int) { return }

func _(x int) {
	_ = f /* ERROR 2-valued f */ () + 1
	_ = x + f /* ERROR 2-valued f */ ()
	_ = f /* ERROR 2-valued f */ () + f
	_ = f /* ERROR 2-valued f */ () + f /* ERROR 2-valued f */ ()
}
