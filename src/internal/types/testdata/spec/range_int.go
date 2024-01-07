// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a subset of the tests in range.go for range over integers,
// with extra tests, and without the need for -goexperiment=range.

package p

type MyInt int32

func _() {
	for range -1 {
	}
	for range 0 {
	}
	for range 1 {
	}
	for range uint8(1) {
	}
	for range int64(1) {
	}
	for range MyInt(1) {
	}
	for range 'x' {
	}
	for range 1.0 /* ERROR "cannot range over 1.0 (untyped float constant 1)" */ {
	}

	var i int
	var mi MyInt
	for i := range 10 {
		_ = i
	}
	for i = range 10 {
		_ = i
	}
	for i, j /* ERROR "range over 10 (untyped int constant) permits only one iteration variable" */ := range 10 {
		_, _ = i, j
	}
	for i /* ERROR "cannot use i (value of type MyInt) as int value in assignment" */ = range MyInt(10) {
		_ = i
	}
	for mi := range MyInt(10) {
		_ = mi
	}
	for mi = range MyInt(10) {
		_ = mi
	}
}

func _[T int | string](x T) {
	for range x /* ERROR "cannot range over x (variable of type T constrained by int | string): no core type" */ {
	}
}

func _[T int | int64](x T) {
	for range x /* ERROR "cannot range over x (variable of type T constrained by int | int64): no core type" */ {
	}
}

func _[T ~int](x T) {
	for range x { // ok
	}
}
