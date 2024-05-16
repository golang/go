// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is a subset of the tests in range.go for range over integers,
// with extra tests, and without the need for -goexperiment=range.

package p

// test framework assumes 64-bit int/uint sizes by default
const (
	maxInt  = 1<<63 - 1
	maxUint = 1<<64 - 1
)

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
	for i = range MyInt /* ERROR "cannot use MyInt(10) (constant 10 of type MyInt) as int value in range clause" */ (10) {
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

func issue65133() {
	for range maxInt {
	}
	for range maxInt /* ERROR "cannot use maxInt + 1 (untyped int constant 9223372036854775808) as int value in range clause (overflows)" */ + 1 {
	}
	for range maxUint /* ERROR "cannot use maxUint (untyped int constant 18446744073709551615) as int value in range clause (overflows)" */ {
	}

	for i := range maxInt {
		_ = i
	}
	for i := range maxInt /* ERROR "cannot use maxInt + 1 (untyped int constant 9223372036854775808) as int value in range clause (overflows)" */ + 1 {
		_ = i
	}
	for i := range maxUint /* ERROR "cannot use maxUint (untyped int constant 18446744073709551615) as int value in range clause (overflows)" */ {
		_ = i
	}

	var i int
	_ = i
	for i = range maxInt {
	}
	for i = range maxInt /* ERROR "cannot use maxInt + 1 (untyped int constant 9223372036854775808) as int value in range clause (overflows)" */ + 1 {
	}
	for i = range maxUint /* ERROR "cannot use maxUint (untyped int constant 18446744073709551615) as int value in range clause (overflows)" */ {
	}

	var j uint
	_ = j
	for j = range maxInt {
	}
	for j = range maxInt + 1 {
	}
	for j = range maxUint {
	}
	for j = range maxUint /* ERROR "cannot use maxUint + 1 (untyped int constant 18446744073709551616) as uint value in range clause (overflows)" */ + 1 {
	}

	for range 256 {
	}
	for _ = range 256 {
	}
	for i = range 256 {
	}
	for i := range 256 {
		_ = i
	}

	var u8 uint8
	_ = u8
	for u8 = range - /* ERROR "cannot use -1 (untyped int constant) as uint8 value in range clause (overflows)" */ 1 {
	}
	for u8 = range 0 {
	}
	for u8 = range 255 {
	}
	for u8 = range 256 /* ERROR "cannot use 256 (untyped int constant) as uint8 value in range clause (overflows)" */ {
	}
}

func issue64471() {
	for i := range 'a' {
		var _ *rune = &i // ensure i has type rune
	}
}

func issue66561() {
	for range 10.0 /* ERROR "cannot range over 10.0 (untyped float constant 10)" */ {
	}
	for range 1e3 /* ERROR "cannot range over 1e3 (untyped float constant 1000)" */ {
	}
	for range 1 /* ERROR "cannot range over 1 + 0i (untyped complex constant (1 + 0i))" */ + 0i {
	}

	for range 1.1 /* ERROR "cannot range over 1.1 (untyped float constant)" */ {
	}
	for range 1i /* ERROR "cannot range over 1i (untyped complex constant (0 + 1i))" */ {
	}

	for i := range 10.0 /* ERROR "cannot range over 10.0 (untyped float constant 10)" */ {
		_ = i
	}
	for i := range 1e3 /* ERROR "cannot range over 1e3 (untyped float constant 1000)" */ {
		_ = i
	}
	for i := range 1 /* ERROR "cannot range over 1 + 0i (untyped complex constant (1 + 0i))" */ + 0i {
		_ = i
	}

	for i := range 1.1 /* ERROR "cannot range over 1.1 (untyped float constant)" */ {
		_ = i
	}
	for i := range 1i /* ERROR "cannot range over 1i (untyped complex constant (0 + 1i))" */ {
		_ = i
	}

	var j float64
	_ = j
	for j /* ERROR "cannot use iteration variable of type float64" */ = range 1 {
	}
	for j = range 1.1 /* ERROR "cannot range over 1.1 (untyped float constant)" */ {
	}
	for j = range 10.0 /* ERROR "cannot range over 10.0 (untyped float constant 10)" */ {
	}

	// There shouldn't be assignment errors if there are more iteration variables than permitted.
	var i int
	_ = i
	for i, j /* ERROR "range over 10 (untyped int constant) permits only one iteration variable" */ = range 10 {
	}
}

func issue67027() {
	var i float64
	_ = i
	for i /* ERROR "cannot use iteration variable of type float64" */ = range 10 {
	}
}
