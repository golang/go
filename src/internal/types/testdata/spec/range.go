// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type MyInt int32
type MyBool bool
type MyString string
type MyFunc1 func(func(int) bool)
type MyFunc2 func(int) bool
type MyFunc3 func(MyFunc2)

type T struct{}

func (*T) PM() {}
func (T) M()   {}

func f1()                             {}
func f2(func())                       {}
func f4(func(int) bool)               {}
func f5(func(int, string) bool)       {}
func f7(func(int) MyBool)             {}
func f8(func(MyInt, MyString) MyBool) {}

func test() {
	// TODO: Would be nice to test 'for range T.M' and 'for range (*T).PM' directly,
	// but there is no gofmt-friendly way to write the error pattern in the right place.
	m1 := T.M
	for range m1 /* ERROR "cannot range over m1 (variable of type func(T)): func must be func(yield func(...) bool): argument is not func" */ {
	}
	m2 := (*T).PM
	for range m2 /* ERROR "cannot range over m2 (variable of type func(*T)): func must be func(yield func(...) bool): argument is not func" */ {
	}
	for range f1 /* ERROR "cannot range over f1 (value of type func()): func must be func(yield func(...) bool): wrong argument count" */ {
	}
	for range f2 /* ERROR "cannot range over f2 (value of type func(func())): func must be func(yield func(...) bool): yield func does not return bool" */ {
	}
	for range f4 {
	}
	for _ = range f4 {
	}
	for _, _ = range f5 {
	}
	for _ = range f7 {
	}
	for _, _ = range f8 {
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
	for _ = range MyFunc1(nil) {
	}
	for _ = range MyFunc3(nil) {
	}
	for _ = range (func(MyFunc2))(nil) {
	}

	var i int
	var s string
	var mi MyInt
	var ms MyString
	for i := range f4 {
		_ = i
	}
	for i = range f4 {
		_ = i
	}
	for i, s := range f5 {
		_, _ = i, s
	}
	for i, s = range f5 {
		_, _ = i, s
	}
	for i, _ := range f5 {
		_ = i
	}
	for i, _ = range f5 {
		_ = i
	}
	for i := range f7 {
		_ = i
	}
	for i = range f7 {
		_ = i
	}
	for mi, _ := range f8 {
		_ = mi
	}
	for mi, _ = range f8 {
		_ = mi
	}
	for mi, ms := range f8 {
		_, _ = mi, ms
	}
	for i /* ERROR "cannot use i (value of int32 type MyInt) as int value in assignment" */, s /* ERROR "cannot use s (value of string type MyString) as string value in assignment" */ = range f8 {
		_, _ = mi, ms
	}
	for mi, ms := range f8 {
		i, s = mi /* ERROR "cannot use mi (variable of int32 type MyInt) as int value in assignment" */, ms /* ERROR "cannot use ms (variable of string type MyString) as string value in assignment" */
	}
	for mi, ms = range f8 {
		_, _ = mi, ms
	}

	for i := range 10 {
		_ = i
	}
	for i = range 10 {
		_ = i
	}
	for i, j /* ERROR "range over 10 (untyped int constant) permits only one iteration variable" */ := range 10 {
		_, _ = i, j
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

func _[T any](x func(func(T) bool)) {
	for _ = range x { // ok
	}
}

func _[T ~func(func(int) bool)](x T) {
	for _ = range x { // ok
	}
}

// go.dev/issue/65236

func seq0(func() bool) {}
func seq1(func(int) bool) {}
func seq2(func(int, int) bool) {}

func _() {
	for range seq0 {
	}
	for _ /* ERROR "range over seq0 (value of type func(func() bool)) permits no iteration variables" */ = range seq0 {
	}

	for range seq1 {
	}
	for _ = range seq1 {
	}
	for _, _ /* ERROR "range over seq1 (value of type func(func(int) bool)) permits only one iteration variable" */ = range seq1 {
	}

	for range seq2 {
	}
	for _ = range seq2 {
	}
	for _, _ = range seq2 {
	}
	// Note: go/types reports a parser error in this case, hence the different error messages.
	for _, _, _ /* ERRORx "(range clause permits at most two iteration variables|expected at most 2 expressions)" */ = range seq2 {
	}
}
