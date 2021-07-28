// compile -G=3

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests built-in calls on generic types.

// derived and expanded from cmd/compile/internal/types2/testdata/check/builtins.go2

package builtins

// close

type C0 interface{ int }
type C1 interface{ chan int }
type C2 interface{ chan int | <-chan int }
type C3 interface{ chan int | chan float32 }
type C4 interface{ chan int | chan<- int }
type C5[T any] interface{ ~chan T | chan<- T }

func _[T C1](ch T) {
	close(ch)
}

func _[T C3](ch T) {
	close(ch)
}

func _[T C4](ch T) {
	close(ch)
}

func _[T C5[X], X any](ch T) {
	close(ch)
}

// delete

type M0 interface{ int }
type M1 interface{ map[string]int }
type M2 interface {
	map[string]int | map[string]float64
}
type M3 interface{ map[string]int | map[rune]int }
type M4[K comparable, V any] interface{ map[K]V | map[rune]V }

func _[T M1](m T) {
	delete(m, "foo")
}

func _[T M2](m T) {
	delete(m, "foo")
}

func _[T M4[rune, V], V any](m T) {
	delete(m, 'k')
}

// make

type Bmc interface {
	~map[rune]string | ~chan int
}

type Bms interface {
	~map[string]int | ~[]int
}

type Bcs interface {
	~chan bool | ~[]float64
}

type Bss interface {
	~[]int | ~[]string
}

func _[T Bmc]() {
	_ = make(T)
	_ = make(T, 10)
}

func _[T Bms]() {
	_ = make(T, 10)
}

func _[T Bcs]() {
	_ = make(T, 10)
}

func _[T Bss]() {
	_ = make(T, 10)
	_ = make(T, 10, 20)
}

// len/cap

type Slice[T any] interface {
	type []T
}

func _[T any, S Slice[T]]() {
	x := make(S, 5, 10)
	_ = len(x)
	_ = cap(x)
}

// append

func _[T any, S Slice[T]]() {
	x := make(S, 5)
	y := make(S, 2)
	var z T
	_ = append(x, y...)
	_ = append(x, z)
}
