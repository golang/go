// compile

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

func f1[T C1](ch T) {
	close(ch)
}

func f2[T C3](ch T) {
	close(ch)
}

func f3[T C4](ch T) {
	close(ch)
}

func f4[T C5[X], X any](ch T) {
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

func g1[T M1](m T) {
	delete(m, "foo")
}

func g2[T M2](m T) {
	delete(m, "foo")
}

func g3[T M4[rune, V], V any](m T) {
	delete(m, 'k')
}

// make

func m1[
	S1 interface{ []int },
	S2 interface{ []int | chan int },

	M1 interface{ map[string]int },
	M2 interface{ map[string]int | chan int },

	C1 interface{ chan int },
	C2 interface{ chan int | chan string },
]() {
	_ = make([]int, 10)
	_ = make(m1S0, 10)
	_ = make(S1, 10)
	_ = make(S1, 10, 20)

	_ = make(map[string]int)
	_ = make(m1M0)
	_ = make(M1)
	_ = make(M1, 10)

	_ = make(chan int)
	_ = make(m1C0)
	_ = make(C1)
	_ = make(C1, 10)
}
// TODO: put these type declarations back inside m1 when issue 47631 is fixed.
type m1S0 []int
type m1M0 map[string]int
type m1C0 chan int

// len/cap

type Slice[T any] interface {
	[]T
}

func c1[T any, S Slice[T]]() {
	x := make(S, 5, 10)
	_ = len(x)
	_ = cap(x)
}

// append

func a1[T any, S Slice[T]]() {
	x := make(S, 5)
	y := make(S, 2)
	var z T
	_ = append(x, y...)
	_ = append(x, z)
}
