// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

type C0 interface{ int }
type C1 interface{ chan int }
type C2 interface{ chan int | <-chan int }
type C3 interface{ chan int | chan float32 }
type C4 interface{ chan int | chan<- int }
type C5[T any] interface{ ~chan T | chan<- T }

func _[T any](ch T) {
	ch <- /* ERRORx `cannot send to ch .* no core type` */ 0
}

func _[T C0](ch T) {
	ch <- /* ERROR "cannot send to non-channel" */ 0
}

func _[T C1](ch T) {
	ch <- 0
}

func _[T C2](ch T) {
	ch  <-/* ERROR "cannot send to receive-only channel" */ 0
}

func _[T C3](ch T) {
	ch <- /* ERRORx `cannot send to ch .* no core type` */ 0
}

func _[T C4](ch T) {
	ch <- 0
}

func _[T C5[X], X any](ch T, x X) {
	ch <- x
}
