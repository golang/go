// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file tests built-in calls on generic types.

package builtins

import "unsafe"

// close

type C0 interface{ int }
type C1 interface{ chan int }
type C2 interface{ chan int | <-chan int }
type C3 interface{ chan int | chan float32 }
type C4 interface{ chan int | chan<- int }
type C5[T any] interface{ ~chan T | chan<- T }

func _[T any](ch T) {
	close(ch /* ERROR cannot close non-channel */)
}

func _[T C0](ch T) {
	close(ch /* ERROR cannot close non-channel */)
}

func _[T C1](ch T) {
	close(ch)
}

func _[T C2](ch T) {
	close(ch /* ERROR cannot close receive-only channel */)
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

// copy

func _[T any](x, y T) {
	copy(x /* ERROR copy expects slice arguments */ , y)
}

func _[T ~[]byte](x, y T) {
	copy(x, y)
	copy(x, "foo")
	copy("foo" /* ERROR expects slice arguments */ , y)

	var x2 []byte
	copy(x2, y) // element types are identical
	copy(y, x2) // element types are identical

	type myByte byte
	var x3 []myByte
	copy(x3 /* ERROR different element types */ , y)
	copy(y, x3 /* ERROR different element types */ )
}

func _[T ~[]E, E any](x T, y []E) {
	copy(x, y)
	copy(x /* ERROR different element types */ , "foo")
}

func _[T ~string](x []byte, y T) {
	copy(x, y)
	copy(y /* ERROR expects slice arguments */ , x)
}

func _[T ~[]byte|~string](x T, y []byte) {
	copy(x /* ERROR expects slice arguments */ , y)
	copy(y, x)
}

type L0 []int
type L1 []int

func _[T L0 | L1](x, y T) {
	copy(x, y)
}

// delete

type M0 interface{ int }
type M1 interface{ map[string]int }
type M2 interface { map[string]int | map[string]float64 }
type M3 interface{ map[string]int | map[rune]int }
type M4[K comparable, V any] interface{ map[K]V | map[rune]V }

func _[T any](m T) {
	delete(m /* ERROR not a map */, "foo")
}

func _[T M0](m T) {
	delete(m /* ERROR not a map */, "foo")
}

func _[T M1](m T) {
	delete(m, "foo")
}

func _[T M2](m T) {
	delete(m, "foo")
	delete(m, 0 /* ERROR cannot use .* as string */)
}

func _[T M3](m T) {
	delete(m /* ERROR must have identical key types */, "foo")
}

func _[T M4[rune, V], V any](m T) {
	delete(m, 'k')
}

func _[T M4[K, V], K comparable, V any](m T) {
	delete(m /* ERROR must have identical key types */, "foo")
}

// make

type myChan chan int

func _[
	S1 ~[]int,
	S2 ~[]int | ~chan int,

	M1 ~map[string]int,
	M2 ~map[string]int | ~chan int,

	C1 ~chan int,
	C2 ~chan int | ~chan string,
	C3 chan int | myChan, // single underlying type
]() {
	type S0 []int
	_ = make([]int, 10)
	_ = make(S0, 10)
	_ = make(S1, 10)
	_ = make /* ERROR not enough arguments */ ()
	_ = make /* ERROR expects 2 or 3 arguments */ (S1)
	_ = make(S1, 10, 20)
	_ = make /* ERROR expects 2 or 3 arguments */ (S1, 10, 20, 30)
	_ = make(S2 /* ERROR cannot make S2: no core type */ , 10)

	type M0 map[string]int
	_ = make(map[string]int)
	_ = make(M0)
	_ = make(M1)
	_ = make(M1, 10)
	_ = make/* ERROR expects 1 or 2 arguments */(M1, 10, 20)
	_ = make(M2 /* ERROR cannot make M2: no core type */ )

	type C0 chan int
	_ = make(chan int)
	_ = make(C0)
	_ = make(C1)
	_ = make(C1, 10)
	_ = make/* ERROR expects 1 or 2 arguments */(C1, 10, 20)
	_ = make(C2 /* ERROR cannot make C2: no core type */ )
	_ = make(C3)
}

// unsafe.Alignof

func _[T comparable]() {
	var (
		b int64
		a [10]T
		s struct{ f T }
		p *T
		l []T
		f func(T)
		i interface{ m() T }
		c chan T
		m map[T]T
		t T
	)

	const bb = unsafe.Alignof(b)
	assert(bb == 8)
	const _ = unsafe /* ERROR not constant */ .Alignof(a)
	const _ = unsafe /* ERROR not constant */ .Alignof(s)
	const pp = unsafe.Alignof(p)
	assert(pp == 8)
	const ll = unsafe.Alignof(l)
	assert(ll == 8)
	const ff = unsafe.Alignof(f)
	assert(ff == 8)
	const ii = unsafe.Alignof(i)
	assert(ii == 8)
	const cc = unsafe.Alignof(c)
	assert(cc == 8)
	const mm = unsafe.Alignof(m)
	assert(mm == 8)
	const _ = unsafe /* ERROR not constant */ .Alignof(t)
}

// unsafe.Offsetof

func _[T comparable]() {
	var (
		b struct{ _, f int64 }
		a struct{ _, f [10]T }
		s struct{ _, f struct{ f T } }
		p struct{ _, f *T }
		l struct{ _, f []T }
		f struct{ _, f func(T) }
		i struct{ _, f interface{ m() T } }
		c struct{ _, f chan T }
		m struct{ _, f map[T]T }
		t struct{ _, f T }
	)

	const bb = unsafe.Offsetof(b.f)
	assert(bb == 8)
	const _ = unsafe /* ERROR not constant */ .Alignof(a)
	const _ = unsafe /* ERROR not constant */ .Alignof(s)
	const pp = unsafe.Offsetof(p.f)
	assert(pp == 8)
	const ll = unsafe.Offsetof(l.f)
	assert(ll == 24)
	const ff = unsafe.Offsetof(f.f)
	assert(ff == 8)
	const ii = unsafe.Offsetof(i.f)
	assert(ii == 16)
	const cc = unsafe.Offsetof(c.f)
	assert(cc == 8)
	const mm = unsafe.Offsetof(m.f)
	assert(mm == 8)
	const _ = unsafe /* ERROR not constant */ .Alignof(t)
}

// unsafe.Sizeof

func _[T comparable]() {
	var (
		b int64
		a [10]T
		s struct{ f T }
		p *T
		l []T
		f func(T)
		i interface{ m() T }
		c chan T
		m map[T]T
		t T
	)

	const bb = unsafe.Sizeof(b)
	assert(bb == 8)
	const _ = unsafe /* ERROR not constant */ .Alignof(a)
	const _ = unsafe /* ERROR not constant */ .Alignof(s)
	const pp = unsafe.Sizeof(p)
	assert(pp == 8)
	const ll = unsafe.Sizeof(l)
	assert(ll == 24)
	const ff = unsafe.Sizeof(f)
	assert(ff == 8)
	const ii = unsafe.Sizeof(i)
	assert(ii == 16)
	const cc = unsafe.Sizeof(c)
	assert(cc == 8)
	const mm = unsafe.Sizeof(m)
	assert(mm == 8)
	const _ = unsafe /* ERROR not constant */ .Alignof(t)
}
