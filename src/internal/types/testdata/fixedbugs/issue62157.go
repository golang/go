// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func f[T any](...T) T { var x T; return x }

// Test case 1

func _() {
	var a chan string
	var b <-chan string
	f(a, b)
	f(b, a)
}

// Test case 2

type F[T any] func(T) bool

func g[T any](T) F[<-chan T] { return nil }

func f1[T any](T, F[T]) {}
func f2[T any](F[T], T) {}

func _() {
	var ch chan string
	f1(ch, g(""))
	f2(g(""), ch)
}

// Test case 3: named and directional types combined

func _() {
	type namedA chan int
	type namedB chan<- int

	var a chan int
	var A namedA
	var b chan<- int
	var B namedB

	// Defined types win over channel types irrespective of channel direction.
	f(A, b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */)
	f(b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */, A)

	f(a, b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */, A)
	f(a, A, b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */)
	f(b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */, A, a)
	f(b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */, a, A)
	f(A, a, b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */)
	f(A, b /* ERROR "cannot use b (variable of type chan<- int) as namedA value in argument to f" */, a)

	// Unnamed directed channels win over bidirectional channels.
	b = f(a, b)
	b = f(b, a)

	// Defined directed channels win over defined bidirectional channels.
	A = f(A, a)
	A = f(a, A)
	B = f(B, b)
	B = f(b, B)

	f(a, b, B)
	f(a, B, b)
	f(b, B, a)
	f(b, a, B)
	f(B, a, b)
	f(B, b, a)

	// Differently named channel types conflict irrespective of channel direction.
	f(A, B /* ERROR "type namedB of B does not match inferred type namedA for T" */)
	f(B, A /* ERROR "type namedA of A does not match inferred type namedB for T" */)

	// Ensure that all combinations of directional and
	// bidirectional channels with a named directional
	// channel lead to the correct (named) directional
	// channel.
	B = f(a, b)
	B = f(a, B)
	B = f(b, a)
	B = f(B, a)

	B = f(a, b, B)
	B = f(a, B, b)
	B = f(b, B, a)
	B = f(b, a, B)
	B = f(B, a, b)
	B = f(B, b, a)

	// verify type error
	A = f /* ERROR "cannot use f(B, b, a) (value of chan type namedB) as namedA value in assignment" */ (B, b, a)
}

// Test case 4: some more combinations

func _() {
	type A chan int
	type B chan int
	type C = chan int
	type D = chan<- int

	var a A
	var b B
	var c C
	var d D

	f(a, b /* ERROR "type B of b does not match inferred type A for T" */, c)
	f(c, a, b /* ERROR "type B of b does not match inferred type A for T" */)
	f(a, b /* ERROR "type B of b does not match inferred type A for T" */, d)
	f(d, a, b /* ERROR "type B of b does not match inferred type A for T" */)
}

// Simplified test case from issue

type Matcher[T any] func(T) bool

func Produces[T any](T) Matcher[<-chan T] { return nil }

func Assert1[T any](Matcher[T], T) {}
func Assert2[T any](T, Matcher[T]) {}

func _() {
	var ch chan string
	Assert1(Produces(""), ch)
	Assert2(ch, Produces(""))
}
