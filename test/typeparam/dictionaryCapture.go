// run -gcflags=-G=3

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test situations where functions/methods are not
// immediately called and we need to capture the dictionary
// required for later invocation.

package main

import (
	"fmt"
)

func main() {
	functions()
	methodExpressions()
	genMethodExpressions[int](7)
	methodValues()
	genMethodValues[int](7)
	interfaceMethods()
	globals()
	recursive()
}

func g0[T any](x T) {
}
func g1[T any](x T) T {
	return x
}
func g2[T any](x T) (T, T) {
	return x, x
}

func functions() {
	f0 := g0[int]
	f0(7)
	f1 := g1[int]
	is7(f1(7))
	f2 := g2[int]
	is77(f2(7))
}

func is7(x int) {
	if x != 7 {
		println(x)
		panic("assertion failed")
	}
}
func is77(x, y int) {
	if x != 7 || y != 7 {
		println(x, y)
		panic("assertion failed")
	}
}

type s[T any] struct {
	a T
}

func (x s[T]) g0() {
}
func (x s[T]) g1() T {
	return x.a
}
func (x s[T]) g2() (T, T) {
	return x.a, x.a
}

func methodExpressions() {
	x := s[int]{a: 7}
	f0 := s[int].g0
	f0(x)
	f0p := (*s[int]).g0
	f0p(&x)
	f1 := s[int].g1
	is7(f1(x))
	f1p := (*s[int]).g1
	is7(f1p(&x))
	f2 := s[int].g2
	is77(f2(x))
	f2p := (*s[int]).g2
	is77(f2p(&x))
}

func genMethodExpressions[T comparable](want T) {
	x := s[T]{a: want}
	f0 := s[T].g0
	f0(x)
	f0p := (*s[T]).g0
	f0p(&x)
	f1 := s[T].g1
	if got := f1(x); got != want {
		panic(fmt.Sprintf("f1(x) == %d, want %d", got, want))
	}
	f1p := (*s[T]).g1
	if got := f1p(&x); got != want {
		panic(fmt.Sprintf("f1p(&x) == %d, want %d", got, want))
	}
	f2 := s[T].g2
	if got1, got2 := f2(x); got1 != want || got2 != want {
		panic(fmt.Sprintf("f2(x) == %d, %d, want %d, %d", got1, got2, want, want))
	}
}

func methodValues() {
	x := s[int]{a: 7}
	f0 := x.g0
	f0()
	f1 := x.g1
	is7(f1())
	f2 := x.g2
	is77(f2())
}

func genMethodValues[T comparable](want T) {
	x := s[T]{a: want}
	f0 := x.g0
	f0()
	f1 := x.g1
	if got := f1(); got != want {
		panic(fmt.Sprintf("f1() == %d, want %d", got, want))
	}
	f2 := x.g2
	if got1, got2 := f2(); got1 != want || got2 != want {
		panic(fmt.Sprintf("f2() == %d, %d, want %d, %d", got1, got2, want, want))
	}
}

var x interface {
	g0()
	g1() int
	g2() (int, int)
} = s[int]{a: 7}
var y interface{} = s[int]{a: 7}

func interfaceMethods() {
	x.g0()
	is7(x.g1())
	is77(x.g2())
	y.(interface{ g0() }).g0()
	is7(y.(interface{ g1() int }).g1())
	is77(y.(interface{ g2() (int, int) }).g2())
}

// Also check for instantiations outside functions.
var gg0 = g0[int]
var gg1 = g1[int]
var gg2 = g2[int]

var hh0 = s[int].g0
var hh1 = s[int].g1
var hh2 = s[int].g2

var xtop = s[int]{a: 7}
var ii0 = x.g0
var ii1 = x.g1
var ii2 = x.g2

func globals() {
	gg0(7)
	is7(gg1(7))
	is77(gg2(7))
	x := s[int]{a: 7}
	hh0(x)
	is7(hh1(x))
	is77(hh2(x))
	ii0()
	is7(ii1())
	is77(ii2())
}

func recursive() {
	if got, want := recur1[int](5), 110; got != want {
		panic(fmt.Sprintf("recur1[int](5) = %d, want = %d", got, want))
	}
}

type Integer interface {
	int | int32 | int64
}

func recur1[T Integer](n T) T {
	if n == 0 || n == 1 {
		return T(1)
	} else {
		return n * recur2(n-1)
	}
}

func recur2[T Integer](n T) T {
	list := make([]T, n)
	for i, _ := range list {
		list[i] = T(i + 1)
	}
	var sum T
	for _, elt := range list {
		sum += elt
	}
	return sum + recur1(n-1)
}
