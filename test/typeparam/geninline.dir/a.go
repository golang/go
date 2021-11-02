// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

type IVal[T comparable] interface {
	check(want T)
}

type Val[T comparable] struct {
	val T
}

//go:noinline
func (l *Val[T]) check(want T) {
	if l.val != want {
		panic("hi")
	}
}

func Test1() {
	var l Val[int]
	if l.val != 0 {
		panic("hi")
	}
	_ = IVal[int](&l)
}

func Test2() {
	var l Val[float64]
	l.val = 3.0
	l.check(float64(3))
	_ = IVal[float64](&l)
}

type privateVal[T comparable] struct {
	val T
}

//go:noinline
func (l *privateVal[T]) check(want T) {
	if l.val != want {
		panic("hi")
	}
}

type Outer struct {
	val privateVal[string]
}

func Test3() {
	var o Outer
	o.val.check("")
	_ = IVal[string](&o.val)
}
