// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "./a"

// The two instantiations share a shape but require different runtime types
// and method implementations in their dictionaries.
type first int
type second int

func (first) Value() int  { return 11 }
func (second) Value() int { return 22 }

// Keep the returned closures across a call boundary, so the test needs their
// captured dictionaries even when their generic factories are inlined.
//
//go:noinline
func call[R any](f func() R) R {
	return f()
}

//go:noinline
func callAssert(f func(any) any, x any) any {
	return f(x)
}

//go:noinline
func callVoid(f func()) {
	f()
}

func check[T interface {
	comparable
	Value() int
}](want int) {
	if _, ok := call(call(a.Nested[T]())).([]T); !ok {
		panic("nested dictionary capture")
	}
	plain, typed := a.Siblings[T]()
	if call(plain) != 7 {
		panic("dictionary-free sibling")
	}
	if _, ok := call(typed).([]T); !ok {
		panic("typed sibling")
	}
	var value T
	if got := callAssert(a.Assert[T](), value); got != any(value) {
		panic("type assertion")
	}
	if _, ok := call(a.Subcall[T]()).([]T); !ok {
		panic("generic subcall")
	}
	if call(a.Method[T]()) != want {
		panic("method constraint")
	}
	if call(call(a.MethodValue[T]())) != want {
		panic("method value")
	}
	m := call(a.Map[T]()).(map[T]int)
	m[value] = want
	if m[value] != want {
		panic("map dictionary")
	}

	ch := make(chan any, 2)
	calls := 0
	callVoid(a.GoDefer[T](ch, func() int {
		calls++
		return calls
	}))
	if calls != 2 {
		panic("go/defer argument evaluation")
	}
	x := (<-ch).(a.Message[T]).Index
	y := (<-ch).(a.Message[T]).Index
	if x+y != 3 || x == y {
		panic("go/defer captured arguments")
	}
	if _, ok := call(a.Range[T]()).([]T); !ok {
		panic("range function dictionary")
	}
}

func main() {
	check[first](11)
	check[second](22)
}
