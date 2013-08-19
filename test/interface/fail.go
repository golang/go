// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that interface conversion fails when method is missing.

package main

type I interface {
	Foo()
}

func main() {
	shouldPanic(p1)
	shouldPanic(p2)
}

func p1() {
	var s *S
	var i I
	var e interface{}
	e = s
	i = e.(I)
	_ = i
}

type S struct{}

func (s *S) _() {}

type B interface {
	_()
}

func p2() {
	var s *S
	var b B
	var e interface{}
	e = s
	b = e.(B)
	_ = b
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function should panic")
		}
	}()
	f()
}
