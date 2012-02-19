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
}

func p1() {
	var s *S
	var i I
	var e interface {}
	e = s
	i = e.(I)
	_ = i
}

type S struct {
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function should panic")
		}
	}()
	f()
}
