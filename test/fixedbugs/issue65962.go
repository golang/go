// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	test1()
	test2()
}

type I interface {
	f()
	g()
	h()
}

//go:noinline
func ld[T any]() {
	var x I
	if _, ok := x.(T); ok {
	}
}

func isI(x any) {
	_ = x.(I)
}

func test1() {
	defer func() { recover() }()
	ld[bool]() // add <bool,I> itab to binary
	_ = any(false).(I)
}

type B bool

func (B) f() {
}
func (B) g() {
}

func test2() {
	defer func() { recover() }()
	ld[B]() // add <B,I> itab to binary
	_ = any(B(false)).(I)
}
