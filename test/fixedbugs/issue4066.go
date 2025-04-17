// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// issue 4066: return values not being spilled eagerly enough

package main

func main() {
	n := foo()
	if n != 2 {
		println(n)
		panic("wrong return value")
	}
}

type terr struct{}

func foo() (val int) {
	val = 0
	defer func() {
		if x := recover(); x != nil {
			_ = x.(terr)
		}
	}()
	for {
		val = 2
		foo1()
	}
	panic("unreachable")
}

func foo1() {
	panic(terr{})
}
