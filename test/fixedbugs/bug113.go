// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface{}

func foo1(i int) int     { return i }
func foo2(i int32) int32 { return i }
func main() {
	var i I
	i = 1
	var v1 = i.(int)
	if foo1(v1) != 1 {
		panic(1)
	}
	var v2 = int32(i.(int))
	if foo2(v2) != 1 {
		panic(2)
	}
	
	shouldPanic(p1)
}

func p1() {
	var i I
	i = 1
	var v3 = i.(int32) // This type conversion should fail at runtime.
	if foo2(v3) != 1 {
		panic(3)
	}
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function should panic")
		}
	}()
	f()
}
