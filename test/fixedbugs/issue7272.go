// compile

// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 7272: test builtin functions in statement context and in
// go/defer functions.

package p

func F() {
	var a []int
	var c chan int
	var m map[int]int

	close(c)
	copy(a, a)
	delete(m, 0)
	panic(0)
	print("foo")
	println("bar")
	recover()

	(close(c))
	(copy(a, a))
	(delete(m, 0))
	(panic(0))
	(print("foo"))
	(println("bar"))
	(recover())

	go close(c)
	go copy(a, a)
	go delete(m, 0)
	go panic(0)
	go print("foo")
	go println("bar")
	go recover()

	defer close(c)
	defer copy(a, a)
	defer delete(m, 0)
	defer panic(0)
	defer print("foo")
	defer println("bar")
	defer recover()
}
