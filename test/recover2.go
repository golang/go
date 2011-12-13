// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of recover for run-time errors.

// TODO(rsc):
//	null pointer accesses

package main

import "strings"

var x = make([]byte, 10)

func main() {
	test1()
	test2()
	test3()
	test4()
	test5()
	test6()
	test7()
}

func mustRecover(s string) {
	v := recover()
	if v == nil {
		panic("expected panic")
	}
	if e := v.(error).Error(); strings.Index(e, s) < 0 {
		panic("want: " + s + "; have: " + e)
	}
}

func test1() {
	defer mustRecover("index")
	println(x[123])
}

func test2() {
	defer mustRecover("slice")
	println(x[5:15])
}

func test3() {
	defer mustRecover("slice")
	var lo = 11
	var hi = 9
	println(x[lo:hi])
}

func test4() {
	defer mustRecover("interface")
	var x interface{} = 1
	println(x.(float32))
}

type T struct {
	a, b int
	c    []int
}

func test5() {
	defer mustRecover("uncomparable")
	var x T
	var z interface{} = x
	println(z != z)
}

func test6() {
	defer mustRecover("unhashable")
	var x T
	var z interface{} = x
	m := make(map[interface{}]int)
	m[z] = 1
}

func test7() {
	defer mustRecover("divide by zero")
	var x, y int
	println(x / y)
}
