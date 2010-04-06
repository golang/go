// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test of recover for run-time errors.

// TODO(rsc):
//	integer divide by zero?
//	null pointer accesses

package main

import (
	"os"
	"strings"
	"syscall"
)

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
	if e := v.(os.Error).String(); strings.Index(e, s) < 0 {
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
	println(x[11:9])
}

func test4() {
	defer mustRecover("interface")
	var x interface{} = 1
	println(x.(float))
}

type T struct {
	a, b int
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
	if syscall.ARCH == "arm" {
		// ARM doesn't have floating point yet
		return
	}
	defer mustRecover("complex divide by zero")
	var x, y complex
	println(x / y)
}
