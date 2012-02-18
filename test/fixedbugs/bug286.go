// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test case for issue 849.

package main

type I interface {
	f()
}

var callee string
var error_ bool

type T int

func (t *T) f() { callee = "f" }
func (i *T) g() { callee = "g" }

// test1 and test2 are the same except that in the interface J
// the entries are swapped. test2 and test3 are the same except
// that in test3 the interface J is declared outside the function.
//
// Error: test2 calls g instead of f

func test1(x I) {
	type J interface {
		I
		g()
	}
	x.(J).f()
	if callee != "f" {
		println("test1 called", callee)
		error_ = true
	}
}

func test2(x I) {
	type J interface {
		g()
		I
	}
	x.(J).f()
	if callee != "f" {
		println("test2 called", callee)
		error_ = true
	}
}

type J interface {
	g()
	I
}

func test3(x I) {
	x.(J).f()
	if callee != "f" {
		println("test3 called", callee)
		error_ = true
	}
}

func main() {
	x := new(T)
	test1(x)
	test2(x)
	test3(x)
	if error_ {
		panic("wrong method called")
	}
}

/*
6g bug286.go && 6l bug286.6 && 6.out
test2 called g
panic: wrong method called

panic PC=0x24e040
runtime.panic+0x7c /home/gri/go1/src/pkg/runtime/proc.c:1012
	runtime.panic(0x0, 0x24e0a0)
main.main+0xef /home/gri/go1/test/bugs/bug286.go:76
	main.main()
mainstart+0xf /home/gri/go1/src/pkg/runtime/amd64/asm.s:60
	mainstart()
goexit /home/gri/go1/src/pkg/runtime/proc.c:145
	goexit()
*/
