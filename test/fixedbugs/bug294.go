// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// http://code.google.com/p/go/issues/detail?id=800

package main

var log string

type T int

func (t T) a(s string) T {
	log += "a(" + s + ")"
	return t
}

func (T) b(s string) string {
	log += "b"
	return s
}

type F func(s string) F

func a(s string) F {
	log += "a(" + s + ")"
	return F(a)
}

func b(s string) string {
	log += "b"
	return s
}

type I interface {
	a(s string) I
	b(s string) string
}

type T1 int

func (t T1) a(s string) I {
	log += "a(" + s + ")"
	return t
}

func (T1) b(s string) string {
	log += "b"
	return s
}

var ok = true

func bad() {
	if !ok {
		println("BUG")
		ok = false
	}
	println(log)
}

func main() {
	var t T
	if t.a("1").a(t.b("2")); log != "a(1)ba(2)" {
		bad()
	}
	log = ""
	if a("3")(b("4"))(b("5")); log != "a(3)ba(4)ba(5)" {
		bad()
	}
	log = ""
	var i I = T1(0)
	if i.a("6").a(i.b("7")).a(i.b("8")).a(i.b("9")); log != "a(6)ba(7)ba(8)ba(9)" {
		bad()
	}
}

