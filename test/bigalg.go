// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the internal "algorithms" for objects larger than a word: hashing, equality etc.

package main

type T struct {
	a float64
	b int64
	c string
	d byte
}

var a = []int{ 1, 2, 3 }
var NIL []int

func arraycmptest() {
	if NIL != nil {
		println("fail1:", NIL, "!= nil")
	}
	if nil != NIL {
		println("fail2: nil !=", NIL)
	}
	if a == nil || nil == a {
		println("fail3:", a, "== nil")
	}
}

func SameArray(a, b []int) bool {
	if len(a) != len(b) || cap(a) != cap(b) {
		return false
	}
	if len(a) > 0 && &a[0] != &b[0] {
		return false
	}
	return true
}

var t = T{1.5, 123, "hello", 255}
var mt = make(map[int]T)
var ma = make(map[int][]int)

func maptest() {
	mt[0] = t
	t1 := mt[0]
	if t1.a != t.a || t1.b != t.b || t1.c != t.c || t1.d != t.d {
		println("fail: map val struct", t1.a, t1.b, t1.c, t1.d)
	}

	ma[1] = a
	a1 := ma[1]
	if !SameArray(a, a1) {
		println("fail: map val array", a, a1)
	}
}

var ct = make(chan T)
var ca = make(chan []int)

func send() {
	ct <- t
	ca <- a
}

func chantest() {
	go send()

	t1 := <-ct
	if t1.a != t.a || t1.b != t.b || t1.c != t.c || t1.d != t.d {
		println("fail: map val struct", t1.a, t1.b, t1.c, t1.d)
	}

	a1 := <-ca
	if !SameArray(a, a1) {
		println("fail: map val array", a, a1)
	}
}

type E struct { }
var e E

func interfacetest() {
	var i interface{}

	i = a
	a1 := i.([]int)
	if !SameArray(a, a1) {
		println("interface <-> []int", a, a1)
	}
	pa := new([]int)
	*pa = a
	i = pa
	a1 = *i.(*[]int)
	if !SameArray(a, a1) {
		println("interface <-> *[]int", a, a1)
	}

	i = t
	t1 := i.(T)
	if t1.a != t.a || t1.b != t.b || t1.c != t.c || t1.d != t.d {
		println("interface <-> struct", t1.a, t1.b, t1.c, t1.d)
	}

	i = e
	e1 := i.(E)
	// nothing to check; just verify it doesn't crash
	_ = e1
}

func main() {
	arraycmptest()
	maptest()
	chantest()
	interfacetest()
}
