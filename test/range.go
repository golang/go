// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// test range over channels

func gen(c chan int, lo, hi int) {
	for i := lo; i <= hi; i++ {
		c <- i
	}
	close(c)
}

func seq(lo, hi int) chan int {
	c := make(chan int)
	go gen(c, lo, hi)
	return c
}

func testchan() {
	s := ""
	for i := range seq('a', 'z') {
		s += string(i)
	}
	if s != "abcdefghijklmnopqrstuvwxyz" {
		println("Wanted lowercase alphabet; got", s)
		panic("fail")
	}
}

// test that range over slice only evaluates
// the expression after "range" once.

var nmake = 0

func makeslice() []int {
	nmake++
	return []int{1, 2, 3, 4, 5}
}

func testslice() {
	s := 0
	nmake = 0
	for _, v := range makeslice() {
		s += v
	}
	if nmake != 1 {
		println("range called makeslice", nmake, "times")
		panic("fail")
	}
	if s != 15 {
		println("wrong sum ranging over makeslice")
		panic("fail")
	}
}

func testslice1() {
	s := 0
	nmake = 0
	for i := range makeslice() {
		s += i
	}
	if nmake != 1 {
		println("range called makeslice", nmake, "times")
		panic("fail")
	}
	if s != 10 {
		println("wrong sum ranging over makeslice")
		panic("fail")
	}
}

// test that range over array only evaluates
// the expression after "range" once.

func makearray() [5]int {
	nmake++
	return [5]int{1, 2, 3, 4, 5}
}

func testarray() {
	s := 0
	nmake = 0
	for _, v := range makearray() {
		s += v
	}
	if nmake != 1 {
		println("range called makearray", nmake, "times")
		panic("fail")
	}
	if s != 15 {
		println("wrong sum ranging over makearray")
		panic("fail")
	}
}

func testarray1() {
	s := 0
	nmake = 0
	for i := range makearray() {
		s += i
	}
	if nmake != 1 {
		println("range called makearray", nmake, "times")
		panic("fail")
	}
	if s != 10 {
		println("wrong sum ranging over makearray")
		panic("fail")
	}
}

func makearrayptr() *[5]int {
	nmake++
	return &[5]int{1, 2, 3, 4, 5}
}

func testarrayptr() {
	nmake = 0
	x := len(makearrayptr())
	if x != 5 || nmake != 1 {
		println("len called makearrayptr", nmake, "times and got len", x)
		panic("fail")
	}
	nmake = 0
	x = cap(makearrayptr())
	if x != 5 || nmake != 1 {
		println("cap called makearrayptr", nmake, "times and got len", x)
		panic("fail")
	}
	s := 0
	nmake = 0
	for _, v := range makearrayptr() {
		s += v
	}
	if nmake != 1 {
		println("range called makearrayptr", nmake, "times")
		panic("fail")
	}
	if s != 15 {
		println("wrong sum ranging over makearrayptr")
		panic("fail")
	}
}

func testarrayptr1() {
	s := 0
	nmake = 0
	for i := range makearrayptr() {
		s += i
	}
	if nmake != 1 {
		println("range called makearrayptr", nmake, "times")
		panic("fail")
	}
	if s != 10 {
		println("wrong sum ranging over makearrayptr")
		panic("fail")
	}
}

// test that range over string only evaluates
// the expression after "range" once.

func makestring() string {
	nmake++
	return "abcd☺"
}

func teststring() {
	var s rune
	nmake = 0
	for _, v := range makestring() {
		s += v
	}
	if nmake != 1 {
		println("range called makestring", nmake, "times")
		panic("fail")
	}
	if s != 'a'+'b'+'c'+'d'+'☺' {
		println("wrong sum ranging over makestring")
		panic("fail")
	}
}

func teststring1() {
	s := 0
	nmake = 0
	for i := range makestring() {
		s += i
	}
	if nmake != 1 {
		println("range called makestring", nmake, "times")
		panic("fail")
	}
	if s != 10 {
		println("wrong sum ranging over makestring")
		panic("fail")
	}
}

// test that range over map only evaluates
// the expression after "range" once.

func makemap() map[int]int {
	nmake++
	return map[int]int{0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: '☺'}
}

func testmap() {
	s := 0
	nmake = 0
	for _, v := range makemap() {
		s += v
	}
	if nmake != 1 {
		println("range called makemap", nmake, "times")
		panic("fail")
	}
	if s != 'a'+'b'+'c'+'d'+'☺' {
		println("wrong sum ranging over makemap")
		panic("fail")
	}
}

func testmap1() {
	s := 0
	nmake = 0
	for i := range makemap() {
		s += i
	}
	if nmake != 1 {
		println("range called makemap", nmake, "times")
		panic("fail")
	}
	if s != 10 {
		println("wrong sum ranging over makemap")
		panic("fail")
	}
}

// test that range evaluates the index and value expressions
// exactly once per iteration.

var ncalls = 0

func getvar(p *int) *int {
	ncalls++
	return p
}

func testcalls() {
	var i, v int
	si := 0
	sv := 0
	for *getvar(&i), *getvar(&v) = range [2]int{1, 2} {
		si += i
		sv += v
	}
	if ncalls != 4 {
		println("wrong number of calls:", ncalls, "!= 4")
		panic("fail")
	}
	if si != 1 || sv != 3 {
		println("wrong sum in testcalls", si, sv)
		panic("fail")
	}

	ncalls = 0
	for *getvar(&i), *getvar(&v) = range [0]int{} {
		println("loop ran on empty array")
		panic("fail")
	}
	if ncalls != 0 {
		println("wrong number of calls:", ncalls, "!= 0")
		panic("fail")
	}
}

func main() {
	testchan()
	testarray()
	testarray1()
	testarrayptr()
	testarrayptr1()
	testslice()
	testslice1()
	teststring()
	teststring1()
	testmap()
	testmap1()
	testcalls()
}
