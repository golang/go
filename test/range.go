// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the 'for range' construct.

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

const alphabet = "abcdefghijklmnopqrstuvwxyz"

func testblankvars() {
	n := 0
	for range alphabet {
		n++
	}
	if n != 26 {
		println("for range: wrong count", n, "want 26")
		panic("fail")
	}
	n = 0
	for _ = range alphabet {
		n++
	}
	if n != 26 {
		println("for _ = range: wrong count", n, "want 26")
		panic("fail")
	}
	n = 0
	for _, _ = range alphabet {
		n++
	}
	if n != 26 {
		println("for _, _ = range: wrong count", n, "want 26")
		panic("fail")
	}
	s := 0
	for i, _ := range alphabet {
		s += i
	}
	if s != 325 {
		println("for i, _ := range: wrong sum", s, "want 325")
		panic("fail")
	}
	r := rune(0)
	for _, v := range alphabet {
		r += v
	}
	if r != 2847 {
		println("for _, v := range: wrong sum", r, "want 2847")
		panic("fail")
	}
}

func testchan() {
	s := ""
	for i := range seq('a', 'z') {
		s += string(i)
	}
	if s != alphabet {
		println("Wanted lowercase alphabet; got", s)
		panic("fail")
	}
	n := 0
	for range seq('a', 'z') {
		n++
	}
	if n != 26 {
		println("testchan wrong count", n, "want 26")
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
		println("wrong sum ranging over makeslice", s)
		panic("fail")
	}

	x := []int{10, 20}
	y := []int{99}
	i := 1
	for i, x[i] = range y {
		break
	}
	if i != 0 || x[0] != 10 || x[1] != 99 {
		println("wrong parallel assignment", i, x[0], x[1])
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
		println("wrong sum ranging over makeslice", s)
		panic("fail")
	}
}

func testslice2() {
	n := 0
	nmake = 0
	for range makeslice() {
		n++
	}
	if nmake != 1 {
		println("range called makeslice", nmake, "times")
		panic("fail")
	}
	if n != 5 {
		println("wrong count ranging over makeslice", n)
		panic("fail")
	}
}

// test that range over []byte(string) only evaluates
// the expression after "range" once.

func makenumstring() string {
	nmake++
	return "\x01\x02\x03\x04\x05"
}

func testslice3() {
	s := byte(0)
	nmake = 0
	for _, v := range []byte(makenumstring()) {
		s += v
	}
	if nmake != 1 {
		println("range called makenumstring", nmake, "times")
		panic("fail")
	}
	if s != 15 {
		println("wrong sum ranging over []byte(makenumstring)", s)
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
		println("wrong sum ranging over makearray", s)
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
		println("wrong sum ranging over makearray", s)
		panic("fail")
	}
}

func testarray2() {
	n := 0
	nmake = 0
	for range makearray() {
		n++
	}
	if nmake != 1 {
		println("range called makearray", nmake, "times")
		panic("fail")
	}
	if n != 5 {
		println("wrong count ranging over makearray", n)
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
		println("wrong sum ranging over makearrayptr", s)
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
		println("wrong sum ranging over makearrayptr", s)
		panic("fail")
	}
}

func testarrayptr2() {
	n := 0
	nmake = 0
	for range makearrayptr() {
		n++
	}
	if nmake != 1 {
		println("range called makearrayptr", nmake, "times")
		panic("fail")
	}
	if n != 5 {
		println("wrong count ranging over makearrayptr", n)
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
		println("wrong sum ranging over makestring", s)
		panic("fail")
	}

	x := []rune{'a', 'b'}
	i := 1
	for i, x[i] = range "c" {
		break
	}
	if i != 0 || x[0] != 'a' || x[1] != 'c' {
		println("wrong parallel assignment", i, x[0], x[1])
		panic("fail")
	}

	y := []int{1, 2, 3}
	r := rune(1)
	for y[r], r = range "\x02" {
		break
	}
	if r != 2 || y[0] != 1 || y[1] != 0 || y[2] != 3 {
		println("wrong parallel assignment", r, y[0], y[1], y[2])
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
		println("wrong sum ranging over makestring", s)
		panic("fail")
	}
}

func teststring2() {
	n := 0
	nmake = 0
	for range makestring() {
		n++
	}
	if nmake != 1 {
		println("range called makestring", nmake, "times")
		panic("fail")
	}
	if n != 5 {
		println("wrong count ranging over makestring", n)
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
		println("wrong sum ranging over makemap", s)
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
		println("wrong sum ranging over makemap", s)
		panic("fail")
	}
}

func testmap2() {
	n := 0
	nmake = 0
	for range makemap() {
		n++
	}
	if nmake != 1 {
		println("range called makemap", nmake, "times")
		panic("fail")
	}
	if n != 5 {
		println("wrong count ranging over makemap", n)
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
	testblankvars()
	testchan()
	testarray()
	testarray1()
	testarray2()
	testarrayptr()
	testarrayptr1()
	testarrayptr2()
	testslice()
	testslice1()
	testslice2()
	testslice3()
	teststring()
	teststring1()
	teststring2()
	testmap()
	testmap1()
	testmap2()
	testcalls()
}
