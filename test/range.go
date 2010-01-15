// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// test range over channels

func gen(c chan int, lo, hi int) {
	for i := lo; i <= hi; i++ {
		c <- i;
	}
	close(c);
}

func seq(lo, hi int) chan int {
	c := make(chan int);
	go gen(c, lo, hi);
	return c;
}

func testchan() {
	s := "";
	for i := range seq('a', 'z') {
		s += string(i);
	}
	if s != "abcdefghijklmnopqrstuvwxyz" {
		panicln("Wanted lowercase alphabet; got", s);
	}
}

// test that range over array only evaluates
// the expression after "range" once.

var nmake = 0;
func makearray() []int {
	nmake++;
	return []int{1,2,3,4,5};
}

func testarray() {
	s := 0;
	for _, v := range makearray() {
		s += v;
	}
	if nmake != 1 {
		panicln("range called makearray", nmake, "times");
	}
	if s != 15 {
		panicln("wrong sum ranging over makearray");
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
		panicln("wrong number of calls:", ncalls, "!= 4")
	}
	if si != 1 || sv != 3 {
		panicln("wrong sum in testcalls", si, sv)
	}

	ncalls = 0
	for *getvar(&i), *getvar(&v) = range [0]int{} {
		panicln("loop ran on empty array")
	}
	if ncalls != 0 {
		panicln("wrong number of calls:", ncalls, "!= 0")
	}
}

func main() {
	testchan();
	testarray();
	testcalls();
}
