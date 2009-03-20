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
	for k, v := range makearray() {
		s += v;
	}
	if nmake != 1 {
		panicln("range called makearray", nmake, "times");
	}
	if s != 15 {
		panicln("wrong sum ranging over makearray");
	}
}

func main() {
	testchan();
	testarray();
}
