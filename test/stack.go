// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Try to tickle stack splitting bugs by doing
// go, defer, and closure calls at different stack depths.

package main

type T [20] int;

func g(c chan int, t T) {
	s := 0;
	for i := 0; i < len(t); i++ {
		s += t[i];
	}
	c <- s;
}

func d(t T) {
	s := 0;
	for i := 0; i < len(t); i++ {
		s += t[i];
	}
	if s != len(t) {
		panicln("bad defer", s);
	}
}

var c = make(chan int);
var t T;

func recur(n int) {
	go g(c, t);
	s := <-c;
	if s != len(t) {
		panicln("bad go", s);
	}
	f := func(t T) int {
		s := 0;
		for i := 0; i < len(t); i++ {
			s += t[i];
		}
		s += n;
		return s;
	};
	s = f(t);
	if s != len(t) + n {
		panicln("bad func", s, "at level", n);
	}
	if n > 0 {
		recur(n-1);
	}
	defer d(t);
}

func main() {
	for i := 0; i < len(t); i++ {
		t[i] = 1;
	}
	recur(10000);
}
