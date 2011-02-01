// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Try to tickle stack splitting bugs by doing
// go, defer, and closure calls at different stack depths.

package main

type T [20]int

func g(c chan int, t T) {
	s := 0
	for i := 0; i < len(t); i++ {
		s += t[i]
	}
	c <- s
}

func d(t T) {
	s := 0
	for i := 0; i < len(t); i++ {
		s += t[i]
	}
	if s != len(t) {
		println("bad defer", s)
		panic("fail")
	}
}

func f0() {
	// likely to make a new stack for f0,
	// because the call to f1 puts 3000 bytes
	// in our frame.
	f1()
}

func f1() [3000]byte {
	// likely to make a new stack for f1,
	// because 3000 bytes were used by f0
	// and we need 3000 more for the call
	// to f2.  if the call to morestack in f1
	// does not pass the frame size, the new
	// stack (default size 5k) will not be big
	// enough for the frame, and the morestack
	// check in f2 will die, if we get that far 
	// without faulting.
	f2()
	return [3000]byte{}
}

func f2() [3000]byte {
	// just take up space
	return [3000]byte{}
}

var c = make(chan int)
var t T
var b = []byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}

func recur(n int) {
	ss := string(b)
	if len(ss) != len(b) {
		panic("bad []byte -> string")
	}
	go g(c, t)
	f0()
	s := <-c
	if s != len(t) {
		println("bad go", s)
		panic("fail")
	}
	f := func(t T) int {
		s := 0
		for i := 0; i < len(t); i++ {
			s += t[i]
		}
		s += n
		return s
	}
	s = f(t)
	if s != len(t)+n {
		println("bad func", s, "at level", n)
		panic("fail")
	}
	if n > 0 {
		recur(n - 1)
	}
	defer d(t)
}

func main() {
	for i := 0; i < len(t); i++ {
		t[i] = 1
	}
	recur(8000)
}
