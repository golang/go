// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// check for correct heap-moving of escaped variables.
// it is hard to check for the allocations, but it is easy
// to check that if you call the function twice at the
// same stack level, the pointers returned should be
// different.

var bad = false

var allptr = make([]*int, 0, 100)

func noalias(p, q *int, s string) {
	n := len(allptr)
	*p = -(n+1)
	*q = -(n+2)
	allptr = allptr[0:n+2]
	allptr[n] = p
	allptr[n+1] = q
	n += 2
	for i := 0; i < n; i++ {
		if allptr[i] != nil && *allptr[i] != -(i+1) {
			println("aliased pointers", -(i+1), *allptr[i], "after", s)
			allptr[i] = nil
			bad = true
		}
	}
}

func val(p, q *int, v int, s string) {
	if *p != v {
		println("wrong value want", v, "got", *p, "after", s)
		bad = true
	}
	if *q != v+1 {
		println("wrong value want", v+1, "got", *q, "after", s)
		bad = true
	}
}

func chk(p, q *int, v int, s string) {
	val(p, q, v, s)
	noalias(p, q, s)
}

func chkalias(p, q *int, v int, s string) {
	if p != q {
		println("want aliased pointers but got different after", s)
	}
	if *q != v+1 {
		println("wrong value want", v+1, "got", *q, "after", s)
	}
}

func i_escapes(x int) *int {
	var i int
	i = x
	return &i
}

func j_escapes(x int) *int {
	var j int = x
	j = x
	return &j
}

func k_escapes(x int) *int {
	k := x
	return &k
}

func in_escapes(x int) *int {
	return &x
}

func send(c chan int, x int) {
	c <- x
}

func select_escapes(x int) *int {
	c := make(chan int)
	go send(c, x)
	select {
	case req := <-c:
		return &req
	}
	return nil
}

func select_escapes1(x int, y int) (*int, *int) {
	c := make(chan int)
	var a [2]int
	var p [2]*int
	a[0] = x
	a[1] = y
	for i := 0; i < 2; i++ {
		go send(c, a[i])
		select {
		case req := <-c:
			p[i] = &req
		}
	}
	return p[0], p[1]
}

func range_escapes(x int) *int {
	var a [1]int
	a[0] = x
	for _, v := range a {
		return &v
	}
	return nil
}

// *is* aliased
func range_escapes2(x, y int) (*int, *int) {
	var a [2]int
	var p [2]*int
	a[0] = x
	a[1] = y
	for k, v := range a {
		p[k] = &v
	}
	return p[0], p[1]
}

// *is* aliased
func for_escapes2(x int, y int) (*int, *int) {
	var p [2]*int
	n := 0
	for i := x; n < 2; i = y {
		p[n] = &i
		n++
	}
	return p[0], p[1]
}

func out_escapes(i int) (x int, p *int) {
	x = i
	p = &x	// ERROR "address of out parameter"
	return
}

func out_escapes_2(i int) (x int, p *int) {
	x = i
	return x, &x	// ERROR "address of out parameter"
}

func defer1(i int) (x int) {
	c := make(chan int)
	go func() { x = i; c <- 1 }()
	<-c
	return
}

func main() {
	p, q := i_escapes(1), i_escapes(2)
	chk(p, q, 1, "i_escapes")

	p, q = j_escapes(3), j_escapes(4)
	chk(p, q, 3, "j_escapes")

	p, q = k_escapes(5), k_escapes(6)
	chk(p, q, 5, "k_escapes")

	p, q = in_escapes(7), in_escapes(8)
	chk(p, q, 7, "in_escapes")

	p, q = select_escapes(9), select_escapes(10)
	chk(p, q, 9, "select_escapes")

	p, q = select_escapes1(11, 12)
	chk(p, q, 11, "select_escapes1")

	p, q = range_escapes(13), range_escapes(14)
	chk(p, q, 13, "range_escapes")

	p, q = range_escapes2(101, 102)
	chkalias(p, q, 101, "range_escapes2")

	p, q = for_escapes2(103, 104)
	chkalias(p, q, 103, "for_escapes2")

	_, p = out_escapes(15)
	_, q = out_escapes(16)
	chk(p, q, 15, "out_escapes")

	_, p = out_escapes_2(17)
	_, q = out_escapes_2(18)
	chk(p, q, 17, "out_escapes_2")

	x := defer1(20)
	if x != 20 {
		println("defer failed", x)
		bad = true
	}

	if bad {
		panic("BUG: no escape")
	}
}
