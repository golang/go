// run

// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test variadic functions and calls (dot-dot-dot).

package main

func sum(args ...int) int {
	s := 0
	for _, v := range args {
		s += v
	}
	return s
}

func sumC(args ...int) int { return func() int { return sum(args...) }() }

var sumD = func(args ...int) int { return sum(args...) }

var sumE = func() func(...int) int { return func(args ...int) int { return sum(args...) } }()

var sumF = func(args ...int) func() int { return func() int { return sum(args...) } }

func sumA(args []int) int {
	s := 0
	for _, v := range args {
		s += v
	}
	return s
}

func sumB(args []int) int { return sum(args...) }

func sum2(args ...int) int { return 2 * sum(args...) }

func sum3(args ...int) int { return 3 * sumA(args) }

func sum4(args ...int) int { return 4 * sumB(args) }

func intersum(args ...interface{}) int {
	s := 0
	for _, v := range args {
		s += v.(int)
	}
	return s
}

type T []T

func ln(args ...T) int { return len(args) }

func ln2(args ...T) int { return 2 * ln(args...) }

func (*T) Sum(args ...int) int { return sum(args...) }

type U struct {
	*T
}

type I interface {
	Sum(...int) int
}

func main() {
	if x := sum(1, 2, 3); x != 6 {
		println("sum 6", x)
		panic("fail")
	}
	if x := sum(); x != 0 {
		println("sum 0", x)
		panic("fail")
	}
	if x := sum(10); x != 10 {
		println("sum 10", x)
		panic("fail")
	}
	if x := sum(1, 8); x != 9 {
		println("sum 9", x)
		panic("fail")
	}
	if x := sumC(4, 5, 6); x != 15 {
		println("sumC 15", x)
		panic("fail")
	}
	if x := sumD(4, 5, 7); x != 16 {
		println("sumD 16", x)
		panic("fail")
	}
	if x := sumE(4, 5, 8); x != 17 {
		println("sumE 17", x)
		panic("fail")
	}
	if x := sumF(4, 5, 9)(); x != 18 {
		println("sumF 18", x)
		panic("fail")
	}
	if x := sum2(1, 2, 3); x != 2*6 {
		println("sum 6", x)
		panic("fail")
	}
	if x := sum2(); x != 2*0 {
		println("sum 0", x)
		panic("fail")
	}
	if x := sum2(10); x != 2*10 {
		println("sum 10", x)
		panic("fail")
	}
	if x := sum2(1, 8); x != 2*9 {
		println("sum 9", x)
		panic("fail")
	}
	if x := sum3(1, 2, 3); x != 3*6 {
		println("sum 6", x)
		panic("fail")
	}
	if x := sum3(); x != 3*0 {
		println("sum 0", x)
		panic("fail")
	}
	if x := sum3(10); x != 3*10 {
		println("sum 10", x)
		panic("fail")
	}
	if x := sum3(1, 8); x != 3*9 {
		println("sum 9", x)
		panic("fail")
	}
	if x := sum4(1, 2, 3); x != 4*6 {
		println("sum 6", x)
		panic("fail")
	}
	if x := sum4(); x != 4*0 {
		println("sum 0", x)
		panic("fail")
	}
	if x := sum4(10); x != 4*10 {
		println("sum 10", x)
		panic("fail")
	}
	if x := sum4(1, 8); x != 4*9 {
		println("sum 9", x)
		panic("fail")
	}
	if x := intersum(1, 2, 3); x != 6 {
		println("intersum 6", x)
		panic("fail")
	}
	if x := intersum(); x != 0 {
		println("intersum 0", x)
		panic("fail")
	}
	if x := intersum(10); x != 10 {
		println("intersum 10", x)
		panic("fail")
	}
	if x := intersum(1, 8); x != 9 {
		println("intersum 9", x)
		panic("fail")
	}

	if x := ln(nil, nil, nil); x != 3 {
		println("ln 3", x)
		panic("fail")
	}
	if x := ln([]T{}); x != 1 {
		println("ln 1", x)
		panic("fail")
	}
	if x := ln2(nil, nil, nil); x != 2*3 {
		println("ln2 3", x)
		panic("fail")
	}
	if x := ln2([]T{}); x != 2*1 {
		println("ln2 1", x)
		panic("fail")
	}
	if x := ((*T)(nil)).Sum(1, 3, 5, 7); x != 16 {
		println("(*T)(nil).Sum", x)
		panic("fail")
	}
	if x := (*T).Sum(nil, 1, 3, 5, 6); x != 15 {
		println("(*T).Sum", x)
		panic("fail")
	}
	if x := (&U{}).Sum(1, 3, 5, 5); x != 14 {
		println("(&U{}).Sum", x)
		panic("fail")
	}
	var u U
	if x := u.Sum(1, 3, 5, 4); x != 13 {
		println("u.Sum", x)
		panic("fail")
	}
	if x := (&u).Sum(1, 3, 5, 3); x != 12 {
		println("(&u).Sum", x)
		panic("fail")
	}
	var i interface {
		Sum(...int) int
	} = &u
	if x := i.Sum(2, 3, 5, 7); x != 17 {
		println("i(=&u).Sum", x)
		panic("fail")
	}
	i = u
	if x := i.Sum(2, 3, 5, 6); x != 16 {
		println("i(=u).Sum", x)
		panic("fail")
	}
	var s struct {
		I
	}
	s.I = &u
	if x := s.Sum(2, 3, 5, 8); x != 18 {
		println("s{&u}.Sum", x)
		panic("fail")
	}
	if x := (*U).Sum(&U{}, 1, 3, 5, 2); x != 11 {
		println("(*U).Sum", x)
		panic("fail")
	}
	if x := U.Sum(U{}, 1, 3, 5, 1); x != 10 {
		println("U.Sum", x)
		panic("fail")
	}
}
