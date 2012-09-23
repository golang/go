// $G $D/method4a.go && $G $D/$F.go && $L $F.$A && ./$A.out

// NOTE: This test is not run by 'run.go' and so not run by all.bash.
// To run this test you must use the ./run shell script.

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test method expressions with arguments.

package main

import "./method4a"

type T1 int

type T2 struct {
	f int
}

type I1 interface {
	Sum([]int, int) int
}

type I2 interface {
	Sum(a []int, b int) int
}

func (i T1) Sum(a []int, b int) int {
	r := int(i) + b
	for _, v := range a {
		r += v
	}
	return r
}

func (p *T2) Sum(a []int, b int) int {
	r := p.f + b
	for _, v := range a {
		r += v
	}
	return r
}

func eq(v1, v2 int) {
	if v1 != v2 {
		panic(0)
	}
}

func main() {
	a := []int{1, 2, 3}
	t1 := T1(4)
	t2 := &T2{4}

	eq(t1.Sum(a, 5), 15)
	eq(t2.Sum(a, 6), 16)

	eq(T1.Sum(t1, a, 7), 17)
	eq((*T2).Sum(t2, a, 8), 18)

	f1 := T1.Sum
	eq(f1(t1, a, 9), 19)
	f2 := (*T2).Sum
	eq(f2(t2, a, 10), 20)

	eq(I1.Sum(t1, a, 11), 21)
	eq(I1.Sum(t2, a, 12), 22)

	f3 := I1.Sum
	eq(f3(t1, a, 13), 23)
	eq(f3(t2, a, 14), 24)

	eq(I2.Sum(t1, a, 15), 25)
	eq(I2.Sum(t2, a, 16), 26)

	f4 := I2.Sum
	eq(f4(t1, a, 17), 27)
	eq(f4(t2, a, 18), 28)
	
	mt1 := method4a.T1(4)
	mt2 := &method4a.T2{4}

	eq(mt1.Sum(a, 30), 40)
	eq(mt2.Sum(a, 31), 41)

	eq(method4a.T1.Sum(mt1, a, 32), 42)
	eq((*method4a.T2).Sum(mt2, a, 33), 43)

	g1 := method4a.T1.Sum
	eq(g1(mt1, a, 34), 44)
	g2 := (*method4a.T2).Sum
	eq(g2(mt2, a, 35), 45)

	eq(method4a.I1.Sum(mt1, a, 36), 46)
	eq(method4a.I1.Sum(mt2, a, 37), 47)

	g3 := method4a.I1.Sum
	eq(g3(mt1, a, 38), 48)
	eq(g3(mt2, a, 39), 49)

	eq(method4a.I2.Sum(mt1, a, 40), 50)
	eq(method4a.I2.Sum(mt2, a, 41), 51)

	g4 := method4a.I2.Sum
	eq(g4(mt1, a, 42), 52)
	eq(g4(mt2, a, 43), 53)
}
