// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: tuple evaluation order

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test order of evaluation in tuple assignments.

package main

var i byte = 0
var a [30]byte

func f() *byte {
	i++
	return &a[i-1]
}
func gbyte() byte {
	i++
	return 'a' + i - 1
}
func gint() byte {
	i++
	return i - 1
}
func x() (byte, byte) {
	i++
	return 'a' + i - 1, 'a' + i - 1
}
func e1(c chan byte, expected byte) chan byte {
	if i != expected {
		println("e1: got", i, "expected", expected)
		panic("fail")
	}
	i++
	return c
}

type Empty interface{}
type I interface {
	Get() byte
}
type S1 struct {
	i byte
}

func (p S1) Get() byte { return p.i }

type S2 struct {
	i byte
}

func e2(p Empty, expected byte) Empty {
	if i != expected {
		println("e2: got", i, "expected", expected)
		panic("fail")
	}
	i++
	return p
}
func e3(p *I, expected byte) *I {
	if i != expected {
		println("e3: got", i, "expected", expected)
		panic("fail")
	}
	i++
	return p
}

func main() {
	for i := range a {
		a[i] = ' '
	}

	// 0     1     2     3        4        5
	*f(), *f(), *f() = gbyte(), gbyte(), gbyte()

	// 6     7     8
	*f(), *f() = x()

	m := make(map[byte]byte)
	m[10] = 'A'
	var p1, p2 bool
	// 9           10
	*f(), p1 = m[gint()]
	// 11          12
	*f(), p2 = m[gint()]
	a[11] += '0'
	if !p1 || p2 {
		println("bad map check", i, p1, p2)
		panic("fail")
	}

	m[13] = 'B'
	//  13        14
	m[gint()] = gbyte(), false
	if _, present := m[13]; present {
		println("bad map removal")
		panic("fail")
	}

	c := make(chan byte, 1)
	c <- 'C'
	// 15          16
	*f(), p1 = <-e1(c, 16)
	// 17          18
	*f(), p2 = <-e1(c, 18)
	a[17] += '0'
	if !p1 || p2 {
		println("bad chan check", i, p1, p2)
		panic("fail")
	}

	s1 := S1{'D'}
	s2 := S2{'E'}
	var iv I
	// 19                20
	*e3(&iv, 19), p1 = e2(s1, 20).(I)
	// 21                22
	*e3(&iv, 21), p2 = e2(s2, 22).(I)
	if !p1 || p2 {
		println("bad interface check", i, p1, p2)
		panic("fail")
	}

	s := string(a[0:i])
	if s != "def   ii A 0   C 0     " {
		println("bad array results:", s)
		panic("fail")
	}
}
