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
		panicln("e1: got", i, "expected", expected)
	}
	i++
	return c
}

type Empty interface {}
type I interface {
	Get() byte
}
type S1 struct {
	i byte
}
func (p S1) Get() byte {
	return p.i
}
type S2 struct {
	i byte
}
func e2(p Empty, expected byte) Empty {
	if i != expected {
		panicln("e2: got", i, "expected", expected)
	}
	i++
	return p
}
func e3(p *I, expected byte) *I {
	if i != expected {
		panicln("e3: got", i, "expected", expected)
	}
	i++
	return p
}

func main() {
	for i := range a {
		a[i] = ' '
	}

	*f(), *f(), *f() = gbyte(), gbyte(), gbyte()

	*f(), *f() = x()

	m := make(map[byte]byte)
	m[10] = 'A'
	var p1, p2 bool
	*f(), p1 = m[gint()]
	*f(), p2 = m[gint()]
	if !p1 || p2 {
		panicln("bad map check", i, p1, p2)
	}

	m[13] = 'B'
	m[gint()] = gbyte(), false
	if _, present := m[13]; present {
		panicln("bad map removal")
	}

	c := make(chan byte, 1)
	c <- 'C'
	*f(), p1 = <-e1(c, 16)
	*f(), p2 = <-e1(c, 18)
	if !p1 || p2 {
		panicln("bad chan check", i, p1, p2)
	}

	s1 := S1{'D'}
	s2 := S2{'E'}
	var iv I
	*e3(&iv, 19), p1 = e2(s1, 20).(I)
	*e3(&iv, 21), p2 = e2(s2, 22).(I)
	if !p1 || p2 {
		panicln("bad interface check", i, p1, p2)
	}

	s := string(a[0:i])
	if s != "def   ii A     C       " {
		panicln("bad array results:", s)
	}
}
