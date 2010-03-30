// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Check uses of all the different interface
// conversion runtime functions.

package main

type Stringer interface {
	String() string
}
type StringLengther interface {
	String() string
	Length() int
}
type Empty interface{}

type T string

func (t T) String() string {
	return string(t)
}
func (t T) Length() int {
	return len(t)
}

type U string

func (u U) String() string {
	return string(u)
}

var t = T("hello")
var u = U("goodbye")
var e Empty
var s Stringer = t
var sl StringLengther = t
var i int
var ok bool

func hello(s string) {
	if s != "hello" {
		println("not hello: ", s)
		panic("fail")
	}
}

func five(i int) {
	if i != 5 {
		println("not 5: ", i)
		panic("fail")
	}
}

func true(ok bool) {
	if !ok {
		panic("not true")
	}
}

func false(ok bool) {
	if ok {
		panic("not false")
	}
}

func main() {
	// T2I
	s = t
	hello(s.String())

	// I2T
	t = s.(T)
	hello(t.String())

	// T2E
	e = t

	// E2T
	t = e.(T)
	hello(t.String())

	// T2I again
	sl = t
	hello(sl.String())
	five(sl.Length())

	// I2I static
	s = sl
	hello(s.String())

	// I2I dynamic
	sl = s.(StringLengther)
	hello(sl.String())
	five(sl.Length())

	// I2E (and E2T)
	e = s
	hello(e.(T).String())

	// E2I
	s = e.(Stringer)
	hello(s.String())

	// I2T2 true
	t, ok = s.(T)
	true(ok)
	hello(t.String())

	// I2T2 false
	_, ok = s.(U)
	false(ok)

	// I2I2 true
	sl, ok = s.(StringLengther)
	true(ok)
	hello(sl.String())
	five(sl.Length())

	// I2I2 false (and T2I)
	s = u
	sl, ok = s.(StringLengther)
	false(ok)

	// E2T2 true
	t, ok = e.(T)
	true(ok)
	hello(t.String())

	// E2T2 false
	i, ok = e.(int)
	false(ok)

	// E2I2 true
	sl, ok = e.(StringLengther)
	true(ok)
	hello(sl.String())
	five(sl.Length())

	// E2I2 false (and T2E)
	e = u
	sl, ok = e.(StringLengther)
	false(ok)
}
