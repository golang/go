// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S int

type T struct {
	a int
	S
}

//go:noinline
func (s *S) M(a int, x [2]int, b float64, y [2]float64) (S, int, [2]int, float64, [2]float64) {
	return *s, a, x, b, y
}

var s S = 42
var t = &T{S: s}

var fn = (*T).M // force a method wrapper
var fn0 = (*T0).NilCheck
var fn1 = (*T1).NilCheck
var valueFn0 = (*ValueT0).ValueNilCheck
var valueFn1 = (*ValueT1).ValueNilCheck
var onlyReaderFn = (*TOnlyReader).Read
var closeReaderFn = (*TCloseReader).Read

func main() {
	a := 123
	x := [2]int{456, 789}
	b := 1.2
	y := [2]float64{3.4, 5.6}
	s1, a1, x1, b1, y1 := fn(t, a, x, b, y)
	if a1 != a || x1 != x || b1 != b || y1 != y || s1 != s {
		panic("FAIL")
	}
	testNilWrapper("offset zero", (*T0)(nil))
	testNilWrapper("offset non-zero", (*T1)(nil))
	testNilMethodExpr("offset zero method expression", func() { fn0((*T0)(nil)) })
	testNilMethodExpr("offset non-zero method expression", func() { fn1((*T1)(nil)) })
	testValueNilWrapper("value offset zero", (*ValueT0)(nil))
	testValueNilWrapper("value offset non-zero", (*ValueT1)(nil))
	testNilMethodExpr("value offset zero method expression", func() { valueFn0((*ValueT0)(nil)) })
	testNilMethodExpr("value offset non-zero method expression", func() { valueFn1((*ValueT1)(nil)) })
	testEmbeddedInterfaceWrapper()
}

type E struct{}

var called bool

func (*E) NilCheck() {
	called = true
}

func (E) ValueNilCheck() {
	called = true
}

type T0 struct {
	E
}

type T1 struct {
	x int
	E
}

type I interface {
	NilCheck()
}

type ValueT0 struct {
	E
}

type ValueT1 struct {
	x int
	E
}

type ValueI interface {
	ValueNilCheck()
}

func testNilWrapper(name string, x I) {
	wantNilPanic(name, func() { x.NilCheck() })
}

func testValueNilWrapper(name string, x ValueI) {
	wantNilPanic(name, func() { x.ValueNilCheck() })
}

func testNilMethodExpr(name string, call func()) {
	wantNilPanic(name, call)
}

func wantNilPanic(name string, call func()) {
	called = false
	defer func() {
		if recover() == nil {
			panic(name + ": want panic")
		}
		if called {
			panic(name + ": called embedded method")
		}
	}()
	call()
}

type onlyReader interface {
	Read([]byte) (int, error)
}

type TOnlyReader struct {
	onlyReader
}

type closeReader interface {
	Close() error
	Read([]byte) (int, error)
}

type TCloseReader struct {
	closeReader
}

type embeddedReader byte

func (r embeddedReader) Read(p []byte) (int, error) {
	p[0] = byte(r)
	return 1, nil
}

type embeddedReadCloser byte

func (embeddedReadCloser) Close() error {
	panic("called Close")
}

func (r embeddedReadCloser) Read(p []byte) (int, error) {
	p[0] = byte(r)
	return 1, nil
}

func testEmbeddedInterfaceWrapper() {
	var x interface {
		Read([]byte) (int, error)
	} = &TOnlyReader{embeddedReader(1)}
	buf := []byte{0}
	if n, err := x.Read(buf); n != 1 || err != nil || buf[0] != 1 {
		panic("bad embedded interface wrapper, slot 0")
	}

	x = &TCloseReader{embeddedReadCloser(2)}
	buf[0] = 0
	if n, err := x.Read(buf); n != 1 || err != nil || buf[0] != 2 {
		panic("bad embedded interface wrapper, non-zero slot")
	}

	buf[0] = 0
	if n, err := onlyReaderFn(&TOnlyReader{embeddedReader(3)}, buf); n != 1 || err != nil || buf[0] != 3 {
		panic("bad embedded interface method expression, slot 0")
	}

	buf[0] = 0
	if n, err := closeReaderFn(&TCloseReader{embeddedReadCloser(4)}, buf); n != 1 || err != nil || buf[0] != 4 {
		panic("bad embedded interface method expression, non-zero slot")
	}
}
