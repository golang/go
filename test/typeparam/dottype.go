// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f[T any](x interface{}) T {
	return x.(T)
}
func f2[T any](x interface{}) (T, bool) {
	t, ok := x.(T)
	return t, ok
}

type I interface {
	foo()
}

type myint int

func (myint) foo() {
}

type myfloat float64

func (myfloat) foo() {
}

func g[T I](x I) T {
	return x.(T)
}
func g2[T I](x I) (T, bool) {
	t, ok := x.(T)
	return t, ok
}

func h[T any](x interface{}) struct{ a, b T } {
	return x.(struct{ a, b T })
}

func k[T any](x interface{}) interface{ bar() T } {
	return x.(interface{ bar() T })
}

type mybar int

func (x mybar) bar() int {
	return int(x)
}

func main() {
	var i interface{} = int(3)
	var j I = myint(3)
	var x interface{} = float64(3)
	var y I = myfloat(3)

	println(f[int](i))
	shouldpanic(func() { f[int](x) })
	println(f2[int](i))
	println(f2[int](x))

	println(g[myint](j))
	shouldpanic(func() { g[myint](y) })
	println(g2[myint](j))
	println(g2[myint](y))

	println(h[int](struct{ a, b int }{3, 5}).a)

	println(k[int](mybar(3)).bar())

	type large struct {a,b,c,d,e,f int}
	println(f[large](large{}).a)
	l2, ok := f2[large](large{})
	println(l2.a, ok)
}
func shouldpanic(x func()) {
	defer func() {
		e := recover()
		if e == nil {
			panic("didn't panic")
		}
	}()
	x()
}
