// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var global bool
func use(b bool) { global = b }

func stringptr(s string) uintptr { return *(*uintptr)(unsafe.Pointer(&s)) }

func isfalse(b bool) {
	if b {
		// stack will explain where
		panic("wanted false, got true")
	}
}

func istrue(b bool) {
	if !b {
		// stack will explain where
		panic("wanted true, got false")
	}
}

type T *int

func main() {
	var a []int
	var b map[string]int

	var c string = "hello"
	var d string = "hel" // try to get different pointer
	d = d + "lo"
	if stringptr(c) == stringptr(d) {
		panic("compiler too smart -- got same string")
	}

	var e = make(chan int)

	var ia interface{} = a
	var ib interface{} = b
	var ic interface{} = c
	var id interface{} = d
	var ie interface{} = e

	// these comparisons are okay because
	// string compare is okay and the others
	// are comparisons where the types differ.
	isfalse(ia == ib)
	isfalse(ia == ic)
	isfalse(ia == id)
	isfalse(ib == ic)
	isfalse(ib == id)
	istrue(ic == id)
	istrue(ie == ie)

	istrue(ia != ib)
	istrue(ia != ic)
	istrue(ia != id)
	istrue(ib != ic)
	istrue(ib != id)
	isfalse(ic != id)
	isfalse(ie != ie)

	// these are not okay, because there is no comparison on slices or maps.
	//isfalse(a == ib)
	//isfalse(a == ic)
	//isfalse(a == id)
	//isfalse(b == ic)
	//isfalse(b == id)

	istrue(c == id)
	istrue(e == ie)

	//isfalse(ia == b)
	isfalse(ia == c)
	isfalse(ia == d)
	isfalse(ib == c)
	isfalse(ib == d)
	istrue(ic == d)
	istrue(ie == e)

	//istrue(a != ib)
	//istrue(a != ic)
	//istrue(a != id)
	//istrue(b != ic)
	//istrue(b != id)
	isfalse(c != id)
	isfalse(e != ie)

	//istrue(ia != b)
	istrue(ia != c)
	istrue(ia != d)
	istrue(ib != c)
	istrue(ib != d)
	isfalse(ic != d)
	isfalse(ie != e)

	// 6g used to let this go through as true.
	var g uint64 = 123
	var h int64 = 123
	var ig interface{} = g
	var ih interface{} = h
	isfalse(ig == ih)
	istrue(ig != ih)

	// map of interface should use == on interface values,
	// not memory.
	var m = make(map[interface{}]int)
	m[ic] = 1
	m[id] = 2
	if m[c] != 2 {
		println("m[c] = ", m[c])
		panic("bad m[c]")
	}

	// non-interface comparisons
	{
		c := make(chan int)
		c1 := (<-chan int)(c)
		c2 := (chan<- int)(c)
		istrue(c == c1)
		istrue(c == c2)
		istrue(c1 == c)
		istrue(c2 == c)

		isfalse(c != c1)
		isfalse(c != c2)
		isfalse(c1 != c)
		isfalse(c2 != c)

		d := make(chan int)
		isfalse(c == d)
		isfalse(d == c)
		isfalse(d == c1)
		isfalse(d == c2)
		isfalse(c1 == d)
		isfalse(c2 == d)

		istrue(c != d)
		istrue(d != c)
		istrue(d != c1)
		istrue(d != c2)
		istrue(c1 != d)
		istrue(c2 != d)
	}

	// named types vs not
	{
		var x = new(int)
		var y T
		var z T = x

		isfalse(x == y)
		istrue(x == z)
		isfalse(y == z)

		isfalse(y == x)
		istrue(z == x)
		isfalse(z == y)

		istrue(x != y)
		isfalse(x != z)
		istrue(y != z)

		istrue(y != x)
		isfalse(z != x)
		istrue(z != y)
	}

	// structs
	{
		var x = struct {
			x int
			y string
		}{1, "hi"}
		var y = struct {
			x int
			y string
		}{2, "bye"}
		var z = struct {
			x int
			y string
		}{1, "hi"}

		isfalse(x == y)
		isfalse(y == x)
		isfalse(y == z)
		isfalse(z == y)
		istrue(x == z)
		istrue(z == x)

		istrue(x != y)
		istrue(y != x)
		istrue(y != z)
		istrue(z != y)
		isfalse(x != z)
		isfalse(z != x)

		var m = make(map[struct {
			x int
			y string
		}]int)
		m[x] = 10
		m[y] = 20
		m[z] = 30
		istrue(m[x] == 30)
		istrue(m[y] == 20)
		istrue(m[z] == 30)
		istrue(m[x] != 10)
		isfalse(m[x] != 30)
		isfalse(m[y] != 20)
		isfalse(m[z] != 30)
		isfalse(m[x] == 10)

		var m1 = make(map[struct {
			x int
			y string
		}]struct {
			x int
			y string
		})
		m1[x] = x
		m1[y] = y
		m1[z] = z
		istrue(m1[x] == z)
		istrue(m1[y] == y)
		istrue(m1[z] == z)
		istrue(m1[x] == x)
		isfalse(m1[x] != z)
		isfalse(m1[y] != y)
		isfalse(m1[z] != z)
		isfalse(m1[x] != x)

		var ix, iy, iz interface{} = x, y, z

		isfalse(ix == iy)
		isfalse(iy == ix)
		isfalse(iy == iz)
		isfalse(iz == iy)
		istrue(ix == iz)
		istrue(iz == ix)

		isfalse(x == iy)
		isfalse(y == ix)
		isfalse(y == iz)
		isfalse(z == iy)
		istrue(x == iz)
		istrue(z == ix)

		isfalse(ix == y)
		isfalse(iy == x)
		isfalse(iy == z)
		isfalse(iz == y)
		istrue(ix == z)
		istrue(iz == x)

		istrue(ix != iy)
		istrue(iy != ix)
		istrue(iy != iz)
		istrue(iz != iy)
		isfalse(ix != iz)
		isfalse(iz != ix)

		istrue(x != iy)
		istrue(y != ix)
		istrue(y != iz)
		istrue(z != iy)
		isfalse(x != iz)
		isfalse(z != ix)

		istrue(ix != y)
		istrue(iy != x)
		istrue(iy != z)
		istrue(iz != y)
		isfalse(ix != z)
		isfalse(iz != x)
	}

	// arrays
	{
		var x = [2]string{"1", "hi"}
		var y = [2]string{"2", "bye"}
		var z = [2]string{"1", "hi"}

		isfalse(x == y)
		isfalse(y == x)
		isfalse(y == z)
		isfalse(z == y)
		istrue(x == z)
		istrue(z == x)

		istrue(x != y)
		istrue(y != x)
		istrue(y != z)
		istrue(z != y)
		isfalse(x != z)
		isfalse(z != x)

		var m = make(map[[2]string]int)
		m[x] = 10
		m[y] = 20
		m[z] = 30
		istrue(m[x] == 30)
		istrue(m[y] == 20)
		istrue(m[z] == 30)
		isfalse(m[x] != 30)
		isfalse(m[y] != 20)
		isfalse(m[z] != 30)

		var ix, iy, iz interface{} = x, y, z

		isfalse(ix == iy)
		isfalse(iy == ix)
		isfalse(iy == iz)
		isfalse(iz == iy)
		istrue(ix == iz)
		istrue(iz == ix)

		isfalse(x == iy)
		isfalse(y == ix)
		isfalse(y == iz)
		isfalse(z == iy)
		istrue(x == iz)
		istrue(z == ix)

		isfalse(ix == y)
		isfalse(iy == x)
		isfalse(iy == z)
		isfalse(iz == y)
		istrue(ix == z)
		istrue(iz == x)

		istrue(ix != iy)
		istrue(iy != ix)
		istrue(iy != iz)
		istrue(iz != iy)
		isfalse(ix != iz)
		isfalse(iz != ix)

		istrue(x != iy)
		istrue(y != ix)
		istrue(y != iz)
		istrue(z != iy)
		isfalse(x != iz)
		isfalse(z != ix)

		istrue(ix != y)
		istrue(iy != x)
		istrue(iy != z)
		istrue(iz != y)
		isfalse(ix != z)
		isfalse(iz != x)
	}

	shouldPanic(p1)
	shouldPanic(p2)
	shouldPanic(p3)
	shouldPanic(p4)
}

func p1() {
	var a []int
	var ia interface{} = a
	use(ia == ia)
}

func p2() {
	var b []int
	var ib interface{} = b
	use(ib == ib)
}

func p3() {
	var a []int
	var ia interface{} = a
	var m = make(map[interface{}]int)
	m[ia] = 1
}

func p4() {
	var b []int
	var ib interface{} = b
	var m = make(map[interface{}]int)
	m[ib] = 1
}

func shouldPanic(f func()) {
	defer func() {
		if recover() == nil {
			panic("function should panic")
		}
	}()
	f()
}
