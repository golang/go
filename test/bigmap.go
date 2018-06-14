// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Internally a map holds elements in up to 255 bytes of key+value.
// When key or value or both are too large, it uses pointers to key+value
// instead.  Test all the combinations.

package main

func seq(x, y int) [1000]byte {
	var r [1000]byte
	for i := 0; i < len(r); i++ {
		r[i] = byte(x + i*y)
	}
	return r
}

func cmp(x, y [1000]byte) {
	for i := 0; i < len(x); i++ {
		if x[i] != y[i] {
			panic("BUG mismatch")
		}
	}
}

func main() {
	m := make(map[int][1000]byte)
	m[1] = seq(11, 13)
	m[2] = seq(2, 9)
	m[3] = seq(3, 17)

	cmp(m[1], seq(11, 13))
	cmp(m[2], seq(2, 9))
	cmp(m[3], seq(3, 17))
	

	{
		type T [1]byte
		type V [1]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
  	}
	{
		type T [100]byte
		type V [1]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [1]byte
		type V [100]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [1000]byte
		type V [1]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [1]byte
		type V [1000]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [1000]byte
		type V [1000]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [200]byte
		type V [1]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [1]byte
		type V [200]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
		}
	}
	{
		type T [200]byte
		type V [200]byte
		m := make(map[T]V)
		m[T{}] = V{1}
		m[T{1}] = V{2}
		if x, y := m[T{}][0], m[T{1}][0]; x != 1 || y != 2 {
			println(x, y)
			panic("bad map")
  		}
  	}
}
