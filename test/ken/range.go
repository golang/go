// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test 'for range' on arrays, slices, and maps.

package main

const size = 16

var a [size]byte
var p []byte
var m map[int]byte

func f(k int) byte {
	return byte(k * 10007 % size)
}

func init() {
	p = make([]byte, size)
	m = make(map[int]byte)
	for k := 0; k < size; k++ {
		v := f(k)
		a[k] = v
		p[k] = v
		m[k] = v
	}
}

func main() {
	var i int

	/*
	 * key only
	 */
	i = 0
	for k := range a {
		v := a[k]
		if v != f(k) {
			println("key array range", k, v, a[k])
			panic("fail")
		}
		i++
	}
	if i != size {
		println("key array size", i)
		panic("fail")
	}

	i = 0
	for k := range p {
		v := p[k]
		if v != f(k) {
			println("key pointer range", k, v, p[k])
			panic("fail")
		}
		i++
	}
	if i != size {
		println("key pointer size", i)
		panic("fail")
	}

	i = 0
	for k := range m {
		v := m[k]
		if v != f(k) {
			println("key map range", k, v, m[k])
			panic("fail")
		}
		i++
	}
	if i != size {
		println("key map size", i)
		panic("fail")
	}

	/*
	 * key,value
	 */
	i = 0
	for k, v := range a {
		if v != f(k) {
			println("key:value array range", k, v, a[k])
			panic("fail")
		}
		i++
	}
	if i != size {
		println("key:value array size", i)
		panic("fail")
	}

	i = 0
	for k, v := range p {
		if v != f(k) {
			println("key:value pointer range", k, v, p[k])
			panic("fail")
		}
		i++
	}
	if i != size {
		println("key:value pointer size", i)
		panic("fail")
	}

	i = 0
	for k, v := range m {
		if v != f(k) {
			println("key:value map range", k, v, m[k])
			panic("fail")
		}
		i++
	}
	if i != size {
		println("key:value map size", i)
		panic("fail")
	}
}
