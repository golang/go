// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test compound types made of complex numbers.

package main

var a [12]complex128
var s []complex128
var c chan complex128
var f struct {
	c complex128
}
var m map[complex128]complex128

func main() {
	// array of complex128
	for i := 0; i < len(a); i++ {
		a[i] = complex(float64(i), float64(-i))
	}
	if a[5] != 5-5i {
		panic(a[5])
	}

	// slice of complex128
	s = make([]complex128, len(a))
	for i := 0; i < len(s); i++ {
		s[i] = a[i]
	}
	if s[5] != 5-5i {
		panic(s[5])
	}

	// chan
	c = make(chan complex128)
	go chantest(c)
	vc := <-c
	if vc != 5-5i {
		panic(vc)
	}

	// pointer of complex128
	v := a[5]
	pv := &v
	if *pv != 5-5i {
		panic(*pv)
	}

	// field of complex128
	f.c = a[5]
	if f.c != 5-5i {
		panic(f.c)
	}

	// map of complex128
	m = make(map[complex128]complex128)
	for i := 0; i < len(s); i++ {
		m[-a[i]] = a[i]
	}
	if m[5i-5] != 5-5i {
		panic(m[5i-5])
	}
	vm := m[complex(-5, 5)]
	if vm != 5-5i {
		panic(vm)
	}
}

func chantest(c chan complex128) { c <- a[5] }
