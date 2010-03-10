// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var a [12]complex
var s []complex
var c chan complex
var f struct {
	c complex
}
var m map[complex]complex

func main() {
	// array of complex
	for i := 0; i < len(a); i++ {
		a[i] = cmplx(float(i), float(-i))
	}
	println(a[5])

	// slice of complex
	s = make([]complex, len(a))
	for i := 0; i < len(s); i++ {
		s[i] = a[i]
	}
	println(s[5])

	// chan
	c = make(chan complex)
	go chantest(c)
	println(<-c)

	// pointer of complex
	v := a[5]
	pv := &v
	println(*pv)

	// field of complex
	f.c = a[5]
	println(f.c)

	// map of complex
	m = make(map[complex]complex)
	for i := 0; i < len(s); i++ {
		m[-a[i]] = a[i]
	}
	println(m[5i-5])
	println(m[cmplx(-5, 5)])
}

func chantest(c chan complex) { c <- a[5] }
