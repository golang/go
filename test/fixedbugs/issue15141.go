// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	a := f(1, 99)
	b := g(0xFFFFFFe, 98)
	c := h(0xFFFFFFe, 98)
	println(a[1], b[1], c[1], a[0xFFFFFFe], b[0xFFFFFFe], c[0xFFFFFFe])
}

//go:noinline
func f(i, y int) (a [0xFFFFFFF]byte) {
	a[i] = byte(y)
	return
}

//go:noinline
func g(i, y int) [0xFFFFFFF]byte {
	var a [0xFFFFFFF]byte
	a[i] = byte(y)
	return a
}

//go:noinline
func h(i, y int) (a [0xFFFFFFF]byte) {
	a[i] = byte(y)
	return a
}
