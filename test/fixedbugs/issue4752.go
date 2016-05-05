// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func F(xi, yi interface{}) uint64 {
	x, y := xi.(uint64), yi.(uint64)
	return x &^ y
}

func G(xi, yi interface{}) uint64 {
	return xi.(uint64) &^ yi.(uint64) // generates incorrect code
}

func main() {
	var x, y uint64 = 0, 1 << 63
	f := F(x, y)
	g := G(x, y)
	if f != 0 || g != 0 {
		println("F", f, "G", g)
		panic("bad")
	}
}
