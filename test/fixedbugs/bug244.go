// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var nf int
var ng int

func f() (int, int, int) {
	nf++
	return 1, 2, 3
}

func g() int {
	ng++
	return 4
}

var x, y, z = f()
var m = make(map[int]int)
var v, ok = m[g()]

func main() {
	if x != 1 || y != 2 || z != 3 || nf != 1 || v != 0 || ok != false || ng != 1 {
		println("x=", x, " y=", y, " z=", z, " nf=", nf, " v=", v, " ok=", ok, " ng=", ng)
		panic("fail")
	}
}
