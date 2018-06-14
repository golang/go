// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test trailing commas. DO NOT gofmt THIS FILE.

package main

var a = []int{1, 2, }
var b = [5]int{1, 2, 3, }
var c = []int{1, }
var d = [...]int{1, 2, 3, }

func main() {
	if len(a) != 2 {
		println("len a", len(a))
		panic("fail")
	}
	if len(b) != 5 {
		println("len b", len(b))
		panic("fail")
	}
	if len(c) != 1 {
		println("len d", len(c))
		panic("fail")
	}
	if len(d) != 3 {
		println("len c", len(d))
		panic("fail")
	}

	if a[0] != 1 {
		println("a[0]", a[0])
		panic("fail")
	}
	if a[1] != 2 {
		println("a[1]", a[1])
		panic("fail")
	}

	if b[0] != 1 {
		println("b[0]", b[0])
		panic("fail")
	}
	if b[1] != 2 {
		println("b[1]", b[1])
		panic("fail")
	}
	if b[2] != 3 {
		println("b[2]", b[2])
		panic("fail")
	}
	if b[3] != 0 {
		println("b[3]", b[3])
		panic("fail")
	}
	if b[4] != 0 {
		println("b[4]", b[4])
		panic("fail")
	}

	if c[0] != 1 {
		println("c[0]", c[0])
		panic("fail")
	}

	if d[0] != 1 {
		println("d[0]", d[0])
		panic("fail")
	}
	if d[1] != 2 {
		println("d[1]", d[1])
		panic("fail")
	}
	if d[2] != 3 {
		println("d[2]", d[2])
		panic("fail")
	}
}
