// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var i = 257

func main() {
	var buf [10]byte
	p0 := unsafe.Pointer(&buf[0])
	p1 := unsafe.Pointer(&buf[1])

	if p := unsafe.Add(p0, uint8(i)); p != p1 {
		println("FAIL:", p, "!=", p1)
	}

	var x uint8
	if i != 0 {
		x = 1
	}
	if p := unsafe.Add(p0, x); p != p1 {
		println("FAIL:", p, "!=", p1)
	}
}
