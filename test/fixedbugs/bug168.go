// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var g byte = 123
var f *byte = &g
var b = make([]byte, 5)

func main() {
	b[0:1][0] = *f
	if b[0] != 123 {
		println("want 123 got", b[0])
		panic("fail")
	}
}
