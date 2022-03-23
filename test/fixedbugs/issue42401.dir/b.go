// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	_ "unsafe"

	"./a"
)

//go:linkname s test/a.s
var s string

func main() {
	if a.Get() != "a" {
		panic("FAIL")
	}

	s = "b"
	if a.Get() != "b" {
		panic("FAIL")
	}
}
