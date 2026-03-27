// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"unsafe"
)

type S struct {
	p *byte
	a string
	b string
	c int64
	d int64
}

func main() {
	s := &S{p: nil, a: "foo", b: "foo", c: 0, d: 0}
	s.a = ""
	s.b = "bar"
	s.c = 33

	z := (*[2]uintptr)(unsafe.Pointer(&s.a))
	fmt.Printf("%x %x\n", z[0], z[1])
}
