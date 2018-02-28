// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cgo converts C void* to Go unsafe.Pointer, so despite appearances C
// void** is Go *unsafe.Pointer. This test verifies that we detect the
// problem at build time.

package main

// typedef void v;
// void F(v** p) {}
import "C"

import "unsafe"

type v [0]byte

func f(p **v) {
	C.F((**C.v)(unsafe.Pointer(p))) // ERROR HERE
}

func main() {
	var p *v
	f(&p)
}
