// run fake-arg-to-force-use-of-go-run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo
// +build cgo

package main

// void f(int *p) { *p = 0x12345678; }
import "C"

func main() {
	var x C.int
	func() {
		defer C.f(&x)
	}()
	if x != 0x12345678 {
		panic("FAIL")
	}
}
