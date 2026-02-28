// run -race

//go:build race && cgo

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
   int v[8192];
*/
import "C"

var x [8192]C.int

func main() {
	copy(C.v[:], x[:])
}
