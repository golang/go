// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Linkname "var" to reference newcoro is not allowed.

package main

import "unsafe"

func main() {
	call(&newcoro)
}

//go:linkname newcoro runtime.newcoro
var newcoro unsafe.Pointer

//go:noinline
func call(*unsafe.Pointer) {
	// not implemented
}
