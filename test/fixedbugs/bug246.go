// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug246

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func main() {
	// works
	addr := uintptr(0x234)
	x1 := (*int)(unsafe.Pointer(addr))

	// fails
	x2 := (*int)(unsafe.Pointer(uintptr(0x234)))

	if x1 != x2 {
		println("mismatch", x1, x2)
		panic("fail")
	}
}
