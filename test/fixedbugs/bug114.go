// $G $D/$F.go && $L $F.$A && (./$A.out || echo BUG: bug114 failed)

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const B32 = 1<<32 - 1
const C32 = (-1) & ((1 << 32) - 1)
const D32 = ^0

func main() {
	if B32 != 0xFFFFFFFF {
		println("1<<32 - 1 is", B32, "should be", 0xFFFFFFFF)
		panic("fail")
	}
	if C32 != 0xFFFFFFFF {
		println("(-1) & ((1<<32) - 1) is", C32, "should be", 0xFFFFFFFF)
		panic("fail")
	}
	if D32 != -1 {
		println("^0 is", D32, "should be", -1)
		panic("fail")
	}
}
