// $G $D/$F.go && $L $F.$A && ./$A.out || echo BUG: bug341

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used to panic because 8g was generating incorrect
// code for converting a negative float to a uint64.

package main

func main() {
	var x float32 = -2.5

	_ = uint64(x)
	_ = float32(0)
}
/*
panic: runtime error: floating point error

[signal 0x8 code=0x6 addr=0x8048c64 pc=0x8048c64]
*/
