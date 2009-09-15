// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


func main() {
	var u30 uint64 = 0;
	var u31 uint64 = 1;
	_, _ = u30, u31;
	var u32 uint64 = 18446744073709551615;
	var u33 uint64 = +18446744073709551615;
	if u32 != (1<<64)-1 { panic("u32\n"); }
	if u33 != (1<<64)-1 { panic("u33\n"); }
	var i34 int64 = ^0;  // note: 2's complement means ^0 == -1
	if i34 != -1 { panic("i34") }
}
/*
bug12.go:5: overflow converting constant to <uint64>UINT64
bug12.go:6: overflow converting constant to <uint64>UINT64
bug12.go:7: overflow converting constant to <uint64>UINT64
bug12.go:8: overflow converting constant to <uint64>UINT64
*/
