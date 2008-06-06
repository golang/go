// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main


func main() {
	var u30 uint64 = 0;
	var u31 uint64 = 1;
	var u32 uint64 = 18446744073709551615;
	var u33 uint64 = +18446744073709551615;
	if u32 != ^0 { panic "u32\n"; }
	if u33 != ^0 { panic "u33\n"; }
}
/*
bug12.go:5: overflow converting constant to <uint64>UINT64
bug12.go:6: overflow converting constant to <uint64>UINT64
bug12.go:7: overflow converting constant to <uint64>UINT64
bug12.go:8: overflow converting constant to <uint64>UINT64
*/
