// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	var i, k int;
	outer:
	for k=0; k<2; k++ {
		print("outer loop top k ", k, "\n");
		if k != 0 { panic("k not zero") }  // inner loop breaks this one every time
		for i=0; i<2; i++ {
			if i != 0 { panic("i not zero") }  // loop breaks every time
			print("inner loop top i ", i, "\n");
			if true {
				print("do break\n");
				break outer;
			}
		}
	}
	print("broke\n");
}
