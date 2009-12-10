// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func
main() {
	a := 3;
	for i:=0; i<10; i=i+1 {
		switch(i) {
		case 5:
			print("five");
		case a,7:
			print("a");
		default:
			print(i);
		}
		print("out", i);
	}
	print("\n");
}
