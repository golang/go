// $G $D/$F.go && $L $F.$A && ./$A.out

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
L:
	for i := 0; i < 1; i++ {
L1:
		for {
			break L;
		}
		panic("BUG: not reached - break");
	}

L2:
	for i := 0; i < 1; i++ {
L3:
		for {
			continue L2;
		}
		panic("BUG: not reached - continue");
	}
}
