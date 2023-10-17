// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func main() {
	L: ;  // ';' terminates empty statement => L does not apply to for loop
	for i := 0; i < 10; i++ {
		println(i);
		break L;  // ERROR "L"
	}

	L1: { // L1 labels block => L1 does not apply to for loop
		for i := 0; i < 10; i++ {
			println(i);
			break L1;  // ERROR "L1"
		}
	}
}
