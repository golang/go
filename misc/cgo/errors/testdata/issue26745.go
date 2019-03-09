// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// int a;
// void CF(int i) {}
import "C"

func F1(i int) int {
	return C.a + 1 // ERROR HERE: :13
}

func F2(i int) {
	C.CF(i) // ERROR HERE: :6
}
