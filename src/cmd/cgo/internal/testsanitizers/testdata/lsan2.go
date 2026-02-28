// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <stdlib.h>

int* test() {
   return malloc(sizeof(int));
}

void clearStack(int n) {
  if (n > 0) {
    clearStack(n - 1);
  }
}

*/
import "C"

var p *C.int

//go:noinline
func F() {
	p = C.test()
}

func clearStack(n int) {
	if n > 0 {
		clearStack(n - 1)
	}
}

func main() {
	// Test should pass: memory allocated by C does not leak
	// because a Go global variable points to it.
	F()
	clearStack(100)
	C.clearStack(100)
}
