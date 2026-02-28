// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// typedef struct { int a; void* ptr; } S;
// static void f(S* p) {}
import "C"

func main() {
	C.f(&C.S{
		a: 1+

			(3 + ""), // ERROR HERE

		ptr: nil,
	})
}
