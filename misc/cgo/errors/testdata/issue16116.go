// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// void f(void *p, int x) {}
import "C"

func main() {
	_ = C.f(1) // ERROR HERE
}
