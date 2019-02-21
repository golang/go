// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
typedef struct foo foo_t;
typedef struct bar bar_t;

foo_t *foop;
*/
import "C"

func main() {
	x := (*C.bar_t)(nil)
	C.foop = x // ERROR HERE
}
