// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cgo should reject the use of mangled C names.

package main

/*
typedef struct a {
	int i;
} a;
void fn(void) {}
*/
import "C"

type B _Ctype_struct_a // ERROR HERE

var a _Ctype_struct_a // ERROR HERE

type A struct {
	a *_Ctype_struct_a // ERROR HERE
}

var notExist _Ctype_NotExist // ERROR HERE

func main() {
	_Cfunc_fn() // ERROR HERE
}
