// build

//go:build cgo

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#cgo CFLAGS: -Werror -Wimplicit-function-declaration

#include <stdio.h>

static void CFn(_GoString_ gostr) {
	printf("%.*s\n", _GoStringLen(gostr), _GoStringPtr(gostr));
}
*/
import "C"

func main() {
	C.CFn("hello, world")
}

// The bug only occurs if there is an exported function.
//export Fn
func Fn() {
}
