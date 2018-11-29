// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Crash from call with two arguments that need pointer checking.
// No runtime test; just make sure it compiles.

package cgotest

/*
static void twoargs1(void *p, int n) {}
static void *twoargs2() { return 0; }
static int twoargs3(void * p) { return 0; }
*/
import "C"

import "unsafe"

func twoargsF() {
	v := []string{}
	C.twoargs1(C.twoargs2(), C.twoargs3(unsafe.Pointer(&v)))
}
