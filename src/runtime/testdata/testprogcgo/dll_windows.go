// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

/*
#include <windows.h>

DWORD getthread() {
	return GetCurrentThreadId();
}
*/
import "C"
import "./windows"

func init() {
	register("CgoDLLImportsMain", CgoDLLImportsMain)
}

func CgoDLLImportsMain() {
	C.getthread()
	windows.GetThread()
	println("OK")
}
