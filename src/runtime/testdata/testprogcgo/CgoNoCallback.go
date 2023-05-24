// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !plan9 && !windows
// +build !plan9,!windows

package main

// #cgo noescape/nocallback annotations for a C function means it should not callback to Go.
// But it do callback to go in this test, Go should crash here.

/*
#cgo nocallback runCShouldNotCallback
#cgo noescape runCShouldNotCallback

extern void CallbackToGo();

static void runCShouldNotCallback() {
	CallbackToGo();
}
*/
import "C"

import (
	"fmt"
)

func init() {
	register("CgoNoCallback", CgoNoCallback)
}

//export CallbackToGo
func CallbackToGo() {
}

func CgoNoCallback() {
	C.runCShouldNotCallback()
	fmt.Println("OK")
}
