// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// #cgo nocallback annotations for a C function means it should not callback to Go.
// But it do callback to go in this test, Go should crash here.

/*
// TODO(#56378): #cgo nocallback runCShouldNotCallback
extern void runCShouldNotCallback();
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
