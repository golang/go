// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build unsafe

// This file contains a special test for the printf-checker that tests unsafe.Pointer.

package main

import (
	"fmt"
	"unsafe" // just for test case printing unsafe.Pointer
)

func UnsafePointerPrintfTest() {
	var up *unsafe.Pointer
	fmt.Printf("%p", up)
}
