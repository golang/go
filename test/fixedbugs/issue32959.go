// compile

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis with shifting constant

package main

import "unsafe"

func main() {
	var l uint64
	var p unsafe.Pointer
	_ = unsafe.Pointer(uintptr(p) + (uintptr(l) >> 1))
}
