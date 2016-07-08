// compile

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var x = unsafe.Pointer(uintptr(0))

func main() {
	_ = map[unsafe.Pointer]int{unsafe.Pointer(uintptr(0)): 0}
}
