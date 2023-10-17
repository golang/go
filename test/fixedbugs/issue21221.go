// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func main() {
	if unsafe.Pointer(uintptr(0)) != unsafe.Pointer(nil) {
		panic("fail")
	}
	if (*int)(unsafe.Pointer(uintptr(0))) != (*int)(nil) {
		panic("fail")
	}
}
