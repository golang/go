// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func main() {
	var s = []byte("abc")
	ptr1 := *(*unsafe.Pointer)(unsafe.Pointer(&s))
	ptr2 := unsafe.Pointer(unsafe.SliceData(s))
	println(ptr1 == ptr2)
}
