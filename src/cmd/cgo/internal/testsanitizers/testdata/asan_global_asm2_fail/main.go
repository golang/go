// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var x uint64

func main() {
	bar(&x)
}

func bar(a *uint64) {
	p := (*uint64)(unsafe.Add(unsafe.Pointer(a), 1*unsafe.Sizeof(uint64(1))))
	if *p == 10 { // BOOM
		println("its value is 10")
	}
}
