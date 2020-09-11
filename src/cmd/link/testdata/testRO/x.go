// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that read-only data is indeed read-only. This
// program attempts to modify read-only data, and it
// should fail.

package main

import "unsafe"

var s = "hello"

func main() {
	println(s)
	*(*struct {
		p *byte
		l int
	})(unsafe.Pointer(&s)).p = 'H'
	println(s)
}
