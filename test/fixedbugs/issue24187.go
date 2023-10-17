// +build amd64p32
// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"unsafe"
)

func main() {
	b := make([]byte, 128)
	for i := range b {
		b[i] = 1
	}
	if bytes.IndexByte(b, 0) != -1 {
		panic("found 0")
	}
	for i := range b {
		b[i] = 0
		c := b
		*(*int)(unsafe.Pointer(uintptr(unsafe.Pointer(&c)) + unsafe.Sizeof(uintptr(0)))) = 1<<31 - 1
		if bytes.IndexByte(c, 0) != i {
			panic(fmt.Sprintf("missing 0 at %d\n", i))
		}
		b[i] = 1
	}
}
