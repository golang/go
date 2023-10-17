// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func repro(b []byte, bit int32) {
	_ = b[3]
	v := uint32(b[0]) | uint32(b[1])<<8 | uint32(b[2])<<16 | uint32(b[3])<<24 | 1<<(bit&31)
	b[0] = byte(v)
	b[1] = byte(v >> 8)
	b[2] = byte(v >> 16)
	b[3] = byte(v >> 24)
}

func main() {
	var b [8]byte
	repro(b[:], 32)
	want := [8]byte{1, 0, 0, 0, 0, 0, 0, 0}
	if b != want {
		panic(fmt.Sprintf("got %v, want %v\n", b, want))
	}
}
