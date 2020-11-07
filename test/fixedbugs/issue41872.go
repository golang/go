// run

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

package main

//go:noinline
func f8(x int32) bool {
	return byte(x&0xc0) == 64
}

//go:noinline
func f16(x int32) bool {
	return uint16(x&0x8040) == 64
}

func main() {
	if !f8(64) {
		panic("wanted true, got false")
	}
	if !f16(64) {
		panic("wanted true, got false")
	}
}
