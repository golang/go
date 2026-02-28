// run

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	testMod()
	testMul()
}

//go:noinline
func mod3(x uint32) uint64 {
	return uint64(x % 3)
}

func testMod() {
	got := mod3(1<<32 - 1)
	want := uint64((1<<32 - 1) % 3)
	if got != want {
		fmt.Printf("testMod: got %x want %x\n", got, want)
	}

}

//go:noinline
func mul3(a uint32) uint64 {
	return uint64(a * 3)
}

func testMul() {
	got := mul3(1<<32 - 1)
	want := uint64((1<<32-1)*3 - 2<<32)
	if got != want {
		fmt.Printf("testMul: got %x want %x\n", got, want)
	}
}
