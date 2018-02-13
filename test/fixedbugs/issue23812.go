// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

func main() {
	want := int32(0x3edae8)
	got := foo(1)
	if want != got {
		panic(fmt.Sprintf("want %x, got %x", want, got))
	}
}

func foo(a int32) int32 {
	return shr1(int32(shr2(int64(0x14ff6e2207db5d1f), int(a))), 4)
}

func shr1(n int32, m int) int32 { return n >> uint(m) }

func shr2(n int64, m int) int64 {
	if m < 0 {
		m = -m
	}
	if m >= 64 {
		return n
	}

	return n >> uint(m)
}
