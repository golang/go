// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func f(y int8) bool { return uint64(y) < 0xFFFFFFFF }

func main() {
	if f(-128) {
		fmt.Println("uint64(int8(-128)) < 0xFFFFFFFF = true, want false")
		panic("FAIL")
	}
}
