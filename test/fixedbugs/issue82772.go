// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func f(x, k int8, s uint64) int8 { return (x * k) >> s }

func main() {
	if got := f(127, -15, 8); got != -1 {
		fmt.Printf("int8(127*-15) >> 8 = %d, want -1\n", got)
		panic("FAIL")
	}
}
