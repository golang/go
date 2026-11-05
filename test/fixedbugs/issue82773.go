// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func f(x int8) int8 { return (x - (-x)) >> 8 }

func main() {
	if got := f(-1); got != -1 {
		fmt.Printf("int8(x-(-x))>>8 with x=-1 = %d, want -1\n", got)
		panic("FAIL")
	}
}
