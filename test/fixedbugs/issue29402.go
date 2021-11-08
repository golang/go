// run
  
// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 29402: wrong optimization of comparison of
// constant and shift on MIPS.

package main

//go:noinline
func F(s []int) bool {
	half := len(s) / 2
	return half >= 0
}

func main() {
	b := F([]int{1, 2, 3, 4})
	if !b {
		panic("FAIL")
	}
}
