// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "fmt"

//go:noinline
func bug1(a []int, i int) int {
	if i < 0 || i > 20-len(a) {
		return 0
	}
	diff := len(a) - i
	if diff < 10 {
		return 1
	}
	return 2
}

//go:noinline
func bug2(s []int, i int) int {
	if i < 0 {
		return 0
	}
	if i <= 10-len(s) {
		x := len(s) - i
		return x / 2
	}
	return 0
}

func main() {
	if got := bug1(make([]int, 5), 15); got != 1 {
		panic(fmt.Sprintf("bug1: got %d, want 1", got))
	}
	if got := bug2(make([]int, 3), 7); got != -2 {
		panic(fmt.Sprintf("bug2: got %d, want -2", got))
	}
}
