// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

//go:noinline
func f(a, b uint) int {
	return int(a-b) / 8
}

func main() {
	if x := f(1, 2); x != 0 {
		panic(x)
	}
}
