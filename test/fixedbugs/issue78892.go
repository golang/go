// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f(x int) int {
        if x == 1 {
                return 7
        }
        return 7 / (x - 1)
}

//go:noinline
func g(x int) int {
	r := 0
	for range 5 {
		r += f(x)
	}
	return r
}

func main() {
	g(1)
}
