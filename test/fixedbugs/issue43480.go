// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue #43480: ICE on large uint64 constants in switch cases.

package main

func isPow10(x uint64) bool {
	switch x {
	case 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
		1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19:
		return true
	}
	return false
}

func main() {
	var x uint64 = 1

	for {
		if !isPow10(x) || isPow10(x-1) || isPow10(x+1) {
			panic(x)
		}
		next := x * 10
		if next/10 != x {
			break // overflow
		}
		x = next
	}
}
