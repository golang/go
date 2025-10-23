// errorcheck -0 -d=ssa/prove/debug=2

//go:build amd64 || arm64

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func f0i(x int) int {
	if x == 20 {
		return x // ERROR "Proved.+is constant 20$"
	}

	if (x + 20) == 20 {
		return x + 5 // ERROR "Proved.+is constant 0$" "Proved.+is constant 5$" "x\+d >=? w"
	}

	return x + 1
}

func f0u(x uint) uint {
	if x == 20 {
		return x // ERROR "Proved.+is constant 20$"
	}

	if (x + 20) == 20 {
		return x + 5 // ERROR "Proved.+is constant 0$" "Proved.+is constant 5$" "x\+d >=? w"
	}

	return x + 1
}
