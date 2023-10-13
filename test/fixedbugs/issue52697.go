// errorcheck

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !386 && !amd64p32 && !arm && !mips && !mipsle

package main

func g() { // GC_ERROR "stack frame too large"
	xs := [3000 * 2000][33]int{}
	for _, x := range xs {
		if len(x) > 50 {

		}
	}
}

func main() { // GC_ERROR "stack frame too large"
	defer f()
	g()
}

func f() {}
