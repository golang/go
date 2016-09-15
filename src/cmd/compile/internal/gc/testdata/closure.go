// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// map_ssa.go tests map operations.
package main

import "fmt"

var failed = false

//go:noinline
func testCFunc_ssa() int {
	a := 0
	b := func() {
		switch {
		}
		a++
	}
	b()
	b()
	return a
}

func testCFunc() {
	if want, got := 2, testCFunc_ssa(); got != want {
		fmt.Printf("expected %d, got %d", want, got)
		failed = true
	}
}

func main() {
	testCFunc()

	if failed {
		panic("failed")
	}
}
