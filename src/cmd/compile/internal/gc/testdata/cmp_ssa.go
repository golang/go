// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// cmp_ssa.go tests compare simplification operations.
package main

import "fmt"

var failed = false

//go:noinline
func eq_ssa(a int64) bool {
	return 4+a == 10
}

//go:noinline
func neq_ssa(a int64) bool {
	return 10 != a+4
}

func testCmp() {
	if wanted, got := true, eq_ssa(6); wanted != got {
		fmt.Printf("eq_ssa: expected %v, got %v\n", wanted, got)
		failed = true
	}
	if wanted, got := false, eq_ssa(7); wanted != got {
		fmt.Printf("eq_ssa: expected %v, got %v\n", wanted, got)
		failed = true
	}

	if wanted, got := false, neq_ssa(6); wanted != got {
		fmt.Printf("neq_ssa: expected %v, got %v\n", wanted, got)
		failed = true
	}
	if wanted, got := true, neq_ssa(7); wanted != got {
		fmt.Printf("neq_ssa: expected %v, got %v\n", wanted, got)
		failed = true
	}
}

func main() {
	testCmp()

	if failed {
		panic("failed")
	}
}
