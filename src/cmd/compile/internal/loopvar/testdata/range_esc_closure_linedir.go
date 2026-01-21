// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//line range_esc_closure_linedir.go:5
package main

import "fmt"

var is []func() int

func main() {
	var ints = []int{0, 0, 0}
	for i := range ints {
		is = append(is, func() int { return i })
	}

	for _, f := range is {
		fmt.Println(f())
		if f() != 2 {
			panic("loop variable i: expected shared per-loop, but got distinct per-iteration")
		}
	}
}
