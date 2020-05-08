// errorcheck

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that append arguments requirements are enforced by the
// compiler.

package main

func main() {

	s := make([]int, 8)

	_ = append()           // ERROR "missing arguments to append"
	_ = append(s...)       // ERROR "cannot use ... on first argument"
	_ = append(s, 2, s...) // ERROR "too many arguments to append"

	_ = append(s, make([]int, 0))     // ERROR "cannot use make.* as type int in append"
	_ = append(s, make([]int, -1)...) // ERROR "negative len argument in make"
}
