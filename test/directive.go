// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that misplaced directives are diagnosed.

//go:noinline // ERROR "misplaced compiler directive"

//go:noinline // ERROR "misplaced compiler directive"
package main

//go:nosplit
func f1() {}

//go:nosplit
//go:noinline
func f2() {}

//go:noinline // ERROR "misplaced compiler directive"

//go:noinline // ERROR "misplaced compiler directive"
var x int

//go:noinline // ERROR "misplaced compiler directive"
const c = 1

//go:noinline // ERROR "misplaced compiler directive"
type T int

type (
	//go:noinline // ERROR "misplaced compiler directive"
	T2 int
	//go:noinline // ERROR "misplaced compiler directive"
	T3 int
)

//go:noinline
func f() {
	x := 1

	{
		_ = x
	}
	//go:noinline // ERROR "misplaced compiler directive"
	var y int
	_ = y

	//go:noinline // ERROR "misplaced compiler directive"
	const c = 1

	_ = func() {}

	//go:noinline // ERROR "misplaced compiler directive"
	type T int
}
