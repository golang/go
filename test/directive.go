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

// ok
//go:notinheap
type T1 int

//go:notinheap // ERROR "misplaced compiler directive"
type (
	//go:notinheap
	//go:noinline // ERROR "misplaced compiler directive"
	T2  int //go:notinheap // ERROR "misplaced compiler directive"
	T2b int
	//go:notinheap
	T2c int
	//go:noinline // ERROR "misplaced compiler directive"
	T3 int
)

//go:notinheap // ERROR "misplaced compiler directive"
type (
	//go:notinheap
	T4 int
)

//go:notinheap // ERROR "misplaced compiler directive"
type ()

type T5 int

func g() {} //go:noinline // ERROR "misplaced compiler directive"

// ok: attached to f (duplicated yes, but ok)
//go:noinline

//go:noinline
func f() {
	//go:noinline // ERROR "misplaced compiler directive"
	x := 1

	//go:noinline // ERROR "misplaced compiler directive"
	{
		_ = x //go:noinline // ERROR "misplaced compiler directive"
	}
	//go:noinline // ERROR "misplaced compiler directive"
	var y int //go:noinline // ERROR "misplaced compiler directive"
	//go:noinline // ERROR "misplaced compiler directive"
	_ = y

	//go:noinline // ERROR "misplaced compiler directive"
	const c = 1

	//go:noinline // ERROR "misplaced compiler directive"
	_ = func() {}

	//go:noinline // ERROR "misplaced compiler directive"
	// ok:
	//go:notinheap
	type T int
}

// someday there might be a directive that can apply to type aliases, but go:notinheap doesn't.
//go:notinheap // ERROR "misplaced compiler directive"
type T6 = int

// EOF
//go:noinline // ERROR "misplaced compiler directive"
