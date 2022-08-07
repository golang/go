// errorcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that misplaced directives are diagnosed.

// ok
//go:build !ignore

package main

//go:build bad // ERROR "misplaced compiler directive"

//go:noinline // ERROR "misplaced compiler directive"
type (
	T2  int //go:noinline // ERROR "misplaced compiler directive"
	T2b int
	T2c int
	T3  int
)

//go:noinline // ERROR "misplaced compiler directive"
type (
	T4 int
)

//go:noinline // ERROR "misplaced compiler directive"
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
	var y int //go:noinline // ERROR "misplaced compiler directive"
	//go:noinline // ERROR "misplaced compiler directive"
	_ = y

	const c = 1

	_ = func() {}
}

// EOF
//go:noinline // ERROR "misplaced compiler directive"
