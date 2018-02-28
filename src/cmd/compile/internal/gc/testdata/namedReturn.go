// run

// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This test makes sure that naming named
// return variables in a return statement works.
// See issue #14904.

package main

import (
	"fmt"
	"runtime"
)

// Our heap-allocated object that will be GC'd incorrectly.
// Note that we always check the second word because that's
// where 0xdeaddeaddeaddead is written.
type B [4]int

// small (SSAable) array
type T1 [3]*B

//go:noinline
func f1() (t T1) {
	t[0] = &B{91, 92, 93, 94}
	runtime.GC()
	return t
}

// large (non-SSAable) array
type T2 [8]*B

//go:noinline
func f2() (t T2) {
	t[0] = &B{91, 92, 93, 94}
	runtime.GC()
	return t
}

// small (SSAable) struct
type T3 struct {
	a, b, c *B
}

//go:noinline
func f3() (t T3) {
	t.a = &B{91, 92, 93, 94}
	runtime.GC()
	return t
}

// large (non-SSAable) struct
type T4 struct {
	a, b, c, d, e, f *B
}

//go:noinline
func f4() (t T4) {
	t.a = &B{91, 92, 93, 94}
	runtime.GC()
	return t
}

var sink *B

func f5() int {
	b := &B{91, 92, 93, 94}
	t := T4{b, nil, nil, nil, nil, nil}
	sink = b   // make sure b is heap allocated ...
	sink = nil // ... but not live
	runtime.GC()
	t = t
	return t.a[1]
}

func main() {
	failed := false

	if v := f1()[0][1]; v != 92 {
		fmt.Printf("f1()[0][1]=%d, want 92\n", v)
		failed = true
	}
	if v := f2()[0][1]; v != 92 {
		fmt.Printf("f2()[0][1]=%d, want 92\n", v)
		failed = true
	}
	if v := f3().a[1]; v != 92 {
		fmt.Printf("f3().a[1]=%d, want 92\n", v)
		failed = true
	}
	if v := f4().a[1]; v != 92 {
		fmt.Printf("f4().a[1]=%d, want 92\n", v)
		failed = true
	}
	if v := f5(); v != 92 {
		fmt.Printf("f5()=%d, want 92\n", v)
		failed = true
	}
	if failed {
		panic("bad")
	}
}
