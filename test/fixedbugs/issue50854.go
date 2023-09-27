// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// This checks for incorrect application of CMP(-x,y) -> CMN(x,y) in arm and arm64

//go:noinline
func f(p int64, x, y int64) bool { return -x <= p && p <= y }

//go:noinline
func g(p int32, x, y int32) bool { return -x <= p && p <= y }

// There are some more complicated patterns involving compares and shifts, try to trigger those.

//go:noinline
func h(p int64, x, y int64) bool { return -(x<<1) <= p && p <= y }

//go:noinline
func k(p int32, x, y int32) bool { return -(1<<x) <= p && p <= y }

//go:noinline
func check(b bool) {
	if b {
		return
	}
	panic("FAILURE")
}

func main() {
	check(f(1, -1<<63, 1<<63-1))
	check(g(1, -1<<31, 1<<31-1))
	check(h(1, -1<<62, 1<<63-1))
	check(k(1, 31, 1<<31-1))
}
