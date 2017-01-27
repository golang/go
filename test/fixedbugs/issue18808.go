// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

const lim = 0x80000000

//go:noinline
func eq(x uint32) {
	if x == lim {
		return
	}
	panic("x == lim returned false")
}

//go:noinline
func neq(x uint32) {
	if x != lim {
		panic("x != lim returned true")
	}
}

//go:noinline
func gt(x uint32) {
	if x > lim {
		return
	}
	panic("x > lim returned false")
}

//go:noinline
func gte(x uint32) {
	if x >= lim {
		return
	}
	panic("x >= lim returned false")
}

//go:noinline
func lt(x uint32) {
	if x < lim {
		panic("x < lim returned true")
	}
}

//go:noinline
func lte(x uint32) {
	if x <= lim {
		panic("x <= lim returned true")
	}
}

func main() {
	eq(lim)
	neq(lim)
	gt(lim+1)
	gte(lim+1)
	lt(lim+1)
	lte(lim+1)
}
