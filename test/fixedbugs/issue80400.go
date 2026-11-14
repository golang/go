// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// When the carry of a link that reads its operand from memory has to
// survive a call, flagalloc recomputes it after the call. The
// recomputation must not read memory again, because a store in
// between may have changed it.

package main

import "math/bits"

//go:noinline
func sink() {}

//go:noinline
func add(x, y *[2]uint64, p *uint64, v uint64) uint64 {
	_, c := bits.Add64(x[0], y[0], 0)
	*p = v
	if v > 100 {
		sink()
	}
	hi, _ := bits.Add64(x[1], y[1], c)
	return hi
}

//go:noinline
func sub(x, y *[2]uint64, p *uint64, v uint64) uint64 {
	_, b := bits.Sub64(x[0], y[0], 0)
	*p = v
	if v > 100 {
		sink()
	}
	hi, _ := bits.Sub64(x[1], y[1], b)
	return hi
}

//go:noinline
func add3(x, y *[3]uint64, p *uint64, v uint64) uint64 {
	_, c := bits.Add64(x[0], y[0], 0)
	_, c = bits.Add64(x[1], y[1], c)
	*p = v
	if v > 100 {
		sink()
	}
	hi, _ := bits.Add64(x[2], y[2], c)
	return hi
}

func main() {
	// The store through p overwrites y[0] after the first link has
	// read it. The carry must come from the old value.
	x := [2]uint64{1 << 63, 0}
	y := [2]uint64{1 << 63, 0}
	if got := add(&x, &y, &y[0], 1000); got != 1 {
		panic(got)
	}

	x = [2]uint64{0, 5}
	y = [2]uint64{1, 0}
	if got := sub(&x, &y, &y[0], 1000); got != 4 {
		panic(got)
	}

	x3 := [3]uint64{^uint64(0), ^uint64(0), 0}
	y3 := [3]uint64{1, 0, 0}
	if got := add3(&x3, &y3, &y3[1], 1000); got != 1 {
		panic(got)
	}
	x3 = [3]uint64{^uint64(0), ^uint64(0), 0}
	y3 = [3]uint64{1, 0, 0}
	if got := add3(&x3, &y3, &y3[0], 1000); got != 1 {
		panic(got)
	}
}
