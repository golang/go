// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// On 32-bit archs, one of the f fields of a [2]T
// will be unaligned (address of 4 mod 8).
// Make sure we can access the f fields successfully,
// particularly for load-add combo instructions
// introduced by CL 102036.
type T struct {
	pad uint32
	f float64
}

//go:noinline
func f(t, u *T) float64 {
	return 3.0 + t.f + u.f
}

func main() {
	t := [2]T{{0, 1.0}, {0, 2.0}}
	sink = f(&t[0], &t[1])
}

var sink float64
