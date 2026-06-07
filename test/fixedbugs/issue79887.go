// build

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import "math/bits"

//go:noinline
func add(p, q, x, y uint64) uint64 {
	c := uint64(0)
	if p < q {
		c = 1
	}
	s, _ := bits.Add64(x, y, c)
	return s
}

//go:noinline
func sub(p, q, x, y uint64) uint64 {
	c := uint64(0)
	if p < q {
		c = 1
	}
	s, _ := bits.Sub64(x, y, c)
	return s
}
