// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Don't allow 0 masks in shift lowering rules on ppc64x.
// See issue 42610.

package codegen

func f32(a []int32, i uint32) {
        g := func(p int32) int32 {
                i = uint32(p) * (uint32(p) & (i & 1))
                return 1
        }
        // ppc64x: -"RLWNIM"
        a[0] = g(8) >> 1
}

func f(a []int, i uint) {
	g := func(p int) int {
		i = uint(p) * (uint(p) & (i & 1))
		return 1
	}
	// ppc64x: -"RLDIC"
	a[0] = g(8) >> 1
}
