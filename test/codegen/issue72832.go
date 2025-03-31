// asmcheck

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

type tile1 struct {
	a uint16
	b uint16
	c uint32
}

func store_tile1(t *tile1) {
	// amd64:`MOVQ`
	t.a, t.b, t.c = 1, 1, 1
}

type tile2 struct {
	a, b, c, d, e int8
}

func store_tile2(t *tile2) {
	// amd64:`MOVW`
	t.a, t.b = 1, 1
	// amd64:`MOVW`
	t.d, t.e = 1, 1
}

type tile3 struct {
	a, b uint8
	c    uint16
}

func store_shifted(t *tile3, x uint32) {
	// amd64:`MOVL`
	t.a = uint8(x)
	t.b = uint8(x >> 8)
	t.c = uint16(x >> 16)
}
