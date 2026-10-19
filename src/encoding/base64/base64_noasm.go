// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !riscv64 || purego

package base64

import "unsafe"

func encodeChunk(encode *[64]byte, dst, src *byte, n int) {
	// encodeChunk assembly is written using pointers and n. Back to slices.
	dstb := unsafe.Slice(dst, n/3*4)
	srcb := unsafe.Slice(src, n)

	si := 0
	di := 0
	for si < n {
		// Convert 3x 8bit source bytes into 4 bytes
		val := uint(srcb[si+0])<<16 | uint(srcb[si+1])<<8 | uint(srcb[si+2])

		dstb[di+0] = encode[val>>18&0x3F]
		dstb[di+1] = encode[val>>12&0x3F]
		dstb[di+2] = encode[val>>6&0x3F]
		dstb[di+3] = encode[val&0x3F]

		si += 3
		di += 4
	}
}
