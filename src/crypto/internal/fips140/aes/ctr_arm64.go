// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import "crypto/internal/fips140deps/cpu"

// ctrBulk reports whether xorKeyStreamAt passes its n full blocks, starting at
// a counter whose low 64 bits are ivlo, to ctrBlocksBulk or ctrBlocksBulkEOR3:
// there are AES instructions, n > 0, and the low 64 bits of the counter do not
// wrap within the n blocks.
func ctrBulk(n, ivlo uint64) bool {
	return supportsAES && n > 0 && n-1 <= ^ivlo
}

// supportsAESEOR3 selects ctrBlocksBulkEOR3, which needs FEAT_SHA3.
var supportsAESEOR3 = supportsAES && cpu.ARM64HasSHA3

// ctrBlocksBulk and ctrBlocksBulkEOR3 XOR n blocks of keystream into dst,
// starting at counter ivhi*2^64 + ivlo, with ctrBlocksBulkAsm and
// ctrBlocksBulkEOR3Asm. The low 64 bits of the counter must not wrap within
// the n blocks, and dst and src must be the same buffer or not overlap. They
// are small enough to be inlined into xorKeyStreamAt: a Go function call in
// between cost up to 8% for messages of a few blocks.
func ctrBlocksBulk(b *Block, dst, src *byte, n, ivlo, ivhi uint64) {
	ctrBlocksBulkAsm(b.rounds, &b.enc, dst, src, int(n/12), int(n%12), ivlo, ivhi)
}

func ctrBlocksBulkEOR3(b *Block, dst, src *byte, n, ivlo, ivhi uint64) {
	ctrBlocksBulkEOR3Asm(b.rounds, &b.enc, dst, src, int(n/12), int(n%12), ivlo, ivhi)
}

// ctrBlocksBulkAsm and ctrBlocksBulkEOR3Asm XOR n >= 0 groups of 12 blocks and
// then 0 <= r < 12 more blocks of keystream into dst, starting at counter
// ivhi*2^64 + ivlo.
// The low 64 bits of the counter must not wrap: ivlo + 12*n + r <= 2^64.
// dst and src must be the same buffer or not overlap.
//
//go:noescape
func ctrBlocksBulkAsm(nr int, xk *[60]uint32, dst, src *byte, n, r int, ivlo, ivhi uint64)

//go:noescape
func ctrBlocksBulkEOR3Asm(nr int, xk *[60]uint32, dst, src *byte, n, r int, ivlo, ivhi uint64)
