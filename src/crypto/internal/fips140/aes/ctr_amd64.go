// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import "crypto/internal/fips140deps/cpu"

// ctrBulk reports whether xorKeyStreamAt passes its n full blocks, starting at
// a counter whose low 64 bits are ivlo, to ctrBlocksBulk: there are AES
// instructions, n > 0, and the low 64 bits of the counter do not wrap within
// the n blocks.
func ctrBulk(n, ivlo uint64) bool {
	return supportsAES && n > 0 && n-1 <= ^ivlo
}

// supportsVAES selects the VAES path of ctrBlocksBulkAsm, which needs VAES and
// AVX2.
var supportsVAES = supportsAES && cpu.X86HasVAES && cpu.X86HasAVX2

// ctrBlocksBulk XORs n blocks of keystream into dst, starting at counter
// ivhi*2^64 + ivlo, with ctrBlocksBulkAsm. The low 64 bits of the counter must
// not wrap within the n blocks, and dst and src must be the same buffer or not
// overlap. It is small enough to be inlined into xorKeyStreamAt: a Go function
// call in between costs several percent for messages of a few blocks.
func ctrBlocksBulk(b *Block, dst, src *byte, n, ivlo, ivhi uint64) {
	ctrBlocksBulkAsm(b.rounds, &b.enc, dst, src, int(n), ivlo, ivhi, supportsVAES)
}

// ctrBlocksBulkAsm XORs n >= 1 blocks of keystream into dst, starting at
// counter ivhi*2^64 + ivlo, with VAES and AVX2 if vaes is set and with AES-NI
// otherwise. The low 64 bits of the counter must not wrap: ivlo + n <= 2^64.
// dst and src must be the same buffer or not overlap.
//
//go:noescape
func ctrBlocksBulkAsm(nr int, xk *[60]uint32, dst, src *byte, n int, ivlo, ivhi uint64, vaes bool)
