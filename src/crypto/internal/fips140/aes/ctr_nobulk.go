// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !arm64 || purego

package aes

// ctrBulk reports whether xorKeyStreamAt passes its n full blocks to
// ctrBlocksBulk or ctrBlocksBulkEOR3, which only arm64 has.
func ctrBulk(n, ivlo uint64) bool { return false }

const supportsAESEOR3 = false

func ctrBlocksBulk(b *Block, dst, src *byte, n, ivlo, ivhi uint64) {
	panic("crypto/aes: internal error: ctrBlocksBulk is not available")
}

func ctrBlocksBulkEOR3(b *Block, dst, src *byte, n, ivlo, ivhi uint64) {
	panic("crypto/aes: internal error: ctrBlocksBulkEOR3 is not available")
}
