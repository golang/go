// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha1

import (
	"internal/cpu"
	"unsafe"
)

//go:noescape
func blockAVX2(dig *digest, p *byte, n int)

//go:noescape
func blockAMD64(dig *digest, p *byte, n int)

var useAVX2 = cpu.X86.HasAVX2 && cpu.X86.HasBMI1 && cpu.X86.HasBMI2

func block(dig *digest, p []byte) {
	if useAVX2 && len(p) >= 256 {
		// blockAVX2 calculates sha1 for 2 blocks per iteration
		// it also interleaves precalculation for next block.
		// So it may read up-to 192 bytes past end of p
		// We may add checks inside blockAVX2, but this will
		// just turn it into a copy of blockAMD64,
		// so call it directly, instead.
		safeLen := len(p) - 128
		if safeLen%128 != 0 {
			safeLen -= 64
		}
		blockAVX2(dig, unsafe.SliceData(p), safeLen)
		pRem := p[safeLen:]
		blockAMD64(dig, unsafe.SliceData(pRem), len(pRem))
	} else {
		blockAMD64(dig, unsafe.SliceData(p), len(p))
	}
}

// blockString is a duplicate of block that takes a string.
func blockString(dig *digest, s string) {
	if useAVX2 && len(s) >= 256 {
		safeLen := len(s) - 128
		if safeLen%128 != 0 {
			safeLen -= 64
		}
		blockAVX2(dig, unsafe.StringData(s), safeLen)
		sRem := s[safeLen:]
		blockAMD64(dig, unsafe.StringData(sRem), len(sRem))
	} else {
		blockAMD64(dig, unsafe.StringData(s), len(s))
	}
}
