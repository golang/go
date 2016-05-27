// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha1

//go:noescape

func blockAVX2(dig *digest, p []byte)

//go:noescape
func blockAMD64(dig *digest, p []byte)
func checkAVX2() bool

var hasAVX2 = checkAVX2()

func block(dig *digest, p []byte) {
	if hasAVX2 && len(p) >= 256 {
		// blockAVX2 calculates sha1 for 2 block per iteration
		// it also interleaves precalculation for next block.
		// So it may read up-to 192 bytes past end of p
		// We may add checks inside blockAVX2, but this will
		// just turn it into a copy of blockAMD64,
		// so call it directly, instead.
		safeLen := len(p) - 128
		if safeLen%128 != 0 {
			safeLen -= 64
		}
		blockAVX2(dig, p[:safeLen])
		blockAMD64(dig, p[safeLen:])
	} else {
		blockAMD64(dig, p)
	}
}
