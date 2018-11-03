// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sha1

import "internal/cpu"

var k = []uint32{
	0x5A827999,
	0x6ED9EBA1,
	0x8F1BBCDC,
	0xCA62C1D6,
}

//go:noescape
func sha1block(h []uint32, p []byte, k []uint32)

func block(dig *digest, p []byte) {
	if !cpu.ARM64.HasSHA1 {
		blockGeneric(dig, p)
	} else {
		h := dig.h[:]
		sha1block(h, p, k)
	}
}
