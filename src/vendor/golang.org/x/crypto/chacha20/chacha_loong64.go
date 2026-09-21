// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build gc && !purego

package chacha20

import "golang.org/x/sys/cpu"

const bufSize = 256

//go:noescape
func xorKeyStreamVX(dst, src []byte, key *[8]uint32, nonce *[3]uint32, counter *uint32)

func (c *Cipher) xorKeyStreamBlocks(dst, src []byte) {
	if cpu.Loong64.HasLSX {
		xorKeyStreamVX(dst, src, &c.key, &c.nonce, &c.counter)
	} else {
		c.xorKeyStreamBlocksGeneric(dst, src)
	}
}
