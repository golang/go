// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64 || ppc64 || ppc64le) && !purego

package aes

import (
	"crypto/internal/fipsdeps/cpu"
	"crypto/internal/fipsdeps/godebug"
	"crypto/internal/impl"
)

//go:noescape
func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)

//go:noescape
func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)

//go:noescape
func expandKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)

var supportsAES = cpu.X86HasAES && cpu.X86HasSSE41 && cpu.X86HasSSSE3 ||
	cpu.ARM64HasAES || cpu.PPC64 || cpu.PPC64le

func init() {
	if cpu.AMD64 {
		impl.Register("aes", "AES-NI", &supportsAES)
	}
	if cpu.ARM64 {
		impl.Register("aes", "Armv8.0", &supportsAES)
	}
	if cpu.PPC64 || cpu.PPC64le {
		// The POWER architecture doesn't have a way to turn off AES support
		// at runtime with GODEBUG=cpu.something=off, so introduce a new GODEBUG
		// knob for that. It's intentionally only checked at init() time, to
		// avoid the performance overhead of checking it every time.
		if godebug.Value("#ppc64aes") == "off" {
			supportsAES = false
		}
		impl.Register("aes", "POWER8", &supportsAES)
	}
}

// checkGenericIsExpected is called by the variable-time implementation to make
// sure it is not used when hardware support is available. It shouldn't happen,
// but this way it's more evidently correct.
func checkGenericIsExpected() {
	if supportsAES {
		panic("crypto/aes: internal error: using generic implementation despite hardware support")
	}
}

type block struct {
	blockExpanded
}

func newBlock(c *Block, key []byte) *Block {
	switch len(key) {
	case aes128KeySize:
		c.rounds = aes128Rounds
	case aes192KeySize:
		c.rounds = aes192Rounds
	case aes256KeySize:
		c.rounds = aes256Rounds
	}
	if supportsAES {
		expandKeyAsm(c.rounds, &key[0], &c.enc[0], &c.dec[0])
	} else {
		expandKeyGeneric(&c.blockExpanded, key)
	}
	return c
}

// EncryptionKeySchedule is used from the GCM implementation to access the
// precomputed AES key schedule, to pass to the assembly implementation.
func EncryptionKeySchedule(c *Block) []uint32 {
	return c.enc[:c.roundKeysSize()]
}

func encryptBlock(c *Block, dst, src []byte) {
	if supportsAES {
		encryptBlockAsm(c.rounds, &c.enc[0], &dst[0], &src[0])
	} else {
		encryptBlockGeneric(&c.blockExpanded, dst, src)
	}
}

func decryptBlock(c *Block, dst, src []byte) {
	if supportsAES {
		decryptBlockAsm(c.rounds, &c.dec[0], &dst[0], &src[0])
	} else {
		decryptBlockGeneric(&c.blockExpanded, dst, src)
	}
}
