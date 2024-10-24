// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64 || ppc64 || ppc64le) && !purego

package aes

import (
	"crypto/cipher"
	"crypto/internal/boring"
	"crypto/internal/fips/alias"
	"internal/cpu"
	"internal/goarch"
)

// defined in asm_*.s

//go:noescape
func encryptBlockAsm(nr int, xk *uint32, dst, src *byte)

//go:noescape
func decryptBlockAsm(nr int, xk *uint32, dst, src *byte)

//go:noescape
func expandKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)

type aesCipherAsm struct {
	aesCipher
}

// aesCipherGCM implements crypto/cipher.gcmAble so that crypto/cipher.NewGCM
// will use the optimised implementation in aes_gcm.go when possible.
// Instances of this type only exist when hasGCMAsm returns true. Likewise,
// the gcmAble implementation is in aes_gcm.go.
type aesCipherGCM struct {
	aesCipherAsm
}

var supportsAES = cpu.X86.HasAES || cpu.ARM64.HasAES || goarch.IsPpc64 == 1 || goarch.IsPpc64le == 1
var supportsGFMUL = cpu.X86.HasPCLMULQDQ || cpu.ARM64.HasPMULL

func newCipher(key []byte) (cipher.Block, error) {
	if !supportsAES {
		return newCipherGeneric(key)
	}
	// Note that under certain circumstances, we only return the inner aesCipherAsm.
	// This avoids an unnecessary allocation of the aesCipher struct.
	c := aesCipherGCM{aesCipherAsm{aesCipher{l: uint8(len(key) + 28)}}}
	var rounds int
	switch len(key) {
	case 128 / 8:
		rounds = 10
	case 192 / 8:
		rounds = 12
	case 256 / 8:
		rounds = 14
	default:
		return nil, KeySizeError(len(key))
	}

	expandKeyAsm(rounds, &key[0], &c.enc[0], &c.dec[0])
	if supportsAES && supportsGFMUL {
		return &c, nil
	}
	return &c.aesCipherAsm, nil
}

func (c *aesCipherAsm) BlockSize() int { return BlockSize }

func (c *aesCipherAsm) Encrypt(dst, src []byte) {
	boring.Unreachable()
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	if alias.InexactOverlap(dst[:BlockSize], src[:BlockSize]) {
		panic("crypto/aes: invalid buffer overlap")
	}
	encryptBlockAsm(int(c.l)/4-1, &c.enc[0], &dst[0], &src[0])
}

func (c *aesCipherAsm) Decrypt(dst, src []byte) {
	boring.Unreachable()
	if len(src) < BlockSize {
		panic("crypto/aes: input not full block")
	}
	if len(dst) < BlockSize {
		panic("crypto/aes: output not full block")
	}
	if alias.InexactOverlap(dst[:BlockSize], src[:BlockSize]) {
		panic("crypto/aes: invalid buffer overlap")
	}
	decryptBlockAsm(int(c.l)/4-1, &c.dec[0], &dst[0], &src[0])
}

// expandKey is used by BenchmarkExpand to ensure that the asm implementation
// of key expansion is used for the benchmark when it is available.
func expandKey(key []byte, enc, dec []uint32) {
	if supportsAES {
		rounds := 10 // rounds needed for AES128
		switch len(key) {
		case 192 / 8:
			rounds = 12
		case 256 / 8:
			rounds = 14
		}
		expandKeyAsm(rounds, &key[0], &enc[0], &dec[0])
	} else {
		expandKeyGo(key, enc, dec)
	}
}
