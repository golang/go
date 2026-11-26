// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

package aes

import (
	"bytes"
	"crypto/internal/fips140deps/byteorder"
	"crypto/internal/fips140deps/cpu"
	"testing"
)

// ctrReference is CTR with the generic key schedule and block function.
func ctrReference(key, src []byte, ivlo, ivhi uint64) []byte {
	// The generic code checks that it is not used when AES instructions are.
	old := supportsAES
	defer func() { supportsAES = old }()
	supportsAES = false
	var c blockExpanded
	c.rounds = map[int]int{16: aes128Rounds, 24: aes192Rounds, 32: aes256Rounds}[len(key)]
	expandKeyGeneric(&c, key)
	out := make([]byte, len(src))
	var ctr, ks [BlockSize]byte
	for i := 0; i < len(src); i += BlockSize {
		byteorder.BEPutUint64(ctr[:8], ivhi)
		byteorder.BEPutUint64(ctr[8:], ivlo)
		encryptBlockGeneric(&c, ks[:], ctr[:])
		for j := range BlockSize {
			out[i+j] = src[i+j] ^ ks[j]
		}
		ivlo, ivhi = add128(ivlo, ivhi, 1)
	}
	return out
}

// TestCTRBlocksBulk checks both paths of ctrBlocksBulkAsm (VAES and AES-NI)
// against the generic code: all key sizes, every number of blocks left after
// the loop, in place and not. It calls the assembly directly, and
// XORKeyStreamAt, which also covers counters whose low 64 bits wrap within the
// message (those go to the fixed-size functions).
func TestCTRBlocksBulk(t *testing.T) {
	if !supportsAES {
		t.Skip("no AES instructions")
	}
	old := supportsVAES
	defer func() { supportsVAES = old }()
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i*7 + 1)
	}
	for _, vaes := range []bool{false, true} {
		if vaes && !(cpu.X86HasVAES && cpu.X86HasAVX2) {
			t.Log("skipping the VAES path: no VAES or AVX2")
			continue
		}
		supportsVAES = vaes
		for _, keySize := range []int{16, 24, 32} {
			var b Block
			newBlock(&b, key[:keySize])
			for _, ivhi := range []uint64{0, 3, ^uint64(0)} {
				for _, ivlo := range []uint64{0, 5, 1 << 63, ^uint64(0) - 70, ^uint64(0) - 33,
					^uint64(0) - 16, ^uint64(0) - 15, ^uint64(0) - 1, ^uint64(0)} {
					var iv [BlockSize]byte
					byteorder.BEPutUint64(iv[:8], ivhi)
					byteorder.BEPutUint64(iv[8:], ivlo)
					c := NewCTR(&b, iv[:])
					for n := 1; n <= 64; n++ {
						src := make([]byte, n*BlockSize)
						for i := range src {
							src[i] = byte(i*131 + n)
						}
						want := ctrReference(key[:keySize], src, ivlo, ivhi)
						fail := func(what string) {
							t.Fatalf("vaes=%v key=%d ivhi=%#x ivlo=%#x n=%d: wrong output from %s", vaes, keySize, ivhi, ivlo, n, what)
						}
						got := make([]byte, len(src))
						c.XORKeyStreamAt(got, src, 0)
						if !bytes.Equal(got, want) {
							fail("XORKeyStreamAt")
						}
						if uint64(n)-1 <= ^ivlo {
							clear(got)
							ctrBlocksBulkAsm(b.rounds, &b.enc, &got[0], &src[0], n, ivlo, ivhi, vaes)
							if !bytes.Equal(got, want) {
								fail("the assembly")
							}
							copy(got, src)
							ctrBlocksBulkAsm(b.rounds, &b.enc, &got[0], &got[0], n, ivlo, ivhi, vaes)
							if !bytes.Equal(got, want) {
								fail("the assembly in place")
							}
						}
						c.XORKeyStreamAt(src, src, 0)
						if !bytes.Equal(src, want) {
							fail("XORKeyStreamAt in place")
						}
					}
				}
			}
		}
	}
}
