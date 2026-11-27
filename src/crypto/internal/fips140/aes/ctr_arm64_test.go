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

// TestCTRBlocksBulk checks both variants of the bulk assembly (EOR3 and
// plain) against the generic code: all key sizes, every remainder after the
// 12-block groups, in place and not. It calls the assembly directly, and
// XORKeyStreamAt, which also covers counters whose low 64 bits wrap within
// the message (those go to the fixed-size functions).
func TestCTRBlocksBulk(t *testing.T) {
	if !supportsAES {
		t.Skip("no AES instructions")
	}
	old := supportsAESEOR3
	defer func() { supportsAESEOR3 = old }()
	key := make([]byte, 32)
	for i := range key {
		key[i] = byte(i*7 + 1)
	}
	for _, eor3 := range []bool{false, true} {
		if eor3 && !cpu.ARM64HasSHA3 {
			continue
		}
		supportsAESEOR3 = eor3
		bulk := ctrBlocksBulkAsm
		if eor3 {
			bulk = ctrBlocksBulkEOR3Asm
		}
		for _, keySize := range []int{16, 24, 32} {
			var b Block
			newBlock(&b, key[:keySize])
			for _, ivhi := range []uint64{0, 3, ^uint64(0)} {
				for _, ivlo := range []uint64{0, 5, 1 << 63, ^uint64(0) - 30, ^uint64(0) - 12,
					^uint64(0) - 11, ^uint64(0) - 1, ^uint64(0)} {
					var iv [BlockSize]byte
					byteorder.BEPutUint64(iv[:8], ivhi)
					byteorder.BEPutUint64(iv[8:], ivlo)
					c := NewCTR(&b, iv[:])
					for n := 1; n <= 40; n++ {
						src := make([]byte, n*BlockSize)
						for i := range src {
							src[i] = byte(i*131 + n)
						}
						want := ctrReference(key[:keySize], src, ivlo, ivhi)
						fail := func(what string) {
							t.Fatalf("eor3=%v key=%d ivhi=%#x ivlo=%#x n=%d: wrong output from %s", eor3, keySize, ivhi, ivlo, n, what)
						}
						got := make([]byte, len(src))
						c.XORKeyStreamAt(got, src, 0)
						if !bytes.Equal(got, want) {
							fail("XORKeyStreamAt")
						}
						if uint64(n)-1 <= ^ivlo {
							clear(got)
							bulk(b.rounds, &b.enc, &got[0], &src[0], n/12, n%12, ivlo, ivhi)
							if !bytes.Equal(got, want) {
								fail("the assembly")
							}
							copy(got, src)
							bulk(b.rounds, &b.enc, &got[0], &got[0], n/12, n%12, ivlo, ivhi)
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
