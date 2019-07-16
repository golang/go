// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build s390x,!gccgo

package ecdsa

import (
	"crypto/cipher"
	"crypto/elliptic"
	"internal/cpu"
	"math/big"
)

// s390x accelerated signatures
//go:noescape
func kdsaSig(fc uint64, block *[1720]byte) (errn uint64)

type signverify int

const (
	signing signverify = iota
	verifying
)

// bufferOffsets represents the offset of a particular parameter in
// the buffer passed to the KDSA instruction.
type bufferOffsets struct {
	baseSize       int
	hashSize       int
	offsetHash     int
	offsetKey1     int
	offsetRNorKey2 int
	offsetR        int
	offsetS        int
	functionCode   uint64
}

func canUseKDSA(sv signverify, c elliptic.Curve, bo *bufferOffsets) bool {
	if !cpu.S390X.HasECDSA {
		return false
	}

	switch c.Params().Name {
	case "P-256":
		bo.baseSize = 32
		bo.hashSize = 32
		bo.offsetHash = 64
		bo.offsetKey1 = 96
		bo.offsetRNorKey2 = 128
		bo.offsetR = 0
		bo.offsetS = 32
		if sv == signing {
			bo.functionCode = 137
		} else {
			bo.functionCode = 1
		}
		return true
	case "P-384":
		bo.baseSize = 48
		bo.hashSize = 48
		bo.offsetHash = 96
		bo.offsetKey1 = 144
		bo.offsetRNorKey2 = 192
		bo.offsetR = 0
		bo.offsetS = 48
		if sv == signing {
			bo.functionCode = 138
		} else {
			bo.functionCode = 2
		}
		return true
	case "P-521":
		bo.baseSize = 66
		bo.hashSize = 80
		bo.offsetHash = 160
		bo.offsetKey1 = 254
		bo.offsetRNorKey2 = 334
		bo.offsetR = 14
		bo.offsetS = 94
		if sv == signing {
			bo.functionCode = 139
		} else {
			bo.functionCode = 3
		}
		return true
	}
	return false
}

// zeroExtendAndCopy pads src with leading zeros until it has the size given.
// It then copies the padded src into the dst. Bytes beyond size in dst are
// not modified.
func zeroExtendAndCopy(dst, src []byte, size int) {
	nz := size - len(src)
	if nz < 0 {
		panic("src is too long")
	}
	// the compiler should replace this loop with a memclr call
	z := dst[:nz]
	for i := range z {
		z[i] = 0
	}
	copy(dst[nz:size], src[:size-nz])
	return
}

func sign(priv *PrivateKey, csprng *cipher.StreamReader, c elliptic.Curve, e *big.Int) (r, s *big.Int, err error) {
	var bo bufferOffsets
	if canUseKDSA(signing, c, &bo) && e.Sign() != 0 {
		var buffer [1720]byte
		for {
			var k *big.Int
			k, err = randFieldElement(c, csprng)
			if err != nil {
				return nil, nil, err
			}
			zeroExtendAndCopy(buffer[bo.offsetHash:], e.Bytes(), bo.hashSize)
			zeroExtendAndCopy(buffer[bo.offsetKey1:], priv.D.Bytes(), bo.baseSize)
			zeroExtendAndCopy(buffer[bo.offsetRNorKey2:], k.Bytes(), bo.baseSize)
			errn := kdsaSig(bo.functionCode, &buffer)
			if errn == 2 {
				return nil, nil, errZeroParam
			}
			if errn == 0 { // success == 0 means successful signing
				r = new(big.Int)
				r.SetBytes(buffer[bo.offsetR : bo.offsetR+bo.baseSize])
				s = new(big.Int)
				s.SetBytes(buffer[bo.offsetS : bo.offsetS+bo.baseSize])
				return
			}
			//at this point, it must be that errn == 1: retry
		}
	}
	r, s, err = signGeneric(priv, csprng, c, e)
	return
}

func verify(pub *PublicKey, c elliptic.Curve, e, r, s *big.Int) bool {
	var bo bufferOffsets
	if canUseKDSA(verifying, c, &bo) && e.Sign() != 0 {
		var buffer [1720]byte
		zeroExtendAndCopy(buffer[bo.offsetR:], r.Bytes(), bo.baseSize)
		zeroExtendAndCopy(buffer[bo.offsetS:], s.Bytes(), bo.baseSize)
		zeroExtendAndCopy(buffer[bo.offsetHash:], e.Bytes(), bo.hashSize)
		zeroExtendAndCopy(buffer[bo.offsetKey1:], pub.X.Bytes(), bo.baseSize)
		zeroExtendAndCopy(buffer[bo.offsetRNorKey2:], pub.Y.Bytes(), bo.baseSize)
		errn := kdsaSig(bo.functionCode, &buffer)
		return errn == 0
	}
	return verifyGeneric(pub, c, e, r, s)
}
