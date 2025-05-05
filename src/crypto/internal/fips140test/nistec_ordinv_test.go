// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fipstest

import (
	"bytes"
	"crypto/elliptic"
	"internal/byteorder"
	"math/big"
	"testing"
)

func bytesToLimbs(b []byte) [4]uint64 {
	var l [4]uint64
	l[0] = byteorder.BEUint64(b[24:])
	l[1] = byteorder.BEUint64(b[16:])
	l[2] = byteorder.BEUint64(b[8:])
	l[3] = byteorder.BEUint64(b[:])
	return l
}

func limbsToBytes(l [4]uint64) []byte {
	b := make([]byte, 32)
	byteorder.BEPutUint64(b[24:], l[0])
	byteorder.BEPutUint64(b[16:], l[1])
	byteorder.BEPutUint64(b[8:], l[2])
	byteorder.BEPutUint64(b[:], l[3])
	return b
}

func TestP256OrdInverse(t *testing.T) {
	N := elliptic.P256().Params().N

	// inv(0) is expected to be 0.
	zero := make([]byte, 32)
	k := bytesToLimbs(zero)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), zero) {
		t.Error("unexpected output for inv(0)")
	}

	// inv(N) is also 0 mod N.
	input := make([]byte, 32)
	N.FillBytes(input)
	k = bytesToLimbs(input)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), zero) {
		t.Error("unexpected output for inv(N)")
	}

	// Check inv(1) and inv(N+1) against math/big
	exp := new(big.Int).ModInverse(big.NewInt(1), N).FillBytes(make([]byte, 32))
	big.NewInt(1).FillBytes(input)
	k = bytesToLimbs(input)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), exp) {
		t.Error("unexpected output for inv(1)")
	}

	new(big.Int).Add(N, big.NewInt(1)).FillBytes(input)
	k = bytesToLimbs(input)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), exp) {
		t.Error("unexpected output for inv(N+1)")
	}

	// Check inv(20) and inv(N+20) against math/big
	exp = new(big.Int).ModInverse(big.NewInt(20), N).FillBytes(make([]byte, 32))
	big.NewInt(20).FillBytes(input)
	k = bytesToLimbs(input)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), exp) {
		t.Error("unexpected output for inv(20)")
	}

	new(big.Int).Add(N, big.NewInt(20)).FillBytes(input)
	k = bytesToLimbs(input)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), exp) {
		t.Error("unexpected output for inv(N+20)")
	}

	// Check inv(2^256-1) against math/big
	bigInput := new(big.Int).Lsh(big.NewInt(1), 256)
	bigInput.Sub(bigInput, big.NewInt(1))
	exp = new(big.Int).ModInverse(bigInput, N).FillBytes(make([]byte, 32))
	bigInput.FillBytes(input)
	k = bytesToLimbs(input)
	p256OrdInverse(t, &k)
	if !bytes.Equal(limbsToBytes(k), exp) {
		t.Error("unexpected output for inv(2^256-1)")
	}
}
