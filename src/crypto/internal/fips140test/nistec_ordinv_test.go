// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (amd64 || arm64) && !purego

package fipstest

import (
	"bytes"
	"crypto/elliptic"
	"crypto/internal/fips140/nistec"
	"math/big"
	"testing"
)

func TestP256OrdInverse(t *testing.T) {
	N := elliptic.P256().Params().N

	// inv(0) is expected to be 0.
	zero := make([]byte, 32)
	out, err := nistec.P256OrdInverse(zero)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, zero) {
		t.Error("unexpected output for inv(0)")
	}

	// inv(N) is also 0 mod N.
	input := make([]byte, 32)
	N.FillBytes(input)
	out, err = nistec.P256OrdInverse(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, zero) {
		t.Error("unexpected output for inv(N)")
	}
	if !bytes.Equal(input, N.Bytes()) {
		t.Error("input was modified")
	}

	// Check inv(1) and inv(N+1) against math/big
	exp := new(big.Int).ModInverse(big.NewInt(1), N).FillBytes(make([]byte, 32))
	big.NewInt(1).FillBytes(input)
	out, err = nistec.P256OrdInverse(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, exp) {
		t.Error("unexpected output for inv(1)")
	}
	new(big.Int).Add(N, big.NewInt(1)).FillBytes(input)
	out, err = nistec.P256OrdInverse(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, exp) {
		t.Error("unexpected output for inv(N+1)")
	}

	// Check inv(20) and inv(N+20) against math/big
	exp = new(big.Int).ModInverse(big.NewInt(20), N).FillBytes(make([]byte, 32))
	big.NewInt(20).FillBytes(input)
	out, err = nistec.P256OrdInverse(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, exp) {
		t.Error("unexpected output for inv(20)")
	}
	new(big.Int).Add(N, big.NewInt(20)).FillBytes(input)
	out, err = nistec.P256OrdInverse(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, exp) {
		t.Error("unexpected output for inv(N+20)")
	}

	// Check inv(2^256-1) against math/big
	bigInput := new(big.Int).Lsh(big.NewInt(1), 256)
	bigInput.Sub(bigInput, big.NewInt(1))
	exp = new(big.Int).ModInverse(bigInput, N).FillBytes(make([]byte, 32))
	bigInput.FillBytes(input)
	out, err = nistec.P256OrdInverse(input)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(out, exp) {
		t.Error("unexpected output for inv(2^256-1)")
	}
}
