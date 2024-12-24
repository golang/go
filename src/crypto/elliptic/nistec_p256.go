// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64

package elliptic

import (
	"crypto/internal/fips140/nistec"
	"math/big"
)

func (c p256Curve) Inverse(k *big.Int) *big.Int {
	if k.Sign() < 0 {
		// This should never happen.
		k = new(big.Int).Neg(k)
	}
	if k.Cmp(c.params.N) >= 0 {
		// This should never happen.
		k = new(big.Int).Mod(k, c.params.N)
	}
	scalar := k.FillBytes(make([]byte, 32))
	inverse, err := nistec.P256OrdInverse(scalar)
	if err != nil {
		panic("crypto/elliptic: nistec rejected normalized scalar")
	}
	return new(big.Int).SetBytes(inverse)
}
