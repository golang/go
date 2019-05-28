// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !s390x

package ecdsa

import (
	"crypto/cipher"
	"crypto/elliptic"
	"math/big"
)

func sign(priv *PrivateKey, csprng *cipher.StreamReader, c elliptic.Curve, e *big.Int) (r, s *big.Int, err error) {
	r, s, err = signGeneric(priv, csprng, c, e)
	return
}

func verify(pub *PublicKey, c elliptic.Curve, e, r, s *big.Int) bool {
	return verifyGeneric(pub, c, e, r, s)
}
