// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !s390x || purego

package ecdsa

func sign[P Point[P]](c *Curve[P], priv *PrivateKey, drbg *hmacDRBG, hash []byte) (*Signature, error) {
	return signGeneric(c, priv, drbg, hash)
}

func verify[P Point[P]](c *Curve[P], pub *PublicKey, hash []byte, sig *Signature) error {
	return verifyGeneric(c, pub, hash, sig)
}
