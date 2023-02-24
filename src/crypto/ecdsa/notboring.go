// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !boringcrypto

package ecdsa

import "crypto/internal/boring"

func boringPublicKey(*PublicKey) (*boring.PublicKeyECDSA, error) {
	panic("boringcrypto: not available")
}
func boringPrivateKey(*PrivateKey) (*boring.PrivateKeyECDSA, error) {
	panic("boringcrypto: not available")
}
