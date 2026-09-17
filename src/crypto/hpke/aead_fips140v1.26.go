// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package hpke

import (
	"crypto/cipher"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
	"crypto/subtle"
)

func newAESGCM(key []byte) (cipher.AEAD, error) {
	var g *gcm.GCMWithXORCounterNonce
	var err error
	subtle.WithDataIndependentTiming(func() {
		var b *aes.Block
		b, err = aes.New(key)
		if err != nil {
			return
		}
		g, err = gcm.NewGCMForHPKE(b)
	})
	if err != nil {
		return nil, err
	}
	return g, nil
}
