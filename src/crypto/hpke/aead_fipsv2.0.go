// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !fips140v1.0

package hpke

import (
	"crypto/cipher"
	"crypto/internal/fips140/aes"
	"crypto/internal/fips140/aes/gcm"
)

func newAESGCM(key []byte) (cipher.AEAD, error) {
	b, err := aes.New(key)
	if err != nil {
		return nil, err
	}
	return gcm.NewGCMForHPKE(b)
}
