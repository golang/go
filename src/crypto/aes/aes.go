// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package aes implements AES encryption (formerly Rijndael), as defined in
// U.S. Federal Information Processing Standards Publication 197.
//
// The AES operations in this package are not implemented using constant-time algorithms.
// An exception is when running on systems with enabled hardware support for AES
// that makes these operations constant-time. Examples include amd64 systems using AES-NI
// extensions and s390x systems using Message-Security-Assist extensions.
// On such systems, when the result of NewCipher is passed to cipher.NewGCM,
// the GHASH operation used by GCM is also constant-time.
package aes

import (
	"crypto/cipher"
	"crypto/internal/boring"
	"crypto/internal/fips140/aes"
	"strconv"
)

// The AES block size in bytes.
const BlockSize = 16

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/aes: invalid key size " + strconv.Itoa(int(k))
}

// NewCipher creates and returns a new [cipher.Block].
// The key argument should be the AES key,
// either 16, 24, or 32 bytes to select
// AES-128, AES-192, or AES-256.
func NewCipher(key []byte) (cipher.Block, error) {
	k := len(key)
	switch k {
	default:
		return nil, KeySizeError(k)
	case 16, 24, 32:
		break
	}
	if boring.Enabled {
		return boring.NewAESCipher(key)
	}
	return aes.New(key)
}
