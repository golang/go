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
	"crypto/subtle"
	"internal/runtime/sys"
	"strconv"
	_ "unsafe"
)

// The AES block size in bytes.
const BlockSize = 16

type KeySizeError int

func (k KeySizeError) Error() string {
	return "crypto/aes: invalid key size " + strconv.Itoa(int(k))
}

// NewCipher creates and returns a new [cipher.Block].
// The key argument must be the AES key,
// either 16, 24, or 32 bytes to select
// AES-128, AES-192, or AES-256.
//
// The returned [cipher.Block] is safe for concurrent use.
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
	var block *aes.Block
	var err error
	subtle.WithDataIndependentTiming(func() {
		block, err = aes.New(key)
	})
	if err != nil {
		return nil, err
	}
	if !sys.DITSupported {
		return block, nil
	}
	return aesBlockWrapper{block}, nil
}

type aesBlockWrapper struct {
	b *aes.Block
}

func (w aesBlockWrapper) BlockSize() int {
	return w.b.BlockSize()
}

func (w aesBlockWrapper) Encrypt(dst, src []byte) {
	subtle.WithDataIndependentTiming(func() { w.b.Encrypt(dst, src) })
}

func (w aesBlockWrapper) Decrypt(dst, src []byte) {
	subtle.WithDataIndependentTiming(func() { w.b.Decrypt(dst, src) })
}

func init() {
	registerAESBlockUnwrap(func(b cipher.Block) (*aes.Block, bool) {
		if w, ok := b.(aesBlockWrapper); ok {
			return w.b, true
		}
		return nil, false
	})
}

//go:linkname registerAESBlockUnwrap crypto/cipher.registerAESBlockUnwrap
func registerAESBlockUnwrap(func(cipher.Block) (*aes.Block, bool))
