// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes

import (
	"crypto/cipher"
)

// gcmAble is implemented by cipher.Blocks that can provide an optimized
// implementation of GCM through the AEAD interface.
// See crypto/cipher/gcm.go.
type gcmAble interface {
	NewGCM(size int) (cipher.AEAD, error)
}

// cbcEncAble is implemented by cipher.Blocks that can provide an optimized
// implementation of CBC encryption through the cipher.BlockMode interface.
// See crypto/cipher/cbc.go.
type cbcEncAble interface {
	NewCBCEncrypter(iv []byte) cipher.BlockMode
}

// cbcDecAble is implemented by cipher.Blocks that can provide an optimized
// implementation of CBC decryption through the cipher.BlockMode interface.
// See crypto/cipher/cbc.go.
type cbcDecAble interface {
	NewCBCDecrypter(iv []byte) cipher.BlockMode
}

// ctrAble is implemented by cipher.Blocks that can provide an optimized
// implementation of CTR through the cipher.Stream interface.
// See crypto/cipher/ctr.go.
type ctrAble interface {
	NewCTR(iv []byte) cipher.Stream
}
