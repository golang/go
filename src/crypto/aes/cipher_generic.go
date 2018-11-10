// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !amd64,!s390x,!ppc64le

package aes

import (
	"crypto/cipher"
)

// newCipher calls the newCipherGeneric function
// directly. Platforms with hardware accelerated
// implementations of AES should implement their
// own version of newCipher (which may then call
// newCipherGeneric if needed).
func newCipher(key []byte) (cipher.Block, error) {
	return newCipherGeneric(key)
}

// expandKey is used by BenchmarkExpand and should
// call an assembly implementation if one is available.
func expandKey(key []byte, enc, dec []uint32) {
	expandKeyGo(key, enc, dec)
}
