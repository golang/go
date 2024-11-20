// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package aes_test

import (
	"crypto/cipher"
	"crypto/internal/fips/aes"
)

var _ cipher.Block = (*aes.Block)(nil)
var _ cipher.Stream = (*aes.CTR)(nil)
var _ cipher.BlockMode = (*aes.CBCDecrypter)(nil)
var _ cipher.BlockMode = (*aes.CBCEncrypter)(nil)
