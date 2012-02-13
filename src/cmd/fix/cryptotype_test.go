// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(cryptotypeTests, cryptotypeFix.f)
}

var cryptotypeTests = []testCase{
	{
		Name: "cryptotype.0",
		In: `package main

import (
	"crypto/aes"
	"crypto/des"
)

var (
	_ *aes.Cipher
	_ *des.Cipher
	_ *des.TripleDESCipher
	_ = aes.New()
)
`,
		Out: `package main

import (
	"crypto/aes"
	"crypto/cipher"
)

var (
	_ cipher.Block
	_ cipher.Block
	_ cipher.Block
	_ = aes.New()
)
`,
	},
}
