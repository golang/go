// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(go1renameTests, go1renameFix.f)
}

var go1renameTests = []testCase{
	{
		Name: "go1rename.0",
		In: `package main

import (
	"crypto/aes"
	"crypto/des"
	"net/url"
	"os"
)

var (
	_ *aes.Cipher
	_ *des.Cipher
	_ *des.TripleDESCipher
	_ = aes.New()
	_ = url.Parse
	_ = url.ParseWithReference
	_ = url.ParseRequest
	_ = os.Exec
)
`,
		Out: `package main

import (
	"crypto/aes"
	"crypto/cipher"
	"net/url"
	"syscall"
)

var (
	_ cipher.Block
	_ cipher.Block
	_ cipher.Block
	_ = aes.New()
	_ = url.Parse
	_ = url.Parse
	_ = url.ParseRequestURI
	_ = syscall.Exec
)
`,
	},
}
