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
	"encoding/json"
	"net/url"
	"os"
	"runtime"
)

var (
	_ *aes.Cipher
	_ *des.Cipher
	_ *des.TripleDESCipher
	_ = json.MarshalForHTML
	_ = aes.New()
	_ = url.Parse
	_ = url.ParseWithReference
	_ = url.ParseRequest
	_ = os.Exec
	_ = runtime.Cgocalls
	_ = runtime.Goroutines
)
`,
		Out: `package main

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/json"
	"net/url"
	"runtime"
	"syscall"
)

var (
	_ cipher.Block
	_ cipher.Block
	_ cipher.Block
	_ = json.Marshal
	_ = aes.New()
	_ = url.Parse
	_ = url.Parse
	_ = url.ParseRequestURI
	_ = syscall.Exec
	_ = runtime.NumCgoCall
	_ = runtime.NumGoroutine
)
`,
	},
}
