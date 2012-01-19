// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(hmacNewTests, hmacnew)
}

var hmacNewTests = []testCase{
	{
		Name: "hmacnew.0",
		In: `package main

import "crypto/hmac"

var f = hmac.NewSHA1([]byte("some key"))
`,
		Out: `package main

import (
	"crypto/hmac"
	"crypto/sha1"
)

var f = hmac.New(sha1.New, []byte("some key"))
`,
	},
	{
		Name: "hmacnew.1",
		In: `package main

import "crypto/hmac"

var key = make([]byte, 8)
var f = hmac.NewSHA1(key)
`,
		Out: `package main

import (
	"crypto/hmac"
	"crypto/sha1"
)

var key = make([]byte, 8)
var f = hmac.New(sha1.New, key)
`,
	},
	{
		Name: "hmacnew.2",
		In: `package main

import "crypto/hmac"

var f = hmac.NewMD5([]byte("some key"))
`,
		Out: `package main

import (
	"crypto/hmac"
	"crypto/md5"
)

var f = hmac.New(md5.New, []byte("some key"))
`,
	},
	{
		Name: "hmacnew.3",
		In: `package main

import "crypto/hmac"

var f = hmac.NewSHA256([]byte("some key"))
`,
		Out: `package main

import (
	"crypto/hmac"
	"crypto/sha256"
)

var f = hmac.New(sha256.New, []byte("some key"))
`,
	},
	{
		Name: "hmacnew.4",
		In: `package main

import (
	"crypto/hmac"
	"crypto/sha1"
)

var f = hmac.New(sha1.New, []byte("some key"))
`,
		Out: `package main

import (
	"crypto/hmac"
	"crypto/sha1"
)

var f = hmac.New(sha1.New, []byte("some key"))
`,
	},
}
