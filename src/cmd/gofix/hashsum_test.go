// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(hashSumTests, hashSumFn)
}

var hashSumTests = []testCase{
	{
		Name: "hashsum.0",
		In: `package main

import "crypto/sha256"

func f() []byte {
	h := sha256.New()
	return h.Sum()
}
`,
		Out: `package main

import "crypto/sha256"

func f() []byte {
	h := sha256.New()
	return h.Sum(nil)
}
`,
	},

	{
		Name: "hashsum.1",
		In: `package main

func f(h hash.Hash) []byte {
	return h.Sum()
}
`,
		Out: `package main

func f(h hash.Hash) []byte {
	return h.Sum(nil)
}
`,
	},

	{
		Name: "hashsum.0",
		In: `package main

import "crypto/sha256"

func f() []byte {
	h := sha256.New()
	h.Write([]byte("foo"))
	digest := h.Sum()
}
`,
		Out: `package main

import "crypto/sha256"

func f() []byte {
	h := sha256.New()
	h.Write([]byte("foo"))
	digest := h.Sum(nil)
}
`,
	},

	{
		Name: "hashsum.0",
		In: `package main

import _ "crypto/sha256"
import "crypto"

func f() []byte {
	hashType := crypto.SHA256
	h := hashType.New()
	digest := h.Sum()
}
`,
		Out: `package main

import _ "crypto/sha256"
import "crypto"

func f() []byte {
	hashType := crypto.SHA256
	h := hashType.New()
	digest := h.Sum(nil)
}
`,
	},
}
