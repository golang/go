// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(ioCopyNTests, ioCopyN)
}

var ioCopyNTests = []testCase{
	{
		Name: "io.CopyN.0",
		In: `package main

import (
	"io"
)

func f() {
	io.Copyn(dst, src)
	foo.Copyn(dst, src)
}
`,
		Out: `package main

import (
	"io"
)

func f() {
	io.CopyN(dst, src)
	foo.Copyn(dst, src)
}
`,
	},
}
