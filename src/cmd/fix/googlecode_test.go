// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(googlecodeTests, googlecode)
}

var googlecodeTests = []testCase{
	{
		Name: "googlecode.0",
		In: `package main

import (
	"foo.googlecode.com/hg/bar"
	"go-qux-23.googlecode.com/svn"
	"zap.googlecode.com/git/some/path"
)
`,
		Out: `package main

import (
	"code.google.com/p/foo/bar"
	"code.google.com/p/go-qux-23"
	"code.google.com/p/zap/some/path"
)
`,
	},
}
