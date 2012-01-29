// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(htmlerrTests, htmlerr)
}

var htmlerrTests = []testCase{
	{
		Name: "htmlerr.0",
		In: `package main

import (
	"html"
)

func f() {
	e := errors.New("")
	t := html.NewTokenizer(r)
	_, _ = e.Error(), t.Error()
}
`,
		Out: `package main

import (
	"html"
)

func f() {
	e := errors.New("")
	t := html.NewTokenizer(r)
	_, _ = e.Error(), t.Err()
}
`,
	},
}
