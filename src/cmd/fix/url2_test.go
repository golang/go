// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(url2Tests, url2)
}

var url2Tests = []testCase{
	{
		Name: "url2.0",
		In: `package main

import "net/url"

func f() {
	url.ParseWithReference("foo")
}
`,
		Out: `package main

import "net/url"

func f() {
	url.ParseWithFragment("foo")
}
`,
	},
}
