// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(httpFileSystemTests, httpfs)
}

var httpFileSystemTests = []testCase{
	{
		Name: "httpfs.0",
		In: `package httpfs

import (
	"http"
)

func f() {
	_ = http.FileServer("/var/www/foo", "/")
	_ = http.FileServer("/var/www/foo", "")
	_ = http.FileServer("/var/www/foo/bar", "/bar")
	s := "/foo"
	_ = http.FileServer(s, "/")
	prefix := "/p"
	_ = http.FileServer(s, prefix)
}
`,
		Out: `package httpfs

import (
	"http"
)

func f() {
	_ = http.FileServer(http.Dir("/var/www/foo"))
	_ = http.FileServer(http.Dir("/var/www/foo"))
	_ = http.StripPrefix("/bar", http.FileServer(http.Dir("/var/www/foo/bar")))
	s := "/foo"
	_ = http.FileServer(http.Dir(s))
	prefix := "/p"
	_ = http.StripPrefix(prefix, http.FileServer(http.Dir(s)))
}
`,
	},
}
