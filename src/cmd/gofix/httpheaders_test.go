// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(httpHeadersTests, httpheaders)
}

var httpHeadersTests = []testCase{
	{
		Name: "httpheaders.0",
		In: `package headertest

import (
	"http"
)

type Other struct {
	Referer   string
	UserAgent string
	Cookie    []*http.Cookie
}

func f(req *http.Request, res *http.Response, other *Other) {
	_ = req.Referer
	_ = req.UserAgent
	_ = req.Cookie

	_ = res.Cookie

	_ = other.Referer
	_ = other.UserAgent
	_ = other.Cookie

	_ = req.Referer()
	_ = req.UserAgent()
	_ = req.Cookies()
	_ = res.Cookies()
}
`,
		Out: `package headertest

import (
	"http"
)

type Other struct {
	Referer   string
	UserAgent string
	Cookie    []*http.Cookie
}

func f(req *http.Request, res *http.Response, other *Other) {
	_ = req.Referer()
	_ = req.UserAgent()
	_ = req.Cookies()

	_ = res.Cookies()

	_ = other.Referer
	_ = other.UserAgent
	_ = other.Cookie

	_ = req.Referer()
	_ = req.UserAgent()
	_ = req.Cookies()
	_ = res.Cookies()
}
`,
	},
}
