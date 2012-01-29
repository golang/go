// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(urlTests, url)
}

var urlTests = []testCase{
	{
		Name: "url.0",
		In: `package main

import (
	"http"
)

func f() {
	var _ http.URL
	http.ParseURL(a)
	http.ParseURLReference(a)
	http.ParseQuery(a)
	m := http.Values{a: b}
	http.URLEscape(a)
	http.URLUnescape(a)
	var x http.URLError
	var y http.URLEscapeError
}
`,
		Out: `package main

import "url"

func f() {
	var _ url.URL
	url.Parse(a)
	url.ParseWithReference(a)
	url.ParseQuery(a)
	m := url.Values{a: b}
	url.QueryEscape(a)
	url.QueryUnescape(a)
	var x url.Error
	var y url.EscapeError
}
`,
	},
	{
		Name: "url.1",
		In: `package main

import (
	"http"
)

func f() {
	http.ParseURL(a)
	var x http.Request
}
`,
		Out: `package main

import (
	"http"
	"url"
)

func f() {
	url.Parse(a)
	var x http.Request
}
`,
	},
	{
		Name: "url.2",
		In: `package main

import (
	"http"
)

type U struct{ url int }
type M map[int]int

func f() {
	http.ParseURL(a)
	var url = 23
	url, x := 45, y
	_ = U{url: url}
	_ = M{url + 1: url}
}

func g(url string) string {
	return url
}

func h() (url string) {
	return url
}
`,
		Out: `package main

import "url"

type U struct{ url_ int }
type M map[int]int

func f() {
	url.Parse(a)
	var url_ = 23
	url_, x := 45, y
	_ = U{url_: url_}
	_ = M{url_ + 1: url_}
}

func g(url_ string) string {
	return url_
}

func h() (url_ string) {
	return url_
}
`,
	},
	{
		Name: "url.3",
		In: `package main

import "http"

type U struct{ url string }

func f() {
	var u U
	u.url = "x"
}

func (url *T) m() string {
	return url
}
`,
		Out: `package main

import "http"

type U struct{ url string }

func f() {
	var u U
	u.url = "x"
}

func (url *T) m() string {
	return url
}
`,
	},
}
