// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(httpserverTests, httpserver)
}

var httpserverTests = []testCase{
	{
		Name: "httpserver.0",
		In: `package main

import "http"

func f(xyz http.ResponseWriter, abc *http.Request, b string) {
	xyz.SetHeader("foo", "bar")
	xyz.SetHeader("baz", "")
	xyz.Hijack()
	xyz.Flush()
	go xyz.Hijack()
	defer xyz.Flush()
	_ = xyz.UsingTLS()
	_ = true == xyz.UsingTLS()
	_ = xyz.RemoteAddr()
	_ = xyz.RemoteAddr() == "hello"
	if xyz.UsingTLS() {
	}
}
`,
		Out: `package main

import "http"

func f(xyz http.ResponseWriter, abc *http.Request, b string) {
	xyz.Header().Set("foo", "bar")
	xyz.Header().Del("baz")
	xyz.(http.Hijacker).Hijack()
	xyz.(http.Flusher).Flush()
	go xyz.(http.Hijacker).Hijack()
	defer xyz.(http.Flusher).Flush()
	_ = abc.TLS != nil
	_ = true == (abc.TLS != nil)
	_ = abc.RemoteAddr
	_ = abc.RemoteAddr == "hello"
	if abc.TLS != nil {
	}
}
`,
	},
}
