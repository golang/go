// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(httputilTests, httputil)
}

var httputilTests = []testCase{
	{
		Name: "httputil.0",
		In: `package main

import "net/http"

func f() {
	http.DumpRequest(nil, false)
	http.DumpRequestOut(nil, false)
	http.DumpResponse(nil, false)
	http.NewChunkedReader(nil)
	http.NewChunkedWriter(nil)
	http.NewClientConn(nil, nil)
	http.NewProxyClientConn(nil, nil)
	http.NewServerConn(nil, nil)
	http.NewSingleHostReverseProxy(nil)
}
`,
		Out: `package main

import "net/http/httputil"

func f() {
	httputil.DumpRequest(nil, false)
	httputil.DumpRequestOut(nil, false)
	httputil.DumpResponse(nil, false)
	httputil.NewChunkedReader(nil)
	httputil.NewChunkedWriter(nil)
	httputil.NewClientConn(nil, nil)
	httputil.NewProxyClientConn(nil, nil)
	httputil.NewServerConn(nil, nil)
	httputil.NewSingleHostReverseProxy(nil)
}
`,
	},
	{
		Name: "httputil.1",
		In: `package main

import "net/http"

func f() {
	http.DumpRequest(nil, false)
	http.DumpRequestOut(nil, false)
	http.DumpResponse(nil, false)
	http.NewChunkedReader(nil)
	http.NewChunkedWriter(nil)
	http.NewClientConn(nil, nil)
	http.NewProxyClientConn(nil, nil)
	http.NewServerConn(nil, nil)
	http.NewSingleHostReverseProxy(nil)
}
`,
		Out: `package main

import "net/http/httputil"

func f() {
	httputil.DumpRequest(nil, false)
	httputil.DumpRequestOut(nil, false)
	httputil.DumpResponse(nil, false)
	httputil.NewChunkedReader(nil)
	httputil.NewChunkedWriter(nil)
	httputil.NewClientConn(nil, nil)
	httputil.NewProxyClientConn(nil, nil)
	httputil.NewServerConn(nil, nil)
	httputil.NewSingleHostReverseProxy(nil)
}
`,
	},
	{
		Name: "httputil.2",
		In: `package main

import "net/http"

func f() {
	http.DumpRequest(nil, false)
	http.DumpRequestOut(nil, false)
	http.DumpResponse(nil, false)
	http.NewChunkedReader(nil)
	http.NewChunkedWriter(nil)
	http.NewClientConn(nil, nil)
	http.NewProxyClientConn(nil, nil)
	http.NewServerConn(nil, nil)
	http.NewSingleHostReverseProxy(nil)
	http.Get("")
}
`,
		Out: `package main

import (
	"net/http"
	"net/http/httputil"
)

func f() {
	httputil.DumpRequest(nil, false)
	httputil.DumpRequestOut(nil, false)
	httputil.DumpResponse(nil, false)
	httputil.NewChunkedReader(nil)
	httputil.NewChunkedWriter(nil)
	httputil.NewClientConn(nil, nil)
	httputil.NewProxyClientConn(nil, nil)
	httputil.NewServerConn(nil, nil)
	httputil.NewSingleHostReverseProxy(nil)
	http.Get("")
}
`,
	},
}
