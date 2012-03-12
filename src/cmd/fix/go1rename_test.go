// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

func init() {
	addTestCases(go1renameTests, go1renameFix.f)
}

var go1renameTests = []testCase{
	{
		Name: "go1rename.0",
		In: `package main

import (
	"crypto/aes"
	"crypto/des"
	"encoding/json"
	"net/http"
	"net/url"
	"os"
	"runtime"
)

var (
	_ *aes.Cipher
	_ *des.Cipher
	_ *des.TripleDESCipher
	_ = json.MarshalForHTML
	_ = aes.New()
	_ = url.Parse
	_ = url.ParseWithReference
	_ = url.ParseRequest
	_ = os.Exec
	_ = runtime.Cgocalls
	_ = runtime.Goroutines
	_ = http.ErrPersistEOF
	_ = http.ErrPipeline
	_ = http.ErrClosed
	_ = http.NewSingleHostReverseProxy
	_ = http.NewChunkedReader
	_ = http.NewChunkedWriter
	_ *http.ReverseProxy
	_ *http.ClientConn
	_ *http.ServerConn
)
`,
		Out: `package main

import (
	"crypto/aes"
	"crypto/cipher"
	"encoding/json"
	"net/http/httputil"
	"net/url"
	"runtime"
	"syscall"
)

var (
	_ cipher.Block
	_ cipher.Block
	_ cipher.Block
	_ = json.Marshal
	_ = aes.New()
	_ = url.Parse
	_ = url.Parse
	_ = url.ParseRequestURI
	_ = syscall.Exec
	_ = runtime.NumCgoCall
	_ = runtime.NumGoroutine
	_ = httputil.ErrPersistEOF
	_ = httputil.ErrPipeline
	_ = httputil.ErrClosed
	_ = httputil.NewSingleHostReverseProxy
	_ = httputil.NewChunkedReader
	_ = httputil.NewChunkedWriter
	_ *httputil.ReverseProxy
	_ *httputil.ClientConn
	_ *httputil.ServerConn
)
`,
	},
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
