// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgi

import (
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"path"
	"slices"
	"strings"
	"time"
)

func cgiMain() {
	switch path.Join(os.Getenv("SCRIPT_NAME"), os.Getenv("PATH_INFO")) {
	case "/bar", "/test.cgi", "/myscript/bar", "/test.cgi/extrapath":
		testCGI()
		return
	}
	childCGIProcess()
}

// testCGI is a CGI program translated from a Perl program to complete host_test.
// test cases in host_test should be provided by testCGI.
func testCGI() {
	req, err := Request()
	if err != nil {
		panic(err)
	}

	err = req.ParseForm()
	if err != nil {
		panic(err)
	}

	params := req.Form
	if params.Get("loc") != "" {
		fmt.Printf("Location: %s\r\n\r\n", params.Get("loc"))
		return
	}

	fmt.Printf("Content-Type: text/html\r\n")
	fmt.Printf("X-CGI-Pid: %d\r\n", os.Getpid())
	fmt.Printf("X-Test-Header: X-Test-Value\r\n")
	fmt.Printf("\r\n")

	if params.Get("writestderr") != "" {
		fmt.Fprintf(os.Stderr, "Hello, stderr!\n")
	}

	if params.Get("bigresponse") != "" {
		// 17 MB, for OS X: golang.org/issue/4958
		line := strings.Repeat("A", 1024)
		for i := 0; i < 17*1024; i++ {
			fmt.Printf("%s\r\n", line)
		}
		return
	}

	fmt.Printf("test=Hello CGI\r\n")

	for _, key := range slices.Sorted(maps.Keys(params)) {
		fmt.Printf("param-%s=%s\r\n", key, params.Get(key))
	}

	envs := envMap(os.Environ())
	for _, key := range slices.Sorted(maps.Keys(envs)) {
		fmt.Printf("env-%s=%s\r\n", key, envs[key])
	}

	cwd, _ := os.Getwd()
	fmt.Printf("cwd=%s\r\n", cwd)
}

type neverEnding byte

func (b neverEnding) Read(p []byte) (n int, err error) {
	for i := range p {
		p[i] = byte(b)
	}
	return len(p), nil
}

// childCGIProcess is used by integration_test to complete unit tests.
func childCGIProcess() {
	if os.Getenv("REQUEST_METHOD") == "" {
		// Not in a CGI environment; skipping test.
		return
	}
	switch os.Getenv("REQUEST_URI") {
	case "/immediate-disconnect":
		os.Exit(0)
	case "/no-content-type":
		fmt.Printf("Content-Length: 6\n\nHello\n")
		os.Exit(0)
	case "/empty-headers":
		fmt.Printf("\nHello")
		os.Exit(0)
	}
	Serve(http.HandlerFunc(func(rw http.ResponseWriter, req *http.Request) {
		if req.FormValue("nil-request-body") == "1" {
			fmt.Fprintf(rw, "nil-request-body=%v\n", req.Body == nil)
			return
		}
		rw.Header().Set("X-Test-Header", "X-Test-Value")
		req.ParseForm()
		if req.FormValue("no-body") == "1" {
			return
		}
		if eb, ok := req.Form["exact-body"]; ok {
			io.WriteString(rw, eb[0])
			return
		}
		if req.FormValue("write-forever") == "1" {
			io.Copy(rw, neverEnding('a'))
			for {
				time.Sleep(5 * time.Second) // hang forever, until killed
			}
		}
		fmt.Fprintf(rw, "test=Hello CGI-in-CGI\n")
		for k, vv := range req.Form {
			for _, v := range vv {
				fmt.Fprintf(rw, "param-%s=%s\n", k, v)
			}
		}
		for _, kv := range os.Environ() {
			fmt.Fprintf(rw, "env-%s\n", kv)
		}
	}))
	os.Exit(0)
}
