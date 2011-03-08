// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements CGI from the perspective of a child
// process.

package cgi

import (
	"bufio"
	"fmt"
	"http"
	"io"
	"os"
	"strconv"
	"strings"
)

// Request returns the HTTP request as represented in the current
// environment. This assumes the current program is being run
// by a web server in a CGI environment.
func Request() (*http.Request, os.Error) {
	return requestFromEnvironment(envMap(os.Environ()))
}

func envMap(env []string) map[string]string {
	m := make(map[string]string)
	for _, kv := range env {
		if idx := strings.Index(kv, "="); idx != -1 {
			m[kv[:idx]] = kv[idx+1:]
		}
	}
	return m
}

// These environment variables are manually copied into Request
var skipHeader = map[string]bool{
	"HTTP_HOST":       true,
	"HTTP_REFERER":    true,
	"HTTP_USER_AGENT": true,
}

func requestFromEnvironment(env map[string]string) (*http.Request, os.Error) {
	r := new(http.Request)
	r.Method = env["REQUEST_METHOD"]
	if r.Method == "" {
		return nil, os.NewError("cgi: no REQUEST_METHOD in environment")
	}
	r.Close = true
	r.Trailer = http.Header{}
	r.Header = http.Header{}

	r.Host = env["HTTP_HOST"]
	r.Referer = env["HTTP_REFERER"]
	r.UserAgent = env["HTTP_USER_AGENT"]

	// CGI doesn't allow chunked requests, so these should all be accurate:
	r.Proto = "HTTP/1.0"
	r.ProtoMajor = 1
	r.ProtoMinor = 0
	r.TransferEncoding = nil

	if lenstr := env["CONTENT_LENGTH"]; lenstr != "" {
		clen, err := strconv.Atoi64(lenstr)
		if err != nil {
			return nil, os.NewError("cgi: bad CONTENT_LENGTH in environment: " + lenstr)
		}
		r.ContentLength = clen
		r.Body = nopCloser{io.LimitReader(os.Stdin, clen)}
	}

	// Copy "HTTP_FOO_BAR" variables to "Foo-Bar" Headers
	for k, v := range env {
		if !strings.HasPrefix(k, "HTTP_") || skipHeader[k] {
			continue
		}
		r.Header.Add(strings.Replace(k[5:], "_", "-", -1), v)
	}

	// TODO: cookies.  parsing them isn't exported, though.

	if r.Host != "" {
		// Hostname is provided, so we can reasonably construct a URL,
		// even if we have to assume 'http' for the scheme.
		r.RawURL = "http://" + r.Host + env["REQUEST_URI"]
		url, err := http.ParseURL(r.RawURL)
		if err != nil {
			return nil, os.NewError("cgi: failed to parse host and REQUEST_URI into a URL: " + r.RawURL)
		}
		r.URL = url
	}
	// Fallback logic if we don't have a Host header or the URL
	// failed to parse
	if r.URL == nil {
		r.RawURL = env["REQUEST_URI"]
		url, err := http.ParseURL(r.RawURL)
		if err != nil {
			return nil, os.NewError("cgi: failed to parse REQUEST_URI into a URL: " + r.RawURL)
		}
		r.URL = url
	}
	return r, nil
}

// TODO: move this to ioutil or something.  It's copy/pasted way too often.
type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() os.Error { return nil }

// Serve executes the provided Handler on the currently active CGI
// request, if any. If there's no current CGI environment
// an error is returned. The provided handler may be nil to use
// http.DefaultServeMux.
func Serve(handler http.Handler) os.Error {
	req, err := Request()
	if err != nil {
		return err
	}
	if handler == nil {
		handler = http.DefaultServeMux
	}
	rw := &response{
		req:    req,
		header: make(http.Header),
		bufw:   bufio.NewWriter(os.Stdout),
	}
	handler.ServeHTTP(rw, req)
	if err = rw.bufw.Flush(); err != nil {
		return err
	}
	return nil
}

type response struct {
	req        *http.Request
	header     http.Header
	bufw       *bufio.Writer
	headerSent bool
}

func (r *response) Flush() {
	r.bufw.Flush()
}

func (r *response) RemoteAddr() string {
	return os.Getenv("REMOTE_ADDR")
}

func (r *response) SetHeader(k, v string) {
	if v == "" {
		r.header.Del(k)
	} else {
		r.header.Set(k, v)
	}
}

func (r *response) Write(p []byte) (n int, err os.Error) {
	if !r.headerSent {
		r.WriteHeader(http.StatusOK)
	}
	return r.bufw.Write(p)
}

func (r *response) WriteHeader(code int) {
	if r.headerSent {
		// Note: explicitly using Stderr, as Stdout is our HTTP output.
		fmt.Fprintf(os.Stderr, "CGI attempted to write header twice on request for %s", r.req.URL)
		return
	}
	r.headerSent = true
	fmt.Fprintf(r.bufw, "Status: %d %s\r\n", code, http.StatusText(code))

	// Set a default Content-Type
	if _, hasType := r.header["Content-Type"]; !hasType {
		r.header.Add("Content-Type", "text/html; charset=utf-8")
	}

	// TODO: add a method on http.Header to write itself to an io.Writer?
	// This is duplicated code.
	for k, vv := range r.header {
		for _, v := range vv {
			v = strings.Replace(v, "\n", "", -1)
			v = strings.Replace(v, "\r", "", -1)
			v = strings.TrimSpace(v)
			fmt.Fprintf(r.bufw, "%s: %s\r\n", k, v)
		}
	}
	r.bufw.Write([]byte("\r\n"))
	r.bufw.Flush()
}

func (r *response) UsingTLS() bool {
	// There's apparently a de-facto standard for this.
	// http://docstore.mik.ua/orelly/linux/cgi/ch03_02.htm#ch03-35636
	if s := os.Getenv("HTTPS"); s == "on" || s == "ON" || s == "1" {
		return true
	}
	return false
}
