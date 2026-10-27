// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements CGI from the perspective of a child
// process.

package cgi

import (
	"bufio"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
)

// Request returns the HTTP request as represented in the current
// environment. This assumes the current program is being run
// by a web server in a CGI environment.
// The returned Request's Body is populated, if applicable.
func Request() (*http.Request, error) {
	r, err := RequestFromMap(envMap(os.Environ()))
	if err != nil {
		return nil, err
	}
	if r.ContentLength > 0 {
		r.Body = io.NopCloser(io.LimitReader(os.Stdin, r.ContentLength))
	}
	return r, nil
}

func envMap(env []string) map[string]string {
	m := make(map[string]string)
	for _, kv := range env {
		if k, v, ok := strings.Cut(kv, "="); ok {
			m[k] = v
		}
	}
	return m
}

// RequestFromMap creates an [http.Request] from CGI variables.
// The returned Request's Body field is not populated.
func RequestFromMap(params map[string]string) (*http.Request, error) {
	r := new(http.Request)
	r.Method = params["REQUEST_METHOD"]
	if r.Method == "" {
		return nil, errors.New("cgi: no REQUEST_METHOD in environment")
	}

	r.Proto = params["SERVER_PROTOCOL"]
	var ok bool
	if r.Proto == "INCLUDED" {
		// SSI (Server Side Include) use case
		// CGI Specification RFC 3875 - section 4.1.16
		r.ProtoMajor, r.ProtoMinor = 1, 0
	} else if r.ProtoMajor, r.ProtoMinor, ok = http.ParseHTTPVersion(r.Proto); !ok {
		return nil, errors.New("cgi: invalid SERVER_PROTOCOL version")
	}

	r.Close = true
	r.Trailer = http.Header{}
	r.Header = http.Header{}

	r.Host = params["HTTP_HOST"]

	if lenstr := params["CONTENT_LENGTH"]; lenstr != "" {
		clen, err := strconv.ParseInt(lenstr, 10, 64)
		if err != nil {
			return nil, errors.New("cgi: bad CONTENT_LENGTH in environment: " + lenstr)
		}
		r.ContentLength = clen
	}

	if ct := params["CONTENT_TYPE"]; ct != "" {
		r.Header.Set("Content-Type", ct)
	}

	// Copy "HTTP_FOO_BAR" variables to "Foo-Bar" Headers
	for k, v := range params {
		if k == "HTTP_HOST" {
			continue
		}
		if after, found := strings.CutPrefix(k, "HTTP_"); found {
			r.Header.Add(strings.ReplaceAll(after, "_", "-"), v)
		}
	}

	uriStr := params["REQUEST_URI"]
	if uriStr == "" {
		// Fallback to SCRIPT_NAME, PATH_INFO and QUERY_STRING.
		uriStr = params["SCRIPT_NAME"] + params["PATH_INFO"]
		if uriStr == "" {
			uriStr = "/"
		}
		s := params["QUERY_STRING"]
		if s != "" {
			uriStr += "?" + s
		}
	}

	// There's apparently a de-facto standard for this.
	// https://web.archive.org/web/20170105004655/http://docstore.mik.ua/orelly/linux/cgi/ch03_02.htm#ch03-35636
	if s := params["HTTPS"]; s == "on" || s == "ON" || s == "1" {
		r.TLS = &tls.ConnectionState{HandshakeComplete: true}
	}

	var err error
	r.URL, err = url.ParseRequestURI(uriStr)
	if err != nil {
		return nil, errors.New("cgi: failed to parse REQUEST_URI into a URL: " + uriStr)
	}
	if r.Host != "" {
		// Historical behavior: Scheme is only set if there is a host.
		// (We could probably change this.)
		if r.TLS == nil {
			r.URL.Scheme = "http"
		} else {
			r.URL.Scheme = "https"
		}
		r.URL.Host = r.Host
		r.URL.User = nil
	}

	// Request.RemoteAddr has its port set by Go's standard http
	// server, so we do here too.
	remotePort, _ := strconv.Atoi(params["REMOTE_PORT"]) // zero if unset or invalid
	r.RemoteAddr = net.JoinHostPort(params["REMOTE_ADDR"], strconv.Itoa(remotePort))

	return r, nil
}

// Serve executes the provided [Handler] on the currently active CGI
// request, if any. If there's no current CGI environment
// an error is returned. The provided handler may be nil to use
// [http.DefaultServeMux].
func Serve(handler http.Handler) error {
	req, err := Request()
	if err != nil {
		return err
	}
	if req.Body == nil {
		req.Body = http.NoBody
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
	rw.Write(nil) // make sure a response is sent
	if err = rw.bufw.Flush(); err != nil {
		return err
	}
	return nil
}

type response struct {
	req            *http.Request
	header         http.Header
	code           int
	wroteHeader    bool
	wroteCGIHeader bool
	bufw           *bufio.Writer
}

func (r *response) Flush() {
	r.bufw.Flush()
}

func (r *response) Header() http.Header {
	return r.header
}

func (r *response) Write(p []byte) (n int, err error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	if !r.wroteCGIHeader {
		r.writeCGIHeader(p)
	}
	return r.bufw.Write(p)
}

func (r *response) WriteHeader(code int) {
	if r.wroteHeader {
		// Note: explicitly using Stderr, as Stdout is our HTTP output.
		fmt.Fprintf(os.Stderr, "CGI attempted to write header twice on request for %s", r.req.URL)
		return
	}
	r.wroteHeader = true
	r.code = code
}

// writeCGIHeader finalizes the header sent to the client and writes it to the output.
// p is not written by writeHeader, but is the first chunk of the body
// that will be written. It is sniffed for a Content-Type if none is
// set explicitly.
func (r *response) writeCGIHeader(p []byte) {
	if r.wroteCGIHeader {
		return
	}
	r.wroteCGIHeader = true
	fmt.Fprintf(r.bufw, "Status: %d %s\r\n", r.code, http.StatusText(r.code))
	if _, hasType := r.header["Content-Type"]; !hasType {
		r.header.Set("Content-Type", http.DetectContentType(p))
	}
	r.header.Write(r.bufw)
	r.bufw.WriteString("\r\n")
	r.bufw.Flush()
}
