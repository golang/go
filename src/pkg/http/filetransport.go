// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"fmt"
	"io"
	"os"
)

// fileTransport implements RoundTripper for the 'file' protocol.
type fileTransport struct {
	fh fileHandler
}

// NewFileTransport returns a new RoundTripper, serving the provided
// FileSystem. The returned RoundTripper ignores the URL host in its
// incoming requests, as well as most other properties of the
// request.
//
// The typical use case for NewFileTransport is to register the "file"
// protocol with a Transport, as in:
//
//   t := &http.Transport{}
//   t.RegisterProtocol("file", http.NewFileTransport(http.Dir("/")))
//   c := &http.Client{Transport: t}
//   res, err := c.Get("file:///etc/passwd")
//   ...
func NewFileTransport(fs FileSystem) RoundTripper {
	return fileTransport{fileHandler{fs}}
}

func (t fileTransport) RoundTrip(req *Request) (resp *Response, err os.Error) {
	// We start ServeHTTP in a goroutine, which may take a long
	// time if the file is large.  The newPopulateResponseWriter
	// call returns a channel which either ServeHTTP or finish()
	// sends our *Response on, once the *Response itself has been
	// populated (even if the body itself is still being
	// written to the res.Body, a pipe)
	rw, resc := newPopulateResponseWriter()
	go func() {
		t.fh.ServeHTTP(rw, req)
		rw.finish()
	}()
	return <-resc, nil
}

func newPopulateResponseWriter() (*populateResponse, <-chan *Response) {
	pr, pw := io.Pipe()
	rw := &populateResponse{
		ch: make(chan *Response),
		pw: pw,
		res: &Response{
			Proto:      "HTTP/1.0",
			ProtoMajor: 1,
			Header:     make(Header),
			Close:      true,
			Body:       pr,
		},
	}
	return rw, rw.ch
}

// populateResponse is a ResponseWriter that populates the *Response
// in res, and writes its body to a pipe connected to the response
// body. Once writes begin or finish() is called, the response is sent
// on ch.
type populateResponse struct {
	res          *Response
	ch           chan *Response
	wroteHeader  bool
	hasContent   bool
	sentResponse bool
	pw           *io.PipeWriter
}

func (pr *populateResponse) finish() {
	if !pr.wroteHeader {
		pr.WriteHeader(500)
	}
	if !pr.sentResponse {
		pr.sendResponse()
	}
	pr.pw.Close()
}

func (pr *populateResponse) sendResponse() {
	if pr.sentResponse {
		return
	}
	pr.sentResponse = true

	if pr.hasContent {
		pr.res.ContentLength = -1
	}
	pr.ch <- pr.res
}

func (pr *populateResponse) Header() Header {
	return pr.res.Header
}

func (pr *populateResponse) WriteHeader(code int) {
	if pr.wroteHeader {
		return
	}
	pr.wroteHeader = true

	pr.res.StatusCode = code
	pr.res.Status = fmt.Sprintf("%d %s", code, StatusText(code))
}

func (pr *populateResponse) Write(p []byte) (n int, err os.Error) {
	if !pr.wroteHeader {
		pr.WriteHeader(StatusOK)
	}
	pr.hasContent = true
	if !pr.sentResponse {
		pr.sendResponse()
	}
	return pr.pw.Write(p)
}
