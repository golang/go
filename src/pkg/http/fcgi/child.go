// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fcgi

// This file implements FastCGI from the perspective of a child process.

import (
	"fmt"
	"http"
	"io"
	"net"
	"os"
	"strconv"
	"strings"
	"time"
)

// request holds the state for an in-progress request. As soon as it's complete,
// it's converted to an http.Request.
type request struct {
	pw        *io.PipeWriter
	reqId     uint16
	params    map[string]string
	buf       [1024]byte
	rawParams []byte
	keepConn  bool
}

func newRequest(reqId uint16, flags uint8) *request {
	r := &request{
		reqId:    reqId,
		params:   map[string]string{},
		keepConn: flags&flagKeepConn != 0,
	}
	r.rawParams = r.buf[:0]
	return r
}

// TODO(eds): copied from http/cgi
var skipHeader = map[string]bool{
	"HTTP_HOST":       true,
	"HTTP_REFERER":    true,
	"HTTP_USER_AGENT": true,
}

// httpRequest converts r to an http.Request.
// TODO(eds): this is very similar to http/cgi's requestFromEnvironment
func (r *request) httpRequest(body io.ReadCloser) (*http.Request, os.Error) {
	req := &http.Request{
		Method:  r.params["REQUEST_METHOD"],
		RawURL:  r.params["REQUEST_URI"],
		Body:    body,
		Header:  http.Header{},
		Trailer: http.Header{},
		Proto:   r.params["SERVER_PROTOCOL"],
	}

	var ok bool
	req.ProtoMajor, req.ProtoMinor, ok = http.ParseHTTPVersion(req.Proto)
	if !ok {
		return nil, os.NewError("fcgi: invalid HTTP version")
	}

	req.Host = r.params["HTTP_HOST"]
	req.Referer = r.params["HTTP_REFERER"]
	req.UserAgent = r.params["HTTP_USER_AGENT"]

	if lenstr := r.params["CONTENT_LENGTH"]; lenstr != "" {
		clen, err := strconv.Atoi64(r.params["CONTENT_LENGTH"])
		if err != nil {
			return nil, os.NewError("fcgi: bad CONTENT_LENGTH parameter: " + lenstr)
		}
		req.ContentLength = clen
	}

	if req.Host != "" {
		req.RawURL = "http://" + req.Host + r.params["REQUEST_URI"]
		url, err := http.ParseURL(req.RawURL)
		if err != nil {
			return nil, os.NewError("fcgi: failed to parse host and REQUEST_URI into a URL: " + req.RawURL)
		}
		req.URL = url
	}
	if req.URL == nil {
		req.RawURL = r.params["REQUEST_URI"]
		url, err := http.ParseURL(req.RawURL)
		if err != nil {
			return nil, os.NewError("fcgi: failed to parse REQUEST_URI into a URL: " + req.RawURL)
		}
		req.URL = url
	}

	for key, val := range r.params {
		if strings.HasPrefix(key, "HTTP_") && !skipHeader[key] {
			req.Header.Add(strings.Replace(key[5:], "_", "-", -1), val)
		}
	}
	return req, nil
}

// parseParams reads an encoded []byte into Params.
func (r *request) parseParams() {
	text := r.rawParams
	r.rawParams = nil
	for len(text) > 0 {
		keyLen, n := readSize(text)
		if n == 0 {
			return
		}
		text = text[n:]
		valLen, n := readSize(text)
		if n == 0 {
			return
		}
		text = text[n:]
		key := readString(text, keyLen)
		text = text[keyLen:]
		val := readString(text, valLen)
		text = text[valLen:]
		r.params[key] = val
	}
}

// response implements http.ResponseWriter.
type response struct {
	req         *request
	header      http.Header
	w           *bufWriter
	wroteHeader bool
}

func newResponse(c *child, req *request) *response {
	return &response{
		req:    req,
		header: http.Header{},
		w:      newWriter(c.conn, typeStdout, req.reqId),
	}
}

func (r *response) Header() http.Header {
	return r.header
}

func (r *response) Write(data []byte) (int, os.Error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	return r.w.Write(data)
}

func (r *response) WriteHeader(code int) {
	if r.wroteHeader {
		return
	}
	r.wroteHeader = true
	if code == http.StatusNotModified {
		// Must not have body.
		r.header.Del("Content-Type")
		r.header.Del("Content-Length")
		r.header.Del("Transfer-Encoding")
	} else if r.header.Get("Content-Type") == "" {
		r.header.Set("Content-Type", "text/html; charset=utf-8")
	}

	if r.header.Get("Date") == "" {
		r.header.Set("Date", time.UTC().Format(http.TimeFormat))
	}

	fmt.Fprintf(r.w, "Status: %d %s\r\n", code, http.StatusText(code))
	r.header.Write(r.w)
	r.w.WriteString("\r\n")
}

func (r *response) Flush() {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	r.w.Flush()
}

func (r *response) Close() os.Error {
	r.Flush()
	return r.w.Close()
}

type child struct {
	conn    *conn
	handler http.Handler
}

func newChild(rwc net.Conn, handler http.Handler) *child {
	return &child{newConn(rwc), handler}
}

func (c *child) serve() {
	requests := map[uint16]*request{}
	defer c.conn.Close()
	var rec record
	var br beginRequest
	for {
		if err := rec.read(c.conn.rwc); err != nil {
			return
		}

		req, ok := requests[rec.h.Id]
		if !ok && rec.h.Type != typeBeginRequest && rec.h.Type != typeGetValues {
			// The spec says to ignore unknown request IDs.
			continue
		}
		if ok && rec.h.Type == typeBeginRequest {
			// The server is trying to begin a request with the same ID
			// as an in-progress request. This is an error.
			return
		}

		switch rec.h.Type {
		case typeBeginRequest:
			if err := br.read(rec.content()); err != nil {
				return
			}
			if br.role != roleResponder {
				c.conn.writeEndRequest(rec.h.Id, 0, statusUnknownRole)
				break
			}
			requests[rec.h.Id] = newRequest(rec.h.Id, br.flags)
		case typeParams:
			// NOTE(eds): Technically a key-value pair can straddle the boundary
			// between two packets. We buffer until we've received all parameters.
			if len(rec.content()) > 0 {
				req.rawParams = append(req.rawParams, rec.content()...)
				break
			}
			req.parseParams()
		case typeStdin:
			content := rec.content()
			if req.pw == nil {
				var body io.ReadCloser
				if len(content) > 0 {
					// body could be an io.LimitReader, but it shouldn't matter
					// as long as both sides are behaving.
					body, req.pw = io.Pipe()
				}
				go c.serveRequest(req, body)
			}
			if len(content) > 0 {
				// TODO(eds): This blocks until the handler reads from the pipe.
				// If the handler takes a long time, it might be a problem.
				req.pw.Write(content)
			} else if req.pw != nil {
				req.pw.Close()
			}
		case typeGetValues:
			values := map[string]string{"FCGI_MPXS_CONNS": "1"}
			c.conn.writePairs(0, typeGetValuesResult, values)
		case typeData:
			// If the filter role is implemented, read the data stream here.
		case typeAbortRequest:
			requests[rec.h.Id] = nil, false
			c.conn.writeEndRequest(rec.h.Id, 0, statusRequestComplete)
			if !req.keepConn {
				// connection will close upon return
				return
			}
		default:
			b := make([]byte, 8)
			b[0] = rec.h.Type
			c.conn.writeRecord(typeUnknownType, 0, b)
		}
	}
}

func (c *child) serveRequest(req *request, body io.ReadCloser) {
	r := newResponse(c, req)
	httpReq, err := req.httpRequest(body)
	if err != nil {
		// there was an error reading the request
		r.WriteHeader(http.StatusInternalServerError)
		c.conn.writeRecord(typeStderr, req.reqId, []byte(err.String()))
	} else {
		c.handler.ServeHTTP(r, httpReq)
	}
	if body != nil {
		body.Close()
	}
	r.Close()
	c.conn.writeEndRequest(req.reqId, 0, statusRequestComplete)
	if !req.keepConn {
		c.conn.Close()
	}
}

// Serve accepts incoming FastCGI connections on the listener l, creating a new
// service thread for each. The service threads read requests and then call handler
// to reply to them.
// If l is nil, Serve accepts connections on stdin.
// If handler is nil, http.DefaultServeMux is used.
func Serve(l net.Listener, handler http.Handler) os.Error {
	if l == nil {
		var err os.Error
		l, err = net.FileListener(os.Stdin)
		if err != nil {
			return err
		}
		defer l.Close()
	}
	if handler == nil {
		handler = http.DefaultServeMux
	}
	for {
		rw, err := l.Accept()
		if err != nil {
			return err
		}
		c := newChild(rw, handler)
		go c.serve()
	}
	panic("unreachable")
}
