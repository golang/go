// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fcgi

// This file implements FastCGI from the perspective of a child process.

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/cgi"
	"os"
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

// envVarsContextKey uniquely identifies a mapping of CGI
// environment variables to their values in a request context
type envVarsContextKey struct{}

func newRequest(reqId uint16, flags uint8) *request {
	r := &request{
		reqId:    reqId,
		params:   map[string]string{},
		keepConn: flags&flagKeepConn != 0,
	}
	r.rawParams = r.buf[:0]
	return r
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
		if int(keyLen)+int(valLen) > len(text) {
			return
		}
		key := readString(text, keyLen)
		text = text[keyLen:]
		val := readString(text, valLen)
		text = text[valLen:]
		r.params[key] = val
	}
}

// response implements http.ResponseWriter.
type response struct {
	req            *request
	header         http.Header
	code           int
	wroteHeader    bool
	wroteCGIHeader bool
	w              *bufWriter
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

func (r *response) Write(p []byte) (n int, err error) {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	if !r.wroteCGIHeader {
		r.writeCGIHeader(p)
	}
	return r.w.Write(p)
}

func (r *response) WriteHeader(code int) {
	if r.wroteHeader {
		return
	}
	r.wroteHeader = true
	r.code = code
	if code == http.StatusNotModified {
		// Must not have body.
		r.header.Del("Content-Type")
		r.header.Del("Content-Length")
		r.header.Del("Transfer-Encoding")
	}
	if r.header.Get("Date") == "" {
		r.header.Set("Date", time.Now().UTC().Format(http.TimeFormat))
	}
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
	fmt.Fprintf(r.w, "Status: %d %s\r\n", r.code, http.StatusText(r.code))
	if _, hasType := r.header["Content-Type"]; r.code != http.StatusNotModified && !hasType {
		r.header.Set("Content-Type", http.DetectContentType(p))
	}
	r.header.Write(r.w)
	r.w.WriteString("\r\n")
	r.w.Flush()
}

func (r *response) Flush() {
	if !r.wroteHeader {
		r.WriteHeader(http.StatusOK)
	}
	r.w.Flush()
}

func (r *response) Close() error {
	r.Flush()
	return r.w.Close()
}

type child struct {
	conn    *conn
	handler http.Handler

	requests map[uint16]*request // keyed by request ID
}

func newChild(rwc io.ReadWriteCloser, handler http.Handler) *child {
	return &child{
		conn:     newConn(rwc),
		handler:  handler,
		requests: make(map[uint16]*request),
	}
}

func (c *child) serve() {
	defer c.conn.Close()
	defer c.cleanUp()
	var rec record
	for {
		if err := rec.read(c.conn.rwc); err != nil {
			return
		}
		if err := c.handleRecord(&rec); err != nil {
			return
		}
	}
}

var errCloseConn = errors.New("fcgi: connection should be closed")

var emptyBody = io.NopCloser(strings.NewReader(""))

// ErrRequestAborted is returned by Read when a handler attempts to read the
// body of a request that has been aborted by the web server.
var ErrRequestAborted = errors.New("fcgi: request aborted by web server")

// ErrConnClosed is returned by Read when a handler attempts to read the body of
// a request after the connection to the web server has been closed.
var ErrConnClosed = errors.New("fcgi: connection to web server closed")

func (c *child) handleRecord(rec *record) error {
	req, ok := c.requests[rec.h.Id]
	if !ok && rec.h.Type != typeBeginRequest && rec.h.Type != typeGetValues {
		// The spec says to ignore unknown request IDs.
		return nil
	}

	switch rec.h.Type {
	case typeBeginRequest:
		if req != nil {
			// The server is trying to begin a request with the same ID
			// as an in-progress request. This is an error.
			return errors.New("fcgi: received ID that is already in-flight")
		}

		var br beginRequest
		if err := br.read(rec.content()); err != nil {
			return err
		}
		if br.role != roleResponder {
			c.conn.writeEndRequest(rec.h.Id, 0, statusUnknownRole)
			return nil
		}
		req = newRequest(rec.h.Id, br.flags)
		c.requests[rec.h.Id] = req
		return nil
	case typeParams:
		// NOTE(eds): Technically a key-value pair can straddle the boundary
		// between two packets. We buffer until we've received all parameters.
		if len(rec.content()) > 0 {
			req.rawParams = append(req.rawParams, rec.content()...)
			return nil
		}
		req.parseParams()
		return nil
	case typeStdin:
		content := rec.content()
		if req.pw == nil {
			var body io.ReadCloser
			if len(content) > 0 {
				// body could be an io.LimitReader, but it shouldn't matter
				// as long as both sides are behaving.
				body, req.pw = io.Pipe()
			} else {
				body = emptyBody
			}
			go c.serveRequest(req, body)
		}
		if len(content) > 0 {
			// TODO(eds): This blocks until the handler reads from the pipe.
			// If the handler takes a long time, it might be a problem.
			req.pw.Write(content)
		} else {
			delete(c.requests, req.reqId)
			if req.pw != nil {
				req.pw.Close()
			}
		}
		return nil
	case typeGetValues:
		values := map[string]string{"FCGI_MPXS_CONNS": "1"}
		c.conn.writePairs(typeGetValuesResult, 0, values)
		return nil
	case typeData:
		// If the filter role is implemented, read the data stream here.
		return nil
	case typeAbortRequest:
		delete(c.requests, rec.h.Id)
		c.conn.writeEndRequest(rec.h.Id, 0, statusRequestComplete)
		if req.pw != nil {
			req.pw.CloseWithError(ErrRequestAborted)
		}
		if !req.keepConn {
			// connection will close upon return
			return errCloseConn
		}
		return nil
	default:
		b := make([]byte, 8)
		b[0] = byte(rec.h.Type)
		c.conn.writeRecord(typeUnknownType, 0, b)
		return nil
	}
}

// filterOutUsedEnvVars returns a new map of env vars without the
// variables in the given envVars map that are read for creating each http.Request
func filterOutUsedEnvVars(envVars map[string]string) map[string]string {
	withoutUsedEnvVars := make(map[string]string)
	for k, v := range envVars {
		if addFastCGIEnvToContext(k) {
			withoutUsedEnvVars[k] = v
		}
	}
	return withoutUsedEnvVars
}

func (c *child) serveRequest(req *request, body io.ReadCloser) {
	r := newResponse(c, req)
	httpReq, err := cgi.RequestFromMap(req.params)
	if err != nil {
		// there was an error reading the request
		r.WriteHeader(http.StatusInternalServerError)
		c.conn.writeRecord(typeStderr, req.reqId, []byte(err.Error()))
	} else {
		httpReq.Body = body
		withoutUsedEnvVars := filterOutUsedEnvVars(req.params)
		envVarCtx := context.WithValue(httpReq.Context(), envVarsContextKey{}, withoutUsedEnvVars)
		httpReq = httpReq.WithContext(envVarCtx)
		c.handler.ServeHTTP(r, httpReq)
	}
	// Make sure we serve something even if nothing was written to r
	r.Write(nil)
	r.Close()
	c.conn.writeEndRequest(req.reqId, 0, statusRequestComplete)

	// Consume the entire body, so the host isn't still writing to
	// us when we close the socket below in the !keepConn case,
	// otherwise we'd send a RST. (golang.org/issue/4183)
	// TODO(bradfitz): also bound this copy in time. Or send
	// some sort of abort request to the host, so the host
	// can properly cut off the client sending all the data.
	// For now just bound it a little and
	io.CopyN(io.Discard, body, 100<<20)
	body.Close()

	if !req.keepConn {
		c.conn.Close()
	}
}

func (c *child) cleanUp() {
	for _, req := range c.requests {
		if req.pw != nil {
			// race with call to Close in c.serveRequest doesn't matter because
			// Pipe(Reader|Writer).Close are idempotent
			req.pw.CloseWithError(ErrConnClosed)
		}
	}
}

// Serve accepts incoming FastCGI connections on the listener l, creating a new
// goroutine for each. The goroutine reads requests and then calls handler
// to reply to them.
// If l is nil, Serve accepts connections from os.Stdin.
// If handler is nil, [http.DefaultServeMux] is used.
func Serve(l net.Listener, handler http.Handler) error {
	if l == nil {
		var err error
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
}

// ProcessEnv returns FastCGI environment variables associated with the request r
// for which no effort was made to be included in the request itself - the data
// is hidden in the request's context. As an example, if REMOTE_USER is set for a
// request, it will not be found anywhere in r, but it will be included in
// ProcessEnv's response (via r's context).
func ProcessEnv(r *http.Request) map[string]string {
	env, _ := r.Context().Value(envVarsContextKey{}).(map[string]string)
	return env
}

// addFastCGIEnvToContext reports whether to include the FastCGI environment variable s
// in the http.Request.Context, accessible via ProcessEnv.
func addFastCGIEnvToContext(s string) bool {
	// Exclude things supported by net/http natively:
	switch s {
	case "CONTENT_LENGTH", "CONTENT_TYPE", "HTTPS",
		"PATH_INFO", "QUERY_STRING", "REMOTE_ADDR",
		"REMOTE_HOST", "REMOTE_PORT", "REQUEST_METHOD",
		"REQUEST_URI", "SCRIPT_NAME", "SERVER_PROTOCOL":
		return false
	}
	if strings.HasPrefix(s, "HTTP_") {
		return false
	}
	// Explicitly include FastCGI-specific things.
	// This list is redundant with the default "return true" below.
	// Consider this documentation of the sorts of things we expect
	// to maybe see.
	switch s {
	case "REMOTE_USER":
		return true
	}
	// Unknown, so include it to be safe.
	return true
}
