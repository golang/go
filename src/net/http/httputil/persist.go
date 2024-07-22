// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package httputil

import (
	"bufio"
	"errors"
	"io"
	"net"
	"net/http"
	"net/textproto"
	"sync"
)

var (
	// Deprecated: No longer used.
	ErrPersistEOF = &http.ProtocolError{ErrorString: "persistent connection closed"}

	// Deprecated: No longer used.
	ErrClosed = &http.ProtocolError{ErrorString: "connection closed by user"}

	// Deprecated: No longer used.
	ErrPipeline = &http.ProtocolError{ErrorString: "pipeline error"}
)

// This is an API usage error - the local side is closed.
// ErrPersistEOF (above) reports that the remote side is closed.
var errClosed = errors.New("i/o operation on closed connection")

// ServerConn is an artifact of Go's early HTTP implementation.
// It is low-level, old, and unused by Go's current HTTP stack.
// We should have deleted it before Go 1.
//
// Deprecated: Use the Server in package [net/http] instead.
type ServerConn struct {
	mu              sync.Mutex // read-write protects the following fields
	c               net.Conn
	r               *bufio.Reader
	re, we          error // read/write errors
	lastbody        io.ReadCloser
	nread, nwritten int
	pipereq         map[*http.Request]uint

	pipe textproto.Pipeline
}

// NewServerConn is an artifact of Go's early HTTP implementation.
// It is low-level, old, and unused by Go's current HTTP stack.
// We should have deleted it before Go 1.
//
// Deprecated: Use the Server in package [net/http] instead.
func NewServerConn(c net.Conn, r *bufio.Reader) *ServerConn {
	if r == nil {
		r = bufio.NewReader(c)
	}
	return &ServerConn{c: c, r: r, pipereq: make(map[*http.Request]uint)}
}

// Hijack detaches the [ServerConn] and returns the underlying connection as well
// as the read-side bufio which may have some left over data. Hijack may be
// called before Read has signaled the end of the keep-alive logic. The user
// should not call Hijack while [ServerConn.Read] or [ServerConn.Write] is in progress.
func (sc *ServerConn) Hijack() (net.Conn, *bufio.Reader) {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	c := sc.c
	r := sc.r
	sc.c = nil
	sc.r = nil
	return c, r
}

// Close calls [ServerConn.Hijack] and then also closes the underlying connection.
func (sc *ServerConn) Close() error {
	c, _ := sc.Hijack()
	if c != nil {
		return c.Close()
	}
	return nil
}

// Read returns the next request on the wire. An [ErrPersistEOF] is returned if
// it is gracefully determined that there are no more requests (e.g. after the
// first request on an HTTP/1.0 connection, or after a Connection:close on a
// HTTP/1.1 connection).
func (sc *ServerConn) Read() (*http.Request, error) {
	var req *http.Request
	var err error

	// Ensure ordered execution of Reads and Writes
	id := sc.pipe.Next()
	sc.pipe.StartRequest(id)
	defer func() {
		sc.pipe.EndRequest(id)
		if req == nil {
			sc.pipe.StartResponse(id)
			sc.pipe.EndResponse(id)
		} else {
			// Remember the pipeline id of this request
			sc.mu.Lock()
			sc.pipereq[req] = id
			sc.mu.Unlock()
		}
	}()

	sc.mu.Lock()
	if sc.we != nil { // no point receiving if write-side broken or closed
		defer sc.mu.Unlock()
		return nil, sc.we
	}
	if sc.re != nil {
		defer sc.mu.Unlock()
		return nil, sc.re
	}
	if sc.r == nil { // connection closed by user in the meantime
		defer sc.mu.Unlock()
		return nil, errClosed
	}
	r := sc.r
	lastbody := sc.lastbody
	sc.lastbody = nil
	sc.mu.Unlock()

	// Make sure body is fully consumed, even if user does not call body.Close
	if lastbody != nil {
		// body.Close is assumed to be idempotent and multiple calls to
		// it should return the error that its first invocation
		// returned.
		err = lastbody.Close()
		if err != nil {
			sc.mu.Lock()
			defer sc.mu.Unlock()
			sc.re = err
			return nil, err
		}
	}

	req, err = http.ReadRequest(r)
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if err != nil {
		if errors.Is(err, io.ErrUnexpectedEOF) {
			// A close from the opposing client is treated as a
			// graceful close, even if there was some unparse-able
			// data before the close.
			sc.re = ErrPersistEOF
			return nil, sc.re
		} else {
			sc.re = err
			return req, err
		}
	}
	sc.lastbody = req.Body
	sc.nread++
	if req.Close {
		sc.re = ErrPersistEOF
		return req, sc.re
	}
	return req, err
}

// Pending returns the number of unanswered requests
// that have been received on the connection.
func (sc *ServerConn) Pending() int {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	return sc.nread - sc.nwritten
}

// Write writes resp in response to req. To close the connection gracefully, set the
// Response.Close field to true. Write should be considered operational until
// it returns an error, regardless of any errors returned on the [ServerConn.Read] side.
func (sc *ServerConn) Write(req *http.Request, resp *http.Response) error {

	// Retrieve the pipeline ID of this request/response pair
	sc.mu.Lock()
	id, ok := sc.pipereq[req]
	delete(sc.pipereq, req)
	if !ok {
		sc.mu.Unlock()
		return ErrPipeline
	}
	sc.mu.Unlock()

	// Ensure pipeline order
	sc.pipe.StartResponse(id)
	defer sc.pipe.EndResponse(id)

	sc.mu.Lock()
	if sc.we != nil {
		defer sc.mu.Unlock()
		return sc.we
	}
	if sc.c == nil { // connection closed by user in the meantime
		defer sc.mu.Unlock()
		return ErrClosed
	}
	c := sc.c
	if sc.nread <= sc.nwritten {
		defer sc.mu.Unlock()
		return errors.New("persist server pipe count")
	}
	if resp.Close {
		// After signaling a keep-alive close, any pipelined unread
		// requests will be lost. It is up to the user to drain them
		// before signaling.
		sc.re = ErrPersistEOF
	}
	sc.mu.Unlock()

	err := resp.Write(c)
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if err != nil {
		sc.we = err
		return err
	}
	sc.nwritten++

	return nil
}

// ClientConn is an artifact of Go's early HTTP implementation.
// It is low-level, old, and unused by Go's current HTTP stack.
// We should have deleted it before Go 1.
//
// Deprecated: Use Client or Transport in package [net/http] instead.
type ClientConn struct {
	mu              sync.Mutex // read-write protects the following fields
	c               net.Conn
	r               *bufio.Reader
	re, we          error // read/write errors
	lastbody        io.ReadCloser
	nread, nwritten int
	pipereq         map[*http.Request]uint

	pipe     textproto.Pipeline
	writeReq func(*http.Request, io.Writer) error
}

// NewClientConn is an artifact of Go's early HTTP implementation.
// It is low-level, old, and unused by Go's current HTTP stack.
// We should have deleted it before Go 1.
//
// Deprecated: Use the Client or Transport in package [net/http] instead.
func NewClientConn(c net.Conn, r *bufio.Reader) *ClientConn {
	if r == nil {
		r = bufio.NewReader(c)
	}
	return &ClientConn{
		c:        c,
		r:        r,
		pipereq:  make(map[*http.Request]uint),
		writeReq: (*http.Request).Write,
	}
}

// NewProxyClientConn is an artifact of Go's early HTTP implementation.
// It is low-level, old, and unused by Go's current HTTP stack.
// We should have deleted it before Go 1.
//
// Deprecated: Use the Client or Transport in package [net/http] instead.
func NewProxyClientConn(c net.Conn, r *bufio.Reader) *ClientConn {
	cc := NewClientConn(c, r)
	cc.writeReq = (*http.Request).WriteProxy
	return cc
}

// Hijack detaches the [ClientConn] and returns the underlying connection as well
// as the read-side bufio which may have some left over data. Hijack may be
// called before the user or Read have signaled the end of the keep-alive
// logic. The user should not call Hijack while [ClientConn.Read] or ClientConn.Write is in progress.
func (cc *ClientConn) Hijack() (c net.Conn, r *bufio.Reader) {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	c = cc.c
	r = cc.r
	cc.c = nil
	cc.r = nil
	return
}

// Close calls [ClientConn.Hijack] and then also closes the underlying connection.
func (cc *ClientConn) Close() error {
	c, _ := cc.Hijack()
	if c != nil {
		return c.Close()
	}
	return nil
}

// Write writes a request. An [ErrPersistEOF] error is returned if the connection
// has been closed in an HTTP keep-alive sense. If req.Close equals true, the
// keep-alive connection is logically closed after this request and the opposing
// server is informed. An ErrUnexpectedEOF indicates the remote closed the
// underlying TCP connection, which is usually considered as graceful close.
func (cc *ClientConn) Write(req *http.Request) error {
	var err error

	// Ensure ordered execution of Writes
	id := cc.pipe.Next()
	cc.pipe.StartRequest(id)
	defer func() {
		cc.pipe.EndRequest(id)
		if err != nil {
			cc.pipe.StartResponse(id)
			cc.pipe.EndResponse(id)
		} else {
			// Remember the pipeline id of this request
			cc.mu.Lock()
			cc.pipereq[req] = id
			cc.mu.Unlock()
		}
	}()

	cc.mu.Lock()
	if cc.re != nil { // no point sending if read-side closed or broken
		defer cc.mu.Unlock()
		return cc.re
	}
	if cc.we != nil {
		defer cc.mu.Unlock()
		return cc.we
	}
	if cc.c == nil { // connection closed by user in the meantime
		defer cc.mu.Unlock()
		return errClosed
	}
	c := cc.c
	if req.Close {
		// We write the EOF to the write-side error, because there
		// still might be some pipelined reads
		cc.we = ErrPersistEOF
	}
	cc.mu.Unlock()

	err = cc.writeReq(req, c)
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if err != nil {
		cc.we = err
		return err
	}
	cc.nwritten++

	return nil
}

// Pending returns the number of unanswered requests
// that have been sent on the connection.
func (cc *ClientConn) Pending() int {
	cc.mu.Lock()
	defer cc.mu.Unlock()
	return cc.nwritten - cc.nread
}

// Read reads the next response from the wire. A valid response might be
// returned together with an [ErrPersistEOF], which means that the remote
// requested that this be the last request serviced. Read can be called
// concurrently with [ClientConn.Write], but not with another Read.
func (cc *ClientConn) Read(req *http.Request) (resp *http.Response, err error) {
	// Retrieve the pipeline ID of this request/response pair
	cc.mu.Lock()
	id, ok := cc.pipereq[req]
	delete(cc.pipereq, req)
	if !ok {
		cc.mu.Unlock()
		return nil, ErrPipeline
	}
	cc.mu.Unlock()

	// Ensure pipeline order
	cc.pipe.StartResponse(id)
	defer cc.pipe.EndResponse(id)

	cc.mu.Lock()
	if cc.re != nil {
		defer cc.mu.Unlock()
		return nil, cc.re
	}
	if cc.r == nil { // connection closed by user in the meantime
		defer cc.mu.Unlock()
		return nil, errClosed
	}
	r := cc.r
	lastbody := cc.lastbody
	cc.lastbody = nil
	cc.mu.Unlock()

	// Make sure body is fully consumed, even if user does not call body.Close
	if lastbody != nil {
		// body.Close is assumed to be idempotent and multiple calls to
		// it should return the error that its first invocation
		// returned.
		err = lastbody.Close()
		if err != nil {
			cc.mu.Lock()
			defer cc.mu.Unlock()
			cc.re = err
			return nil, err
		}
	}

	resp, err = http.ReadResponse(r, req)
	cc.mu.Lock()
	defer cc.mu.Unlock()
	if err != nil {
		cc.re = err
		return resp, err
	}
	cc.lastbody = resp.Body

	cc.nread++

	if resp.Close {
		cc.re = ErrPersistEOF // don't send any more requests
		return resp, cc.re
	}
	return resp, err
}

// Do is convenience method that writes a request and reads a response.
func (cc *ClientConn) Do(req *http.Request) (*http.Response, error) {
	err := cc.Write(req)
	if err != nil {
		return nil, err
	}
	return cc.Read(req)
}
