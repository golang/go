// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"errors"
	"io"
	"net"
	"net/textproto"
	"os"
	"sync"
)

var (
	ErrPersistEOF = &ProtocolError{"persistent connection closed"}
	ErrPipeline   = &ProtocolError{"pipeline error"}
)

// A ServerConn reads requests and sends responses over an underlying
// connection, until the HTTP keepalive logic commands an end. ServerConn
// also allows hijacking the underlying connection by calling Hijack
// to regain control over the connection. ServerConn supports pipe-lining,
// i.e. requests can be read out of sync (but in the same order) while the
// respective responses are sent.
//
// ServerConn is low-level and should not be needed by most applications.
// See Server.
type ServerConn struct {
	lk              sync.Mutex // read-write protects the following fields
	c               net.Conn
	r               *bufio.Reader
	re, we          error // read/write errors
	lastbody        io.ReadCloser
	nread, nwritten int
	pipereq         map[*Request]uint

	pipe textproto.Pipeline
}

// NewServerConn returns a new ServerConn reading and writing c.  If r is not
// nil, it is the buffer to use when reading c.
func NewServerConn(c net.Conn, r *bufio.Reader) *ServerConn {
	if r == nil {
		r = bufio.NewReader(c)
	}
	return &ServerConn{c: c, r: r, pipereq: make(map[*Request]uint)}
}

// Hijack detaches the ServerConn and returns the underlying connection as well
// as the read-side bufio which may have some left over data. Hijack may be
// called before Read has signaled the end of the keep-alive logic. The user
// should not call Hijack while Read or Write is in progress.
func (sc *ServerConn) Hijack() (c net.Conn, r *bufio.Reader) {
	sc.lk.Lock()
	defer sc.lk.Unlock()
	c = sc.c
	r = sc.r
	sc.c = nil
	sc.r = nil
	return
}

// Close calls Hijack and then also closes the underlying connection
func (sc *ServerConn) Close() error {
	c, _ := sc.Hijack()
	if c != nil {
		return c.Close()
	}
	return nil
}

// Read returns the next request on the wire. An ErrPersistEOF is returned if
// it is gracefully determined that there are no more requests (e.g. after the
// first request on an HTTP/1.0 connection, or after a Connection:close on a
// HTTP/1.1 connection).
func (sc *ServerConn) Read() (req *Request, err error) {

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
			sc.lk.Lock()
			sc.pipereq[req] = id
			sc.lk.Unlock()
		}
	}()

	sc.lk.Lock()
	if sc.we != nil { // no point receiving if write-side broken or closed
		defer sc.lk.Unlock()
		return nil, sc.we
	}
	if sc.re != nil {
		defer sc.lk.Unlock()
		return nil, sc.re
	}
	if sc.r == nil { // connection closed by user in the meantime
		defer sc.lk.Unlock()
		return nil, os.EBADF
	}
	r := sc.r
	lastbody := sc.lastbody
	sc.lastbody = nil
	sc.lk.Unlock()

	// Make sure body is fully consumed, even if user does not call body.Close
	if lastbody != nil {
		// body.Close is assumed to be idempotent and multiple calls to
		// it should return the error that its first invocation
		// returned.
		err = lastbody.Close()
		if err != nil {
			sc.lk.Lock()
			defer sc.lk.Unlock()
			sc.re = err
			return nil, err
		}
	}

	req, err = ReadRequest(r)
	sc.lk.Lock()
	defer sc.lk.Unlock()
	if err != nil {
		if err == io.ErrUnexpectedEOF {
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
	sc.lk.Lock()
	defer sc.lk.Unlock()
	return sc.nread - sc.nwritten
}

// Write writes resp in response to req. To close the connection gracefully, set the
// Response.Close field to true. Write should be considered operational until
// it returns an error, regardless of any errors returned on the Read side.
func (sc *ServerConn) Write(req *Request, resp *Response) error {

	// Retrieve the pipeline ID of this request/response pair
	sc.lk.Lock()
	id, ok := sc.pipereq[req]
	delete(sc.pipereq, req)
	if !ok {
		sc.lk.Unlock()
		return ErrPipeline
	}
	sc.lk.Unlock()

	// Ensure pipeline order
	sc.pipe.StartResponse(id)
	defer sc.pipe.EndResponse(id)

	sc.lk.Lock()
	if sc.we != nil {
		defer sc.lk.Unlock()
		return sc.we
	}
	if sc.c == nil { // connection closed by user in the meantime
		defer sc.lk.Unlock()
		return os.EBADF
	}
	c := sc.c
	if sc.nread <= sc.nwritten {
		defer sc.lk.Unlock()
		return errors.New("persist server pipe count")
	}
	if resp.Close {
		// After signaling a keep-alive close, any pipelined unread
		// requests will be lost. It is up to the user to drain them
		// before signaling.
		sc.re = ErrPersistEOF
	}
	sc.lk.Unlock()

	err := resp.Write(c)
	sc.lk.Lock()
	defer sc.lk.Unlock()
	if err != nil {
		sc.we = err
		return err
	}
	sc.nwritten++

	return nil
}

// A ClientConn sends request and receives headers over an underlying
// connection, while respecting the HTTP keepalive logic. ClientConn
// supports hijacking the connection calling Hijack to
// regain control of the underlying net.Conn and deal with it as desired.
//
// ClientConn is low-level and should not be needed by most applications.
// See Client.
type ClientConn struct {
	lk              sync.Mutex // read-write protects the following fields
	c               net.Conn
	r               *bufio.Reader
	re, we          error // read/write errors
	lastbody        io.ReadCloser
	nread, nwritten int
	pipereq         map[*Request]uint

	pipe     textproto.Pipeline
	writeReq func(*Request, io.Writer) error
}

// NewClientConn returns a new ClientConn reading and writing c.  If r is not
// nil, it is the buffer to use when reading c.
func NewClientConn(c net.Conn, r *bufio.Reader) *ClientConn {
	if r == nil {
		r = bufio.NewReader(c)
	}
	return &ClientConn{
		c:        c,
		r:        r,
		pipereq:  make(map[*Request]uint),
		writeReq: (*Request).Write,
	}
}

// NewProxyClientConn works like NewClientConn but writes Requests
// using Request's WriteProxy method.
func NewProxyClientConn(c net.Conn, r *bufio.Reader) *ClientConn {
	cc := NewClientConn(c, r)
	cc.writeReq = (*Request).WriteProxy
	return cc
}

// Hijack detaches the ClientConn and returns the underlying connection as well
// as the read-side bufio which may have some left over data. Hijack may be
// called before the user or Read have signaled the end of the keep-alive
// logic. The user should not call Hijack while Read or Write is in progress.
func (cc *ClientConn) Hijack() (c net.Conn, r *bufio.Reader) {
	cc.lk.Lock()
	defer cc.lk.Unlock()
	c = cc.c
	r = cc.r
	cc.c = nil
	cc.r = nil
	return
}

// Close calls Hijack and then also closes the underlying connection
func (cc *ClientConn) Close() error {
	c, _ := cc.Hijack()
	if c != nil {
		return c.Close()
	}
	return nil
}

// Write writes a request. An ErrPersistEOF error is returned if the connection
// has been closed in an HTTP keepalive sense. If req.Close equals true, the
// keepalive connection is logically closed after this request and the opposing
// server is informed. An ErrUnexpectedEOF indicates the remote closed the
// underlying TCP connection, which is usually considered as graceful close.
func (cc *ClientConn) Write(req *Request) (err error) {

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
			cc.lk.Lock()
			cc.pipereq[req] = id
			cc.lk.Unlock()
		}
	}()

	cc.lk.Lock()
	if cc.re != nil { // no point sending if read-side closed or broken
		defer cc.lk.Unlock()
		return cc.re
	}
	if cc.we != nil {
		defer cc.lk.Unlock()
		return cc.we
	}
	if cc.c == nil { // connection closed by user in the meantime
		defer cc.lk.Unlock()
		return os.EBADF
	}
	c := cc.c
	if req.Close {
		// We write the EOF to the write-side error, because there
		// still might be some pipelined reads
		cc.we = ErrPersistEOF
	}
	cc.lk.Unlock()

	err = cc.writeReq(req, c)
	cc.lk.Lock()
	defer cc.lk.Unlock()
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
	cc.lk.Lock()
	defer cc.lk.Unlock()
	return cc.nwritten - cc.nread
}

// Read reads the next response from the wire. A valid response might be
// returned together with an ErrPersistEOF, which means that the remote
// requested that this be the last request serviced. Read can be called
// concurrently with Write, but not with another Read.
func (cc *ClientConn) Read(req *Request) (*Response, error) {
	return cc.readUsing(req, ReadResponse)
}

// readUsing is the implementation of Read with a replaceable
// ReadResponse-like function, used by the Transport.
func (cc *ClientConn) readUsing(req *Request, readRes func(*bufio.Reader, *Request) (*Response, error)) (resp *Response, err error) {
	// Retrieve the pipeline ID of this request/response pair
	cc.lk.Lock()
	id, ok := cc.pipereq[req]
	delete(cc.pipereq, req)
	if !ok {
		cc.lk.Unlock()
		return nil, ErrPipeline
	}
	cc.lk.Unlock()

	// Ensure pipeline order
	cc.pipe.StartResponse(id)
	defer cc.pipe.EndResponse(id)

	cc.lk.Lock()
	if cc.re != nil {
		defer cc.lk.Unlock()
		return nil, cc.re
	}
	if cc.r == nil { // connection closed by user in the meantime
		defer cc.lk.Unlock()
		return nil, os.EBADF
	}
	r := cc.r
	lastbody := cc.lastbody
	cc.lastbody = nil
	cc.lk.Unlock()

	// Make sure body is fully consumed, even if user does not call body.Close
	if lastbody != nil {
		// body.Close is assumed to be idempotent and multiple calls to
		// it should return the error that its first invokation
		// returned.
		err = lastbody.Close()
		if err != nil {
			cc.lk.Lock()
			defer cc.lk.Unlock()
			cc.re = err
			return nil, err
		}
	}

	resp, err = readRes(r, req)
	cc.lk.Lock()
	defer cc.lk.Unlock()
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
func (cc *ClientConn) Do(req *Request) (resp *Response, err error) {
	err = cc.Write(req)
	if err != nil {
		return
	}
	return cc.Read(req)
}
