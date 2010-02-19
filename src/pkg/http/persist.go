// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http

import (
	"bufio"
	"container/list"
	"io"
	"net"
	"os"
	"sync"
)

var ErrPersistEOF = &ProtocolError{"persistent connection closed"}

// A ServerConn reads requests and sends responses over an underlying
// connection, until the HTTP keepalive logic commands an end. ServerConn
// does not close the underlying connection. Instead, the user calls Close
// and regains control over the connection. ServerConn supports pipe-lining,
// i.e. requests can be read out of sync (but in the same order) while the
// respective responses are sent.
type ServerConn struct {
	c               net.Conn
	r               *bufio.Reader
	clsd            bool     // indicates a graceful close
	re, we          os.Error // read/write errors
	lastBody        io.ReadCloser
	nread, nwritten int
	lk              sync.Mutex // protected read/write to re,we
}

// NewServerConn returns a new ServerConn reading and writing c.  If r is not
// nil, it is the buffer to use when reading c.
func NewServerConn(c net.Conn, r *bufio.Reader) *ServerConn {
	if r == nil {
		r = bufio.NewReader(c)
	}
	return &ServerConn{c: c, r: r}
}

// Close detaches the ServerConn and returns the underlying connection as well
// as the read-side bufio which may have some left over data. Close may be
// called before Read has signaled the end of the keep-alive logic. The user
// should not call Close while Read or Write is in progress.
func (sc *ServerConn) Close() (c net.Conn, r *bufio.Reader) {
	sc.lk.Lock()
	defer sc.lk.Unlock()
	c = sc.c
	r = sc.r
	sc.c = nil
	sc.r = nil
	return
}

// Read returns the next request on the wire. An ErrPersistEOF is returned if
// it is gracefully determined that there are no more requests (e.g. after the
// first request on an HTTP/1.0 connection, or after a Connection:close on a
// HTTP/1.1 connection). Read can be called concurrently with Write, but not
// with another Read.
func (sc *ServerConn) Read() (req *Request, err os.Error) {

	sc.lk.Lock()
	if sc.we != nil { // no point receiving if write-side broken or closed
		defer sc.lk.Unlock()
		return nil, sc.we
	}
	if sc.re != nil {
		defer sc.lk.Unlock()
		return nil, sc.re
	}
	sc.lk.Unlock()

	// Make sure body is fully consumed, even if user does not call body.Close
	if sc.lastBody != nil {
		// body.Close is assumed to be idempotent and multiple calls to
		// it should return the error that its first invokation
		// returned.
		err = sc.lastBody.Close()
		sc.lastBody = nil
		if err != nil {
			sc.lk.Lock()
			defer sc.lk.Unlock()
			sc.re = err
			return nil, err
		}
	}

	req, err = ReadRequest(sc.r)
	if err != nil {
		sc.lk.Lock()
		defer sc.lk.Unlock()
		if err == io.ErrUnexpectedEOF {
			// A close from the opposing client is treated as a
			// graceful close, even if there was some unparse-able
			// data before the close.
			sc.re = ErrPersistEOF
			return nil, sc.re
		} else {
			sc.re = err
			return
		}
	}
	sc.lastBody = req.Body
	sc.nread++
	if req.Close {
		sc.lk.Lock()
		defer sc.lk.Unlock()
		sc.re = ErrPersistEOF
		return req, sc.re
	}
	return
}

// Pending returns the number of unanswered requests
// that have been received on the connection.
func (sc *ServerConn) Pending() int {
	sc.lk.Lock()
	defer sc.lk.Unlock()
	return sc.nread - sc.nwritten
}

// Write writes a repsonse. To close the connection gracefully, set the
// Response.Close field to true. Write should be considered operational until
// it returns an error, regardless of any errors returned on the Read side.
// Write can be called concurrently with Read, but not with another Write.
func (sc *ServerConn) Write(resp *Response) os.Error {

	sc.lk.Lock()
	if sc.we != nil {
		defer sc.lk.Unlock()
		return sc.we
	}
	sc.lk.Unlock()
	if sc.nread <= sc.nwritten {
		return os.NewError("persist server pipe count")
	}

	if resp.Close {
		// After signaling a keep-alive close, any pipelined unread
		// requests will be lost. It is up to the user to drain them
		// before signaling.
		sc.lk.Lock()
		sc.re = ErrPersistEOF
		sc.lk.Unlock()
	}

	err := resp.Write(sc.c)
	if err != nil {
		sc.lk.Lock()
		defer sc.lk.Unlock()
		sc.we = err
		return err
	}
	sc.nwritten++

	return nil
}

// A ClientConn sends request and receives headers over an underlying
// connection, while respecting the HTTP keepalive logic. ClientConn is not
// responsible for closing the underlying connection. One must call Close to
// regain control of that connection and deal with it as desired.
type ClientConn struct {
	c               net.Conn
	r               *bufio.Reader
	re, we          os.Error // read/write errors
	lastBody        io.ReadCloser
	nread, nwritten int
	reqm            list.List  // request methods in order of execution
	lk              sync.Mutex // protects read/write to reqm,re,we
}

// NewClientConn returns a new ClientConn reading and writing c.  If r is not
// nil, it is the buffer to use when reading c.
func NewClientConn(c net.Conn, r *bufio.Reader) *ClientConn {
	if r == nil {
		r = bufio.NewReader(c)
	}
	return &ClientConn{c: c, r: r}
}

// Close detaches the ClientConn and returns the underlying connection as well
// as the read-side bufio which may have some left over data. Close may be
// called before the user or Read have signaled the end of the keep-alive
// logic. The user should not call Close while Read or Write is in progress.
func (cc *ClientConn) Close() (c net.Conn, r *bufio.Reader) {
	cc.lk.Lock()
	c = cc.c
	r = cc.r
	cc.c = nil
	cc.r = nil
	cc.reqm.Init()
	cc.lk.Unlock()
	return
}

// Write writes a request. An ErrPersistEOF error is returned if the connection
// has been closed in an HTTP keepalive sense. If req.Close equals true, the
// keepalive connection is logically closed after this request and the opposing
// server is informed. An ErrUnexpectedEOF indicates the remote closed the
// underlying TCP connection, which is usually considered as graceful close.
// Write can be called concurrently with Read, but not with another Write.
func (cc *ClientConn) Write(req *Request) os.Error {

	cc.lk.Lock()
	if cc.re != nil { // no point sending if read-side closed or broken
		defer cc.lk.Unlock()
		return cc.re
	}
	if cc.we != nil {
		defer cc.lk.Unlock()
		return cc.we
	}
	cc.lk.Unlock()

	if req.Close {
		// We write the EOF to the write-side error, because there
		// still might be some pipelined reads
		cc.lk.Lock()
		cc.we = ErrPersistEOF
		cc.lk.Unlock()
	}

	err := req.Write(cc.c)
	if err != nil {
		cc.lk.Lock()
		defer cc.lk.Unlock()
		cc.we = err
		return err
	}
	cc.nwritten++
	cc.lk.Lock()
	cc.reqm.PushBack(req.Method)
	cc.lk.Unlock()

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
func (cc *ClientConn) Read() (resp *Response, err os.Error) {

	cc.lk.Lock()
	if cc.re != nil {
		defer cc.lk.Unlock()
		return nil, cc.re
	}
	cc.lk.Unlock()

	if cc.nread >= cc.nwritten {
		return nil, os.NewError("persist client pipe count")
	}

	// Make sure body is fully consumed, even if user does not call body.Close
	if cc.lastBody != nil {
		// body.Close is assumed to be idempotent and multiple calls to
		// it should return the error that its first invokation
		// returned.
		err = cc.lastBody.Close()
		cc.lastBody = nil
		if err != nil {
			cc.lk.Lock()
			defer cc.lk.Unlock()
			cc.re = err
			return nil, err
		}
	}

	cc.lk.Lock()
	m := cc.reqm.Front()
	cc.reqm.Remove(m)
	cc.lk.Unlock()
	resp, err = ReadResponse(cc.r, m.Value.(string))
	if err != nil {
		cc.lk.Lock()
		defer cc.lk.Unlock()
		cc.re = err
		return
	}
	cc.lastBody = resp.Body

	cc.nread++

	if resp.Close {
		cc.lk.Lock()
		defer cc.lk.Unlock()
		cc.re = ErrPersistEOF // don't send any more requests
		return resp, cc.re
	}
	return
}
