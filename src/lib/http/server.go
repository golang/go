// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP server.  See RFC 2616.

// TODO(rsc):
//	logging
//	cgi support
//	post support

package http

import (
	"bufio";
	"fmt";
	"http";
	"io";
	"log";
	"net";
	"os";
	"strconv";
)

var ErrWriteAfterFlush = os.NewError("Conn.Write called after Flush")
var ErrHijacked = os.NewError("Conn has been hijacked")

type Conn struct

// Interface implemented by servers using this library.
type Handler interface {
	ServeHTTP(*Conn, *Request);
}

// Active HTTP connection (server side).
type Conn struct {
	RemoteAddr string;	// network address of remote side
	Req *Request;	// current HTTP request

	fd io.ReadWriteClose;	// i/o connection
	buf *bufio.BufReadWrite;	// buffered fd
	handler Handler;	// request handler
	hijacked bool;	// connection has been hijacked by handler

	// state for the current reply
	closeAfterReply bool;	// close connection after this reply
	chunking bool;	// using chunked transfer encoding for reply body
	wroteHeader bool;	// reply header has been written
	header map[string] string;	// reply header parameters
}

// Create new connection from rwc.
func newConn(rwc io.ReadWriteClose, raddr string, handler Handler) (c *Conn, err *os.Error) {
	c = new(Conn);
	c.RemoteAddr = raddr;
	c.handler = handler;
	c.fd = rwc;
	br := bufio.NewBufRead(rwc);
	bw := bufio.NewBufWrite(rwc);
	c.buf = bufio.NewBufReadWrite(br, bw);
	return c, nil
}

func (c *Conn) SetHeader(hdr, val string)

// Read next request from connection.
func (c *Conn) readRequest() (req *Request, err *os.Error) {
	if c.hijacked {
		return nil, ErrHijacked
	}
	if req, err = ReadRequest(c.buf.BufRead); err != nil {
		return nil, err
	}

	// Reset per-request connection state.
	c.header = make(map[string] string);
	c.wroteHeader = false;
	c.Req = req;

	// Default output is HTML encoded in UTF-8.
	c.SetHeader("Content-Type", "text/html; charset=utf-8");

	if req.ProtoAtLeast(1, 1) {
		// HTTP/1.1 or greater: use chunked transfer encoding
		// to avoid closing the connection at EOF.
		c.chunking = true;
		c.SetHeader("Transfer-Encoding", "chunked");
	} else {
		// HTTP version < 1.1: cannot do chunked transfer
		// encoding, so signal EOF by closing connection.
		// Could avoid closing the connection if there is
		// a Content-Length: header in the response,
		// but everyone who expects persistent connections
		// does HTTP/1.1 now.
		c.closeAfterReply = true;
		c.chunking = false;
	}

	return req, nil
}

func (c *Conn) SetHeader(hdr, val string) {
	c.header[CanonicalHeaderKey(hdr)] = val;
}

// Write header.
func (c *Conn) WriteHeader(code int) {
	if c.hijacked {
		log.Stderr("http: Conn.WriteHeader on hijacked connection");
		return
	}
	if c.wroteHeader {
		log.Stderr("http: multiple Conn.WriteHeader calls");
		return
	}
	c.wroteHeader = true;
	if !c.Req.ProtoAtLeast(1, 0) {
		return
	}
	proto := "HTTP/1.0";
	if c.Req.ProtoAtLeast(1, 1) {
		proto = "HTTP/1.1";
	}
	codestring := strconv.Itoa(code);
	text, ok := statusText[code];
	if !ok {
		text = "status code " + codestring;
	}
	io.WriteString(c.buf, proto + " " + codestring + " " + text + "\r\n");
	for k,v := range c.header {
		io.WriteString(c.buf, k + ": " + v + "\r\n");
	}
	io.WriteString(c.buf, "\r\n");
}

// TODO(rsc): BUG in 6g: must return "nn int" not "n int"
// so that the implicit struct assignment in
// return c.buf.Write(data) works.  oops
func (c *Conn) Write(data []byte) (nn int, err *os.Error) {
	if c.hijacked {
		log.Stderr("http: Conn.Write on hijacked connection");
		return 0, ErrHijacked
	}
	if !c.wroteHeader {
		c.WriteHeader(StatusOK);
	}
	if len(data) == 0 {
		return 0, nil
	}

	// TODO(rsc): if chunking happened after the buffering,
	// then there would be fewer chunk headers.
	// On the other hand, it would make hijacking more difficult.
	if c.chunking {
		fmt.Fprintf(c.buf, "%x\r\n", len(data));	// TODO(rsc): use strconv not fmt
	}
	return c.buf.Write(data);
}

func (c *Conn) flush() {
	if !c.wroteHeader {
		c.WriteHeader(StatusOK);
	}
	if c.chunking {
		io.WriteString(c.buf, "0\r\n");
		// trailer key/value pairs, followed by blank line
		io.WriteString(c.buf, "\r\n");
	}
	c.buf.Flush();
}

// Close the connection.
func (c *Conn) close() {
	if c.buf != nil {
		c.buf.Flush();
		c.buf = nil;
	}
	if c.fd != nil {
		c.fd.Close();
		c.fd = nil;
	}
}

// Serve a new connection.
func (c *Conn) serve() {
	for {
		req, err := c.readRequest();
		if err != nil {
			break
		}
		// HTTP cannot have multiple simultaneous active requests.
		// Until the server replies to this request, it can't read another,
		// so we might as well run the handler in this thread.
		c.handler.ServeHTTP(c, req);
		if c.hijacked {
			return;
		}
		c.flush();
		if c.closeAfterReply {
			break;
		}
	}
	c.close();
}

// Allow client to take over the connection.
// After a handler calls c.Hijack(), the HTTP server library
// will never touch the connection again.
// It is the caller's responsibility to manage and close
// the connection.
func (c *Conn) Hijack() (fd io.ReadWriteClose, buf *bufio.BufReadWrite, err *os.Error) {
	if c.hijacked {
		return nil, nil, ErrHijacked;
	}
	c.hijacked = true;
	fd = c.fd;
	buf = c.buf;
	c.fd = nil;
	c.buf = nil;
	return;
}

// Adapter: can use HandlerFunc(f) as Handler
type HandlerFunc func(*Conn, *Request)
func (f HandlerFunc) ServeHTTP(c *Conn, req *Request) {
	f(c, req);
}

// Helper handlers

// 404 not found
func notFound(c *Conn, req *Request) {
	c.SetHeader("Content-Type", "text/plain; charset=utf-8");
	c.WriteHeader(StatusNotFound);
	io.WriteString(c, "404 page not found\n");
}

var NotFoundHandler = HandlerFunc(notFound)

// Redirect to a fixed URL
type redirectHandler struct {
	to string;
}
func (h *redirectHandler) ServeHTTP(c *Conn, req *Request) {
	c.SetHeader("Location", h.to);
	c.WriteHeader(StatusMovedPermanently);
}

func RedirectHandler(to string) Handler {
	return &redirectHandler{to};
}

// Path-based HTTP request multiplexer.
// Patterns name fixed paths, like "/favicon.ico",
// or subtrees, like "/images/".
// For now, patterns must begin with /.
// Eventually, might want to allow host name
// at beginning of pattern, so that you could register
//	/codesearch
//	codesearch.google.com/
// but not take over /.

type ServeMux struct {
	m map[string] Handler
}

func NewServeMux() *ServeMux {
	return &ServeMux{make(map[string] Handler)};
}

var DefaultServeMux = NewServeMux();

// Does path match pattern?
func pathMatch(pattern, path string) bool {
	if len(pattern) == 0 {
		// should not happen
		return false
	}
	n := len(pattern);
	if pattern[n-1] != '/' {
		return pattern == path
	}
	return len(path) >= n && path[0:n] == pattern;
}

func (mux *ServeMux) ServeHTTP(c *Conn, req *Request) {
	// Most-specific (longest) pattern wins.
	var h Handler;
	var n = 0;
	for k, v := range mux.m {
		if !pathMatch(k, req.Url.Path) {
			continue;
		}
		if h == nil || len(k) > n {
			n = len(k);
			h = v;
		}
	}
	if h == nil {
		h = NotFoundHandler;
	}
	h.ServeHTTP(c, req);
}

func (mux *ServeMux) Handle(pattern string, handler Handler) {
	if pattern == "" || pattern[0] != '/' {
		panicln("http: invalid pattern", pattern);
	}

	mux.m[pattern] = handler;

	// Helpful behavior:
	// If pattern is /tree/, insert redirect for /tree.
	n := len(pattern);
	if n > 0 && pattern[n-1] == '/' {
		mux.m[pattern[0:n-1]] = RedirectHandler(pattern);
	}
}

func Handle(pattern string, h Handler) {
	DefaultServeMux.Handle(pattern, h);
}


// Web server: listening on l, call handler.ServeHTTP for each request.
func Serve(l net.Listener, handler Handler) *os.Error {
	if handler == nil {
		handler = DefaultServeMux;
	}
	for {
		rw, raddr, e := l.Accept();
		if e != nil {
			return e
		}
		c, err := newConn(rw, raddr, handler);
		if err != nil {
			continue;
		}
		go c.serve();
	}
	panic("not reached")
}

// Web server: listen on address, call f for each request.
func ListenAndServe(addr string, handler Handler) *os.Error {
	l, e := net.Listen("tcp", addr);
	if e != nil {
		return e
	}
	e = Serve(l, handler);
	l.Close();
	return e
}

