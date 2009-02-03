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
	"net";
	"os";
	"strconv";
)

var ErrWriteAfterFlush = os.NewError("Conn.Write called after Flush")

type Conn struct

// Interface implemented by servers using this library.
type Handler interface {
	ServeHTTP(*Conn, *Request);
}

// Active HTTP connection (server side).
type Conn struct {
	Fd io.ReadWriteClose;
	RemoteAddr string;
	Req *Request;
	Br *bufio.BufRead;

	br *bufio.BufRead;
	bw *bufio.BufWrite;
	close bool;
	chunking bool;
	flushed bool;
	header map[string] string;
	wroteHeader bool;
	handler Handler;
}

// HTTP response codes.
// TODO(rsc): Maybe move these to their own file, so that
// clients can use them too.

const (
	StatusContinue = 100;
	StatusSwitchingProtocols = 101;

	StatusOK = 200;
	StatusCreated = 201;
	StatusAccepted = 202;
	StatusNonAuthoritativeInfo = 203;
	StatusNoContent = 204;
	StatusResetContent = 205;
	StatusPartialContent = 206;

	StatusMultipleChoices = 300;
	StatusMovedPermanently = 301;
	StatusFound = 302;
	StatusSeeOther = 303;
	StatusNotModified = 304;
	StatusUseProxy = 305;
	StatusTemporaryRedirect = 307;

	StatusBadRequest = 400;
	StatusUnauthorized = 401;
	StatusPaymentRequired = 402;
	StatusForbidden = 403;
	StatusNotFound = 404;
	StatusMethodNotAllowed = 405;
	StatusNotAcceptable = 406;
	StatusProxyAuthRequired = 407;
	StatusRequestTimeout = 408;
	StatusConflict = 409;
	StatusGone = 410;
	StatusLengthRequired = 411;
	StatusPreconditionFailed = 412;
	StatusRequestEntityTooLarge = 413;
	StatusRequestURITooLong = 414;
	StatusUnsupportedMediaType = 415;
	StatusRequestedRangeNotSatisfiable = 416;
	StatusExpectationFailed = 417;

	StatusInternalServerError = 500;
	StatusNotImplemented = 501;
	StatusBadGateway = 502;
	StatusServiceUnavailable = 503;
	StatusGatewayTimeout = 504;
	StatusHTTPVersionNotSupported = 505;
)

var statusText = map[int]string {
	StatusContinue:			"Continue",
	StatusSwitchingProtocols:	"Switching Protocols",

	StatusOK:			"OK",
	StatusCreated:			"Created",
	StatusAccepted:			"Accepted",
	StatusNonAuthoritativeInfo:	"Non-Authoritative Information",
	StatusNoContent:		"No Content",
	StatusResetContent:		"Reset Content",
	StatusPartialContent:		"Partial Content",

	StatusMultipleChoices:		"Multiple Choices",
	StatusMovedPermanently:		"Moved Permanently",
	StatusFound:			"Found",
	StatusSeeOther:			"See Other",
	StatusNotModified:		"Not Modified",
	StatusUseProxy:			"Use Proxy",
	StatusTemporaryRedirect:	"Temporary Redirect",

	StatusBadRequest:		"Bad Request",
	StatusUnauthorized:		"Unauthorized",
	StatusPaymentRequired:		"Payment Required",
	StatusForbidden:		"Forbidden",
	StatusNotFound:			"Not Found",
	StatusMethodNotAllowed:		"Method Not Allowed",
	StatusNotAcceptable:		"Not Acceptable",
	StatusProxyAuthRequired:	"Proxy Authentication Required",
	StatusRequestTimeout:		"Request Timeout",
	StatusConflict:			"Conflict",
	StatusGone:			"Gone",
	StatusLengthRequired:		"Length Required",
	StatusPreconditionFailed:	"Precondition Failed",
	StatusRequestEntityTooLarge:	"Request Entity Too Large",
	StatusRequestURITooLong:	"Request URI Too Long",
	StatusUnsupportedMediaType:	"Unsupported Media Type",
	StatusRequestedRangeNotSatisfiable:	"Requested Range Not Satisfiable",
	StatusExpectationFailed:	"Expectation Failed",

	StatusInternalServerError:	"Internal Server Error",
	StatusNotImplemented:		"Not Implemented",
	StatusBadGateway:		"Bad Gateway",
	StatusServiceUnavailable:	"Service Unavailable",
	StatusGatewayTimeout:		"Gateway Timeout",
	StatusHTTPVersionNotSupported:	"HTTP Version Not Supported",
}

// Create new connection from rwc.
func newConn(rwc io.ReadWriteClose, raddr string, handler Handler) (c *Conn, err *os.Error) {
	c = new(Conn);
	c.Fd = rwc;
	c.RemoteAddr = raddr;
	c.handler = handler;
	if c.br, err = bufio.NewBufRead(rwc.(io.Read)); err != nil {
		return nil, err
	}
c.Br = c.br;
	if c.bw, err = bufio.NewBufWrite(rwc); err != nil {
		return nil, err
	}
	return c, nil
}

func (c *Conn) SetHeader(hdr, val string)

// Read next request from connection.
func (c *Conn) readRequest() (req *Request, err *os.Error) {
	if req, err = ReadRequest(c.br); err != nil {
		return nil, err
	}

	// Reset per-request connection state.
	c.header = make(map[string] string);
	c.wroteHeader = false;
	c.flushed = false;
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
		c.close = true;
		c.chunking = false;
	}

	return req, nil
}

func (c *Conn) SetHeader(hdr, val string) {
	c.header[CanonicalHeaderKey(hdr)] = val;
}

// Write header.
func (c *Conn) WriteHeader(code int) {
	if c.wroteHeader {
		// TODO(rsc): log
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
	io.WriteString(c.bw, proto + " " + codestring + " " + text + "\r\n");
	for k,v := range c.header {
		io.WriteString(c.bw, k + ": " + v + "\r\n");
	}
	io.WriteString(c.bw, "\r\n");
}

// TODO(rsc): BUG in 6g: must return "nn int" not "n int"
// so that the implicit struct assignment in
// return c.bw.Write(data) works.  oops
func (c *Conn) Write(data []byte) (nn int, err *os.Error) {
	if c.flushed {
		return 0, ErrWriteAfterFlush
	}
	if !c.wroteHeader {
		c.WriteHeader(StatusOK);
	}
	if len(data) == 0 {
		return 0, nil
	}

	// TODO(rsc): if chunking happened after the buffering,
	// then there would be fewer chunk headers
	if c.chunking {
		fmt.Fprintf(c.bw, "%x\r\n", len(data));	// TODO(rsc): use strconv not fmt
	}
	return c.bw.Write(data);
}

func (c *Conn) Flush() {
	if c.flushed {
		return
	}
	if !c.wroteHeader {
		c.WriteHeader(StatusOK);
	}
	if c.chunking {
		io.WriteString(c.bw, "0\r\n");
		// trailer key/value pairs, followed by blank line
		io.WriteString(c.bw, "\r\n");
	}
	c.bw.Flush();
	c.flushed = true;
}

// Close the connection.
func (c *Conn) Close() {
	if c.bw != nil {
		c.bw.Flush();
		c.bw = nil;
	}
	if c.Fd != nil {
		c.Fd.Close();
		c.Fd = nil;
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
		if c.Fd == nil {
			// Handler took over the connection.
			return;
		}
		if !c.flushed {
			c.Flush();
		}
		if c.close {
			break;
		}
	}
	c.Close();
}

// Adapter: can use RequestFunction(f) as Handler
type handlerFunc struct {
	f func(*Conn, *Request)
}
func (h handlerFunc) ServeHTTP(c *Conn, req *Request) {
	h.f(c, req)
}
func HandlerFunc(f func(*Conn, *Request)) Handler {
	return handlerFunc{f}
}

/* simpler version of above, not accepted by 6g:

type HandlerFunc func(*Conn, *Request)
func (f HandlerFunc) ServeHTTP(c *Conn, req *Request) {
	f(c, req);
}
*/

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

