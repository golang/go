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
	"bufio"
	"crypto/rand"
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"net"
	"os"
	"path"
	"strconv"
	"strings"
	"time"
)

// Errors introduced by the HTTP server.
var (
	ErrWriteAfterFlush = os.NewError("Conn.Write called after Flush")
	ErrBodyNotAllowed  = os.NewError("http: response status code does not allow body")
	ErrHijacked        = os.NewError("Conn has been hijacked")
)

// Objects implementing the Handler interface can be
// registered to serve a particular path or subtree
// in the HTTP server.
//
// ServeHTTP should write reply headers and data to the ResponseWriter
// and then return.  Returning signals that the request is finished
// and that the HTTP server can move on to the next request on
// the connection.
type Handler interface {
	ServeHTTP(ResponseWriter, *Request)
}

// A ResponseWriter interface is used by an HTTP handler to
// construct an HTTP response.
type ResponseWriter interface {
	// RemoteAddr returns the address of the client that sent the current request
	RemoteAddr() string

	// UsingTLS returns true if the client is connected using TLS
	UsingTLS() bool

	// SetHeader sets a header line in the eventual response.
	// For example, SetHeader("Content-Type", "text/html; charset=utf-8")
	// will result in the header line
	//
	//	Content-Type: text/html; charset=utf-8
	//
	// being sent.  UTF-8 encoded HTML is the default setting for
	// Content-Type in this library, so users need not make that
	// particular call.  Calls to SetHeader after WriteHeader (or Write)
	// are ignored.
	SetHeader(string, string)

	// Write writes the data to the connection as part of an HTTP reply.
	// If WriteHeader has not yet been called, Write calls WriteHeader(http.StatusOK)
	// before writing the data.
	Write([]byte) (int, os.Error)

	// WriteHeader sends an HTTP response header with status code.
	// If WriteHeader is not called explicitly, the first call to Write
	// will trigger an implicit WriteHeader(http.StatusOK).
	// Thus explicit calls to WriteHeader are mainly used to
	// send error codes.
	WriteHeader(int)

	// Flush sends any buffered data to the client.
	Flush()

	// Hijack lets the caller take over the connection.
	// After a call to Hijack(), the HTTP server library
	// will not do anything else with the connection.
	// It becomes the caller's responsibility to manage
	// and close the connection.
	Hijack() (io.ReadWriteCloser, *bufio.ReadWriter, os.Error)
}

// A conn represents the server side of an HTTP connection.
type conn struct {
	remoteAddr string             // network address of remote side
	handler    Handler            // request handler
	rwc        io.ReadWriteCloser // i/o connection
	buf        *bufio.ReadWriter  // buffered rwc
	hijacked   bool               // connection has been hijacked by handler
	usingTLS   bool               // a flag indicating connection over TLS
}

// A response represents the server side of an HTTP response.
type response struct {
	conn          *conn
	req           *Request          // request for this response
	chunking      bool              // using chunked transfer encoding for reply body
	wroteHeader   bool              // reply header has been written
	wroteContinue bool              // 100 Continue response was written
	header        map[string]string // reply header parameters
	written       int64             // number of bytes written in body
	status        int               // status code passed to WriteHeader

	// close connection after this reply.  set on request and
	// updated after response from handler if there's a
	// "Connection: keep-alive" response header and a
	// Content-Length.
	closeAfterReply bool
}

// Create new connection from rwc.
func newConn(rwc net.Conn, handler Handler) (c *conn, err os.Error) {
	c = new(conn)
	c.remoteAddr = rwc.RemoteAddr().String()
	c.handler = handler
	c.rwc = rwc
	_, c.usingTLS = rwc.(*tls.Conn)
	br := bufio.NewReader(rwc)
	bw := bufio.NewWriter(rwc)
	c.buf = bufio.NewReadWriter(br, bw)
	return c, nil
}

// wrapper around io.ReaderCloser which on first read, sends an
// HTTP/1.1 100 Continue header
type expectContinueReader struct {
	resp       *response
	readCloser io.ReadCloser
}

func (ecr *expectContinueReader) Read(p []byte) (n int, err os.Error) {
	if !ecr.resp.wroteContinue && !ecr.resp.conn.hijacked {
		ecr.resp.wroteContinue = true
		io.WriteString(ecr.resp.conn.buf, "HTTP/1.1 100 Continue\r\n\r\n")
		ecr.resp.conn.buf.Flush()
	}
	return ecr.readCloser.Read(p)
}

func (ecr *expectContinueReader) Close() os.Error {
	return ecr.readCloser.Close()
}

// TimeFormat is the time format to use with
// time.Parse and time.Time.Format when parsing
// or generating times in HTTP headers.
// It is like time.RFC1123 but hard codes GMT as the time zone.
const TimeFormat = "Mon, 02 Jan 2006 15:04:05 GMT"

// Read next request from connection.
func (c *conn) readRequest() (w *response, err os.Error) {
	if c.hijacked {
		return nil, ErrHijacked
	}
	var req *Request
	if req, err = ReadRequest(c.buf.Reader); err != nil {
		return nil, err
	}

	w = new(response)
	w.conn = c
	w.req = req
	w.header = make(map[string]string)

	// Expect 100 Continue support
	if req.expectsContinue() && req.ProtoAtLeast(1, 1) {
		// Wrap the Body reader with one that replies on the connection
		req.Body = &expectContinueReader{readCloser: req.Body, resp: w}
	}

	// Default output is HTML encoded in UTF-8.
	w.SetHeader("Content-Type", "text/html; charset=utf-8")
	w.SetHeader("Date", time.UTC().Format(TimeFormat))

	if req.Method == "HEAD" {
		// do nothing
	} else if req.ProtoAtLeast(1, 1) {
		// HTTP/1.1 or greater: use chunked transfer encoding
		// to avoid closing the connection at EOF.
		w.chunking = true
		w.SetHeader("Transfer-Encoding", "chunked")
	} else {
		// HTTP version < 1.1: cannot do chunked transfer
		// encoding, so signal EOF by closing connection.
		// Will be overridden if the HTTP handler ends up
		// writing a Content-Length and the client requested
		// "Connection: keep-alive"
		w.closeAfterReply = true
	}

	return w, nil
}

// UsingTLS implements the ResponseWriter.UsingTLS
func (w *response) UsingTLS() bool {
	return w.conn.usingTLS
}

// RemoteAddr implements the ResponseWriter.RemoteAddr method
func (w *response) RemoteAddr() string { return w.conn.remoteAddr }

// SetHeader implements the ResponseWriter.SetHeader method
func (w *response) SetHeader(hdr, val string) { w.header[CanonicalHeaderKey(hdr)] = val }

// WriteHeader implements the ResponseWriter.WriteHeader method
func (w *response) WriteHeader(code int) {
	if w.conn.hijacked {
		log.Print("http: response.WriteHeader on hijacked connection")
		return
	}
	if w.wroteHeader {
		log.Print("http: multiple response.WriteHeader calls")
		return
	}
	w.wroteHeader = true
	w.status = code
	if code == StatusNotModified {
		// Must not have body.
		w.header["Content-Type"] = "", false
		w.header["Transfer-Encoding"] = "", false
		w.chunking = false
	}
	// Cannot use Content-Length with non-identity Transfer-Encoding.
	if w.chunking {
		w.header["Content-Length"] = "", false
	}
	if !w.req.ProtoAtLeast(1, 0) {
		return
	}
	proto := "HTTP/1.0"
	if w.req.ProtoAtLeast(1, 1) {
		proto = "HTTP/1.1"
	}
	codestring := strconv.Itoa(code)
	text, ok := statusText[code]
	if !ok {
		text = "status code " + codestring
	}
	io.WriteString(w.conn.buf, proto+" "+codestring+" "+text+"\r\n")
	for k, v := range w.header {
		io.WriteString(w.conn.buf, k+": "+v+"\r\n")
	}
	io.WriteString(w.conn.buf, "\r\n")
}

// Write implements the ResponseWriter.Write method
func (w *response) Write(data []byte) (n int, err os.Error) {
	if w.conn.hijacked {
		log.Print("http: response.Write on hijacked connection")
		return 0, ErrHijacked
	}
	if !w.wroteHeader {
		if w.req.wantsHttp10KeepAlive() {
			_, hasLength := w.header["Content-Length"]
			if hasLength {
				_, connectionHeaderSet := w.header["Connection"]
				if !connectionHeaderSet {
					w.header["Connection"] = "keep-alive"
				}
			}
		}
		w.WriteHeader(StatusOK)
	}
	if len(data) == 0 {
		return 0, nil
	}

	if w.status == StatusNotModified || w.req.Method == "HEAD" {
		// Must not have body.
		return 0, ErrBodyNotAllowed
	}

	w.written += int64(len(data)) // ignoring errors, for errorKludge

	// TODO(rsc): if chunking happened after the buffering,
	// then there would be fewer chunk headers.
	// On the other hand, it would make hijacking more difficult.
	if w.chunking {
		fmt.Fprintf(w.conn.buf, "%x\r\n", len(data)) // TODO(rsc): use strconv not fmt
	}
	n, err = w.conn.buf.Write(data)
	if err == nil && w.chunking {
		if n != len(data) {
			err = io.ErrShortWrite
		}
		if err == nil {
			io.WriteString(w.conn.buf, "\r\n")
		}
	}

	return n, err
}

// If this is an error reply (4xx or 5xx)
// and the handler wrote some data explaining the error,
// some browsers (i.e., Chrome, Internet Explorer)
// will show their own error instead unless the error is
// long enough.  The minimum lengths used in those
// browsers are in the 256-512 range.
// Pad to 1024 bytes.
func errorKludge(w *response) {
	const min = 1024

	// Is this an error?
	if kind := w.status / 100; kind != 4 && kind != 5 {
		return
	}

	// Did the handler supply any info?  Enough?
	if w.written == 0 || w.written >= min {
		return
	}

	// Is it a broken browser?
	var msg string
	switch agent := w.req.UserAgent; {
	case strings.Contains(agent, "MSIE"):
		msg = "Internet Explorer"
	case strings.Contains(agent, "Chrome/"):
		msg = "Chrome"
	default:
		return
	}
	msg += " would ignore this error page if this text weren't here.\n"

	// Is it text?  ("Content-Type" is always in the map)
	baseType := strings.Split(w.header["Content-Type"], ";", 2)[0]
	switch baseType {
	case "text/html":
		io.WriteString(w, "<!-- ")
		for w.written < min {
			io.WriteString(w, msg)
		}
		io.WriteString(w, " -->")
	case "text/plain":
		io.WriteString(w, "\n")
		for w.written < min {
			io.WriteString(w, msg)
		}
	}
}

func (w *response) finishRequest() {
	// If this was an HTTP/1.0 request with keep-alive and we sent a Content-Length
	// back, we can make this a keep-alive response ...
	if w.req.wantsHttp10KeepAlive() {
		_, sentLength := w.header["Content-Length"]
		if sentLength && w.header["Connection"] == "keep-alive" {
			w.closeAfterReply = false
		}
	}
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	errorKludge(w)
	if w.chunking {
		io.WriteString(w.conn.buf, "0\r\n")
		// trailer key/value pairs, followed by blank line
		io.WriteString(w.conn.buf, "\r\n")
	}
	w.conn.buf.Flush()
	w.req.Body.Close()
}

// Flush implements the ResponseWriter.Flush method.
func (w *response) Flush() {
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	w.conn.buf.Flush()
}

// Close the connection.
func (c *conn) close() {
	if c.buf != nil {
		c.buf.Flush()
		c.buf = nil
	}
	if c.rwc != nil {
		c.rwc.Close()
		c.rwc = nil
	}
}

// Serve a new connection.
func (c *conn) serve() {
	for {
		w, err := c.readRequest()
		if err != nil {
			break
		}
		// HTTP cannot have multiple simultaneous active requests.[*]
		// Until the server replies to this request, it can't read another,
		// so we might as well run the handler in this goroutine.
		// [*] Not strictly true: HTTP pipelining.  We could let them all process
		// in parallel even if their responses need to be serialized.
		c.handler.ServeHTTP(w, w.req)
		if c.hijacked {
			return
		}
		w.finishRequest()
		if w.closeAfterReply {
			break
		}
	}
	c.close()
}

// Hijack impements the ResponseWriter.Hijack method.
func (w *response) Hijack() (rwc io.ReadWriteCloser, buf *bufio.ReadWriter, err os.Error) {
	if w.conn.hijacked {
		return nil, nil, ErrHijacked
	}
	w.conn.hijacked = true
	rwc = w.conn.rwc
	buf = w.conn.buf
	w.conn.rwc = nil
	w.conn.buf = nil
	return
}

// The HandlerFunc type is an adapter to allow the use of
// ordinary functions as HTTP handlers.  If f is a function
// with the appropriate signature, HandlerFunc(f) is a
// Handler object that calls f.
type HandlerFunc func(ResponseWriter, *Request)

// ServeHTTP calls f(w, req).
func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
	f(w, r)
}

// Helper handlers

// Error replies to the request with the specified error message and HTTP code.
func Error(w ResponseWriter, error string, code int) {
	w.SetHeader("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(code)
	fmt.Fprintln(w, error)
}

// NotFound replies to the request with an HTTP 404 not found error.
func NotFound(w ResponseWriter, r *Request) { Error(w, "404 page not found", StatusNotFound) }

// NotFoundHandler returns a simple request handler
// that replies to each request with a ``404 page not found'' reply.
func NotFoundHandler() Handler { return HandlerFunc(NotFound) }

// Redirect replies to the request with a redirect to url,
// which may be a path relative to the request path.
func Redirect(w ResponseWriter, r *Request, url string, code int) {
	if u, err := ParseURL(url); err == nil {
		// If url was relative, make absolute by
		// combining with request path.
		// The browser would probably do this for us,
		// but doing it ourselves is more reliable.

		// NOTE(rsc): RFC 2616 says that the Location
		// line must be an absolute URI, like
		// "http://www.google.com/redirect/",
		// not a path like "/redirect/".
		// Unfortunately, we don't know what to
		// put in the host name section to get the
		// client to connect to us again, so we can't
		// know the right absolute URI to send back.
		// Because of this problem, no one pays attention
		// to the RFC; they all send back just a new path.
		// So do we.
		oldpath := r.URL.Path
		if oldpath == "" { // should not happen, but avoid a crash if it does
			oldpath = "/"
		}
		if u.Scheme == "" {
			// no leading http://server
			if url == "" || url[0] != '/' {
				// make relative path absolute
				olddir, _ := path.Split(oldpath)
				url = olddir + url
			}

			// clean up but preserve trailing slash
			trailing := url[len(url)-1] == '/'
			url = path.Clean(url)
			if trailing && url[len(url)-1] != '/' {
				url += "/"
			}
		}
	}

	w.SetHeader("Location", url)
	w.WriteHeader(code)

	// RFC2616 recommends that a short note "SHOULD" be included in the
	// response because older user agents may not understand 301/307.
	// Shouldn't send the response for POST or HEAD; that leaves GET.
	if r.Method == "GET" {
		note := "<a href=\"" + htmlEscape(url) + "\">" + statusText[code] + "</a>.\n"
		fmt.Fprintln(w, note)
	}
}

func htmlEscape(s string) string {
	s = strings.Replace(s, "&", "&amp;", -1)
	s = strings.Replace(s, "<", "&lt;", -1)
	s = strings.Replace(s, ">", "&gt;", -1)
	s = strings.Replace(s, "\"", "&quot;", -1)
	s = strings.Replace(s, "'", "&apos;", -1)
	return s
}

// Redirect to a fixed URL
type redirectHandler struct {
	url  string
	code int
}

func (rh *redirectHandler) ServeHTTP(w ResponseWriter, r *Request) {
	Redirect(w, r, rh.url, rh.code)
}

// RedirectHandler returns a request handler that redirects
// each request it receives to the given url using the given
// status code.
func RedirectHandler(url string, code int) Handler {
	return &redirectHandler{url, code}
}

// ServeMux is an HTTP request multiplexer.
// It matches the URL of each incoming request against a list of registered
// patterns and calls the handler for the pattern that
// most closely matches the URL.
//
// Patterns named fixed paths, like "/favicon.ico",
// or subtrees, like "/images/" (note the trailing slash).
// Patterns must begin with /.
// Longer patterns take precedence over shorter ones, so that
// if there are handlers registered for both "/images/"
// and "/images/thumbnails/", the latter handler will be
// called for paths beginning "/images/thumbnails/" and the
// former will receiver requests for any other paths in the
// "/images/" subtree.
//
// In the future, the pattern syntax may be relaxed to allow
// an optional host-name at the beginning of the pattern,
// so that a handler might register for the two patterns
// "/codesearch" and "codesearch.google.com/"
// without taking over requests for http://www.google.com/.
//
// ServeMux also takes care of sanitizing the URL request path,
// redirecting any request containing . or .. elements to an
// equivalent .- and ..-free URL.
type ServeMux struct {
	m map[string]Handler
}

// NewServeMux allocates and returns a new ServeMux.
func NewServeMux() *ServeMux { return &ServeMux{make(map[string]Handler)} }

// DefaultServeMux is the default ServeMux used by Serve.
var DefaultServeMux = NewServeMux()

// Does path match pattern?
func pathMatch(pattern, path string) bool {
	if len(pattern) == 0 {
		// should not happen
		return false
	}
	n := len(pattern)
	if pattern[n-1] != '/' {
		return pattern == path
	}
	return len(path) >= n && path[0:n] == pattern
}

// Return the canonical path for p, eliminating . and .. elements.
func cleanPath(p string) string {
	if p == "" {
		return "/"
	}
	if p[0] != '/' {
		p = "/" + p
	}
	np := path.Clean(p)
	// path.Clean removes trailing slash except for root;
	// put the trailing slash back if necessary.
	if p[len(p)-1] == '/' && np != "/" {
		np += "/"
	}
	return np
}

// ServeHTTP dispatches the request to the handler whose
// pattern most closely matches the request URL.
func (mux *ServeMux) ServeHTTP(w ResponseWriter, r *Request) {
	// Clean path to canonical form and redirect.
	if p := cleanPath(r.URL.Path); p != r.URL.Path {
		w.SetHeader("Location", p)
		w.WriteHeader(StatusMovedPermanently)
		return
	}

	// Most-specific (longest) pattern wins.
	var h Handler
	var n = 0
	for k, v := range mux.m {
		if !pathMatch(k, r.URL.Path) {
			continue
		}
		if h == nil || len(k) > n {
			n = len(k)
			h = v
		}
	}
	if h == nil {
		h = NotFoundHandler()
	}
	h.ServeHTTP(w, r)
}

// Handle registers the handler for the given pattern.
func (mux *ServeMux) Handle(pattern string, handler Handler) {
	if pattern == "" || pattern[0] != '/' {
		panic("http: invalid pattern " + pattern)
	}

	mux.m[pattern] = handler

	// Helpful behavior:
	// If pattern is /tree/, insert permanent redirect for /tree.
	n := len(pattern)
	if n > 0 && pattern[n-1] == '/' {
		mux.m[pattern[0:n-1]] = RedirectHandler(pattern, StatusMovedPermanently)
	}
}

// HandleFunc registers the handler function for the given pattern.
func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	mux.Handle(pattern, HandlerFunc(handler))
}

// Handle registers the handler for the given pattern
// in the DefaultServeMux.
func Handle(pattern string, handler Handler) { DefaultServeMux.Handle(pattern, handler) }

// HandleFunc registers the handler function for the given pattern
// in the DefaultServeMux.
func HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	DefaultServeMux.HandleFunc(pattern, handler)
}

// Serve accepts incoming HTTP connections on the listener l,
// creating a new service thread for each.  The service threads
// read requests and then call handler to reply to them.
// Handler is typically nil, in which case the DefaultServeMux is used.
func Serve(l net.Listener, handler Handler) os.Error {
	if handler == nil {
		handler = DefaultServeMux
	}
	for {
		rw, e := l.Accept()
		if e != nil {
			return e
		}
		c, err := newConn(rw, handler)
		if err != nil {
			continue
		}
		go c.serve()
	}
	panic("not reached")
}

// ListenAndServe listens on the TCP network address addr
// and then calls Serve with handler to handle requests
// on incoming connections.  Handler is typically nil,
// in which case the DefaultServeMux is used.
//
// A trivial example server is:
//
//	package main
//
//	import (
//		"http"
//		"io"
//		"log"
//	)
//
//	// hello world, the web server
//	func HelloServer(w http.ResponseWriter, req *http.Request) {
//		io.WriteString(w, "hello, world!\n")
//	}
//
//	func main() {
//		http.HandleFunc("/hello", HelloServer)
//		err := http.ListenAndServe(":12345", nil)
//		if err != nil {
//			log.Exit("ListenAndServe: ", err.String())
//		}
//	}
func ListenAndServe(addr string, handler Handler) os.Error {
	l, e := net.Listen("tcp", addr)
	if e != nil {
		return e
	}
	e = Serve(l, handler)
	l.Close()
	return e
}

// ListenAndServeTLS acts identically to ListenAndServe, except that it
// expects HTTPS connections. Additionally, files containing a certificate and
// matching private key for the server must be provided.
//
// A trivial example server is:
//
//	import (
//		"http"
//		"log"
//	)
//
//	func handler(w http.ResponseWriter, req *http.Request) {
//		w.SetHeader("Content-Type", "text/plain")
//		w.Write([]byte("This is an example server.\n"))
//	}
//
//	func main() {
//		http.HandleFunc("/", handler)
//		log.Printf("About to listen on 10443. Go to https://127.0.0.1:10443/")
//		err := http.ListenAndServeTLS(":10443", "cert.pem", "key.pem", nil)
//		if err != nil {
//			log.Exit(err)
//		}
//	}
//
// One can use generate_cert.go in crypto/tls to generate cert.pem and key.pem.
func ListenAndServeTLS(addr string, certFile string, keyFile string, handler Handler) os.Error {
	config := &tls.Config{
		Rand:       rand.Reader,
		Time:       time.Seconds,
		NextProtos: []string{"http/1.1"},
	}

	var err os.Error
	config.Certificates = make([]tls.Certificate, 1)
	config.Certificates[0], err = tls.LoadX509KeyPair(certFile, keyFile)
	if err != nil {
		return err
	}

	conn, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}

	tlsListener := tls.NewListener(conn, config)
	return Serve(tlsListener, handler)
}
