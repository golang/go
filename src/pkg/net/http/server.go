// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// HTTP server.  See RFC 2616.

// TODO(rsc):
//	logging

package http

import (
	"bufio"
	"bytes"
	"crypto/rand"
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/url"
	"path"
	"runtime/debug"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Errors introduced by the HTTP server.
var (
	ErrWriteAfterFlush = errors.New("Conn.Write called after Flush")
	ErrBodyNotAllowed  = errors.New("http: response status code does not allow body")
	ErrHijacked        = errors.New("Conn has been hijacked")
	ErrContentLength   = errors.New("Conn.Write wrote more than the declared Content-Length")
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
	// Header returns the header map that will be sent by WriteHeader.
	// Changing the header after a call to WriteHeader (or Write) has
	// no effect.
	Header() Header

	// Write writes the data to the connection as part of an HTTP reply.
	// If WriteHeader has not yet been called, Write calls WriteHeader(http.StatusOK)
	// before writing the data.  If the Header does not contain a
	// Content-Type line, Write adds a Content-Type set to the result of passing
	// the initial 512 bytes of written data to DetectContentType.
	Write([]byte) (int, error)

	// WriteHeader sends an HTTP response header with status code.
	// If WriteHeader is not called explicitly, the first call to Write
	// will trigger an implicit WriteHeader(http.StatusOK).
	// Thus explicit calls to WriteHeader are mainly used to
	// send error codes.
	WriteHeader(int)
}

// The Flusher interface is implemented by ResponseWriters that allow
// an HTTP handler to flush buffered data to the client.
//
// Note that even for ResponseWriters that support Flush,
// if the client is connected through an HTTP proxy,
// the buffered data may not reach the client until the response
// completes.
type Flusher interface {
	// Flush sends any buffered data to the client.
	Flush()
}

// The Hijacker interface is implemented by ResponseWriters that allow
// an HTTP handler to take over the connection.
type Hijacker interface {
	// Hijack lets the caller take over the connection.
	// After a call to Hijack(), the HTTP server library
	// will not do anything else with the connection.
	// It becomes the caller's responsibility to manage
	// and close the connection.
	Hijack() (net.Conn, *bufio.ReadWriter, error)
}

// A conn represents the server side of an HTTP connection.
type conn struct {
	remoteAddr string               // network address of remote side
	server     *Server              // the Server on which the connection arrived
	rwc        net.Conn             // i/o connection
	lr         *io.LimitedReader    // io.LimitReader(rwc)
	buf        *bufio.ReadWriter    // buffered(lr,rwc), reading from bufio->limitReader->rwc
	hijacked   bool                 // connection has been hijacked by handler
	tlsState   *tls.ConnectionState // or nil when not using TLS
	body       []byte
}

// A response represents the server side of an HTTP response.
type response struct {
	conn          *conn
	req           *Request // request for this response
	chunking      bool     // using chunked transfer encoding for reply body
	wroteHeader   bool     // reply header has been written
	wroteContinue bool     // 100 Continue response was written
	header        Header   // reply header parameters
	written       int64    // number of bytes written in body
	contentLength int64    // explicitly-declared Content-Length; or -1
	status        int      // status code passed to WriteHeader
	needSniff     bool     // need to sniff to find Content-Type

	// close connection after this reply.  set on request and
	// updated after response from handler if there's a
	// "Connection: keep-alive" response header and a
	// Content-Length.
	closeAfterReply bool

	// requestBodyLimitHit is set by requestTooLarge when
	// maxBytesReader hits its max size. It is checked in
	// WriteHeader, to make sure we don't consume the the
	// remaining request body to try to advance to the next HTTP
	// request. Instead, when this is set, we stop doing
	// subsequent requests on this connection and stop reading
	// input from it.
	requestBodyLimitHit bool
}

// requestTooLarge is called by maxBytesReader when too much input has
// been read from the client.
func (w *response) requestTooLarge() {
	w.closeAfterReply = true
	w.requestBodyLimitHit = true
	if !w.wroteHeader {
		w.Header().Set("Connection", "close")
	}
}

type writerOnly struct {
	io.Writer
}

func (w *response) ReadFrom(src io.Reader) (n int64, err error) {
	// Call WriteHeader before checking w.chunking if it hasn't
	// been called yet, since WriteHeader is what sets w.chunking.
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	if !w.chunking && w.bodyAllowed() && !w.needSniff {
		w.Flush()
		if rf, ok := w.conn.rwc.(io.ReaderFrom); ok {
			n, err = rf.ReadFrom(src)
			w.written += n
			return
		}
	}
	// Fall back to default io.Copy implementation.
	// Use wrapper to hide w.ReadFrom from io.Copy.
	return io.Copy(writerOnly{w}, src)
}

// noLimit is an effective infinite upper bound for io.LimitedReader
const noLimit int64 = (1 << 63) - 1

// Create new connection from rwc.
func (srv *Server) newConn(rwc net.Conn) (c *conn, err error) {
	c = new(conn)
	c.remoteAddr = rwc.RemoteAddr().String()
	c.server = srv
	c.rwc = rwc
	c.body = make([]byte, sniffLen)
	c.lr = io.LimitReader(rwc, noLimit).(*io.LimitedReader)
	br := bufio.NewReader(c.lr)
	bw := bufio.NewWriter(rwc)
	c.buf = bufio.NewReadWriter(br, bw)
	return c, nil
}

// DefaultMaxHeaderBytes is the maximum permitted size of the headers
// in an HTTP request.
// This can be overridden by setting Server.MaxHeaderBytes.
const DefaultMaxHeaderBytes = 1 << 20 // 1 MB

func (srv *Server) maxHeaderBytes() int {
	if srv.MaxHeaderBytes > 0 {
		return srv.MaxHeaderBytes
	}
	return DefaultMaxHeaderBytes
}

// wrapper around io.ReaderCloser which on first read, sends an
// HTTP/1.1 100 Continue header
type expectContinueReader struct {
	resp       *response
	readCloser io.ReadCloser
	closed     bool
}

func (ecr *expectContinueReader) Read(p []byte) (n int, err error) {
	if ecr.closed {
		return 0, errors.New("http: Read after Close on request Body")
	}
	if !ecr.resp.wroteContinue && !ecr.resp.conn.hijacked {
		ecr.resp.wroteContinue = true
		io.WriteString(ecr.resp.conn.buf, "HTTP/1.1 100 Continue\r\n\r\n")
		ecr.resp.conn.buf.Flush()
	}
	return ecr.readCloser.Read(p)
}

func (ecr *expectContinueReader) Close() error {
	ecr.closed = true
	return ecr.readCloser.Close()
}

// TimeFormat is the time format to use with
// time.Parse and time.Time.Format when parsing
// or generating times in HTTP headers.
// It is like time.RFC1123 but hard codes GMT as the time zone.
const TimeFormat = "Mon, 02 Jan 2006 15:04:05 GMT"

var errTooLarge = errors.New("http: request too large")

// Read next request from connection.
func (c *conn) readRequest() (w *response, err error) {
	if c.hijacked {
		return nil, ErrHijacked
	}
	c.lr.N = int64(c.server.maxHeaderBytes()) + 4096 /* bufio slop */
	var req *Request
	if req, err = ReadRequest(c.buf.Reader); err != nil {
		if c.lr.N == 0 {
			return nil, errTooLarge
		}
		return nil, err
	}
	c.lr.N = noLimit

	req.RemoteAddr = c.remoteAddr
	req.TLS = c.tlsState

	w = new(response)
	w.conn = c
	w.req = req
	w.header = make(Header)
	w.contentLength = -1
	c.body = c.body[:0]
	return w, nil
}

func (w *response) Header() Header {
	return w.header
}

// maxPostHandlerReadBytes is the max number of Request.Body bytes not
// consumed by a handler that the server will read from the client
// in order to keep a connection alive.  If there are more bytes than
// this then the server to be paranoid instead sends a "Connection:
// close" response.
//
// This number is approximately what a typical machine's TCP buffer
// size is anyway.  (if we have the bytes on the machine, we might as
// well read them)
const maxPostHandlerReadBytes = 256 << 10

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

	// Check for a explicit (and valid) Content-Length header.
	var hasCL bool
	var contentLength int64
	if clenStr := w.header.Get("Content-Length"); clenStr != "" {
		var err error
		contentLength, err = strconv.ParseInt(clenStr, 10, 64)
		if err == nil {
			hasCL = true
		} else {
			log.Printf("http: invalid Content-Length of %q sent", clenStr)
			w.header.Del("Content-Length")
		}
	}

	if w.req.wantsHttp10KeepAlive() && (w.req.Method == "HEAD" || hasCL) {
		_, connectionHeaderSet := w.header["Connection"]
		if !connectionHeaderSet {
			w.header.Set("Connection", "keep-alive")
		}
	} else if !w.req.ProtoAtLeast(1, 1) {
		// Client did not ask to keep connection alive.
		w.closeAfterReply = true
	}

	if w.header.Get("Connection") == "close" {
		w.closeAfterReply = true
	}

	// Per RFC 2616, we should consume the request body before
	// replying, if the handler hasn't already done so.  But we
	// don't want to do an unbounded amount of reading here for
	// DoS reasons, so we only try up to a threshold.
	if w.req.ContentLength != 0 && !w.closeAfterReply {
		ecr, isExpecter := w.req.Body.(*expectContinueReader)
		if !isExpecter || ecr.resp.wroteContinue {
			n, _ := io.CopyN(ioutil.Discard, w.req.Body, maxPostHandlerReadBytes+1)
			if n >= maxPostHandlerReadBytes {
				w.requestTooLarge()
				w.header.Set("Connection", "close")
			} else {
				w.req.Body.Close()
			}
		}
	}

	if code == StatusNotModified {
		// Must not have body.
		for _, header := range []string{"Content-Type", "Content-Length", "Transfer-Encoding"} {
			if w.header.Get(header) != "" {
				// TODO: return an error if WriteHeader gets a return parameter
				// or set a flag on w to make future Writes() write an error page?
				// for now just log and drop the header.
				log.Printf("http: StatusNotModified response with header %q defined", header)
				w.header.Del(header)
			}
		}
	} else {
		// If no content type, apply sniffing algorithm to body.
		if w.header.Get("Content-Type") == "" && w.req.Method != "HEAD" {
			w.needSniff = true
		}
	}

	if _, ok := w.header["Date"]; !ok {
		w.Header().Set("Date", time.Now().UTC().Format(TimeFormat))
	}

	te := w.header.Get("Transfer-Encoding")
	hasTE := te != ""
	if hasCL && hasTE && te != "identity" {
		// TODO: return an error if WriteHeader gets a return parameter
		// For now just ignore the Content-Length.
		log.Printf("http: WriteHeader called with both Transfer-Encoding of %q and a Content-Length of %d",
			te, contentLength)
		w.header.Del("Content-Length")
		hasCL = false
	}

	if w.req.Method == "HEAD" || code == StatusNotModified {
		// do nothing
	} else if hasCL {
		w.contentLength = contentLength
		w.header.Del("Transfer-Encoding")
	} else if w.req.ProtoAtLeast(1, 1) {
		// HTTP/1.1 or greater: use chunked transfer encoding
		// to avoid closing the connection at EOF.
		// TODO: this blows away any custom or stacked Transfer-Encoding they
		// might have set.  Deal with that as need arises once we have a valid
		// use case.
		w.chunking = true
		w.header.Set("Transfer-Encoding", "chunked")
	} else {
		// HTTP version < 1.1: cannot do chunked transfer
		// encoding and we don't know the Content-Length so
		// signal EOF by closing connection.
		w.closeAfterReply = true
		w.header.Del("Transfer-Encoding") // in case already set
	}

	// Cannot use Content-Length with non-identity Transfer-Encoding.
	if w.chunking {
		w.header.Del("Content-Length")
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
	w.header.Write(w.conn.buf)

	// If we need to sniff the body, leave the header open.
	// Otherwise, end it here.
	if !w.needSniff {
		io.WriteString(w.conn.buf, "\r\n")
	}
}

// sniff uses the first block of written data,
// stored in w.conn.body, to decide the Content-Type
// for the HTTP body.
func (w *response) sniff() {
	if !w.needSniff {
		return
	}
	w.needSniff = false

	data := w.conn.body
	fmt.Fprintf(w.conn.buf, "Content-Type: %s\r\n\r\n", DetectContentType(data))

	if len(data) == 0 {
		return
	}
	if w.chunking {
		fmt.Fprintf(w.conn.buf, "%x\r\n", len(data))
	}
	_, err := w.conn.buf.Write(data)
	if w.chunking && err == nil {
		io.WriteString(w.conn.buf, "\r\n")
	}
}

// bodyAllowed returns true if a Write is allowed for this response type.
// It's illegal to call this before the header has been flushed.
func (w *response) bodyAllowed() bool {
	if !w.wroteHeader {
		panic("")
	}
	return w.status != StatusNotModified && w.req.Method != "HEAD"
}

func (w *response) Write(data []byte) (n int, err error) {
	if w.conn.hijacked {
		log.Print("http: response.Write on hijacked connection")
		return 0, ErrHijacked
	}
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	if len(data) == 0 {
		return 0, nil
	}
	if !w.bodyAllowed() {
		return 0, ErrBodyNotAllowed
	}

	w.written += int64(len(data)) // ignoring errors, for errorKludge
	if w.contentLength != -1 && w.written > w.contentLength {
		return 0, ErrContentLength
	}

	var m int
	if w.needSniff {
		// We need to sniff the beginning of the output to
		// determine the content type.  Accumulate the
		// initial writes in w.conn.body.
		// Cap m so that append won't allocate.
		m = cap(w.conn.body) - len(w.conn.body)
		if m > len(data) {
			m = len(data)
		}
		w.conn.body = append(w.conn.body, data[:m]...)
		data = data[m:]
		if len(data) == 0 {
			// Copied everything into the buffer.
			// Wait for next write.
			return m, nil
		}

		// Filled the buffer; more data remains.
		// Sniff the content (flushes the buffer)
		// and then proceed with the remainder
		// of the data as a normal Write.
		// Calling sniff clears needSniff.
		w.sniff()
	}

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

	return m + n, err
}

func (w *response) finishRequest() {
	// If this was an HTTP/1.0 request with keep-alive and we sent a Content-Length
	// back, we can make this a keep-alive response ...
	if w.req.wantsHttp10KeepAlive() {
		sentLength := w.header.Get("Content-Length") != ""
		if sentLength && w.header.Get("Connection") == "keep-alive" {
			w.closeAfterReply = false
		}
	}
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	if w.needSniff {
		w.sniff()
	}
	if w.chunking {
		io.WriteString(w.conn.buf, "0\r\n")
		// trailer key/value pairs, followed by blank line
		io.WriteString(w.conn.buf, "\r\n")
	}
	w.conn.buf.Flush()
	// Close the body, unless we're about to close the whole TCP connection
	// anyway.
	if !w.closeAfterReply {
		w.req.Body.Close()
	}
	if w.req.MultipartForm != nil {
		w.req.MultipartForm.RemoveAll()
	}

	if w.contentLength != -1 && w.contentLength != w.written {
		// Did not write enough. Avoid getting out of sync.
		w.closeAfterReply = true
	}
}

func (w *response) Flush() {
	if !w.wroteHeader {
		w.WriteHeader(StatusOK)
	}
	w.sniff()
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
	defer func() {
		err := recover()
		if err == nil {
			return
		}

		var buf bytes.Buffer
		fmt.Fprintf(&buf, "http: panic serving %v: %v\n", c.remoteAddr, err)
		buf.Write(debug.Stack())
		log.Print(buf.String())

		if c.rwc != nil { // may be nil if connection hijacked
			c.rwc.Close()
		}
	}()

	if tlsConn, ok := c.rwc.(*tls.Conn); ok {
		if err := tlsConn.Handshake(); err != nil {
			c.close()
			return
		}
		c.tlsState = new(tls.ConnectionState)
		*c.tlsState = tlsConn.ConnectionState()
	}

	for {
		w, err := c.readRequest()
		if err != nil {
			msg := "400 Bad Request"
			if err == errTooLarge {
				// Their HTTP client may or may not be
				// able to read this if we're
				// responding to them and hanging up
				// while they're still writing their
				// request.  Undefined behavior.
				msg = "413 Request Entity Too Large"
			} else if err == io.ErrUnexpectedEOF {
				break // Don't reply
			} else if neterr, ok := err.(net.Error); ok && neterr.Timeout() {
				break // Don't reply
			}
			fmt.Fprintf(c.rwc, "HTTP/1.1 %s\r\n\r\n", msg)
			break
		}

		// Expect 100 Continue support
		req := w.req
		if req.expectsContinue() {
			if req.ProtoAtLeast(1, 1) {
				// Wrap the Body reader with one that replies on the connection
				req.Body = &expectContinueReader{readCloser: req.Body, resp: w}
			}
			if req.ContentLength == 0 {
				w.Header().Set("Connection", "close")
				w.WriteHeader(StatusBadRequest)
				w.finishRequest()
				break
			}
			req.Header.Del("Expect")
		} else if req.Header.Get("Expect") != "" {
			// TODO(bradfitz): let ServeHTTP handlers handle
			// requests with non-standard expectation[s]? Seems
			// theoretical at best, and doesn't fit into the
			// current ServeHTTP model anyway.  We'd need to
			// make the ResponseWriter an optional
			// "ExpectReplier" interface or something.
			//
			// For now we'll just obey RFC 2616 14.20 which says
			// "If a server receives a request containing an
			// Expect field that includes an expectation-
			// extension that it does not support, it MUST
			// respond with a 417 (Expectation Failed) status."
			w.Header().Set("Connection", "close")
			w.WriteHeader(StatusExpectationFailed)
			w.finishRequest()
			break
		}

		handler := c.server.Handler
		if handler == nil {
			handler = DefaultServeMux
		}

		// HTTP cannot have multiple simultaneous active requests.[*]
		// Until the server replies to this request, it can't read another,
		// so we might as well run the handler in this goroutine.
		// [*] Not strictly true: HTTP pipelining.  We could let them all process
		// in parallel even if their responses need to be serialized.
		handler.ServeHTTP(w, w.req)
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

// Hijack implements the Hijacker.Hijack method. Our response is both a ResponseWriter
// and a Hijacker.
func (w *response) Hijack() (rwc net.Conn, buf *bufio.ReadWriter, err error) {
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

// ServeHTTP calls f(w, r).
func (f HandlerFunc) ServeHTTP(w ResponseWriter, r *Request) {
	f(w, r)
}

// Helper handlers

// Error replies to the request with the specified error message and HTTP code.
func Error(w ResponseWriter, error string, code int) {
	w.Header().Set("Content-Type", "text/plain; charset=utf-8")
	w.WriteHeader(code)
	fmt.Fprintln(w, error)
}

// NotFound replies to the request with an HTTP 404 not found error.
func NotFound(w ResponseWriter, r *Request) { Error(w, "404 page not found", StatusNotFound) }

// NotFoundHandler returns a simple request handler
// that replies to each request with a ``404 page not found'' reply.
func NotFoundHandler() Handler { return HandlerFunc(NotFound) }

// StripPrefix returns a handler that serves HTTP requests
// by removing the given prefix from the request URL's Path
// and invoking the handler h. StripPrefix handles a
// request for a path that doesn't begin with prefix by
// replying with an HTTP 404 not found error.
func StripPrefix(prefix string, h Handler) Handler {
	return HandlerFunc(func(w ResponseWriter, r *Request) {
		if !strings.HasPrefix(r.URL.Path, prefix) {
			NotFound(w, r)
			return
		}
		r.URL.Path = r.URL.Path[len(prefix):]
		h.ServeHTTP(w, r)
	})
}

// Redirect replies to the request with a redirect to url,
// which may be a path relative to the request path.
func Redirect(w ResponseWriter, r *Request, urlStr string, code int) {
	if u, err := url.Parse(urlStr); err == nil {
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
			if urlStr == "" || urlStr[0] != '/' {
				// make relative path absolute
				olddir, _ := path.Split(oldpath)
				urlStr = olddir + urlStr
			}

			var query string
			if i := strings.Index(urlStr, "?"); i != -1 {
				urlStr, query = urlStr[:i], urlStr[i:]
			}

			// clean up but preserve trailing slash
			trailing := urlStr[len(urlStr)-1] == '/'
			urlStr = path.Clean(urlStr)
			if trailing && urlStr[len(urlStr)-1] != '/' {
				urlStr += "/"
			}
			urlStr += query
		}
	}

	w.Header().Set("Location", urlStr)
	w.WriteHeader(code)

	// RFC2616 recommends that a short note "SHOULD" be included in the
	// response because older user agents may not understand 301/307.
	// Shouldn't send the response for POST or HEAD; that leaves GET.
	if r.Method == "GET" {
		note := "<a href=\"" + htmlEscape(urlStr) + "\">" + statusText[code] + "</a>.\n"
		fmt.Fprintln(w, note)
	}
}

var htmlReplacer = strings.NewReplacer(
	"&", "&amp;",
	"<", "&lt;",
	">", "&gt;",
	`"`, "&quot;",
	"'", "&apos;",
)

func htmlEscape(s string) string {
	return htmlReplacer.Replace(s)
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
// Patterns named fixed, rooted paths, like "/favicon.ico",
// or rooted subtrees, like "/images/" (note the trailing slash).
// Longer patterns take precedence over shorter ones, so that
// if there are handlers registered for both "/images/"
// and "/images/thumbnails/", the latter handler will be
// called for paths beginning "/images/thumbnails/" and the
// former will receiver requests for any other paths in the
// "/images/" subtree.
//
// Patterns may optionally begin with a host name, restricting matches to
// URLs on that host only.  Host-specific patterns take precedence over
// general patterns, so that a handler might register for the two patterns
// "/codesearch" and "codesearch.google.com/" without also taking over
// requests for "http://www.google.com/".
//
// ServeMux also takes care of sanitizing the URL request path,
// redirecting any request containing . or .. elements to an
// equivalent .- and ..-free URL.
type ServeMux struct {
	mu sync.RWMutex
	m  map[string]muxEntry
}

type muxEntry struct {
	explicit bool
	h        Handler
}

// NewServeMux allocates and returns a new ServeMux.
func NewServeMux() *ServeMux { return &ServeMux{m: make(map[string]muxEntry)} }

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

// Find a handler on a handler map given a path string
// Most-specific (longest) pattern wins
func (mux *ServeMux) match(path string) Handler {
	var h Handler
	var n = 0
	for k, v := range mux.m {
		if !pathMatch(k, path) {
			continue
		}
		if h == nil || len(k) > n {
			n = len(k)
			h = v.h
		}
	}
	return h
}

// handler returns the handler to use for the request r.
func (mux *ServeMux) handler(r *Request) Handler {
	mux.mu.RLock()
	defer mux.mu.RUnlock()

	// Host-specific pattern takes precedence over generic ones
	h := mux.match(r.Host + r.URL.Path)
	if h == nil {
		h = mux.match(r.URL.Path)
	}
	if h == nil {
		h = NotFoundHandler()
	}
	return h
}

// ServeHTTP dispatches the request to the handler whose
// pattern most closely matches the request URL.
func (mux *ServeMux) ServeHTTP(w ResponseWriter, r *Request) {
	// Clean path to canonical form and redirect.
	if p := cleanPath(r.URL.Path); p != r.URL.Path {
		w.Header().Set("Location", p)
		w.WriteHeader(StatusMovedPermanently)
		return
	}
	mux.handler(r).ServeHTTP(w, r)
}

// Handle registers the handler for the given pattern.
// If a handler already exists for pattern, Handle panics.
func (mux *ServeMux) Handle(pattern string, handler Handler) {
	mux.mu.Lock()
	defer mux.mu.Unlock()

	if pattern == "" {
		panic("http: invalid pattern " + pattern)
	}
	if handler == nil {
		panic("http: nil handler")
	}
	if mux.m[pattern].explicit {
		panic("http: multiple registrations for " + pattern)
	}

	mux.m[pattern] = muxEntry{explicit: true, h: handler}

	// Helpful behavior:
	// If pattern is /tree/, insert an implicit permanent redirect for /tree.
	// It can be overridden by an explicit registration.
	n := len(pattern)
	if n > 0 && pattern[n-1] == '/' && !mux.m[pattern[0:n-1]].explicit {
		mux.m[pattern[0:n-1]] = muxEntry{h: RedirectHandler(pattern, StatusMovedPermanently)}
	}
}

// HandleFunc registers the handler function for the given pattern.
func (mux *ServeMux) HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	mux.Handle(pattern, HandlerFunc(handler))
}

// Handle registers the handler for the given pattern
// in the DefaultServeMux.
// The documentation for ServeMux explains how patterns are matched.
func Handle(pattern string, handler Handler) { DefaultServeMux.Handle(pattern, handler) }

// HandleFunc registers the handler function for the given pattern
// in the DefaultServeMux.
// The documentation for ServeMux explains how patterns are matched.
func HandleFunc(pattern string, handler func(ResponseWriter, *Request)) {
	DefaultServeMux.HandleFunc(pattern, handler)
}

// Serve accepts incoming HTTP connections on the listener l,
// creating a new service thread for each.  The service threads
// read requests and then call handler to reply to them.
// Handler is typically nil, in which case the DefaultServeMux is used.
func Serve(l net.Listener, handler Handler) error {
	srv := &Server{Handler: handler}
	return srv.Serve(l)
}

// A Server defines parameters for running an HTTP server.
type Server struct {
	Addr           string        // TCP address to listen on, ":http" if empty
	Handler        Handler       // handler to invoke, http.DefaultServeMux if nil
	ReadTimeout    time.Duration // maximum duration before timing out read of the request
	WriteTimeout   time.Duration // maximum duration before timing out write of the response
	MaxHeaderBytes int           // maximum size of request headers, DefaultMaxHeaderBytes if 0
}

// ListenAndServe listens on the TCP network address srv.Addr and then
// calls Serve to handle requests on incoming connections.  If
// srv.Addr is blank, ":http" is used.
func (srv *Server) ListenAndServe() error {
	addr := srv.Addr
	if addr == "" {
		addr = ":http"
	}
	l, e := net.Listen("tcp", addr)
	if e != nil {
		return e
	}
	return srv.Serve(l)
}

// Serve accepts incoming connections on the Listener l, creating a
// new service thread for each.  The service threads read requests and
// then call srv.Handler to reply to them.
func (srv *Server) Serve(l net.Listener) error {
	defer l.Close()
	var tempDelay time.Duration // how long to sleep on accept failure
	for {
		rw, e := l.Accept()
		if e != nil {
			if ne, ok := e.(net.Error); ok && ne.Temporary() {
				if tempDelay == 0 {
					tempDelay = 5 * time.Millisecond
				} else {
					tempDelay *= 2
				}
				if max := 1 * time.Second; tempDelay > max {
					tempDelay = max
				}
				log.Printf("http: Accept error: %v; retrying in %v", e, tempDelay)
				time.Sleep(tempDelay)
				continue
			}
			return e
		}
		tempDelay = 0
		if srv.ReadTimeout != 0 {
			rw.SetReadDeadline(time.Now().Add(srv.ReadTimeout))
		}
		if srv.WriteTimeout != 0 {
			rw.SetWriteDeadline(time.Now().Add(srv.WriteTimeout))
		}
		c, err := srv.newConn(rw)
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
//		"io"
//		"net/http"
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
//			log.Fatal("ListenAndServe: ", err)
//		}
//	}
func ListenAndServe(addr string, handler Handler) error {
	server := &Server{Addr: addr, Handler: handler}
	return server.ListenAndServe()
}

// ListenAndServeTLS acts identically to ListenAndServe, except that it
// expects HTTPS connections. Additionally, files containing a certificate and
// matching private key for the server must be provided. If the certificate
// is signed by a certificate authority, the certFile should be the concatenation
// of the server's certificate followed by the CA's certificate.
//
// A trivial example server is:
//
//	import (
//		"log"
//		"net/http"
//	)
//
//	func handler(w http.ResponseWriter, req *http.Request) {
//		w.Header().Set("Content-Type", "text/plain")
//		w.Write([]byte("This is an example server.\n"))
//	}
//
//	func main() {
//		http.HandleFunc("/", handler)
//		log.Printf("About to listen on 10443. Go to https://127.0.0.1:10443/")
//		err := http.ListenAndServeTLS(":10443", "cert.pem", "key.pem", nil)
//		if err != nil {
//			log.Fatal(err)
//		}
//	}
//
// One can use generate_cert.go in crypto/tls to generate cert.pem and key.pem.
func ListenAndServeTLS(addr string, certFile string, keyFile string, handler Handler) error {
	server := &Server{Addr: addr, Handler: handler}
	return server.ListenAndServeTLS(certFile, keyFile)
}

// ListenAndServeTLS listens on the TCP network address srv.Addr and
// then calls Serve to handle requests on incoming TLS connections.
//
// Filenames containing a certificate and matching private key for
// the server must be provided. If the certificate is signed by a
// certificate authority, the certFile should be the concatenation
// of the server's certificate followed by the CA's certificate.
//
// If srv.Addr is blank, ":https" is used.
func (srv *Server) ListenAndServeTLS(certFile, keyFile string) error {
	addr := srv.Addr
	if addr == "" {
		addr = ":https"
	}
	config := &tls.Config{
		Rand:       rand.Reader,
		NextProtos: []string{"http/1.1"},
	}

	var err error
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
	return srv.Serve(tlsListener)
}

// TimeoutHandler returns a Handler that runs h with the given time limit.
//
// The new Handler calls h.ServeHTTP to handle each request, but if a
// call runs for more than ns nanoseconds, the handler responds with
// a 503 Service Unavailable error and the given message in its body.
// (If msg is empty, a suitable default message will be sent.)
// After such a timeout, writes by h to its ResponseWriter will return
// ErrHandlerTimeout.
func TimeoutHandler(h Handler, dt time.Duration, msg string) Handler {
	f := func() <-chan time.Time {
		return time.After(dt)
	}
	return &timeoutHandler{h, f, msg}
}

// ErrHandlerTimeout is returned on ResponseWriter Write calls
// in handlers which have timed out.
var ErrHandlerTimeout = errors.New("http: Handler timeout")

type timeoutHandler struct {
	handler Handler
	timeout func() <-chan time.Time // returns channel producing a timeout
	body    string
}

func (h *timeoutHandler) errorBody() string {
	if h.body != "" {
		return h.body
	}
	return "<html><head><title>Timeout</title></head><body><h1>Timeout</h1></body></html>"
}

func (h *timeoutHandler) ServeHTTP(w ResponseWriter, r *Request) {
	done := make(chan bool)
	tw := &timeoutWriter{w: w}
	go func() {
		h.handler.ServeHTTP(tw, r)
		done <- true
	}()
	select {
	case <-done:
		return
	case <-h.timeout():
		tw.mu.Lock()
		defer tw.mu.Unlock()
		if !tw.wroteHeader {
			tw.w.WriteHeader(StatusServiceUnavailable)
			tw.w.Write([]byte(h.errorBody()))
		}
		tw.timedOut = true
	}
}

type timeoutWriter struct {
	w ResponseWriter

	mu          sync.Mutex
	timedOut    bool
	wroteHeader bool
}

func (tw *timeoutWriter) Header() Header {
	return tw.w.Header()
}

func (tw *timeoutWriter) Write(p []byte) (int, error) {
	tw.mu.Lock()
	timedOut := tw.timedOut
	tw.mu.Unlock()
	if timedOut {
		return 0, ErrHandlerTimeout
	}
	return tw.w.Write(p)
}

func (tw *timeoutWriter) WriteHeader(code int) {
	tw.mu.Lock()
	if tw.timedOut || tw.wroteHeader {
		tw.mu.Unlock()
		return
	}
	tw.wroteHeader = true
	tw.mu.Unlock()
	tw.w.WriteHeader(code)
}
