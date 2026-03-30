// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"context"
	"crypto/tls"
	"fmt"
	"io"
	"maps"
	"net/http"
	"slices"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/internal/httpcommon"
	"golang.org/x/net/quic"
)

// A server is an HTTP/3 server.
// The zero value for server is a valid server.
type server struct {
	// handler to invoke for requests, http.DefaultServeMux if nil.
	handler http.Handler

	config *quic.Config

	listenQUIC func(addr string, config *quic.Config) (*quic.Endpoint, error)

	initOnce sync.Once

	serveCtx       context.Context
	serveCtxCancel context.CancelFunc

	// connClosed is used to signal that a connection has been unregistered
	// from activeConns. That way, when shutting down gracefully, the server
	// can avoid busy-waiting for activeConns to be empty.
	connClosed  chan any
	mu          sync.Mutex // Guards fields below.
	activeConns map[*serverConn]struct{}
}

// netHTTPHandler is an interface that is implemented by
// net/http.http3ServerHandler in std.
//
// It provides a way for information to be passed between x/net and net/http
// that would otherwise be inaccessible, such as the TLS configs that users
// have supplied to net/http servers.
//
// This allows us to integrate our HTTP/3 server implementation with the
// net/http server when RegisterServer is called.
type netHTTPHandler interface {
	http.Handler
	TLSConfig() *tls.Config
	BaseContext() context.Context
	Addr() string
	ListenErrHook(err error)
	ShutdownContext() context.Context
}

type ServerOpts struct {
	// ListenQUIC determines how the server will open a QUIC endpoint.
	// By default, quic.Listen("udp", addr, config) is used.
	ListenQUIC func(addr string, config *quic.Config) (*quic.Endpoint, error)

	// QUICConfig is the QUIC configuration used by the server.
	// QUICConfig may be nil and should not be modified after calling
	// RegisterServer.
	// If QUICConfig.TLSConfig is nil, the TLSConfig of the net/http Server
	// given to RegisterServer will be used.
	QUICConfig *quic.Config
}

// RegisterServer adds HTTP/3 support to a net/http Server.
//
// RegisterServer must be called before s begins serving, and only affects
// s.ListenAndServeTLS.
func RegisterServer(s *http.Server, opts ServerOpts) {
	if s.TLSNextProto == nil {
		s.TLSNextProto = make(map[string]func(*http.Server, *tls.Conn, http.Handler))
	}
	s.TLSNextProto["http/3"] = func(s *http.Server, c *tls.Conn, h http.Handler) {
		stdHandler, ok := h.(netHTTPHandler)
		if !ok {
			panic("RegisterServer was given a server that does not implement netHTTPHandler")
		}
		if opts.QUICConfig == nil {
			opts.QUICConfig = &quic.Config{}
		}
		if opts.QUICConfig.TLSConfig == nil {
			opts.QUICConfig.TLSConfig = stdHandler.TLSConfig()
		}
		s3 := &server{
			config:     opts.QUICConfig,
			listenQUIC: opts.ListenQUIC,
			handler:    stdHandler,
			serveCtx:   stdHandler.BaseContext(),
		}
		s3.init()
		s.RegisterOnShutdown(func() {
			s3.shutdown(stdHandler.ShutdownContext())
		})
		stdHandler.ListenErrHook(s3.listenAndServe(stdHandler.Addr()))
	}
}

func (s *server) init() {
	s.initOnce.Do(func() {
		s.config = initConfig(s.config)
		if s.handler == nil {
			s.handler = http.DefaultServeMux
		}
		if s.serveCtx == nil {
			s.serveCtx = context.Background()
		}
		if s.listenQUIC == nil {
			s.listenQUIC = func(addr string, config *quic.Config) (*quic.Endpoint, error) {
				return quic.Listen("udp", addr, config)
			}
		}
		s.serveCtx, s.serveCtxCancel = context.WithCancel(s.serveCtx)
		s.activeConns = make(map[*serverConn]struct{})
		s.connClosed = make(chan any, 1)
	})
}

// listenAndServe listens on the UDP network address addr
// and then calls Serve to handle requests on incoming connections.
func (s *server) listenAndServe(addr string) error {
	s.init()
	e, err := s.listenQUIC(addr, s.config)
	if err != nil {
		return err
	}
	go s.serve(e)
	return nil
}

// serve accepts incoming connections on the QUIC endpoint e,
// and handles requests from those connections.
func (s *server) serve(e *quic.Endpoint) error {
	s.init()
	defer e.Close(canceledCtx)
	for {
		qconn, err := e.Accept(s.serveCtx)
		if err != nil {
			return err
		}
		go s.newServerConn(qconn, s.handler)
	}
}

// shutdown attempts a graceful shutdown for the server.
func (s *server) shutdown(ctx context.Context) {
	// Set a reasonable default in case ctx is nil.
	if ctx == nil {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(context.Background(), time.Second)
		defer cancel()
	}

	// Send GOAWAY frames to all active connections to give a chance for them
	// to gracefully terminate.
	s.mu.Lock()
	for sc := range s.activeConns {
		// TODO: Modify x/net/quic stream API so that write errors from context
		// deadline are sticky.
		go sc.sendGoaway()
	}
	s.mu.Unlock()

	// Complete shutdown as soon as there are no more active connections or ctx
	// is done, whichever comes first.
	defer func() {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.serveCtxCancel()
		for sc := range s.activeConns {
			sc.abort(&connectionError{
				code:    errH3NoError,
				message: "server is shutting down",
			})
		}
	}()
	noMoreConns := func() bool {
		s.mu.Lock()
		defer s.mu.Unlock()
		return len(s.activeConns) == 0
	}
	for {
		if noMoreConns() {
			return
		}
		select {
		case <-ctx.Done():
			return
		case <-s.connClosed:
		}
	}
}

func (s *server) registerConn(sc *serverConn) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.activeConns[sc] = struct{}{}
}

func (s *server) unregisterConn(sc *serverConn) {
	s.mu.Lock()
	delete(s.activeConns, sc)
	s.mu.Unlock()
	select {
	case s.connClosed <- struct{}{}:
	default:
		// Channel already full. No need to send more values since we are just
		// using this channel as a simpler sync.Cond.
	}
}

type serverConn struct {
	qconn *quic.Conn

	genericConn // for handleUnidirectionalStream
	enc         qpackEncoder
	dec         qpackDecoder
	handler     http.Handler

	// For handling shutdown.
	controlStream      *stream
	mu                 sync.Mutex // Guards everything below.
	maxRequestStreamID int64
	goawaySent         bool
}

func (s *server) newServerConn(qconn *quic.Conn, handler http.Handler) {
	sc := &serverConn{
		qconn:   qconn,
		handler: handler,
	}
	s.registerConn(sc)
	defer s.unregisterConn(sc)
	sc.enc.init()

	// Create control stream and send SETTINGS frame.
	// TODO: Time out on creating stream.
	var err error
	sc.controlStream, err = newConnStream(context.Background(), sc.qconn, streamTypeControl)
	if err != nil {
		return
	}
	sc.controlStream.writeSettings()
	sc.controlStream.Flush()

	sc.acceptStreams(sc.qconn, sc)
}

func (sc *serverConn) handleControlStream(st *stream) error {
	// "A SETTINGS frame MUST be sent as the first frame of each control stream [...]"
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.4-2
	if err := st.readSettings(func(settingsType, settingsValue int64) error {
		switch settingsType {
		case settingsMaxFieldSectionSize:
			_ = settingsValue // TODO
		case settingsQPACKMaxTableCapacity:
			_ = settingsValue // TODO
		case settingsQPACKBlockedStreams:
			_ = settingsValue // TODO
		default:
			// Unknown settings types are ignored.
		}
		return nil
	}); err != nil {
		return err
	}

	for {
		ftype, err := st.readFrameHeader()
		if err != nil {
			return err
		}
		switch ftype {
		case frameTypeCancelPush:
			// "If a server receives a CANCEL_PUSH frame for a push ID
			// that has not yet been mentioned by a PUSH_PROMISE frame,
			// this MUST be treated as a connection error of type H3_ID_ERROR."
			// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.3-8
			return &connectionError{
				code:    errH3IDError,
				message: "CANCEL_PUSH for unsent push ID",
			}
		case frameTypeGoaway:
			return errH3NoError
		default:
			// Unknown frames are ignored.
			if err := st.discardUnknownFrame(ftype); err != nil {
				return err
			}
		}
	}
}

func (sc *serverConn) handleEncoderStream(*stream) error {
	// TODO
	return nil
}

func (sc *serverConn) handleDecoderStream(*stream) error {
	// TODO
	return nil
}

func (sc *serverConn) handlePushStream(*stream) error {
	// "[...] if a server receives a client-initiated push stream,
	// this MUST be treated as a connection error of type H3_STREAM_CREATION_ERROR."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.2.2-3
	return &connectionError{
		code:    errH3StreamCreationError,
		message: "client created push stream",
	}
}

type pseudoHeader struct {
	method    string
	scheme    string
	path      string
	authority string
}

func (sc *serverConn) parseHeader(st *stream) (http.Header, pseudoHeader, error) {
	ftype, err := st.readFrameHeader()
	if err != nil {
		return nil, pseudoHeader{}, err
	}
	if ftype != frameTypeHeaders {
		return nil, pseudoHeader{}, err
	}
	header := make(http.Header)
	var pHeader pseudoHeader
	var dec qpackDecoder
	if err := dec.decode(st, func(_ indexType, name, value string) error {
		switch name {
		case ":method":
			pHeader.method = value
		case ":scheme":
			pHeader.scheme = value
		case ":path":
			pHeader.path = value
		case ":authority":
			pHeader.authority = value
		default:
			header.Add(name, value)
		}
		return nil
	}); err != nil {
		return nil, pseudoHeader{}, err
	}
	if err := st.endFrame(); err != nil {
		return nil, pseudoHeader{}, err
	}
	return header, pHeader, nil
}

func (sc *serverConn) sendGoaway() {
	sc.mu.Lock()
	if sc.goawaySent || sc.controlStream == nil {
		sc.mu.Unlock()
		return
	}
	sc.goawaySent = true
	sc.mu.Unlock()

	// No lock in this section in case writing to stream blocks. This is safe
	// since sc.maxRequestStreamID is only updated when sc.goawaySent is false.
	sc.controlStream.writeVarint(int64(frameTypeGoaway))
	sc.controlStream.writeVarint(int64(sizeVarint(uint64(sc.maxRequestStreamID))))
	sc.controlStream.writeVarint(sc.maxRequestStreamID)
	sc.controlStream.Flush()
}

// requestShouldGoAway returns true if st has a stream ID that is equal or
// greater than the ID we have sent in a GOAWAY frame, if any.
func (sc *serverConn) requestShouldGoaway(st *stream) bool {
	sc.mu.Lock()
	defer sc.mu.Unlock()
	if sc.goawaySent {
		return st.stream.ID() >= sc.maxRequestStreamID
	} else {
		sc.maxRequestStreamID = max(sc.maxRequestStreamID, st.stream.ID())
		return false
	}
}

func (sc *serverConn) handleRequestStream(st *stream) error {
	if sc.requestShouldGoaway(st) {
		return &streamError{
			code:    errH3RequestRejected,
			message: "GOAWAY request with equal or lower ID than the stream has been sent",
		}
	}
	header, pHeader, err := sc.parseHeader(st)
	if err != nil {
		return err
	}

	reqInfo := httpcommon.NewServerRequest(httpcommon.ServerRequestParam{
		Method:    pHeader.method,
		Scheme:    pHeader.scheme,
		Authority: pHeader.authority,
		Path:      pHeader.path,
		Header:    header,
	})
	if reqInfo.InvalidReason != "" {
		return &streamError{
			code:    errH3MessageError,
			message: reqInfo.InvalidReason,
		}
	}

	var body io.ReadCloser
	contentLength := int64(-1)
	if n, err := strconv.Atoi(header.Get("Content-Length")); err == nil {
		contentLength = int64(n)
	}
	if contentLength != 0 || len(reqInfo.Trailer) != 0 {
		body = &bodyReader{
			st:      st,
			remain:  contentLength,
			trailer: reqInfo.Trailer,
		}
	} else {
		body = http.NoBody
	}

	req := &http.Request{
		Proto:         "HTTP/3.0",
		Method:        pHeader.method,
		Host:          pHeader.authority,
		URL:           reqInfo.URL,
		RequestURI:    reqInfo.RequestURI,
		Trailer:       reqInfo.Trailer,
		ProtoMajor:    3,
		RemoteAddr:    sc.qconn.RemoteAddr().String(),
		Body:          body,
		Header:        header,
		ContentLength: contentLength,
	}
	defer req.Body.Close()

	rw := &responseWriter{
		st:             st,
		headers:        make(http.Header),
		trailer:        make(http.Header),
		bb:             make(bodyBuffer, 0, defaultBodyBufferCap),
		cannotHaveBody: req.Method == "HEAD",
		bw: &bodyWriter{
			st:     st,
			remain: -1,
			flush:  false,
			name:   "response",
			enc:    &sc.enc,
		},
	}
	defer rw.close()
	if reqInfo.NeedsContinue {
		req.Body.(*bodyReader).send100Continue = func() {
			rw.WriteHeader(100)
		}
	}

	// TODO: handle panic coming from the HTTP handler.
	sc.handler.ServeHTTP(rw, req)
	return nil
}

// abort closes the connection with an error.
func (sc *serverConn) abort(err error) {
	if e, ok := err.(*connectionError); ok {
		sc.qconn.Abort(&quic.ApplicationError{
			Code:   uint64(e.code),
			Reason: e.message,
		})
	} else {
		sc.qconn.Abort(err)
	}
}

// responseCanHaveBody reports whether a given response status code permits a
// body. See RFC 7230, section 3.3.
func responseCanHaveBody(status int) bool {
	switch {
	case status >= 100 && status <= 199:
		return false
	case status == 204:
		return false
	case status == 304:
		return false
	}
	return true
}

type responseWriter struct {
	st             *stream
	bw             *bodyWriter
	mu             sync.Mutex
	headers        http.Header
	trailer        http.Header
	bb             bodyBuffer
	wroteHeader    bool // Non-1xx header has been (logically) written.
	statusCode     int  // Status of the response that will be sent in HEADERS frame.
	statusCodeSet  bool // Status of the response has been set via a call to WriteHeader.
	cannotHaveBody bool // Response should not have a body (e.g. response to a HEAD request).
	bodyLenLeft    int  // How much of the content body is left to be sent, set via "Content-Length" header. -1 if unknown.
}

func (rw *responseWriter) Header() http.Header {
	return rw.headers
}

// prepareTrailerForWriteLocked populates any pre-declared trailer header with
// its value, and passes it to bodyWriter so it can be written after body EOF.
// Caller must hold rw.mu.
func (rw *responseWriter) prepareTrailerForWriteLocked() {
	for name := range rw.trailer {
		if val, ok := rw.headers[name]; ok {
			rw.trailer[name] = val
		} else {
			delete(rw.trailer, name)
		}
	}
	if len(rw.trailer) > 0 {
		rw.bw.trailer = rw.trailer
	}
}

// writeHeaderLockedOnce writes the final response header. If rw.wroteHeader is
// true, calling this method is a no-op. Sending informational status headers
// should be done using writeInfoHeaderLocked, rather than this method.
// Caller must hold rw.mu.
func (rw *responseWriter) writeHeaderLockedOnce() {
	if rw.wroteHeader {
		return
	}
	if !responseCanHaveBody(rw.statusCode) {
		rw.cannotHaveBody = true
	}
	// If there is any Trailer declared in headers, save them so we know which
	// trailers have been pre-declared. Also, write back the extracted value,
	// which is canonicalized, to rw.Header for consistency.
	if _, ok := rw.headers["Trailer"]; ok {
		extractTrailerFromHeader(rw.headers, rw.trailer)
		rw.headers.Set("Trailer", strings.Join(slices.Sorted(maps.Keys(rw.trailer)), ", "))
	}

	rw.bb.inferHeader(rw.headers, rw.statusCode)
	encHeaders := rw.bw.enc.encode(func(f func(itype indexType, name, value string)) {
		f(mayIndex, ":status", strconv.Itoa(rw.statusCode))
		for name, values := range rw.headers {
			if !httpguts.ValidHeaderFieldName(name) {
				continue
			}
			for _, val := range values {
				if !httpguts.ValidHeaderFieldValue(val) {
					continue
				}
				// Issue #71374: Consider supporting never-indexed fields.
				f(mayIndex, name, val)
			}
		}
	})

	rw.st.writeVarint(int64(frameTypeHeaders))
	rw.st.writeVarint(int64(len(encHeaders)))
	rw.st.Write(encHeaders)
	rw.wroteHeader = true
}

// writeHeaderLocked writes informational status headers (i.e. status 1XX).
// If a non-informational status header has been written via
// writeHeaderLockedOnce, this method is a no-op.
// Caller must hold rw.mu.
func (rw *responseWriter) writeHeaderLocked(statusCode int) {
	if rw.wroteHeader {
		return
	}
	encHeaders := rw.bw.enc.encode(func(f func(itype indexType, name, value string)) {
		f(mayIndex, ":status", strconv.Itoa(statusCode))
		for name, values := range rw.headers {
			if name == "Content-Length" || name == "Transfer-Encoding" {
				continue
			}
			if !httpguts.ValidHeaderFieldName(name) {
				continue
			}
			for _, val := range values {
				if !httpguts.ValidHeaderFieldValue(val) {
					continue
				}
				// Issue #71374: Consider supporting never-indexed fields.
				f(mayIndex, name, val)
			}
		}
	})
	rw.st.writeVarint(int64(frameTypeHeaders))
	rw.st.writeVarint(int64(len(encHeaders)))
	rw.st.Write(encHeaders)
}

func isInfoStatus(status int) bool {
	return status >= 100 && status < 200
}

// checkWriteHeaderCode is a copy of net/http's checkWriteHeaderCode.
func checkWriteHeaderCode(code int) {
	// Issue 22880: require valid WriteHeader status codes.
	// For now we only enforce that it's three digits.
	// In the future we might block things over 599 (600 and above aren't defined
	// at http://httpwg.org/specs/rfc7231.html#status.codes).
	// But for now any three digits.
	//
	// We used to send "HTTP/1.1 000 0" on the wire in responses but there's
	// no equivalent bogus thing we can realistically send in HTTP/3,
	// so we'll consistently panic instead and help people find their bugs
	// early. (We can't return an error from WriteHeader even if we wanted to.)
	if code < 100 || code > 999 {
		panic(fmt.Sprintf("invalid WriteHeader code %v", code))
	}
}

func (rw *responseWriter) WriteHeader(statusCode int) {
	// TODO: handle sending informational status headers (e.g. 103).
	rw.mu.Lock()
	defer rw.mu.Unlock()
	if rw.statusCodeSet {
		return
	}
	checkWriteHeaderCode(statusCode)

	// Informational headers can be sent multiple times, and should be flushed
	// immediately.
	if isInfoStatus(statusCode) {
		rw.writeHeaderLocked(statusCode)
		rw.st.Flush()
		return
	}

	// Non-informational headers should only be set once, and should be
	// buffered.
	rw.statusCodeSet = true
	rw.statusCode = statusCode
	if n, err := strconv.Atoi(rw.Header().Get("Content-Length")); err == nil {
		rw.bodyLenLeft = n
	} else {
		rw.bodyLenLeft = -1 // Unknown.
	}
}

// trimWriteLocked trims a byte slice, b, such that the length of b will not
// exceed rw.bodyLenLeft. This method will update rw.bodyLenLeft when trimming
// b, and will also return whether b was trimmed or not.
// Caller must hold rw.mu.
func (rw *responseWriter) trimWriteLocked(b []byte) ([]byte, bool) {
	if rw.bodyLenLeft < 0 {
		return b, false
	}
	n := min(len(b), rw.bodyLenLeft)
	rw.bodyLenLeft -= n
	return b[:n], n != len(b)
}

func (rw *responseWriter) Write(b []byte) (n int, err error) {
	// Calling Write implicitly calls WriteHeader(200) if WriteHeader has not
	// been called before.
	rw.WriteHeader(http.StatusOK)
	rw.mu.Lock()
	defer rw.mu.Unlock()

	if rw.statusCode == http.StatusNotModified {
		return 0, http.ErrBodyNotAllowed
	}

	b, trimmed := rw.trimWriteLocked(b)
	if trimmed {
		defer func() {
			err = http.ErrContentLength
		}()
	}

	// If b fits entirely in our body buffer, save it to the buffer and return
	// early so we can coalesce small writes.
	// As a special case, we always want to save b to the buffer even when b is
	// big if we had yet to write our header, so we can infer headers like
	// "Content-Type" with as much information as possible.
	initialBLen := len(b)
	initialBufLen := len(rw.bb)
	if !rw.wroteHeader || len(b) <= cap(rw.bb)-len(rw.bb) {
		b = rw.bb.write(b)
		if len(b) == 0 {
			return initialBLen, nil
		}
	}

	// Reaching this point means that our buffer has been sufficiently filled.
	// Therefore, we now want to:
	// 1. Infer and write response headers based on our body buffer, if not
	// done yet.
	// 2. Write our body buffer and the rest of b (if any).
	// 3. Reset the current body buffer so it can be used again.
	rw.writeHeaderLockedOnce()
	if rw.cannotHaveBody {
		return initialBLen, nil
	}
	if n, err := rw.bw.write(rw.bb, b); err != nil {
		return max(0, n-initialBufLen), err
	}
	rw.bb.discard()
	return initialBLen, nil
}

func (rw *responseWriter) Flush() {
	// Calling Flush implicitly calls WriteHeader(200) if WriteHeader has not
	// been called before.
	rw.WriteHeader(http.StatusOK)
	rw.mu.Lock()
	defer rw.mu.Unlock()
	rw.writeHeaderLockedOnce()
	if !rw.cannotHaveBody {
		rw.bw.Write(rw.bb)
		rw.bb.discard()
	}
	rw.st.Flush()
}

func (rw *responseWriter) close() error {
	rw.Flush()
	rw.mu.Lock()
	defer rw.mu.Unlock()
	rw.prepareTrailerForWriteLocked()
	if err := rw.bw.Close(); err != nil {
		return err
	}
	return rw.st.stream.Close()
}

// defaultBodyBufferCap is the default number of bytes of body that we are
// willing to save in a buffer for the sake of inferring headers and coalescing
// small writes. 512 was chosen to be consistent with how much
// http.DetectContentType is willing to read.
const defaultBodyBufferCap = 512

// bodyBuffer is a buffer used to store body content of a response.
type bodyBuffer []byte

// write writes b to the buffer. It returns a new slice of b, which contains
// any remaining data that could not be written to the buffer, if any.
func (bb *bodyBuffer) write(b []byte) []byte {
	n := min(len(b), cap(*bb)-len(*bb))
	*bb = append(*bb, b[:n]...)
	return b[n:]
}

// discard resets the buffer so it can be used again.
func (bb *bodyBuffer) discard() {
	*bb = (*bb)[:0]
}

// inferHeader populates h with the header values that we can infer from our
// current buffer content, if not already explicitly set. This method should be
// called only once with as much body content as possible in the buffer, before
// a HEADERS frame is sent, and before discard has been called. Doing so
// properly is the responsibility of the caller.
func (bb *bodyBuffer) inferHeader(h http.Header, status int) {
	if _, ok := h["Date"]; !ok {
		h.Set("Date", time.Now().UTC().Format(http.TimeFormat))
	}
	// If the Content-Encoding is non-blank, we shouldn't
	// sniff the body. See Issue golang.org/issue/31753.
	_, hasCE := h["Content-Encoding"]
	_, hasCT := h["Content-Type"]
	if !hasCE && !hasCT && responseCanHaveBody(status) && len(*bb) > 0 {
		h.Set("Content-Type", http.DetectContentType(*bb))
	}
	// We can technically infer Content-Length too here, as long as the entire
	// response body fits within hi.buf and does not require flushing. However,
	// we have chosen not to do so for now as Content-Length is not very
	// important for HTTP/3, and such inconsistent behavior might be confusing.
}
