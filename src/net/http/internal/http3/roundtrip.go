// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptrace"
	"net/http/internal"
	"net/http/internal/ascii"
	"net/http/internal/httpcommon"
	"net/textproto"
	"strconv"
	"strings"
	"sync"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/quic"
)

type roundTripState struct {
	cc *clientConn
	st *stream

	// Request body, provided by the caller.
	onceCloseReqBody sync.Once
	reqBody          io.ReadCloser

	reqBodyWriter bodyWriter

	// Response.Body, provided to the caller.
	respBody io.ReadCloser

	trace       *httptrace.ClientTrace
	headerSize  int64
	trailerSize int64

	mu               sync.Mutex
	active           bool
	done             roundTripDone
	hasUnsent100Body bool
	err              error
}

// abort terminates the RoundTrip.
// It returns the first fatal error encountered by the RoundTrip call.
func (rt *roundTripState) abort(err error) error {
	if err, done := func() (error, bool) {
		rt.mu.Lock()
		defer rt.mu.Unlock()
		if rt.err != nil {
			return rt.err, true
		}
		if !rt.active {
			// Already cleanly finished.
			rt.err = err
			return err, true
		}
		rt.err = err
		rt.active = false
		return err, false
	}(); done {
		return err
	}
	switch e := err.(type) {
	case *connectionError:
		rt.cc.abort(e)
	case *streamError:
		rt.st.CloseRead(uint64(e.code))
		rt.st.Reset(uint64(e.code))
	default:
		rt.st.CloseRead(uint64(errH3NoError))
		rt.st.Reset(uint64(errH3NoError))
	}
	rt.cc.decrActiveAndCallStateHook()
	return err
}

type roundTripDone uint8

const (
	roundTripReadDone roundTripDone = 1 << iota
	roundTripWriteDone
)

func (rt *roundTripState) finish(done roundTripDone) {
	const all = roundTripReadDone | roundTripWriteDone
	finished := false
	unsent100 := false
	func() {
		rt.mu.Lock()
		defer rt.mu.Unlock()
		rt.done |= done
		if rt.active && rt.done&all == all {
			finished = true
			rt.active = false
			unsent100 = rt.hasUnsent100Body
		}
	}()
	if !finished {
		return
	}
	if unsent100 {
		rt.st.Reset(uint64(errH3NoError))
	}
	rt.cc.decrActiveAndCallStateHook()
}

// closeReqBody closes the Request.Body, at most once.
func (rt *roundTripState) closeReqBody() {
	if rt.reqBody != nil {
		rt.onceCloseReqBody.Do(func() {
			rt.reqBody.Close()
		})
	}
}

// TODO: Set up the rest of the hooks that might be in rt.trace.
func (rt *roundTripState) maybeCallGot1xxResponse(status int, h http.Header) error {
	if rt.trace == nil || rt.trace.Got1xxResponse == nil {
		return nil
	}
	return rt.trace.Got1xxResponse(status, textproto.MIMEHeader(h))
}

func (rt *roundTripState) maybeCallGot100Continue() {
	if rt.trace == nil || rt.trace.Got100Continue == nil {
		return
	}
	rt.trace.Got100Continue()
}

func (rt *roundTripState) maybeCallWait100Continue() {
	if rt.trace == nil || rt.trace.Wait100Continue == nil {
		return
	}
	rt.trace.Wait100Continue()
}

// RoundTrip sends a request on the connection.
func (cc *clientConn) RoundTrip(req *http.Request) (_ *http.Response, err error) {
	cc.mu.Lock()
	wasReserved := false
	if cc.reserved > 0 {
		cc.reserved--
		wasReserved = true
	}
	cc.active++
	cc.mu.Unlock()

	// Each request gets its own QUIC stream.
	st, err := newConnStream(req.Context(), cc.qconn, streamTypeRequest)
	if err != nil {
		cc.mu.Lock()
		cc.active--
		closed := cc.closed
		cc.mu.Unlock()
		if wasReserved && !closed {
			cc.maybeCallStateHook()
		}
		return nil, err
	}
	rt := &roundTripState{
		cc:      cc,
		st:      st,
		trace:   httptrace.ContextClientTrace(req.Context()),
		reqBody: req.Body,
		active:  true,
	}
	if rt.reqBody == nil {
		rt.reqBody = http.NoBody
	}
	// wg tracks the writeBodyAndTrailer goroutine, if we start one.
	var wg sync.WaitGroup
	defer func() {
		if err != nil {
			err = rt.abort(err)

			// Close the request body, and wait for writeBodyAndTrailer to
			// finish with it, before returning.
			//
			// Closing the body here wakes up writeBodyAndTrailer if it is
			// blocked reading from it; abort has already reset the stream,
			// so a blocked write fails rather than hanging.
			//
			// net/http inspects the request body as soon as RoundTrip returns,
			// to see whether it was read from or closed, so the close has to
			// happen before we return rather than concurrently with the
			// caller. The HTTP/2 transport does the same thing;
			// see golang/go#60041.
			rt.closeReqBody()
			wg.Wait()
		}
	}()

	// Cancel reads/writes on the stream when the request expires.
	st.stream.SetReadContext(req.Context())
	st.stream.SetWriteContext(req.Context())

	addedGzip := httpcommon.IsRequestGzip(req.Method, req.Header, cc.tr.tr1.DisableCompression)
	headers := cc.enc.encode(func(yield func(itype indexType, name, value string)) {
		_, err = httpcommon.EncodeHeaders(req.Context(), httpcommon.EncodeHeadersParam{
			Request: httpcommon.Request{
				URL:                 req.URL,
				Method:              req.Method,
				Host:                req.Host,
				Header:              req.Header,
				Trailer:             req.Trailer,
				ActualContentLength: actualContentLength(req),
			},
			AddGzipHeader:         addedGzip,
			PeerMaxHeaderListSize: 0,
			DefaultUserAgent:      "Go-http-client/3.0",
		}, func(name, value string) {
			// Issue #71374: Consider supporting never-indexed fields.
			yield(mayIndex, name, value)
		})
	})
	if err != nil {
		return nil, err
	}

	// Write the HEADERS frame.
	st.writeVarint(int64(frameTypeHeaders))
	st.writeVarint(int64(len(headers)))
	st.Write(headers)

	is100ContinueReq := httpguts.HeaderValuesContainsToken(req.Header["Expect"], "100-continue")
	switch {
	case rt.reqBody == http.NoBody && len(req.Trailer) == 0:
		st.CloseWrite()
		// rt.done is normally synchronized via rt.mu, but nothing else can touch it yet.
		rt.done |= roundTripWriteDone
	case is100ContinueReq:
		if err := st.Flush(); err != nil {
			return nil, err
		}
		rt.hasUnsent100Body = true
		rt.maybeCallWait100Continue()
	default:
		if err := st.Flush(); err != nil {
			return nil, err
		}
		wg.Go(func() { cc.writeBodyAndTrailer(rt, req) })
	}

	// Read the response headers.
	for {
		ftype, err := st.readFrameHeader()
		if err != nil {
			return nil, err
		}
		switch ftype {
		case frameTypeHeaders:
			statusCode, h, err := rt.handleHeaders()
			if err != nil {
				return nil, err
			}

			if isInfoStatus(statusCode) {
				if statusCode == 101 {
					return nil, &streamError{errH3MessageError, "unsupported 101 response"}
				}
				if err := rt.maybeCallGot1xxResponse(statusCode, h); err != nil {
					return nil, err
				}
				if statusCode == 100 {
					rt.maybeCallGot100Continue()
					if rt.hasUnsent100Body {
						rt.hasUnsent100Body = false
						wg.Go(func() { cc.writeBodyAndTrailer(rt, req) })
					}
				}
				continue
			}

			if rt.hasUnsent100Body {
				// We're never going to send the body.
				rt.closeReqBody()
				rt.finish(roundTripWriteDone)
			}

			// We have the response headers.
			// Set up the response and return it to the caller.
			contentLength, err := parseResponseContentLength(req.Method, statusCode, h)
			if err != nil {
				return nil, err
			}

			trailer := make(http.Header)
			extractTrailerFromHeader(h, trailer)
			delete(h, "Trailer")

			hasBody := true
			if internal.BodyAllowedForStatus(statusCode) && req.Method != http.MethodHead && (contentLength != 0 || len(trailer) > 0) {
				rt.respBody = &bodyReader{
					st:             st,
					remain:         contentLength,
					trailer:        trailer,
					maxHeaderBytes: cc.tr.maxResponseHeaderBytes(),
				}
			} else {
				hasBody = false
				rt.respBody = http.NoBody
				st.CloseRead(uint64(errH3NoError))
				rt.finish(roundTripReadDone)
			}
			resp := &http.Response{
				Proto:         "HTTP/3.0",
				ProtoMajor:    3,
				Header:        h,
				StatusCode:    statusCode,
				Status:        strconv.Itoa(statusCode) + " " + http.StatusText(statusCode),
				ContentLength: contentLength,
				Trailer:       trailer,
				Body:          rt.respBody,
			}
			if hasBody {
				resp.Body = (*transportResponseBody)(rt)
			}
			if addedGzip && hasBody && ascii.EqualFold(h.Get("Content-Encoding"), "gzip") {
				resp.Body = &httpcommon.GzipReader{Body: resp.Body}
				h.Del("Content-Encoding")
				h.Del("Content-Length")
				resp.ContentLength = -1
				resp.Uncompressed = true
			}
			return resp, nil
		case frameTypePushPromise:
			if err := cc.handlePushPromise(st); err != nil {
				return nil, err
			}
		default:
			if err := st.discardUnknownFrame(ftype); err != nil {
				return nil, err
			}
		}
	}
}

// actualContentLength returns a sanitized version of req.ContentLength,
// where 0 actually means zero (not unknown) and -1 means unknown.
func actualContentLength(req *http.Request) int64 {
	if req.Body == nil || req.Body == http.NoBody {
		return 0
	}
	if req.ContentLength != 0 {
		return req.ContentLength
	}
	return -1
}

// reqBodyIgnored reports whether err is an error caused by the server
// requesting that the client stop sending the request body. Per RFC 9114
// Section 4.1, a server can send a complete response prior to the client
// sending an entire request if the response does not depend on any portion of
// the unsent request. When this happens, the server will use the H3_NO_ERROR
// code, and the client MUST NOT discard the response.
func reqBodyIgnored(err error) bool {
	if streamErr, ok := errors.AsType[quic.StreamError](err); ok {
		return http3Error(streamErr) == errH3NoError
	}
	return false
}

// writeBodyAndTrailer handles writing the body and trailer for a given
// request, if any. This function will close the write direction of the stream.
func (cc *clientConn) writeBodyAndTrailer(rt *roundTripState, req *http.Request) {
	defer rt.finish(roundTripWriteDone)
	defer rt.closeReqBody()

	declaredTrailer := req.Trailer.Clone()

	rt.reqBodyWriter.st = rt.st
	rt.reqBodyWriter.remain = actualContentLength(req)
	rt.reqBodyWriter.flush = true
	rt.reqBodyWriter.name = "request"
	rt.reqBodyWriter.trailer = req.Trailer
	rt.reqBodyWriter.enc = &cc.enc

	if _, err := io.Copy(&rt.reqBodyWriter, rt.reqBody); err != nil {
		if reqBodyIgnored(err) {
			return
		}
		rt.abort(err)
		return
	}
	// Get rid of any trailer that was not declared beforehand, before we
	// close the request body which will cause the trailer headers to be
	// written.
	for name := range req.Trailer {
		if _, ok := declaredTrailer[name]; !ok {
			delete(req.Trailer, name)
		}
	}
	if err := rt.reqBodyWriter.Close(); err != nil {
		rt.abort(err)
	}
}

// transportResponseBody is the Response.Body returned by RoundTrip.
type transportResponseBody roundTripState

// Read is Response.Body.Read.
func (b *transportResponseBody) Read(p []byte) (n int, err error) {
	rt := (*roundTripState)(b)
	n, err = b.respBody.Read(p)
	if err == io.EOF {
		rt.finish(roundTripReadDone)
	} else if err != nil {
		err = rt.abort(err)
		rt.closeReqBody()
	}
	return n, err
}

var errRespBodyClosed = errors.New("response body closed")

// Close is Response.Body.Close.
// Closing the response body is how the caller signals that they're done with a request.
func (b *transportResponseBody) Close() error {
	rt := (*roundTripState)(b)
	// respBody.Close is responsible for closing the receiving half.
	err := rt.abort(errRespBodyClosed)
	// respBody is only *bodyReader or NoBody, neither of which return errors on close.
	rt.respBody.Close()
	// Close the request body, which should wake up writeBodyAndTrailer if it's
	// currently blocked reading the body.
	rt.closeReqBody()
	if err == errRespBodyClosed {
		// No other errors occurred before closing Response.Body,
		// so consider this a successful request.
		return nil
	}
	return err
}

func parseResponseContentLength(method string, statusCode int, h http.Header) (int64, error) {
	clens := h["Content-Length"]
	if len(clens) == 0 {
		return -1, nil
	}

	// We allow duplicate Content-Length headers,
	// but only if they all have the same value.
	for _, v := range clens[1:] {
		if clens[0] != v {
			return -1, &streamError{errH3MessageError, "mismatching Content-Length headers"}
		}
	}

	// "A server MUST NOT send a Content-Length header field in any response
	// with a status code of 1xx (Informational) or 204 (No Content).
	// A server MUST NOT send a Content-Length header field in any 2xx (Successful)
	// response to a CONNECT request [...]"
	// https://www.rfc-editor.org/rfc/rfc9110#section-8.6-8
	if (statusCode >= 100 && statusCode < 200) ||
		statusCode == 204 ||
		(method == "CONNECT" && statusCode >= 200 && statusCode < 300) {
		// This is a protocol violation, but a fairly harmless one.
		// Just ignore the header.
		return -1, nil
	}

	contentLen, err := strconv.ParseUint(clens[0], 10, 63)
	if err != nil {
		return -1, &streamError{errH3MessageError, "invalid Content-Length header"}
	}
	return int64(contentLen), nil
}

func (rt *roundTripState) handleHeaders() (statusCode int, h http.Header, err error) {
	maxHeaderBytes := rt.cc.tr.maxResponseHeaderBytes()
	haveStatus := false
	cookie := ""
	// The QPACK decoder will happily allocate an unbounded buffer, so check
	// the stream read limit prior to decoding.
	if rt.st.lim > maxHeaderBytes-rt.headerSize {
		return 0, nil, errHeadersTooLarge
	}
	// Issue #71374: Consider tracking the never-indexed status of headers
	// with the N bit set in their QPACK encoding.
	err = rt.cc.dec.decode(rt.st, func(_ indexType, name, value string) error {
		if name == "trailer" {
			fieldCount := int64(strings.Count(value, ",") + 1)
			rt.trailerSize += int64(len(value)) - fieldCount + 1 + fieldCount*headerFieldOverhead
		} else {
			rt.headerSize += int64(len(name)) + int64(len(value)) + headerFieldOverhead
		}
		if rt.headerSize > maxHeaderBytes || rt.trailerSize > maxHeaderBytes {
			return errHeadersTooLarge
		}
		if !httpguts.ValidHeaderFieldValue(value) {
			return &streamError{errH3MessageError, "invalid field value"}
		}
		switch {
		case name == ":status":
			if haveStatus {
				return &streamError{errH3MessageError, "duplicate :status"}
			}
			haveStatus = true
			statusCode, err = strconv.Atoi(value)
			if err != nil {
				return &streamError{errH3MessageError, "invalid :status"}
			}
		case name[0] == ':':
			// "Endpoints MUST treat a request or response
			// that contains undefined or invalid
			// pseudo-header fields as malformed."
			// https://www.rfc-editor.org/rfc/rfc9114.html#section-4.3-3
			return &streamError{errH3MessageError, "undefined pseudo-header"}
		case name == "cookie":
			// "If a decompressed field section contains multiple cookie field lines,
			// these MUST be concatenated into a single byte string [...]"
			// using the two-byte delimiter of "; "''
			// https://www.rfc-editor.org/rfc/rfc9114.html#section-4.2.1-2
			if cookie == "" {
				cookie = value
			} else {
				cookie += "; " + value
			}
		default:
			if !validWireHeaderFieldName(name) {
				return &streamError{errH3MessageError, "invalid field name"}
			}
			if h == nil {
				h = make(http.Header)
			}
			// TODO: Use a per-connection canonicalization cache as we do in HTTP/2.
			// Maybe we could put this in the QPACK decoder and have it deliver
			// pre-canonicalized headers to us here?
			cname := httpcommon.CanonicalHeader(name)
			// TODO: Consider using a single []string slice for all headers,
			// as we do in the HTTP/1 and HTTP/2 cases.
			// This is a bit tricky, since we don't know the number of headers
			// at the start of decoding. Perhaps it's worth doing a two-pass decode,
			// or perhaps we should just allocate header value slices in
			// reasonably-sized chunks.
			h[cname] = append(h[cname], value)
		}
		return nil
	})
	if err != nil {
		return 0, nil, err
	}
	if !haveStatus {
		// "[The :status] pseudo-header field MUST be included in all responses [...]"
		// https://www.rfc-editor.org/rfc/rfc9114.html#section-4.3.2-1
		return 0, nil, errH3MessageError
	}
	if cookie != "" {
		if h == nil {
			h = make(http.Header)
		}
		h["Cookie"] = []string{cookie}
	}
	if err := rt.st.endFrame(); err != nil {
		return 0, nil, err
	}
	// go.dev/issue/65035.
	if isInfoStatus(statusCode) && rt.trace != nil && rt.trace.Got1xxResponse != nil {
		rt.headerSize = 0
		rt.trailerSize = 0
	}
	return statusCode, h, nil
}

func (cc *clientConn) handlePushPromise(st *stream) error {
	// "A client MUST treat receipt of a PUSH_PROMISE frame that contains a
	// larger push ID than the client has advertised as a connection error of H3_ID_ERROR."
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.5-5
	return &connectionError{
		code:    errH3IDError,
		message: "PUSH_PROMISE received when no MAX_PUSH_ID has been sent",
	}
}
