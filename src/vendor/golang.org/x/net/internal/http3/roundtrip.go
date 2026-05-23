// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"io"
	"net/http"
	"net/http/httptrace"
	"net/textproto"
	"strconv"
	"sync"

	"golang.org/x/net/http/httpguts"
	"golang.org/x/net/internal/httpcommon"
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

	trace *httptrace.ClientTrace

	errOnce sync.Once
	err     error
}

// abort terminates the RoundTrip.
// It returns the first fatal error encountered by the RoundTrip call.
func (rt *roundTripState) abort(err error) error {
	rt.errOnce.Do(func() {
		rt.err = err
		switch e := err.(type) {
		case *connectionError:
			rt.cc.abort(e)
		case *streamError:
			rt.st.stream.CloseRead()
			rt.st.stream.Reset(uint64(e.code))
		default:
			rt.st.stream.CloseRead()
			rt.st.stream.Reset(uint64(errH3NoError))
		}
	})
	return rt.err
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
	// Each request gets its own QUIC stream.
	st, err := newConnStream(req.Context(), cc.qconn, streamTypeRequest)
	if err != nil {
		return nil, err
	}
	rt := &roundTripState{
		cc:      cc,
		st:      st,
		trace:   httptrace.ContextClientTrace(req.Context()),
		reqBody: req.Body,
	}
	if rt.reqBody == nil {
		rt.reqBody = http.NoBody
	}
	defer func() {
		if err != nil {
			err = rt.abort(err)
		}
	}()

	// Cancel reads/writes on the stream when the request expires.
	st.stream.SetReadContext(req.Context())
	st.stream.SetWriteContext(req.Context())

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
			AddGzipHeader:         false, // TODO: add when appropriate
			PeerMaxHeaderListSize: 0,
			DefaultUserAgent:      "Go-http-client/3",
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
	if err := st.Flush(); err != nil {
		return nil, err
	}

	var bodyAndTrailerWritten bool
	is100ContinueReq := httpguts.HeaderValuesContainsToken(req.Header["Expect"], "100-continue")
	if is100ContinueReq {
		rt.maybeCallWait100Continue()
	} else {
		bodyAndTrailerWritten = true
		go cc.writeBodyAndTrailer(rt, req)
	}

	// Read the response headers.
	for {
		ftype, err := st.readFrameHeader()
		if err != nil {
			return nil, err
		}
		switch ftype {
		case frameTypeHeaders:
			statusCode, h, err := cc.handleHeaders(st)
			if err != nil {
				return nil, err
			}

			// TODO: Handle 1xx responses.
			if isInfoStatus(statusCode) {
				if err := rt.maybeCallGot1xxResponse(statusCode, h); err != nil {
					return nil, err
				}
				switch statusCode {
				case 100:
					rt.maybeCallGot100Continue()
					if is100ContinueReq && !bodyAndTrailerWritten {
						bodyAndTrailerWritten = true
						go cc.writeBodyAndTrailer(rt, req)
						continue
					}
					// If we did not send "Expect: 100-continue" request but
					// received status 100 anyways, just continue per usual and
					// let the caller decide what to do with the response.
				default:
					continue
				}
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

			if (contentLength != 0 && req.Method != http.MethodHead) || len(trailer) > 0 {
				rt.respBody = &bodyReader{
					st:      st,
					remain:  contentLength,
					trailer: trailer,
				}
			} else {
				rt.respBody = http.NoBody
			}
			resp := &http.Response{
				Proto:         "HTTP/3.0",
				ProtoMajor:    3,
				Header:        h,
				StatusCode:    statusCode,
				Status:        strconv.Itoa(statusCode) + " " + http.StatusText(statusCode),
				ContentLength: contentLength,
				Trailer:       trailer,
				Body:          (*transportResponseBody)(rt),
			}
			// TODO: Automatic Content-Type: gzip decoding.
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

// writeBodyAndTrailer handles writing the body and trailer for a given
// request, if any. This function will close the write direction of the stream.
func (cc *clientConn) writeBodyAndTrailer(rt *roundTripState, req *http.Request) {
	defer rt.closeReqBody()

	declaredTrailer := req.Trailer.Clone()

	rt.reqBodyWriter.st = rt.st
	rt.reqBodyWriter.remain = actualContentLength(req)
	rt.reqBodyWriter.flush = true
	rt.reqBodyWriter.name = "request"
	rt.reqBodyWriter.trailer = req.Trailer
	rt.reqBodyWriter.enc = &cc.enc

	if _, err := io.Copy(&rt.reqBodyWriter, rt.reqBody); err != nil {
		rt.abort(err)
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
	return b.respBody.Read(p)
}

var errRespBodyClosed = errors.New("response body closed")

// Close is Response.Body.Close.
// Closing the response body is how the caller signals that they're done with a request.
func (b *transportResponseBody) Close() error {
	rt := (*roundTripState)(b)
	// Close the request body, which should wake up copyRequestBody if it's
	// currently blocked reading the body.
	rt.closeReqBody()
	// Close the request stream, since we're done with the request.
	// Reset closes the sending half of the stream.
	rt.st.stream.Reset(uint64(errH3NoError))
	// respBody.Close is responsible for closing the receiving half.
	err := rt.respBody.Close()
	if err == nil {
		err = errRespBodyClosed
	}
	err = rt.abort(err)
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

func (cc *clientConn) handleHeaders(st *stream) (statusCode int, h http.Header, err error) {
	haveStatus := false
	cookie := ""
	// Issue #71374: Consider tracking the never-indexed status of headers
	// with the N bit set in their QPACK encoding.
	err = cc.dec.decode(st, func(_ indexType, name, value string) error {
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
	if !haveStatus {
		// "[The :status] pseudo-header field MUST be included in all responses [...]"
		// https://www.rfc-editor.org/rfc/rfc9114.html#section-4.3.2-1
		err = errH3MessageError
	}
	if cookie != "" {
		if h == nil {
			h = make(http.Header)
		}
		h["Cookie"] = []string{cookie}
	}
	if err := st.endFrame(); err != nil {
		return 0, nil, err
	}
	return statusCode, h, err
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
