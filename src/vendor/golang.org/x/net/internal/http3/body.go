// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/textproto"
	"strings"
	"sync"

	"golang.org/x/net/http/httpguts"
)

// extractTrailerFromHeader extracts the "Trailer" header values from a header
// map, and populates a trailer map with those values as keys. The extracted
// header values will be canonicalized.
func extractTrailerFromHeader(header, trailer http.Header) {
	for _, names := range header["Trailer"] {
		names = textproto.TrimString(names)
		for name := range strings.SplitSeq(names, ",") {
			name = textproto.CanonicalMIMEHeaderKey(textproto.TrimString(name))
			if !httpguts.ValidTrailerHeader(name) {
				continue
			}
			trailer[name] = nil
		}
	}
}

// A bodyWriter writes a request or response body to a stream
// as a series of DATA frames.
type bodyWriter struct {
	st      *stream
	remain  int64         // -1 when content-length is not known
	flush   bool          // flush the stream after every write
	name    string        // "request" or "response"
	trailer http.Header   // trailer headers that will be written once bodyWriter is closed.
	enc     *qpackEncoder // QPACK encoder used by the connection.
}

func (w *bodyWriter) write(ps ...[]byte) (n int, err error) {
	var size int64
	for _, p := range ps {
		size += int64(len(p))
	}
	// If write is called with empty byte slices, just return instead of
	// sending out a DATA frame containing nothing.
	if size == 0 {
		return 0, nil
	}
	if w.remain >= 0 && size > w.remain {
		return 0, &streamError{
			code:    errH3InternalError,
			message: w.name + " body longer than specified content length",
		}
	}
	w.st.writeVarint(int64(frameTypeData))
	w.st.writeVarint(size)
	for _, p := range ps {
		var n2 int
		n2, err = w.st.Write(p)
		n += n2
		if w.remain >= 0 {
			w.remain -= int64(n)
		}
		if err != nil {
			break
		}
	}
	if w.flush && err == nil {
		err = w.st.Flush()
	}
	if err != nil {
		err = fmt.Errorf("writing %v body: %w", w.name, err)
	}
	return n, err
}

func (w *bodyWriter) Write(p []byte) (n int, err error) {
	return w.write(p)
}

func (w *bodyWriter) Close() error {
	if w.remain > 0 {
		return errors.New(w.name + " body shorter than specified content length")
	}
	if len(w.trailer) > 0 {
		encTrailer := w.enc.encode(func(f func(itype indexType, name, value string)) {
			for name, values := range w.trailer {
				if !httpguts.ValidHeaderFieldName(name) {
					continue
				}
				for _, val := range values {
					if !httpguts.ValidHeaderFieldValue(val) {
						continue
					}
					f(mayIndex, name, val)
				}
			}
		})
		w.st.writeVarint(int64(frameTypeHeaders))
		w.st.writeVarint(int64(len(encTrailer)))
		w.st.Write(encTrailer)
	}
	if w.st != nil && w.st.stream != nil {
		w.st.stream.CloseWrite()
	}
	return nil
}

// A bodyReader reads a request or response body from a stream.
type bodyReader struct {
	st *stream

	mu     sync.Mutex
	remain int64
	err    error
	// If not nil, the body contains an "Expect: 100-continue" header, and
	// send100Continue should be called when Read is invoked for the first
	// time.
	send100Continue func()
	// A map where the key represents the trailer header names we expect. If
	// there is a HEADERS frame after reading DATA frames to EOF, the value of
	// the headers will be written here, provided that the name of the header
	// exists in the map already.
	trailer http.Header
}

func (r *bodyReader) Read(p []byte) (n int, err error) {
	// The HTTP/1 and HTTP/2 implementations both permit concurrent reads from a body,
	// in the sense that the race detector won't complain.
	// Use a mutex here to provide the same behavior.
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.send100Continue != nil {
		r.send100Continue()
		r.send100Continue = nil
	}
	if r.err != nil {
		return 0, r.err
	}
	defer func() {
		if err != nil {
			r.err = err
		}
	}()
	if r.st.lim == 0 {
		// We've finished reading the previous DATA frame, so end it.
		if err := r.st.endFrame(); err != nil {
			return 0, err
		}
	}
	// Read the next DATA frame header,
	// if we aren't already in the middle of one.
	for r.st.lim < 0 {
		ftype, err := r.st.readFrameHeader()
		if err == io.EOF && r.remain > 0 {
			return 0, &streamError{
				code:    errH3MessageError,
				message: "body shorter than content-length",
			}
		}
		if err != nil {
			return 0, err
		}
		switch ftype {
		case frameTypeData:
			if r.remain >= 0 && r.st.lim > r.remain {
				return 0, &streamError{
					code:    errH3MessageError,
					message: "body longer than content-length",
				}
			}
			// Fall out of the loop and process the frame body below.
		case frameTypeHeaders:
			// This HEADERS frame contains the message trailers.
			if r.remain > 0 {
				return 0, &streamError{
					code:    errH3MessageError,
					message: "body shorter than content-length",
				}
			}
			var dec qpackDecoder
			if err := dec.decode(r.st, func(_ indexType, name, value string) error {
				if _, ok := r.trailer[name]; ok {
					r.trailer.Add(name, value)
				}
				return nil
			}); err != nil {
				return 0, err
			}
			if err := r.st.discardFrame(); err != nil {
				return 0, err
			}
			return 0, io.EOF
		default:
			if err := r.st.discardUnknownFrame(ftype); err != nil {
				return 0, err
			}
		}
	}
	// We are now reading the content of a DATA frame.
	// Fill the read buffer or read to the end of the frame,
	// whichever comes first.
	if int64(len(p)) > r.st.lim {
		p = p[:r.st.lim]
	}
	n, err = r.st.Read(p)
	if r.remain > 0 {
		r.remain -= int64(n)
	}
	return n, err
}

func (r *bodyReader) Close() error {
	// Unlike the HTTP/1 and HTTP/2 body readers (at the time of this comment being written),
	// calling Close concurrently with Read will interrupt the read.
	r.st.stream.CloseRead()
	// Make sure that any data that has already been written to bodyReader
	// cannot be read after it has been closed.
	r.err = net.ErrClosed
	r.remain = 0
	return nil
}
