// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"context"
	"io"

	"golang.org/x/net/quic"
)

// A stream wraps a QUIC stream, providing methods to read/write various values.
type stream struct {
	stream *quic.Stream

	// lim is the current read limit.
	// Reading a frame header sets the limit to the end of the frame.
	// Reading past the limit or reading less than the limit and ending the frame
	// results in an error.
	// -1 indicates no limit.
	lim int64
}

// newConnStream creates a new stream on a connection.
// It writes the stream header for unidirectional streams.
//
// The stream returned by newStream is not flushed,
// and will not be sent to the peer until the caller calls
// Flush or writes enough data to the stream.
func newConnStream(ctx context.Context, qconn *quic.Conn, stype streamType) (*stream, error) {
	var qs *quic.Stream
	var err error
	if stype == streamTypeRequest {
		// Request streams are bidirectional.
		qs, err = qconn.NewStream(ctx)
	} else {
		// All other streams are unidirectional.
		qs, err = qconn.NewSendOnlyStream(ctx)
	}
	if err != nil {
		return nil, err
	}
	st := &stream{
		stream: qs,
		lim:    -1, // no limit
	}
	if stype != streamTypeRequest {
		// Unidirectional stream header.
		st.writeVarint(int64(stype))
	}
	return st, err
}

func newStream(qs *quic.Stream) *stream {
	return &stream{
		stream: qs,
		lim:    -1, // no limit
	}
}

// readFrameHeader reads the type and length fields of an HTTP/3 frame.
// It sets the read limit to the end of the frame.
//
// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.1
func (st *stream) readFrameHeader() (ftype frameType, err error) {
	if st.lim >= 0 {
		// We shouldn't call readFrameHeader before ending the previous frame.
		return 0, errH3FrameError
	}
	ftype, err = readVarint[frameType](st)
	if err != nil {
		return 0, err
	}
	size, err := st.readVarint()
	if err != nil {
		return 0, err
	}
	st.lim = size
	return ftype, nil
}

// endFrame is called after reading a frame to reset the read limit.
// It returns an error if the entire contents of a frame have not been read.
func (st *stream) endFrame() error {
	if st.lim != 0 {
		return &connectionError{
			code:    errH3FrameError,
			message: "invalid HTTP/3 frame",
		}
	}
	st.lim = -1
	return nil
}

// readFrameData returns the remaining data in the current frame.
func (st *stream) readFrameData() ([]byte, error) {
	if st.lim < 0 {
		return nil, errH3FrameError
	}
	// TODO: Pool buffers to avoid allocation here.
	b := make([]byte, st.lim)
	_, err := io.ReadFull(st, b)
	if err != nil {
		return nil, err
	}
	return b, nil
}

// ReadByte reads one byte from the stream.
func (st *stream) ReadByte() (b byte, err error) {
	if err := st.recordBytesRead(1); err != nil {
		return 0, err
	}
	b, err = st.stream.ReadByte()
	if err != nil {
		if err == io.EOF && st.lim < 0 {
			return 0, io.EOF
		}
		return 0, errH3FrameError
	}
	return b, nil
}

// Read reads from the stream.
func (st *stream) Read(b []byte) (int, error) {
	n, err := st.stream.Read(b)
	if e2 := st.recordBytesRead(n); e2 != nil {
		return 0, e2
	}
	if err == io.EOF {
		if st.lim == 0 {
			// EOF at end of frame, ignore.
			return n, nil
		} else if st.lim > 0 {
			// EOF inside frame, error.
			return 0, errH3FrameError
		} else {
			// EOF outside of frame, surface to caller.
			return n, io.EOF
		}
	}
	if err != nil {
		return 0, errH3FrameError
	}
	return n, nil
}

// discardUnknownFrame discards an unknown frame.
//
// HTTP/3 requires that unknown frames be ignored on all streams.
// However, a known frame appearing in an unexpected place is a fatal error,
// so this returns an error if the frame is one we know.
func (st *stream) discardUnknownFrame(ftype frameType) error {
	switch ftype {
	case frameTypeData,
		frameTypeHeaders,
		frameTypeCancelPush,
		frameTypeSettings,
		frameTypePushPromise,
		frameTypeGoaway,
		frameTypeMaxPushID:
		return &connectionError{
			code:    errH3FrameUnexpected,
			message: "unexpected " + ftype.String() + " frame",
		}
	}
	return st.discardFrame()
}

// discardFrame discards any remaining data in the current frame and resets the read limit.
func (st *stream) discardFrame() error {
	// TODO: Consider adding a *quic.Stream method to discard some amount of data.
	for range st.lim {
		_, err := st.stream.ReadByte()
		if err != nil {
			return &streamError{errH3FrameError, err.Error()}
		}
	}
	st.lim = -1
	return nil
}

// Write writes to the stream.
func (st *stream) Write(b []byte) (int, error) { return st.stream.Write(b) }

// Flush commits data written to the stream.
func (st *stream) Flush() error { return st.stream.Flush() }

// readVarint reads a QUIC variable-length integer from the stream.
func (st *stream) readVarint() (v int64, err error) {
	b, err := st.stream.ReadByte()
	if err != nil {
		return 0, err
	}
	v = int64(b & 0x3f)
	n := 1 << (b >> 6)
	for i := 1; i < n; i++ {
		b, err := st.stream.ReadByte()
		if err != nil {
			return 0, errH3FrameError
		}
		v = (v << 8) | int64(b)
	}
	if err := st.recordBytesRead(n); err != nil {
		return 0, err
	}
	return v, nil
}

// readVarint reads a varint of a particular type.
func readVarint[T ~int64 | ~uint64](st *stream) (T, error) {
	v, err := st.readVarint()
	return T(v), err
}

// writeVarint writes a QUIC variable-length integer to the stream.
func (st *stream) writeVarint(v int64) {
	switch {
	case v <= (1<<6)-1:
		st.stream.WriteByte(byte(v))
	case v <= (1<<14)-1:
		st.stream.WriteByte((1 << 6) | byte(v>>8))
		st.stream.WriteByte(byte(v))
	case v <= (1<<30)-1:
		st.stream.WriteByte((2 << 6) | byte(v>>24))
		st.stream.WriteByte(byte(v >> 16))
		st.stream.WriteByte(byte(v >> 8))
		st.stream.WriteByte(byte(v))
	case v <= (1<<62)-1:
		st.stream.WriteByte((3 << 6) | byte(v>>56))
		st.stream.WriteByte(byte(v >> 48))
		st.stream.WriteByte(byte(v >> 40))
		st.stream.WriteByte(byte(v >> 32))
		st.stream.WriteByte(byte(v >> 24))
		st.stream.WriteByte(byte(v >> 16))
		st.stream.WriteByte(byte(v >> 8))
		st.stream.WriteByte(byte(v))
	default:
		panic("varint too large")
	}
}

// recordBytesRead records that n bytes have been read.
// It returns an error if the read passes the current limit.
func (st *stream) recordBytesRead(n int) error {
	if st.lim < 0 {
		return nil
	}
	st.lim -= int64(n)
	if st.lim < 0 {
		st.stream = nil // panic if we try to read again
		return &connectionError{
			code:    errH3FrameError,
			message: "invalid HTTP/3 frame",
		}
	}
	return nil
}
