// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"context"
	"io"
	"os"
	"sync"
	"time"

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

	readDeadline  deadline
	writeDeadline deadline
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
	st := newStream(qs)
	if stype != streamTypeRequest {
		// Unidirectional stream header.
		st.writeVarint(int64(stype))
	}
	return st, err
}

func newStream(qs *quic.Stream) *stream {
	readCtx, readCancel := context.WithCancelCause(context.Background())
	writeCtx, writeCancel := context.WithCancelCause(context.Background())
	st := &stream{
		stream: qs,
		lim:    -1, // no limit
		readDeadline: deadline{
			ctx:    readCtx,
			cancel: readCancel,
		},
		writeDeadline: deadline{
			ctx:    writeCtx,
			cancel: writeCancel,
		},
	}
	qs.SetReadContext(readCtx)
	qs.SetWriteContext(writeCtx)
	return st
}

func (st *stream) Close() error {
	st.readDeadline.stop()
	st.writeDeadline.stop()
	return st.stream.Close()
}

func (st *stream) CloseRead() {
	st.readDeadline.stop()
	st.stream.CloseRead()
}

func (st *stream) CloseWrite() {
	st.writeDeadline.stop()
	st.stream.CloseWrite()
}

func (st *stream) Reset(code uint64) {
	st.readDeadline.stop()
	st.writeDeadline.stop()
	st.stream.Reset(code)
}

// deadline manages ctx, and cancels it when timer expires, with
// [os.ErrDeadlineExceeded] as the cause. If the deadline is manually stopped
// before timer expires, the context will be canceled with [context.Canceled]
// as the cause. Once a deadline is exceeded, its timer can no longer be
// extended.
// Practically, this lets the http3 package support time-based deadlines by
// utilizing the quic package's support for context-based deadlines.
type deadline struct {
	ctx    context.Context
	cancel context.CancelCauseFunc

	mu    sync.Mutex // Guards below.
	timer *time.Timer
}

// stopTimerLocked stops the deadline timer and sets it to nil.
// The caller must hold d.mu.
func (d *deadline) stopTimerLocked() {
	if d.timer != nil {
		d.timer.Stop()
		d.timer = nil
	}
}

// stop stops the deadline timer and cancels the context with
// [context.Canceled] as the cause.
func (d *deadline) stop() {
	d.mu.Lock()
	d.stopTimerLocked()
	d.mu.Unlock()
	d.cancel(context.Canceled)
}

// err returns the deadline's context cancelation cause, if any.
func (d *deadline) err() error {
	return context.Cause(d.ctx)
}

// errOf returns the deadline's context cancelation cause if the given err is
// non-nil. This can be used to check whether an error value returned by I/O
// operations at the QUIC layer is non-nil because the deadline has expired.
func (d *deadline) errOf(err error) error {
	if dErr := d.err(); err != nil && dErr != nil {
		return dErr
	}
	return err
}

// set configures a new deadline using the given deadlineTime.
// Once deadline is exceeded, it remains in the expired (sticky) state, and
// subsequent attempts to extend or reset the deadline are ignored.
func (d *deadline) set(deadlineTime time.Time) {
	d.mu.Lock()
	defer d.mu.Unlock()

	if d.ctx.Err() != nil { // Already expired, sticky error.
		return
	}
	if deadlineTime.IsZero() {
		d.stopTimerLocked()
		return
	}
	dur := time.Until(deadlineTime)
	if dur <= 0 {
		d.stopTimerLocked()
		d.cancel(os.ErrDeadlineExceeded)
		return
	}
	if d.timer == nil {
		d.timer = time.AfterFunc(dur, func() {
			d.cancel(os.ErrDeadlineExceeded)
		})
	} else {
		d.timer.Reset(dur)
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
	// Check the deadline before doing I/O operations on the QUIC layer. We do
	// this because the QUIC layer implements a fast path for I/O operations,
	// allowing Read & Write to succeed depending on the state of buffer, even
	// if its context has been canceled. By always checking the deadline here,
	// we make it so that I/O operations fail as soon as its relevant deadline
	// has been exceeded.
	if err := st.readDeadline.err(); err != nil {
		return 0, err
	}
	if err := st.recordBytesRead(1); err != nil {
		return 0, err
	}
	b, err = st.stream.ReadByte()
	if err == io.EOF && st.lim >= 0 {
		return 0, errH3FrameError
	}
	return b, st.readDeadline.errOf(err)
}

// Read reads from the stream.
func (st *stream) Read(b []byte) (int, error) {
	// Check the deadline before doing I/O operations on the QUIC layer. We do
	// this because the QUIC layer implements a fast path for I/O operations,
	// allowing Read & Write to succeed depending on the state of buffer, even
	// if its context has been canceled. By always checking the deadline here,
	// we make it so that I/O operations fail as soon as its relevant deadline
	// has been exceeded.
	if err := st.readDeadline.err(); err != nil {
		return 0, err
	}
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
	return n, st.readDeadline.errOf(err)
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
		_, err := st.ReadByte()
		if err != nil {
			return &streamError{errH3FrameError, err.Error()}
		}
	}
	st.lim = -1
	return nil
}

// Write writes to the stream.
func (st *stream) Write(b []byte) (int, error) {
	// Check the deadline before doing I/O operations on the QUIC layer. We do
	// this because the QUIC layer implements a fast path for I/O operations,
	// allowing Read & Write to succeed depending on the state of buffer, even
	// if its context has been canceled. By always checking the deadline here,
	// we make it so that I/O operations fail as soon as its relevant deadline
	// has been exceeded.
	if err := st.writeDeadline.err(); err != nil {
		return 0, err
	}
	n, err := st.stream.Write(b)
	return n, st.writeDeadline.errOf(err)
}

// Flush commits data written to the stream.
func (st *stream) Flush() error {
	// Check the deadline before doing I/O operations on the QUIC layer. We do
	// this because the QUIC layer implements a fast path for I/O operations,
	// allowing Read & Write to succeed depending on the state of buffer, even
	// if its context has been canceled. By always checking the deadline here,
	// we make it so that I/O operations fail as soon as its relevant deadline
	// has been exceeded.
	if err := st.writeDeadline.err(); err != nil {
		return err
	}
	return st.writeDeadline.errOf(st.stream.Flush())
}

// WriteByte writes one byte to the stream.
func (st *stream) WriteByte(c byte) error {
	// Check the deadline before doing I/O operations on the QUIC layer. We do
	// this because the QUIC layer implements a fast path for I/O operations,
	// allowing Read & Write to succeed depending on the state of buffer, even
	// if its context has been canceled. By always checking the deadline here,
	// we make it so that I/O operations fail as soon as its relevant deadline
	// has been exceeded.
	if err := st.writeDeadline.err(); err != nil {
		return err
	}
	return st.writeDeadline.errOf(st.stream.WriteByte(c))
}

// readVarint reads a QUIC variable-length integer from the stream.
func (st *stream) readVarint() (v int64, err error) {
	b, err := st.ReadByte()
	if err != nil {
		return 0, err
	}
	v = int64(b & 0x3f)
	n := 1 << (b >> 6)
	for i := 1; i < n; i++ {
		b, err := st.ReadByte()
		if err != nil {
			if err == io.EOF {
				return 0, errH3FrameError
			}
			return 0, err
		}
		v = (v << 8) | int64(b)
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
		st.WriteByte(byte(v))
	case v <= (1<<14)-1:
		st.WriteByte((1 << 6) | byte(v>>8))
		st.WriteByte(byte(v))
	case v <= (1<<30)-1:
		st.WriteByte((2 << 6) | byte(v>>24))
		st.WriteByte(byte(v >> 16))
		st.WriteByte(byte(v >> 8))
		st.WriteByte(byte(v))
	case v <= (1<<62)-1:
		st.WriteByte((3 << 6) | byte(v>>56))
		st.WriteByte(byte(v >> 48))
		st.WriteByte(byte(v >> 40))
		st.WriteByte(byte(v >> 32))
		st.WriteByte(byte(v >> 24))
		st.WriteByte(byte(v >> 16))
		st.WriteByte(byte(v >> 8))
		st.WriteByte(byte(v))
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
