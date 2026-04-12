// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"context"
	"io"
	"sync"

	"golang.org/x/net/quic"
)

type streamHandler interface {
	handleControlStream(*stream) error
	handlePushStream(*stream) error
	handleEncoderStream(*stream) error
	handleDecoderStream(*stream) error
	handleRequestStream(*stream) error
	abort(error)
}

type genericConn struct {
	mu sync.Mutex

	// The peer may create exactly one control, encoder, and decoder stream.
	// streamsCreated is a bitset of streams created so far.
	// Bits are 1 << streamType.
	streamsCreated uint8
}

func (c *genericConn) acceptStreams(qconn *quic.Conn, h streamHandler) {
	for {
		// Use context.Background: This blocks until a stream is accepted
		// or the connection closes.
		st, err := qconn.AcceptStream(context.Background())
		if err != nil {
			return // connection closed
		}
		if st.IsReadOnly() {
			go c.handleUnidirectionalStream(newStream(st), h)
		} else {
			go c.handleRequestStream(newStream(st), h)
		}
	}
}

func (c *genericConn) handleUnidirectionalStream(st *stream, h streamHandler) {
	// Unidirectional stream header: One varint with the stream type.
	v, err := st.readVarint()
	if err != nil {
		h.abort(&connectionError{
			code:    errH3StreamCreationError,
			message: "error reading unidirectional stream header",
		})
		return
	}
	stype := streamType(v)
	if err := c.checkStreamCreation(stype); err != nil {
		h.abort(err)
		return
	}
	switch stype {
	case streamTypeControl:
		err = h.handleControlStream(st)
	case streamTypePush:
		err = h.handlePushStream(st)
	case streamTypeEncoder:
		err = h.handleEncoderStream(st)
	case streamTypeDecoder:
		err = h.handleDecoderStream(st)
	default:
		// "Recipients of unknown stream types MUST either abort reading
		// of the stream or discard incoming data without further processing."
		// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.2-7
		//
		// We should send the H3_STREAM_CREATION_ERROR error code,
		// but the quic package currently doesn't allow setting error codes
		// for STOP_SENDING frames.
		// TODO: Should CloseRead take an error code?
		err = nil
	}
	if err == io.EOF {
		err = &connectionError{
			code:    errH3ClosedCriticalStream,
			message: streamType(stype).String() + " stream closed",
		}
	}
	c.handleStreamError(st, h, err)
}

func (c *genericConn) handleRequestStream(st *stream, h streamHandler) {
	c.handleStreamError(st, h, h.handleRequestStream(st))
}

func (c *genericConn) handleStreamError(st *stream, h streamHandler, err error) {
	switch err := err.(type) {
	case *connectionError:
		h.abort(err)
	case nil:
		st.stream.CloseRead()
		st.stream.CloseWrite()
	case *streamError:
		st.stream.CloseRead()
		st.stream.Reset(uint64(err.code))
	default:
		st.stream.CloseRead()
		st.stream.Reset(uint64(errH3InternalError))
	}
}

func (c *genericConn) checkStreamCreation(stype streamType) error {
	switch stype {
	case streamTypeControl, streamTypeEncoder, streamTypeDecoder:
		// The peer may create exactly one control, encoder, and decoder stream.
	default:
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	bit := uint8(1) << stype
	if c.streamsCreated&bit != 0 {
		return &connectionError{
			code:    errH3StreamCreationError,
			message: "multiple " + stype.String() + " streams created",
		}
	}
	c.streamsCreated |= bit
	return nil
}
