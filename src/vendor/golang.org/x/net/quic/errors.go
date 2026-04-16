// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"fmt"
)

// A transportError is a transport error code from RFC 9000 Section 20.1.
//
// The transportError type doesn't implement the error interface to ensure we always
// distinguish between errors sent to and received from the peer.
// See the localTransportError and peerTransportError types below.
type transportError uint64

// https://www.rfc-editor.org/rfc/rfc9000.html#section-20.1
const (
	errNo                   = transportError(0x00)
	errInternal             = transportError(0x01)
	errConnectionRefused    = transportError(0x02)
	errFlowControl          = transportError(0x03)
	errStreamLimit          = transportError(0x04)
	errStreamState          = transportError(0x05)
	errFinalSize            = transportError(0x06)
	errFrameEncoding        = transportError(0x07)
	errTransportParameter   = transportError(0x08)
	errConnectionIDLimit    = transportError(0x09)
	errProtocolViolation    = transportError(0x0a)
	errInvalidToken         = transportError(0x0b)
	errApplicationError     = transportError(0x0c)
	errCryptoBufferExceeded = transportError(0x0d)
	errKeyUpdateError       = transportError(0x0e)
	errAEADLimitReached     = transportError(0x0f)
	errNoViablePath         = transportError(0x10)
	errTLSBase              = transportError(0x0100) // 0x0100-0x01ff; base + TLS code
)

func (e transportError) String() string {
	switch e {
	case errNo:
		return "NO_ERROR"
	case errInternal:
		return "INTERNAL_ERROR"
	case errConnectionRefused:
		return "CONNECTION_REFUSED"
	case errFlowControl:
		return "FLOW_CONTROL_ERROR"
	case errStreamLimit:
		return "STREAM_LIMIT_ERROR"
	case errStreamState:
		return "STREAM_STATE_ERROR"
	case errFinalSize:
		return "FINAL_SIZE_ERROR"
	case errFrameEncoding:
		return "FRAME_ENCODING_ERROR"
	case errTransportParameter:
		return "TRANSPORT_PARAMETER_ERROR"
	case errConnectionIDLimit:
		return "CONNECTION_ID_LIMIT_ERROR"
	case errProtocolViolation:
		return "PROTOCOL_VIOLATION"
	case errInvalidToken:
		return "INVALID_TOKEN"
	case errApplicationError:
		return "APPLICATION_ERROR"
	case errCryptoBufferExceeded:
		return "CRYPTO_BUFFER_EXCEEDED"
	case errKeyUpdateError:
		return "KEY_UPDATE_ERROR"
	case errAEADLimitReached:
		return "AEAD_LIMIT_REACHED"
	case errNoViablePath:
		return "NO_VIABLE_PATH"
	}
	if e >= 0x0100 && e <= 0x01ff {
		return fmt.Sprintf("CRYPTO_ERROR(%v)", uint64(e)&0xff)
	}
	return fmt.Sprintf("ERROR %d", uint64(e))
}

// A localTransportError is an error sent to the peer.
type localTransportError struct {
	code   transportError
	reason string
}

func (e localTransportError) Error() string {
	if e.reason == "" {
		return fmt.Sprintf("closed connection: %v", e.code)
	}
	return fmt.Sprintf("closed connection: %v: %q", e.code, e.reason)
}

// A peerTransportError is an error received from the peer.
type peerTransportError struct {
	code   transportError
	reason string
}

func (e peerTransportError) Error() string {
	return fmt.Sprintf("peer closed connection: %v: %q", e.code, e.reason)
}

// A StreamErrorCode is an application protocol error code (RFC 9000, Section 20.2)
// indicating why a stream is being closed.
type StreamErrorCode uint64

func (e StreamErrorCode) Error() string {
	return fmt.Sprintf("stream error code %v", uint64(e))
}

// An ApplicationError is an application protocol error code (RFC 9000, Section 20.2).
// Application protocol errors may be sent when terminating a stream or connection.
type ApplicationError struct {
	Code   uint64
	Reason string
}

func (e *ApplicationError) Error() string {
	return fmt.Sprintf("peer closed connection: %v: %q", e.Code, e.Reason)
}

// Is reports a match if err is an *ApplicationError with a matching Code.
func (e *ApplicationError) Is(err error) bool {
	e2, ok := err.(*ApplicationError)
	return ok && e2.Code == e.Code
}
