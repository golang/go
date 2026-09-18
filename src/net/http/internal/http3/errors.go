// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import "fmt"

// http3Error is an HTTP/3 error code.
type http3Error int

const (
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-8.1
	errH3NoError              = http3Error(0x0100)
	errH3GeneralProtocolError = http3Error(0x0101)
	errH3InternalError        = http3Error(0x0102)
	errH3StreamCreationError  = http3Error(0x0103)
	errH3ClosedCriticalStream = http3Error(0x0104)
	errH3FrameUnexpected      = http3Error(0x0105)
	errH3FrameError           = http3Error(0x0106)
	errH3ExcessiveLoad        = http3Error(0x0107)
	errH3IDError              = http3Error(0x0108)
	errH3SettingsError        = http3Error(0x0109)
	errH3MissingSettings      = http3Error(0x010a)
	errH3RequestRejected      = http3Error(0x010b)
	errH3RequestCancelled     = http3Error(0x010c)
	errH3RequestIncomplete    = http3Error(0x010d)
	errH3MessageError         = http3Error(0x010e)
	errH3ConnectError         = http3Error(0x010f)
	errH3VersionFallback      = http3Error(0x0110)

	// https://www.rfc-editor.org/rfc/rfc9204.html#section-8.3
	errQPACKDecompressionFailed = http3Error(0x0200)
	errQPACKEncoderStreamError  = http3Error(0x0201)
	errQPACKDecoderStreamError  = http3Error(0x0202)
)

func (e http3Error) Error() string {
	switch e {
	case errH3NoError:
		return "H3_NO_ERROR"
	case errH3GeneralProtocolError:
		return "H3_GENERAL_PROTOCOL_ERROR"
	case errH3InternalError:
		return "H3_INTERNAL_ERROR"
	case errH3StreamCreationError:
		return "H3_STREAM_CREATION_ERROR"
	case errH3ClosedCriticalStream:
		return "H3_CLOSED_CRITICAL_STREAM"
	case errH3FrameUnexpected:
		return "H3_FRAME_UNEXPECTED"
	case errH3FrameError:
		return "H3_FRAME_ERROR"
	case errH3ExcessiveLoad:
		return "H3_EXCESSIVE_LOAD"
	case errH3IDError:
		return "H3_ID_ERROR"
	case errH3SettingsError:
		return "H3_SETTINGS_ERROR"
	case errH3MissingSettings:
		return "H3_MISSING_SETTINGS"
	case errH3RequestRejected:
		return "H3_REQUEST_REJECTED"
	case errH3RequestCancelled:
		return "H3_REQUEST_CANCELLED"
	case errH3RequestIncomplete:
		return "H3_REQUEST_INCOMPLETE"
	case errH3MessageError:
		return "H3_MESSAGE_ERROR"
	case errH3ConnectError:
		return "H3_CONNECT_ERROR"
	case errH3VersionFallback:
		return "H3_VERSION_FALLBACK"
	case errQPACKDecompressionFailed:
		return "QPACK_DECOMPRESSION_FAILED"
	case errQPACKEncoderStreamError:
		return "QPACK_ENCODER_STREAM_ERROR"
	case errQPACKDecoderStreamError:
		return "QPACK_DECODER_STREAM_ERROR"
	}
	return fmt.Sprintf("H3_ERROR_%v", int(e))
}

// A streamError is an error which terminates a stream, but not the connection.
// https://www.rfc-editor.org/rfc/rfc9114.html#section-8-1
type streamError struct {
	code    http3Error
	message string
}

func (e *streamError) Error() string { return e.message }
func (e *streamError) Unwrap() error { return e.code }

// A connectionError is an error which results in the entire connection closing.
// https://www.rfc-editor.org/rfc/rfc9114.html#section-8-2
type connectionError struct {
	code    http3Error
	message string
}

func (e *connectionError) Error() string { return e.message }
func (e *connectionError) Unwrap() error { return e.code }
