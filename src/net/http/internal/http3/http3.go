// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http3

import (
	"context"
	"fmt"
)

// Stream types.
//
// For unidirectional streams, the value is the stream type sent over the wire.
//
// For bidirectional streams (which are always request streams),
// the value is arbitrary and never sent on the wire.
type streamType int64

const (
	// Bidirectional request stream.
	// All bidirectional streams are request streams.
	// This stream type is never sent over the wire.
	//
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.1
	streamTypeRequest = streamType(-1)

	// https://www.rfc-editor.org/rfc/rfc9114.html#section-6.2
	streamTypeControl = streamType(0x00)
	streamTypePush    = streamType(0x01)

	// https://www.rfc-editor.org/rfc/rfc9204.html#section-4.2
	streamTypeEncoder = streamType(0x02)
	streamTypeDecoder = streamType(0x03)
)

// canceledCtx is a canceled Context.
// Used for performing non-blocking QUIC operations.
var canceledCtx = func() context.Context {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	return ctx
}()

func (stype streamType) String() string {
	switch stype {
	case streamTypeRequest:
		return "request"
	case streamTypeControl:
		return "control"
	case streamTypePush:
		return "push"
	case streamTypeEncoder:
		return "encoder"
	case streamTypeDecoder:
		return "decoder"
	default:
		return "unknown"
	}
}

// Frame types.
type frameType int64

const (
	// https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2
	frameTypeData        = frameType(0x00)
	frameTypeHeaders     = frameType(0x01)
	frameTypeCancelPush  = frameType(0x03)
	frameTypeSettings    = frameType(0x04)
	frameTypePushPromise = frameType(0x05)
	frameTypeGoaway      = frameType(0x07)
	frameTypeMaxPushID   = frameType(0x0d)
)

func (ftype frameType) String() string {
	switch ftype {
	case frameTypeData:
		return "DATA"
	case frameTypeHeaders:
		return "HEADERS"
	case frameTypeCancelPush:
		return "CANCEL_PUSH"
	case frameTypeSettings:
		return "SETTINGS"
	case frameTypePushPromise:
		return "PUSH_PROMISE"
	case frameTypeGoaway:
		return "GOAWAY"
	case frameTypeMaxPushID:
		return "MAX_PUSH_ID"
	default:
		return fmt.Sprintf("UNKNOWN_%d", int64(ftype))
	}
}
