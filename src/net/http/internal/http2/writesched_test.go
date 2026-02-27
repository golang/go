// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"fmt"
	"math"
	"reflect"
	"testing"
)

func makeWriteNonStreamRequest() FrameWriteRequest {
	return FrameWriteRequest{writeSettingsAck{}, nil, nil}
}

func makeWriteHeadersRequest(streamID uint32) FrameWriteRequest {
	st := &stream{id: streamID}
	return FrameWriteRequest{&writeResHeaders{streamID: streamID, httpResCode: 200}, st, nil}
}

func makeHandlerPanicRST(streamID uint32) FrameWriteRequest {
	st := &stream{id: streamID}
	return FrameWriteRequest{&handlerPanicRST{StreamID: streamID}, st, nil}
}

func makeWriteRSTStream(streamID uint32) FrameWriteRequest {
	return FrameWriteRequest{write: streamError(streamID, ErrCodeInternal)}
}

func checkConsume(wr FrameWriteRequest, nbytes int32, want []FrameWriteRequest) error {
	consumed, rest, n := wr.Consume(nbytes)
	var wantConsumed, wantRest FrameWriteRequest
	switch len(want) {
	case 0:
	case 1:
		wantConsumed = want[0]
	case 2:
		wantConsumed = want[0]
		wantRest = want[1]
	}
	if !reflect.DeepEqual(consumed, wantConsumed) || !reflect.DeepEqual(rest, wantRest) || n != len(want) {
		return fmt.Errorf("got %v, %v, %v\nwant %v, %v, %v", consumed, rest, n, wantConsumed, wantRest, len(want))
	}
	return nil
}

func TestFrameWriteRequestNonData(t *testing.T) {
	wr := makeWriteNonStreamRequest()
	if got, want := wr.DataSize(), 0; got != want {
		t.Errorf("DataSize: got %v, want %v", got, want)
	}

	// Non-DATA frames are always consumed whole.
	if err := checkConsume(wr, 0, []FrameWriteRequest{wr}); err != nil {
		t.Errorf("Consume:\n%v", err)
	}

	wr = makeWriteRSTStream(123)
	if got, want := wr.DataSize(), 0; got != want {
		t.Errorf("DataSize: got %v, want %v", got, want)
	}

	// RST_STREAM frames are always consumed whole.
	if err := checkConsume(wr, 0, []FrameWriteRequest{wr}); err != nil {
		t.Errorf("Consume:\n%v", err)
	}
}

// #49741 RST_STREAM and Control frames should have more priority than data
// frames to avoid blocking streams caused by clients not able to drain the
// queue.
func TestFrameWriteRequestWithData(t *testing.T) {
	st := &stream{
		id: 1,
		sc: &serverConn{maxFrameSize: 16},
	}
	const size = 32
	wr := FrameWriteRequest{&writeData{st.id, make([]byte, size), true}, st, make(chan error)}
	if got, want := wr.DataSize(), size; got != want {
		t.Errorf("DataSize: got %v, want %v", got, want)
	}

	// No flow-control bytes available: cannot consume anything.
	if err := checkConsume(wr, math.MaxInt32, []FrameWriteRequest{}); err != nil {
		t.Errorf("Consume(limited by flow control):\n%v", err)
	}

	wr = makeWriteNonStreamRequest()
	if got, want := wr.DataSize(), 0; got != want {
		t.Errorf("DataSize: got %v, want %v", got, want)
	}

	// Non-DATA frames are always consumed whole.
	if err := checkConsume(wr, 0, []FrameWriteRequest{wr}); err != nil {
		t.Errorf("Consume:\n%v", err)
	}

	wr = makeWriteRSTStream(1)
	if got, want := wr.DataSize(), 0; got != want {
		t.Errorf("DataSize: got %v, want %v", got, want)
	}

	// RST_STREAM frames are always consumed whole.
	if err := checkConsume(wr, 0, []FrameWriteRequest{wr}); err != nil {
		t.Errorf("Consume:\n%v", err)
	}
}

func TestFrameWriteRequestData(t *testing.T) {
	st := &stream{
		id: 1,
		sc: &serverConn{maxFrameSize: 16},
	}
	const size = 32
	wr := FrameWriteRequest{&writeData{st.id, make([]byte, size), true}, st, make(chan error)}
	if got, want := wr.DataSize(), size; got != want {
		t.Errorf("DataSize: got %v, want %v", got, want)
	}

	// No flow-control bytes available: cannot consume anything.
	if err := checkConsume(wr, math.MaxInt32, []FrameWriteRequest{}); err != nil {
		t.Errorf("Consume(limited by flow control):\n%v", err)
	}

	// Add enough flow-control bytes to consume the entire frame,
	// but we're now restricted by st.sc.maxFrameSize.
	st.flow.add(size)
	want := []FrameWriteRequest{
		{
			write:  &writeData{st.id, make([]byte, st.sc.maxFrameSize), false},
			stream: st,
			done:   nil,
		},
		{
			write:  &writeData{st.id, make([]byte, size-st.sc.maxFrameSize), true},
			stream: st,
			done:   wr.done,
		},
	}
	if err := checkConsume(wr, math.MaxInt32, want); err != nil {
		t.Errorf("Consume(limited by maxFrameSize):\n%v", err)
	}
	rest := want[1]

	// Consume 8 bytes from the remaining frame.
	want = []FrameWriteRequest{
		{
			write:  &writeData{st.id, make([]byte, 8), false},
			stream: st,
			done:   nil,
		},
		{
			write:  &writeData{st.id, make([]byte, size-st.sc.maxFrameSize-8), true},
			stream: st,
			done:   wr.done,
		},
	}
	if err := checkConsume(rest, 8, want); err != nil {
		t.Errorf("Consume(8):\n%v", err)
	}
	rest = want[1]

	// Consume all remaining bytes.
	want = []FrameWriteRequest{
		{
			write:  &writeData{st.id, make([]byte, size-st.sc.maxFrameSize-8), true},
			stream: st,
			done:   wr.done,
		},
	}
	if err := checkConsume(rest, math.MaxInt32, want); err != nil {
		t.Errorf("Consume(remainder):\n%v", err)
	}
}

func TestFrameWriteRequest_StreamID(t *testing.T) {
	const streamID = 123
	wr := FrameWriteRequest{write: streamError(streamID, ErrCodeNo)}
	if got := wr.StreamID(); got != streamID {
		t.Errorf("FrameWriteRequest(StreamError) = %v; want %v", got, streamID)
	}
}
