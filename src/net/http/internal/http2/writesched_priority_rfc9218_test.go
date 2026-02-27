// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"reflect"
	"testing"
)

func TestPrioritySchedulerUrgency(t *testing.T) {
	const maxFrameSize = 16
	sc := &serverConn{maxFrameSize: maxFrameSize}
	ws := newPriorityWriteSchedulerRFC9218()
	streams := make([]*stream, 5)
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 20) // arbitrary large value
		ws.OpenStream(streamID, OpenStreamOptions{
			priority: PriorityParam{
				urgency:     7,
				incremental: 0,
			},
		})
		wr := FrameWriteRequest{
			write: &writeData{
				streamID:  streamID,
				p:         make([]byte, maxFrameSize*(i+1)),
				endStream: false,
			},
			stream: streams[i],
		}
		ws.Push(wr)
	}
	// Raise the urgency of all even-numbered streams.
	for i := range streams {
		streamID := uint32(i) + 1
		if streamID%2 == 1 {
			continue
		}
		ws.AdjustStream(streamID, PriorityParam{
			urgency:     0,
			incremental: 0,
		})
	}
	const controlFrames = 2
	for range controlFrames {
		ws.Push(makeWriteNonStreamRequest())
	}

	// We should get the control frames first.
	for range controlFrames {
		wr, ok := ws.Pop()
		if !ok || wr.StreamID() != 0 {
			t.Fatalf("wr.Pop() = stream %v, %v; want 0, true", wr.StreamID(), ok)
		}
	}

	// Each stream should write maxFrameSize bytes until it runs out of data.
	// Higher-urgency even-numbered streams should come first.
	want := []uint32{2, 2, 4, 4, 4, 4, 1, 3, 3, 3, 5, 5, 5, 5, 5}
	var got []uint32
	for {
		wr, ok := ws.Pop()
		if !ok {
			break
		}
		if wr.DataSize() != maxFrameSize {
			t.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
		}
		got = append(got, wr.StreamID())
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("popped streams %v, want %v", got, want)
	}
}

func TestPrioritySchedulerIncremental(t *testing.T) {
	const maxFrameSize = 16
	sc := &serverConn{maxFrameSize: maxFrameSize}
	ws := newPriorityWriteSchedulerRFC9218()
	streams := make([]*stream, 5)
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 20) // arbitrary large value
		ws.OpenStream(streamID, OpenStreamOptions{
			priority: PriorityParam{
				urgency:     7,
				incremental: 0,
			},
		})
		wr := FrameWriteRequest{
			write: &writeData{
				streamID:  streamID,
				p:         make([]byte, maxFrameSize*(i+1)),
				endStream: false,
			},
			stream: streams[i],
		}
		ws.Push(wr)
	}
	// Make even-numbered streams incremental.
	for i := range streams {
		streamID := uint32(i) + 1
		if streamID%2 == 1 {
			continue
		}
		ws.AdjustStream(streamID, PriorityParam{
			urgency:     7,
			incremental: 1,
		})
	}
	const controlFrames = 2
	for range controlFrames {
		ws.Push(makeWriteNonStreamRequest())
	}

	// We should get the control frames first.
	for range controlFrames {
		wr, ok := ws.Pop()
		if !ok || wr.StreamID() != 0 {
			t.Fatalf("wr.Pop() = stream %v, %v; want 0, true", wr.StreamID(), ok)
		}
	}

	// Each stream should write maxFrameSize bytes until it runs out of data.
	// We should:
	// - Round-robin between even and odd-numbered streams as they have
	// different i but the same u.
	// - Amongst even-numbered streams, round-robin writes as they are
	// incremental.
	// - Among odd-numbered streams, do not round-robin as they are
	// non-incremental.
	want := []uint32{2, 1, 4, 3, 2, 3, 4, 3, 4, 5, 4, 5, 5, 5, 5}
	var got []uint32
	for {
		wr, ok := ws.Pop()
		if !ok {
			break
		}
		if wr.DataSize() != maxFrameSize {
			t.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
		}
		got = append(got, wr.StreamID())
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("popped streams %v, want %v", got, want)
	}
}

func TestPrioritySchedulerUrgencyAndIncremental(t *testing.T) {
	const maxFrameSize = 16
	sc := &serverConn{maxFrameSize: maxFrameSize}
	ws := newPriorityWriteSchedulerRFC9218()
	streams := make([]*stream, 6)
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 20) // arbitrary large value
		ws.OpenStream(streamID, OpenStreamOptions{
			priority: PriorityParam{
				urgency:     7,
				incremental: 0,
			},
		})
		wr := FrameWriteRequest{
			write: &writeData{
				streamID:  streamID,
				p:         make([]byte, maxFrameSize*(i+1)),
				endStream: false,
			},
			stream: streams[i],
		}
		ws.Push(wr)
	}
	// Make even-numbered streams incremental and of higher urgency.
	for i := range streams {
		streamID := uint32(i) + 1
		if streamID%2 == 1 {
			continue
		}
		ws.AdjustStream(streamID, PriorityParam{
			urgency:     0,
			incremental: 1,
		})
	}
	// Close stream 1 and 4
	ws.CloseStream(1)
	ws.CloseStream(4)
	const controlFrames = 2
	for range controlFrames {
		ws.Push(makeWriteNonStreamRequest())
	}

	// We should get the control frames first.
	for range controlFrames {
		wr, ok := ws.Pop()
		if !ok || wr.StreamID() != 0 {
			t.Fatalf("wr.Pop() = stream %v, %v; want 0, true", wr.StreamID(), ok)
		}
	}

	// Each stream should write maxFrameSize bytes until it runs out of data.
	// We should:
	// - Get even-numbered streams first that are written in a round-robin
	// manner as they have higher urgency and are incremental.
	// - Get odd-numbered streams after that are written one-by-one to
	// completion as they are of lower urgency and are not incremental.
	// - Skip stream 1 and 4 that have been closed.
	want := []uint32{2, 6, 2, 6, 6, 6, 6, 6, 3, 3, 3, 5, 5, 5, 5, 5}
	var got []uint32
	for {
		wr, ok := ws.Pop()
		if !ok {
			break
		}
		if wr.DataSize() != maxFrameSize {
			t.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
		}
		got = append(got, wr.StreamID())
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("popped streams %v, want %v", got, want)
	}
}

func TestPrioritySchedulerIdempotentUpdate(t *testing.T) {
	const maxFrameSize = 16
	sc := &serverConn{maxFrameSize: maxFrameSize}
	ws := newPriorityWriteSchedulerRFC9218()
	streams := make([]*stream, 6)
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 20) // arbitrary large value
		ws.OpenStream(streamID, OpenStreamOptions{
			priority: PriorityParam{
				urgency:     7,
				incremental: 0,
			},
		})
		wr := FrameWriteRequest{
			write: &writeData{
				streamID:  streamID,
				p:         make([]byte, maxFrameSize*(i+1)),
				endStream: false,
			},
			stream: streams[i],
		}
		ws.Push(wr)
	}
	// Make even-numbered streams incremental and of higher urgency.
	for i := range streams {
		streamID := uint32(i) + 1
		if streamID%2 == 1 {
			continue
		}
		ws.AdjustStream(streamID, PriorityParam{
			urgency:     0,
			incremental: 1,
		})
	}
	ws.CloseStream(1)
	// Repeat the same priority update to ensure idempotency.
	for i := range streams {
		streamID := uint32(i) + 1
		if streamID%2 == 1 {
			continue
		}
		ws.AdjustStream(streamID, PriorityParam{
			urgency:     0,
			incremental: 1,
		})
	}
	ws.CloseStream(2)
	const controlFrames = 2
	for range controlFrames {
		ws.Push(makeWriteNonStreamRequest())
	}

	// We should get the control frames first.
	for range controlFrames {
		wr, ok := ws.Pop()
		if !ok || wr.StreamID() != 0 {
			t.Fatalf("wr.Pop() = stream %v, %v; want 0, true", wr.StreamID(), ok)
		}
	}

	// Each stream should write maxFrameSize bytes until it runs out of data.
	// We should:
	// - Get even-numbered streams first that are written in a round-robin
	// manner as they have higher urgency and are incremental.
	// - Get odd-numbered streams after that are written one-by-one to
	// completion as they are of lower urgency and are not incremental.
	// - Skip stream 1 and 4 that have been closed.
	want := []uint32{4, 6, 4, 6, 4, 6, 4, 6, 6, 6, 3, 3, 3, 5, 5, 5, 5, 5}
	var got []uint32
	for {
		wr, ok := ws.Pop()
		if !ok {
			break
		}
		if wr.DataSize() != maxFrameSize {
			t.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
		}
		got = append(got, wr.StreamID())
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("popped streams %v, want %v", got, want)
	}
}

func TestPrioritySchedulerBuffersPriorityUpdate(t *testing.T) {
	const maxFrameSize = 16
	sc := &serverConn{maxFrameSize: maxFrameSize}
	ws := newPriorityWriteSchedulerRFC9218()

	// Priorities are adjusted for streams that are not open yet.
	ws.AdjustStream(1, PriorityParam{urgency: 0})
	ws.AdjustStream(5, PriorityParam{urgency: 0})
	for _, streamID := range []uint32{1, 3, 5} {
		stream := &stream{
			id: streamID,
			sc: sc,
		}
		stream.flow.add(1 << 20) // arbitrary large value
		ws.OpenStream(streamID, OpenStreamOptions{
			priority: PriorityParam{
				urgency:     7,
				incremental: 1,
			},
		})
		wr := FrameWriteRequest{
			write: &writeData{
				streamID:  streamID,
				p:         make([]byte, maxFrameSize*(3)),
				endStream: false,
			},
			stream: stream,
		}
		ws.Push(wr)
	}

	const controlFrames = 2
	for range controlFrames {
		ws.Push(makeWriteNonStreamRequest())
	}

	// We should get the control frames first.
	for range controlFrames {
		wr, ok := ws.Pop()
		if !ok || wr.StreamID() != 0 {
			t.Fatalf("wr.Pop() = stream %v, %v; want 0, true", wr.StreamID(), ok)
		}
	}

	// The most recent priority adjustment is buffered and applied. Older ones
	// are ignored.
	want := []uint32{5, 5, 5, 1, 3, 1, 3, 1, 3}
	var got []uint32
	for {
		wr, ok := ws.Pop()
		if !ok {
			break
		}
		if wr.DataSize() != maxFrameSize {
			t.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
		}
		got = append(got, wr.StreamID())
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("popped streams %v, want %v", got, want)
	}
}
