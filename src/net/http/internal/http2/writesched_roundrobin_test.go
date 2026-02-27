// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"reflect"
	"testing"
)

func TestRoundRobinScheduler(t *testing.T) {
	const maxFrameSize = 16
	sc := &serverConn{maxFrameSize: maxFrameSize}
	ws := newRoundRobinWriteScheduler()
	streams := make([]*stream, 4)
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 20) // arbitrary large value
		ws.OpenStream(streamID, OpenStreamOptions{})
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
	const controlFrames = 2
	for i := 0; i < controlFrames; i++ {
		ws.Push(makeWriteNonStreamRequest())
	}

	// We should get the control frames first.
	for i := 0; i < controlFrames; i++ {
		wr, ok := ws.Pop()
		if !ok || wr.StreamID() != 0 {
			t.Fatalf("wr.Pop() = stream %v, %v; want 0, true", wr.StreamID(), ok)
		}
	}

	// Each stream should write maxFrameSize bytes until it runs out of data.
	// Stream 1 has one frame of data, 2 has two frames, etc.
	want := []uint32{1, 2, 3, 4, 2, 3, 4, 3, 4, 4}
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
