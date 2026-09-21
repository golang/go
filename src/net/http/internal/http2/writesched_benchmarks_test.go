// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"testing"
)

func benchmarkThroughput(b *testing.B, wsFunc func() WriteScheduler, priority PriorityParam) {
	const maxFrameSize = 16
	const streamCount = 100

	ws := wsFunc()
	sc := &serverConn{maxFrameSize: maxFrameSize}
	streams := make([]*stream, streamCount)
	// Possible stream payloads. We vary the payload size of different streams
	// to simulate real traffic somewhat.
	streamsFrame := [][]byte{
		make([]byte, maxFrameSize*5),
		make([]byte, maxFrameSize*10),
		make([]byte, maxFrameSize*15),
		make([]byte, maxFrameSize*20),
		make([]byte, maxFrameSize*25),
	}
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 30) // arbitrary large value

		ws.OpenStream(streamID, OpenStreamOptions{
			priority: priority,
		})
	}

	for b.Loop() {
		for i := range streams {
			streamID := uint32(i) + 1
			ws.Push(FrameWriteRequest{
				write: &writeData{
					streamID:  streamID,
					p:         streamsFrame[i%len(streamsFrame)],
					endStream: false,
				},
				stream: streams[i],
			})
		}
		for {
			wr, ok := ws.Pop()
			if !ok {
				break
			}
			if wr.DataSize() != maxFrameSize {
				b.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
			}
		}
	}

	for i := range streams {
		streamID := uint32(i) + 1
		ws.CloseStream(streamID)
	}
}

func benchmarkStreamLifetime(b *testing.B, wsFunc func() WriteScheduler, priority PriorityParam) {
	const maxFrameSize = 16
	const streamCount = 100

	ws := wsFunc()
	sc := &serverConn{maxFrameSize: maxFrameSize}
	streams := make([]*stream, streamCount)
	// Possible stream payloads. We vary the payload size of different streams
	// to simulate real traffic somewhat.
	streamsFrame := [][]byte{
		make([]byte, maxFrameSize*5),
		make([]byte, maxFrameSize*10),
		make([]byte, maxFrameSize*15),
		make([]byte, maxFrameSize*20),
		make([]byte, maxFrameSize*25),
	}
	for i := range streams {
		streamID := uint32(i) + 1
		streams[i] = &stream{
			id: streamID,
			sc: sc,
		}
		streams[i].flow.add(1 << 30) // arbitrary large value
	}

	for b.Loop() {
		for i := range streams {
			streamID := uint32(i) + 1
			ws.OpenStream(streamID, OpenStreamOptions{
				priority: priority,
			})
			ws.Push(FrameWriteRequest{
				write: &writeData{
					streamID:  streamID,
					p:         streamsFrame[i%len(streamsFrame)],
					endStream: false,
				},
				stream: streams[i],
			})
		}
		for {
			wr, ok := ws.Pop()
			if !ok {
				break
			}
			if wr.DataSize() != maxFrameSize {
				b.Fatalf("wr.Pop() = %v data bytes, want %v", wr.DataSize(), maxFrameSize)
			}
		}
		for i := range streams {
			streamID := uint32(i) + 1
			ws.CloseStream(streamID)
		}
	}

}

func BenchmarkWriteSchedulerThroughputRoundRobin(b *testing.B) {
	benchmarkThroughput(b, newRoundRobinWriteScheduler, PriorityParam{})
}

func BenchmarkWriteSchedulerLifetimeRoundRobin(b *testing.B) {
	benchmarkStreamLifetime(b, newRoundRobinWriteScheduler, PriorityParam{})
}

func BenchmarkWriteSchedulerThroughputPriorityRFC9218Incremental(b *testing.B) {
	benchmarkThroughput(b, newPriorityWriteSchedulerRFC9218, PriorityParam{
		incremental: 1,
	})
}

func BenchmarkWriteSchedulerLifetimePriorityRFC9218Incremental(b *testing.B) {
	benchmarkStreamLifetime(b, newPriorityWriteSchedulerRFC9218, PriorityParam{
		incremental: 1,
	})
}

func BenchmarkWriteSchedulerThroughputPriorityRFC9218NonIncremental(b *testing.B) {
	benchmarkThroughput(b, newPriorityWriteSchedulerRFC9218, PriorityParam{
		incremental: 0,
	})
}

func BenchmarkWriteSchedulerLifetimePriorityRFC9218NonIncremental(b *testing.B) {
	benchmarkStreamLifetime(b, newPriorityWriteSchedulerRFC9218, PriorityParam{
		incremental: 0,
	})
}

func BenchmarkWriteQueue(b *testing.B) {
	var qp writeQueuePool
	frameCount := 25
	for b.Loop() {
		q := qp.get()
		for range frameCount {
			q.push(FrameWriteRequest{})
		}
		for !q.empty() {
			// Since we pushed empty frames, consuming 1 byte is enough to
			// consume the entire frame.
			q.consume(1)
		}
		qp.put(q)
	}
}
