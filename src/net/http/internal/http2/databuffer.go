// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package http2

import (
	"errors"
	"fmt"
	"sync"
)

// Buffer chunks are allocated from a pool to reduce pressure on GC.
// The maximum wasted space per dataBuffer is 2x the largest size class,
// which happens when the dataBuffer has multiple chunks and there is
// one unread byte in both the first and last chunks. We use a few size
// classes to minimize overheads for servers that typically receive very
// small request bodies.
//
// TODO: Benchmark to determine if the pools are necessary. The GC may have
// improved enough that we can instead allocate chunks like this:
// make([]byte, max(16<<10, expectedBytesRemaining))
var dataChunkPools = [...]sync.Pool{
	{New: func() interface{} { return new([1 << 10]byte) }},
	{New: func() interface{} { return new([2 << 10]byte) }},
	{New: func() interface{} { return new([4 << 10]byte) }},
	{New: func() interface{} { return new([8 << 10]byte) }},
	{New: func() interface{} { return new([16 << 10]byte) }},
}

func getDataBufferChunk(size int64) []byte {
	switch {
	case size <= 1<<10:
		return dataChunkPools[0].Get().(*[1 << 10]byte)[:]
	case size <= 2<<10:
		return dataChunkPools[1].Get().(*[2 << 10]byte)[:]
	case size <= 4<<10:
		return dataChunkPools[2].Get().(*[4 << 10]byte)[:]
	case size <= 8<<10:
		return dataChunkPools[3].Get().(*[8 << 10]byte)[:]
	default:
		return dataChunkPools[4].Get().(*[16 << 10]byte)[:]
	}
}

func putDataBufferChunk(p []byte) {
	switch len(p) {
	case 1 << 10:
		dataChunkPools[0].Put((*[1 << 10]byte)(p))
	case 2 << 10:
		dataChunkPools[1].Put((*[2 << 10]byte)(p))
	case 4 << 10:
		dataChunkPools[2].Put((*[4 << 10]byte)(p))
	case 8 << 10:
		dataChunkPools[3].Put((*[8 << 10]byte)(p))
	case 16 << 10:
		dataChunkPools[4].Put((*[16 << 10]byte)(p))
	default:
		panic(fmt.Sprintf("unexpected buffer len=%v", len(p)))
	}
}

// dataBuffer is an io.ReadWriter backed by a list of data chunks.
// Each dataBuffer is used to read DATA frames on a single stream.
// The buffer is divided into chunks so the server can limit the
// total memory used by a single connection without limiting the
// request body size on any single stream.
type dataBuffer struct {
	chunks   [][]byte
	r        int   // next byte to read is chunks[0][r]
	w        int   // next byte to write is chunks[len(chunks)-1][w]
	size     int   // total buffered bytes
	expected int64 // we expect at least this many bytes in future Write calls (ignored if <= 0)
}

var errReadEmpty = errors.New("read from empty dataBuffer")

// Read copies bytes from the buffer into p.
// It is an error to read when no data is available.
func (b *dataBuffer) Read(p []byte) (int, error) {
	if b.size == 0 {
		return 0, errReadEmpty
	}
	var ntotal int
	for len(p) > 0 && b.size > 0 {
		readFrom := b.bytesFromFirstChunk()
		n := copy(p, readFrom)
		p = p[n:]
		ntotal += n
		b.r += n
		b.size -= n
		// If the first chunk has been consumed, advance to the next chunk.
		if b.r == len(b.chunks[0]) {
			putDataBufferChunk(b.chunks[0])
			end := len(b.chunks) - 1
			copy(b.chunks[:end], b.chunks[1:])
			b.chunks[end] = nil
			b.chunks = b.chunks[:end]
			b.r = 0
		}
	}
	return ntotal, nil
}

func (b *dataBuffer) bytesFromFirstChunk() []byte {
	if len(b.chunks) == 1 {
		return b.chunks[0][b.r:b.w]
	}
	return b.chunks[0][b.r:]
}

// Len returns the number of bytes of the unread portion of the buffer.
func (b *dataBuffer) Len() int {
	return b.size
}

// Write appends p to the buffer.
func (b *dataBuffer) Write(p []byte) (int, error) {
	ntotal := len(p)
	for len(p) > 0 {
		// If the last chunk is empty, allocate a new chunk. Try to allocate
		// enough to fully copy p plus any additional bytes we expect to
		// receive. However, this may allocate less than len(p).
		want := int64(len(p))
		if b.expected > want {
			want = b.expected
		}
		chunk := b.lastChunkOrAlloc(want)
		n := copy(chunk[b.w:], p)
		p = p[n:]
		b.w += n
		b.size += n
		b.expected -= int64(n)
	}
	return ntotal, nil
}

func (b *dataBuffer) lastChunkOrAlloc(want int64) []byte {
	if len(b.chunks) != 0 {
		last := b.chunks[len(b.chunks)-1]
		if b.w < len(last) {
			return last
		}
	}
	chunk := getDataBufferChunk(want)
	b.chunks = append(b.chunks, chunk)
	b.w = 0
	return chunk
}
