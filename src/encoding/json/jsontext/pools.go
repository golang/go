// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jsonv2

package jsontext

import (
	"bytes"
	"io"
	"math/bits"
	"sync"
)

// TODO(https://go.dev/issue/47657): Use sync.PoolOf.

var (
	// This owns the internal buffer since there is no io.Writer to output to.
	// Since the buffer can get arbitrarily large in normal usage,
	// there is statistical tracking logic to determine whether to recycle
	// the internal buffer or not based on a history of utilization.
	bufferedEncoderPool = &sync.Pool{New: func() any { return new(Encoder) }}

	// This owns the internal buffer, but it is only used to temporarily store
	// buffered JSON before flushing it to the underlying io.Writer.
	// In a sufficiently efficient streaming mode, we do not expect the buffer
	// to grow arbitrarily large. Thus, we avoid recycling large buffers.
	streamingEncoderPool = &sync.Pool{New: func() any { return new(Encoder) }}

	// This does not own the internal buffer since
	// it is taken directly from the provided bytes.Buffer.
	bytesBufferEncoderPool = &sync.Pool{New: func() any { return new(Encoder) }}
)

// bufferStatistics is statistics to track buffer utilization.
// It is used to determine whether to recycle a buffer or not
// to avoid https://go.dev/issue/23199.
type bufferStatistics struct {
	strikes int // number of times the buffer was under-utilized
	prevLen int // length of previous buffer
}

func getBufferedEncoder(opts ...Options) *Encoder {
	e := bufferedEncoderPool.Get().(*Encoder)
	if e.s.Buf == nil {
		// Round up to nearest 2ⁿ to make best use of malloc size classes.
		// See runtime/sizeclasses.go on Go1.15.
		// Logical OR with 63 to ensure 64 as the minimum buffer size.
		n := 1 << bits.Len(uint(e.s.bufStats.prevLen|63))
		e.s.Buf = make([]byte, 0, n)
	}
	e.s.reset(e.s.Buf[:0], nil, opts...)
	return e
}
func putBufferedEncoder(e *Encoder) {
	// Recycle large buffers only if sufficiently utilized.
	// If a buffer is under-utilized enough times sequentially,
	// then it is discarded, ensuring that a single large buffer
	// won't be kept alive by a continuous stream of small usages.
	//
	// The worst case utilization is computed as:
	//	MIN_UTILIZATION_THRESHOLD / (1 + MAX_NUM_STRIKES)
	//
	// For the constants chosen below, this is (25%)/(1+4) ⇒ 5%.
	// This may seem low, but it ensures a lower bound on
	// the absolute worst-case utilization. Without this check,
	// this would be theoretically 0%, which is infinitely worse.
	//
	// See https://go.dev/issue/27735.
	switch {
	case cap(e.s.Buf) <= 4<<10: // always recycle buffers smaller than 4KiB
		e.s.bufStats.strikes = 0
	case cap(e.s.Buf)/4 <= len(e.s.Buf): // at least 25% utilization
		e.s.bufStats.strikes = 0
	case e.s.bufStats.strikes < 4: // at most 4 strikes
		e.s.bufStats.strikes++
	default: // discard the buffer; too large and too often under-utilized
		e.s.bufStats.strikes = 0
		e.s.bufStats.prevLen = len(e.s.Buf) // heuristic for size to allocate next time
		e.s.Buf = nil
	}
	bufferedEncoderPool.Put(e)
}

func getStreamingEncoder(w io.Writer, opts ...Options) *Encoder {
	if _, ok := w.(*bytes.Buffer); ok {
		e := bytesBufferEncoderPool.Get().(*Encoder)
		e.s.reset(nil, w, opts...) // buffer taken from bytes.Buffer
		return e
	} else {
		e := streamingEncoderPool.Get().(*Encoder)
		e.s.reset(e.s.Buf[:0], w, opts...) // preserve existing buffer
		return e
	}
}
func putStreamingEncoder(e *Encoder) {
	if _, ok := e.s.wr.(*bytes.Buffer); ok {
		bytesBufferEncoderPool.Put(e)
	} else {
		if cap(e.s.Buf) > 64<<10 {
			e.s.Buf = nil // avoid pinning arbitrarily large amounts of memory
		}
		streamingEncoderPool.Put(e)
	}
}

var (
	// This does not own the internal buffer since it is externally provided.
	bufferedDecoderPool = &sync.Pool{New: func() any { return new(Decoder) }}

	// This owns the internal buffer, but it is only used to temporarily store
	// buffered JSON fetched from the underlying io.Reader.
	// In a sufficiently efficient streaming mode, we do not expect the buffer
	// to grow arbitrarily large. Thus, we avoid recycling large buffers.
	streamingDecoderPool = &sync.Pool{New: func() any { return new(Decoder) }}

	// This does not own the internal buffer since
	// it is taken directly from the provided bytes.Buffer.
	bytesBufferDecoderPool = bufferedDecoderPool
)

func getBufferedDecoder(b []byte, opts ...Options) *Decoder {
	d := bufferedDecoderPool.Get().(*Decoder)
	d.s.reset(b, nil, opts...)
	return d
}
func putBufferedDecoder(d *Decoder) {
	bufferedDecoderPool.Put(d)
}

func getStreamingDecoder(r io.Reader, opts ...Options) *Decoder {
	if _, ok := r.(*bytes.Buffer); ok {
		d := bytesBufferDecoderPool.Get().(*Decoder)
		d.s.reset(nil, r, opts...) // buffer taken from bytes.Buffer
		return d
	} else {
		d := streamingDecoderPool.Get().(*Decoder)
		d.s.reset(d.s.buf[:0], r, opts...) // preserve existing buffer
		return d
	}
}
func putStreamingDecoder(d *Decoder) {
	if _, ok := d.s.rd.(*bytes.Buffer); ok {
		bytesBufferDecoderPool.Put(d)
	} else {
		if cap(d.s.buf) > 64<<10 {
			d.s.buf = nil // avoid pinning arbitrarily large amounts of memory
		}
		streamingDecoderPool.Put(d)
	}
}
