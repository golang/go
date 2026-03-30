// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package quic

import (
	"sync"
)

// A pipe is a byte buffer used in implementing streams.
//
// A pipe contains a window of stream data.
// Random access reads and writes are supported within the window.
// Writing past the end of the window extends it.
// Data may be discarded from the start of the pipe, advancing the window.
type pipe struct {
	start int64    // stream position of first stored byte
	end   int64    // stream position just past the last stored byte
	head  *pipebuf // if non-nil, then head.off + len(head.b) > start
	tail  *pipebuf // if non-nil, then tail.off + len(tail.b) == end
}

type pipebuf struct {
	off  int64 // stream position of b[0]
	b    []byte
	next *pipebuf
}

func (pb *pipebuf) end() int64 {
	return pb.off + int64(len(pb.b))
}

var pipebufPool = sync.Pool{
	New: func() any {
		return &pipebuf{
			b: make([]byte, 4096),
		}
	},
}

func newPipebuf() *pipebuf {
	return pipebufPool.Get().(*pipebuf)
}

func (b *pipebuf) recycle() {
	b.off = 0
	b.next = nil
	pipebufPool.Put(b)
}

// writeAt writes len(b) bytes to the pipe at offset off.
//
// Writes to offsets before p.start are discarded.
// Writes to offsets after p.end extend the pipe window.
func (p *pipe) writeAt(b []byte, off int64) {
	end := off + int64(len(b))
	if end > p.end {
		p.end = end
	} else if end <= p.start {
		return
	}

	if off < p.start {
		// Discard the portion of b which falls before p.start.
		trim := p.start - off
		b = b[trim:]
		off = p.start
	}

	if p.head == nil {
		p.head = newPipebuf()
		p.head.off = p.start
		p.tail = p.head
	}
	pb := p.head
	if off >= p.tail.off {
		// Common case: Writing past the end of the pipe.
		pb = p.tail
	}
	for {
		pboff := off - pb.off
		if pboff < int64(len(pb.b)) {
			n := copy(pb.b[pboff:], b)
			if n == len(b) {
				return
			}
			off += int64(n)
			b = b[n:]
		}
		if pb.next == nil {
			pb.next = newPipebuf()
			pb.next.off = pb.off + int64(len(pb.b))
			p.tail = pb.next
		}
		pb = pb.next
	}
}

// copy copies len(b) bytes into b starting from off.
// The pipe must contain [off, off+len(b)).
func (p *pipe) copy(off int64, b []byte) {
	dst := b[:0]
	p.read(off, len(b), func(c []byte) error {
		dst = append(dst, c...)
		return nil
	})
}

// read calls f with the data in [off, off+n)
// The data may be provided sequentially across multiple calls to f.
// Note that read (unlike an io.Reader) does not consume the read data.
func (p *pipe) read(off int64, n int, f func([]byte) error) error {
	if off < p.start {
		panic("invalid read range")
	}
	for pb := p.head; pb != nil && n > 0; pb = pb.next {
		if off >= pb.end() {
			continue
		}
		b := pb.b[off-pb.off:]
		if len(b) > n {
			b = b[:n]
		}
		off += int64(len(b))
		n -= len(b)
		if err := f(b); err != nil {
			return err
		}
	}
	if n > 0 {
		panic("invalid read range")
	}
	return nil
}

// peek returns a reference to up to n bytes of internal data buffer, starting at p.start.
// The returned slice is valid until the next call to discardBefore.
// The length of the returned slice will be in the range [0,n].
func (p *pipe) peek(n int64) []byte {
	pb := p.head
	if pb == nil {
		return nil
	}
	b := pb.b[p.start-pb.off:]
	return b[:min(int64(len(b)), n)]
}

// availableBuffer returns the available contiguous, allocated buffer space
// following the pipe window.
//
// This is used by the stream write fast path, which makes multiple writes into the pipe buffer
// without a lock, and then adjusts p.end at a later time with a lock held.
func (p *pipe) availableBuffer() []byte {
	if p.tail == nil {
		return nil
	}
	return p.tail.b[p.end-p.tail.off:]
}

// discardBefore discards all data prior to off.
func (p *pipe) discardBefore(off int64) {
	for p.head != nil && p.head.end() < off {
		head := p.head
		p.head = p.head.next
		head.recycle()
	}
	if p.head == nil {
		p.tail = nil
	}
	p.start = off
	p.end = max(p.end, off)
}
