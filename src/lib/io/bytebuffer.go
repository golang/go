// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

// Simple byte buffer for marshaling data.

import (
	"io";
	"os";
)


// TODO(r): Do better memory management.

func bytecopy(dst []byte, doff int, src []byte, soff int, count int) {
	for i := 0; i < count; i++ {
		dst[doff] = src[soff];
		doff++;
		soff++;
	}
}

// A ByteBuffer is a simple implementation of the io.Read and io.Write interfaces
// connected to a buffer of bytes.
// The zero value for ByteBuffer is an empty buffer ready to use.
type ByteBuffer struct {
	buf	[]byte;
	off	int;	// Read from here
	len	int;	// Write to here
	cap	int;
}

// Reset resets the buffer so it has no content.
func (b *ByteBuffer) Reset() {
	b.off = 0;
	b.len = 0;
}

// Write appends the contents of p to the buffer.  The return
// value is the length of p; err is always nil.
func (b *ByteBuffer) Write(p []byte) (n int, err *os.Error) {
	plen := len(p);
	if len(b.buf) == 0 {
		b.cap = plen + 1024;
		b.buf = make([]byte, b.cap);
		b.len = 0;
	}
	if b.len + len(p) > b.cap {
		b.cap = 2*(b.cap + plen);
		nb := make([]byte, b.cap);
		bytecopy(nb, 0, b.buf, 0, b.len);
		b.buf = nb;
	}
	bytecopy(b.buf, b.len, p, 0, plen);
	b.len += plen;
	return plen, nil;
}

// Read reads the next len(p) bytes from the buffer or until the buffer
// is drained.  The return value is the number of bytes read; err is always nil.
func (b *ByteBuffer) Read(p []byte) (n int, err *os.Error) {
	plen := len(p);
	if len(b.buf) == 0 {
		return 0, nil
	}
	if b.off == b.len {	// empty buffer
		b.Reset();
		return 0, nil
	}
	if plen > b.len - b.off {
		plen = b.len - b.off
	}
	bytecopy(p, 0, b.buf, b.off, plen);
	b.off += plen;
	return plen, nil;
}

// Len returns the length of the underlying buffer.
func (b *ByteBuffer) Len() int {
	return b.len
}

// Off returns the location within the buffer of the next byte to be read.
func (b *ByteBuffer) Off() int {
	return b.off
}

// Data returns the contents of the unread portion of the buffer.
func (b *ByteBuffer) Data() []byte {
	return b.buf[b.off:b.len]
}

// NewByteBufferFromArray creates and initializes a new ByteBuffer
// with buf as its initial contents.
func NewByteBufferFromArray(buf []byte) *ByteBuffer {
	b := new(ByteBuffer);
	b.buf = buf;
	b.off = 0;
	b.len = len(buf);
	b.cap = len(buf);
	return b;
}
