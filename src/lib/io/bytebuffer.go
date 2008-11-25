// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io

// Byte buffer for marshaling nested messages.

import (
	"io";
	"os";
)

// A simple implementation of the io.Read and io.Write interfaces.
// A newly allocated ByteBuffer is ready to use.

// TODO(r): Do better memory management.

func bytecopy(dst *[]byte, doff int, src *[]byte, soff int, count int) {
	for i := 0; i < count; i++ {
		dst[doff] = src[soff];
		doff++;
		soff++;
	}
}

export type ByteBuffer struct {
	buf	*[]byte;
	off	int;	// Read from here
	len	int;	// Write to here
	cap	int;
}

func (b *ByteBuffer) Reset() {
	b.off = 0;
	b.len = 0;
}

func (b *ByteBuffer) Write(p *[]byte) (n int, err *os.Error) {
	plen := len(p);
	if b.buf == nil {
		b.cap = plen + 1024;
		b.buf = new([]byte, b.cap);
		b.len = 0;
	}
	if b.len + len(p) > b.cap {
		b.cap = 2*(b.cap + plen);
		nb := new([]byte, b.cap);
		bytecopy(nb, 0, b.buf, 0, b.len);
		b.buf = nb;
	}
	bytecopy(b.buf, b.len, p, 0, plen);
	b.len += plen;
	return plen, nil;
}

func (b *ByteBuffer) Read(p *[]byte) (n int, err *os.Error) {
	plen := len(p);
	if b.buf == nil {
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

func (b *ByteBuffer) Len() int {
	return b.len
}

func (b *ByteBuffer) Data() *[]byte {
	return b.buf[b.off:b.len]
}


export func NewByteBufferFromArray(buf *[]byte) *ByteBuffer {
	b := new(ByteBuffer);
	b.buf = buf;
	b.off = 0;
	b.len = len(buf);
	b.cap = len(buf);
	return b;
}
