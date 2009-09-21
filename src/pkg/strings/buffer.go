// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "os"

// Efficient construction of large strings and byte arrays.
// Implements io.Reader and io.Writer.

// A Buffer provides efficient construction of large strings
// and slices of bytes.  It implements io.Reader and io.Writer.
// Appends (writes) are efficient.
// The zero value for Buffer is an empty buffer ready to use.
type Buffer struct {
	blk	[]block;
	len	int;
	oneByte	[1]byte;
}

// There are two kinds of block: a string or a []byte.
// When the user writes big strings, we add string blocks;
// when the user writes big byte slices, we add []byte blocks.
// Small writes are coalesced onto the end of the last block,
// whatever it is.
// This strategy is intended to reduce unnecessary allocation.
type block interface {
	Len()	int;
	String()	string;
	appendBytes(s []byte);
	appendString(s string);
	setSlice(m, n int);
}

// stringBlocks represent strings. We use pointer receivers
// so append and setSlice can overwrite the receiver.
type stringBlock string

func (b *stringBlock) Len() int {
	return len(*b)
}

func (b *stringBlock) String() string {
	return string(*b)
}

func (b *stringBlock) appendBytes(s []byte) {
	*b += stringBlock(s)
}

func (b *stringBlock) appendString(s string) {
	*b = stringBlock(s)
}

func (b *stringBlock) setSlice(m, n int) {
	*b = (*b)[m:n]
}

// byteBlock represent slices of bytes.  We use pointer receivers
// so append and setSlice can overwrite the receiver.
type byteBlock []byte

func (b *byteBlock) Len() int {
	return len(*b)
}

func (b *byteBlock) String() string {
	return string(*b)
}

func (b *byteBlock) resize(max int) {
	by := []byte(*b);
	if cap(by) >= max {
		by = by[0:max];
	} else {
		nby := make([]byte, max, 3*(max+10)/2);
		copyBytes(nby, 0, by);
		by = nby;
	}
	*b = by;
}

func (b *byteBlock) appendBytes(s []byte) {
	curLen := b.Len();
	b.resize(curLen + len(s));
	copyBytes([]byte(*b), curLen, s);
}

func (b *byteBlock) appendString(s string) {
	curLen := b.Len();
	b.resize(curLen + len(s));
	copyString([]byte(*b), curLen, s);
}

func (b *byteBlock) setSlice(m, n int) {
	*b = (*b)[m:n]
}

// Because the user may overwrite the contents of byte slices, we need
// to make a copy.  Allocation strategy: leave some space on the end so
// small subsequent writes can avoid another allocation.  The input
// is known to be non-empty.
func newByteBlock(s []byte) *byteBlock {
	l := len(s);
	// Capacity with room to grow.  If small, allocate a mininum.  If medium,
	// double the size.  If huge, use the size plus epsilon (room for a newline,
	// at least).
	c := l;
	switch {
	case l < 32:
		c = 64
	case l < 1<<18:
		c *= 2;
	default:
		c += 8
	}
	b := make([]byte, l, c);
	copyBytes(b, 0, s);
	return &b;
}

// Copy from block to byte array at offset doff.  Assume there's room.
func copy(dst []byte, doff int, src block) {
	switch s := src.(type) {
	case *stringBlock:
		copyString(dst, doff, string(*s));
	case *byteBlock:
		copyBytes(dst, doff, []byte(*s));
	}
}

// Copy from string to byte array at offset doff.  Assume there's room.
func copyString(dst []byte, doff int, str string) {
	for soff := 0; soff < len(str); soff++ {
		dst[doff] = str[soff];
		doff++;
	}
}

// Copy from bytes to byte array at offset doff.  Assume there's room.
func copyBytes(dst []byte, doff int, src []byte) {
	for soff := 0; soff < len(src); soff++ {
		dst[doff] = src[soff];
		doff++;
	}
}

// Bytes returns the contents of the unread portion of the buffer
// as a byte array.
func (b *Buffer) Bytes() []byte {
	n := b.len;
	bytes := make([]byte, n);
	nbytes := 0;
	for _, s := range b.blk {
		copy(bytes, nbytes, s);
		nbytes += s.Len();
	}
	return bytes;
}

// String returns the contents of the unread portion of the buffer
// as a string.
func (b *Buffer) String() string {
	if len(b.blk) == 1 {	// important special case
		return b.blk[0].String()
	}
	return string(b.Bytes())
}

// Len returns the number of bytes in the unread portion of the buffer;
// b.Len() == len(b.Bytes()) == len(b.String()).
func (b *Buffer) Len() int {
	return b.len
}

// Truncate discards all but the first n unread bytes from the buffer.
func (b *Buffer) Truncate(n int) {
	b.len = 0;	// recompute during scan.
	for i, s := range b.blk {
		if n <= 0 {
			b.blk = b.blk[0:i];
			break;
		}
		if l := s.Len(); n < l {
			b.blk[i].setSlice(0, n);
			b.len += n;
			n = 0;
		} else {
			b.len += l;
			n -= l;
		}
	}
}

// Reset resets the buffer so it has no content.
// b.Reset() is the same as b.Truncate(0).
func (b *Buffer) Reset() {
	b.blk = b.blk[0:0];
	b.len = 0;
}

// Can n bytes be appended efficiently to the end of the final string?
func (b *Buffer) canCombine(n int) bool {
	return len(b.blk) > 0 && n+b.blk[len(b.blk)-1].Len() <= 64
}

// WriteString appends string s to the buffer.  The return
// value n is the length of s; err is always nil.
func (b *Buffer) WriteString(s string) (n int, err os.Error) {
	n = len(s);
	if n == 0 {
		return
	}
	b.len += n;
	numStr := len(b.blk);
	// Special case: If the last piece is short and this one is short,
	// combine them and avoid growing the list.
	if b.canCombine(n) {
		b.blk[numStr-1].appendString(s);
		return
	}
	if cap(b.blk) == numStr {
		nstr := make([]block, numStr, 3*(numStr+10)/2);
		for i, s := range b.blk {
			nstr[i] = s;
		}
		b.blk = nstr;
	}
	b.blk = b.blk[0:numStr+1];
	// The string is immutable; no need to make a copy.
	b.blk[numStr] = (*stringBlock)(&s);
	return
}

// Write appends the contents of p to the buffer.  The return
// value n is the length of p; err is always nil.
func (b *Buffer) Write(p []byte) (n int, err os.Error) {
	n = len(p);
	if n == 0 {
		return
	}
	b.len += n;
	numStr := len(b.blk);
	// Special case: If the last piece is short and this one is short,
	// combine them and avoid growing the list.
	if b.canCombine(n) {
		b.blk[numStr-1].appendBytes(p);
		return
	}
	if cap(b.blk) == numStr {
		nstr := make([]block, numStr, 3*(numStr+10)/2);
		for i, s := range b.blk {
			nstr[i] = s;
		}
		b.blk = nstr;
	}
	b.blk = b.blk[0:numStr+1];
	// Need to copy the data - user might overwrite the data.
	b.blk[numStr] = newByteBlock(p);
	return
}

// WriteByte appends the byte c to the buffer.
// The returned error is always nil, but is included
// to match bufio.Writer's WriteByte.
func (b *Buffer) WriteByte(c byte) os.Error {
	b.oneByte[0] = c;
	// For WriteByte, canCombine is almost always true so it's worth
	// doing here.
	if b.canCombine(1) {
		b.blk[len(b.blk)-1].appendBytes(&b.oneByte);
		b.len++;
		return nil
	}
	b.Write(&b.oneByte);
	return nil;
}

// Read reads the next len(p) bytes from the buffer or until the buffer
// is drained.  The return value n is the number of bytes read.  If the
// buffer has no data to return, err is os.EOF even if len(p) is zero;
// otherwise it is nil.
func (b *Buffer) Read(p []byte) (n int, err os.Error) {
	if len(b.blk) == 0 {
		return 0, os.EOF
	}
	for len(b.blk) > 0 {
		blk := b.blk[0];
		m := len(p) - n;
		if l := blk.Len(); m >= l {
			// consume all of this string.
			copy(p, n, blk);
			n += l;
			b.blk = b.blk[1:len(b.blk)];
		} else {
			// consume some of this block; it's the last piece.
			switch b := blk.(type) {
			case *stringBlock:
				copyString(p, n, string(*b)[0:m]);
			case *byteBlock:
				copyBytes(p, n, []byte(*b)[0:m]);
			}
			n += m;
			b.blk[0].setSlice(m, l);
			break;
		}
	}
	b.len -= n;
	return
}

// ReadByte reads and returns the next byte from the buffer.
// If no byte is available, it returns error os.EOF.
func (b *Buffer) ReadByte() (c byte, err os.Error) {
	if _, err := b.Read(&b.oneByte); err != nil {
		return 0, err
	}
	return b.oneByte[0], nil
}

// NewBufferString creates and initializes a new Buffer
// using a string as its initial contents.
func NewBufferString(str string) *Buffer {
	b := new(Buffer);
	b.blk = make([]block, 1, 10);	// room to grow
	b.blk[0] = (*stringBlock)(&str);
	b.len = len(str);
	return b;
}

// NewBuffer creates and initializes a new Buffer
// using a byte slice as its initial contents.
func NewBuffer(by []byte) *Buffer {
	b := new(Buffer);
	b.blk = make([]block, 1, 10);	// room to grow
	b.blk[0] = (*byteBlock)(&by);
	b.len = len(by);
	return b;
}
