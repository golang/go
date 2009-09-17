// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strings

import "os"

// Efficient construction of large strings.
// Implements io.Reader and io.Writer.

// A Buffer is a variable-sized buffer of strings
// with Read and Write methods.  Appends (writes) are efficient.
// The zero value for Buffer is an empty buffer ready to use.
type Buffer struct {
	str	[]string;
	len	int;
	byteBuf	[1]byte;
}

// Copy from string to byte array at offset doff.  Assume there's room.
func copy(dst []byte, doff int, src string) {
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
	for _, s := range b.str {
		copy(bytes, nbytes, s);
		nbytes += len(s);
	}
	return bytes;
}

// String returns the contents of the unread portion of the buffer
// as a string.
func (b *Buffer) String() string {
	if len(b.str) == 1 {	// important special case
		return b.str[0]
	}
	return string(b.Bytes())
}

// Len returns the number of bytes in the unread portion of the buffer;
// b.Len() == len(b.Bytes()) == len(b.String()).
func (b *Buffer) Len() (n int) {
	return b.len
}

// Truncate discards all but the first n unread bytes from the buffer.
func (b *Buffer) Truncate(n int) {
	b.len = 0;	// recompute during scan.
	for i, s := range b.str {
		if n <= 0 {
			b.str = b.str[0:i];
			break;
		}
		if n < len(s) {
			b.str[i] = s[0:n];
			b.len += n;
			n = 0;
		} else {
			b.len += len(s);
			n -= len(s);
		}
	}
}

// Reset resets the buffer so it has no content.
// b.Reset() is the same as b.Truncate(0).
func (b *Buffer) Reset() {
	b.str = b.str[0:0];
	b.len = 0;
}

// Can n bytes be appended efficiently to the end of the final string?
func (b *Buffer) canCombine(n int) bool {
	return len(b.str) > 0 && n+len(b.str[len(b.str)-1]) <= 64
}

// WriteString appends string s to the buffer.  The return
// value n is the length of s; err is always nil.
func (b *Buffer) WriteString(s string) (n int, err os.Error) {
	n = len(s);
	b.len += n;
	numStr := len(b.str);
	// Special case: If the last string is short and this one is short,
	// combine them and avoid growing the list.
	if b.canCombine(n) {
		b.str[numStr-1] += s;
		return
	}
	if cap(b.str) == numStr {
		nstr := make([]string, numStr, 3*(numStr+10)/2);
		for i, s := range b.str {
			nstr[i] = s;
		}
		b.str = nstr;
	}
	b.str = b.str[0:numStr+1];
	b.str[numStr] = s;
	return
}

// Write appends the contents of p to the buffer.  The return
// value n is the length of p; err is always nil.
func (b *Buffer) Write(p []byte) (n int, err os.Error) {
	return b.WriteString(string(p))
}

// WriteByte appends the byte c to the buffer.
// The returned error is always nil, but is included
// to match bufio.Writer's WriteByte.
func (b *Buffer) WriteByte(c byte) os.Error {
	s := string(c);
	// For WriteByte, canCombine is almost always true so it's worth
	// doing here.
	if b.canCombine(1) {
		b.str[len(b.str)-1] += s;
		b.len++;
		return nil
	}
	b.WriteString(s);
	return nil;
}

// Read reads the next len(p) bytes from the buffer or until the buffer
// is drained.  The return value n is the number of bytes read.  If the
// buffer has no data to return, err is os.EOF even if len(p) is zero;
// otherwise it is nil.
func (b *Buffer) Read(p []byte) (n int, err os.Error) {
	if len(b.str) == 0 {
		return 0, os.EOF
	}
	for len(b.str) > 0 {
		s := b.str[0];
		m := len(p) - n;
		if m >= len(s) {
			// consume all of this string.
			copy(p, n, s);
			n += len(s);
			b.str = b.str[1:len(b.str)];
		} else {
			// consume some of this string; it's the last piece.
			copy(p, n, s[0:m]);
			n += m;
			b.str[0] = s[m:len(s)];
			break;
		}
	}
	b.len -= n;
	return
}

// ReadByte reads and returns the next byte from the buffer.
// If no byte is available, it returns error os.EOF.
func (b *Buffer) ReadByte() (c byte, err os.Error) {
	if _, err := b.Read(&b.byteBuf); err != nil {
		return 0, err
	}
	return b.byteBuf[0], nil
}

// NewBuffer creates and initializes a new Buffer
// using str as its initial contents.
func NewBuffer(str string) *Buffer {
	b := new(Buffer);
	b.str = make([]string, 1, 10);	// room to grow
	b.str[0] = str;
	b.len = len(str);
	return b;
}
