// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio

import (
	"bufio";
	"fmt";
	"io";
	"os";
	"syscall";
	"testing";
)

// Should be in language!
func copy(p []byte, q []byte) {
	for i := 0; i < len(p); i++ {
		p[i] = q[i]
	}
}

// Reads from p.
type byteReader struct {
	p []byte
}

func newByteReader(p []byte) io.Read {
	b := new(byteReader);
	b.p = p;
	return b
}

func (b *byteReader) Read(p []byte) (int, *os.Error) {
	n := len(p);
	if n > len(b.p) {
		n = len(b.p)
	}
	copy(p[0:n], b.p[0:n]);
	b.p = b.p[n:len(b.p)];
	return n, nil
}


// Reads from p but only returns half of what you asked for.
type halfByteReader struct {
	p []byte
}

func newHalfByteReader(p []byte) io.Read {
	b := new(halfByteReader);
	b.p = p;
	return b
}

func (b *halfByteReader) Read(p []byte) (int, *os.Error) {
	n := len(p)/2;
	if n == 0 && len(p) > 0 {
		n = 1
	}
	if n > len(b.p) {
		n = len(b.p)
	}
	copy(p[0:n], b.p[0:n]);
	b.p = b.p[n:len(b.p)];
	return n, nil
}

// Reads from a reader and rot13s the result.
type rot13Reader struct {
	r io.Read
}

func newRot13Reader(r io.Read) *rot13Reader {
	r13 := new(rot13Reader);
	r13.r = r;
	return r13
}

func (r13 *rot13Reader) Read(p []byte) (int, *os.Error) {
	n, e := r13.r.Read(p);
	if e != nil {
		return n, e
	}
	for i := 0; i < n; i++ {
		if 'a' <= p[i] && p[i] <= 'z' || 'A' <= p[i] && p[i] <= 'Z' {
			if 'a' <= p[i] && p[i] <= 'm' || 'A' <= p[i] && p[i] <= 'M' {
				p[i] += 13;
			} else {
				p[i] -= 13;
			}
		}
	}
	return n, nil
}

type readMaker struct {
	name string;
	fn func([]byte) io.Read;
}
var readMakers = []readMaker {
	readMaker{ "full", func(p []byte) io.Read { return newByteReader(p) } },
	readMaker{ "half", func(p []byte) io.Read { return newHalfByteReader(p) } },
}

// Call ReadLineString (which ends up calling everything else)
// to accumulate the text of a file.
func readLines(b *BufRead) string {
	s := "";
	for {
		s1, e := b.ReadLineString('\n', true);
		if e == EndOfFile {
			break
		}
		if e != nil {
			panic("GetLines: "+e.String())
		}
		s += s1
	}
	return s
}

// Call ReadByte to accumulate the text of a file
func readBytes(buf *BufRead) string {
	var b [1000]byte;
	nb := 0;
	for {
		c, e := buf.ReadByte();
		if e == EndOfFile {
			break
		}
		if e != nil {
			panic("GetBytes: "+e.String())
		}
		b[nb] = c;
		nb++;
	}
	// BUG return string(b[0:nb]) ?
	return string(b)[0:nb]
}

// Call Read to accumulate the text of a file
func reads(buf *BufRead, m int) string {
	var b [1000]byte;
	nb := 0;
	for {
		n, e := buf.Read(b[nb:nb+m]);
		nb += n;
		if e == EndOfFile {
			break
		}
	}
	return string(b[0:nb])
}

type bufReader struct {
	name string;
	fn func(*BufRead) string;
}
var bufreaders = []bufReader {
	bufReader{ "1", func(b *BufRead) string { return reads(b, 1) } },
	bufReader{ "2", func(b *BufRead) string { return reads(b, 2) } },
	bufReader{ "3", func(b *BufRead) string { return reads(b, 3) } },
	bufReader{ "4", func(b *BufRead) string { return reads(b, 4) } },
	bufReader{ "5", func(b *BufRead) string { return reads(b, 5) } },
	bufReader{ "7", func(b *BufRead) string { return reads(b, 7) } },
	bufReader{ "bytes", readBytes },
	bufReader{ "lines", readLines },
}

var bufsizes = []int {
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	23, 32, 46, 64, 93, 128, 1024, 4096
}

func TestBufReadSimple(t *testing.T) {
	b, e := NewBufRead(newByteReader(io.StringBytes("hello world")));
	if s := readBytes(b); s != "hello world" {
		t.Errorf("simple hello world test failed: got %q", s);
	}

	b, e = NewBufRead(newRot13Reader(newByteReader(io.StringBytes("hello world"))));
	if s := readBytes(b); s != "uryyb jbeyq" {
		t.Error("rot13 hello world test failed: got %q", s);
	}
}

func TestBufRead(t *testing.T) {
	var texts [31]string;
	str := "";
	all := "";
	for i := 0; i < len(texts)-1; i++ {
		texts[i] = str + "\n";
		all += texts[i];
		str += string(i%26+'a')
	}
	texts[len(texts)-1] = all;

	for h := 0; h < len(texts); h++ {
		text := texts[h];
		textbytes := io.StringBytes(text);
		for i := 0; i < len(readMakers); i++ {
			for j := 0; j < len(bufreaders); j++ {
				for k := 0; k < len(bufsizes); k++ {
					readmaker := readMakers[i];
					bufreader := bufreaders[j];
					bufsize := bufsizes[k];
					read := readmaker.fn(textbytes);
					buf, e := NewBufReadSize(read, bufsize);
					s := bufreader.fn(buf);
					if s != text {
						t.Errorf("reader=%s fn=%s bufsize=%d want=%q got=%q",
							readmaker.name, bufreader.name, bufsize, text, s);
					}
				}
			}
		}
	}
}

type writeBuffer interface {
	Write(p []byte) (int, *os.Error);
	GetBytes() []byte
}

// Accumulates bytes into a byte array.
type byteWriter struct {
	p []byte;
	n int
}

func newByteWriter() writeBuffer {
	return new(byteWriter)
}

func (w *byteWriter) Write(p []byte) (int, *os.Error) {
	if w.p == nil {
		w.p = make([]byte, len(p)+100)
	} else if w.n + len(p) >= len(w.p) {
		newp := make([]byte, len(w.p)*2 + len(p));
		copy(newp[0:w.n], w.p[0:w.n]);
		w.p = newp
	}
	copy(w.p[w.n:w.n+len(p)], p);
	w.n += len(p);
	return len(p), nil
}

func (w *byteWriter) GetBytes() []byte {
	return w.p[0:w.n]
}

// Accumulates bytes written into a byte array
// but Write only takes half of what you give it.
// TODO: Could toss this -- Write() is not supposed to do that.
type halfByteWriter struct {
	bw writeBuffer
}

func newHalfByteWriter() writeBuffer {
	w := new(halfByteWriter);
	w.bw = newByteWriter();
	return w
}

func (w *halfByteWriter) Write(p []byte) (int, *os.Error) {
	n := (len(p)+1) / 2;
	// BUG return w.bw.Write(p[0:n])
	r, e := w.bw.Write(p[0:n]);
	return r, e
}

func (w *halfByteWriter) GetBytes() []byte {
	return w.bw.GetBytes()
}

type writeMaker struct {
	name string;
	fn func()writeBuffer;
}
func TestBufWrite(t *testing.T) {
	var data [8192]byte;

	var writers = []writeMaker {
		writeMaker{ "full", newByteWriter },
		writeMaker{ "half", newHalfByteWriter },
	};

	for i := 0; i < len(data); i++ {
		data[i] = byte(' '+ i%('~'-' '));
	}
	for i := 0; i < len(bufsizes); i++ {
		for j := 0; j < len(bufsizes); j++ {
			for k := 0; k < len(writers); k++ {
				nwrite := bufsizes[i];
				bs := bufsizes[j];

				// Write nwrite bytes using buffer size bs.
				// Check that the right amount makes it out
				// and that the data is correct.

				write := writers[k].fn();
				buf, e := NewBufWriteSize(write, bs);
				context := fmt.Sprintf("write=%s nwrite=%d bufsize=%d", writers[k].name, nwrite, bs);
				if e != nil {
					t.Errorf("%s: NewBufWriteSize %d: %v", context, bs, e);
					continue;
				}
				n, e1 := buf.Write(data[0:nwrite]);
				if e1 != nil || n != nwrite {
					t.Errorf("%s: buf.Write %d = %d, %v", context, nwrite, n, e1);
					continue;
				}
				if e = buf.Flush(); e != nil {
					t.Errorf("%s: buf.Flush = %v", context, e);
				}

				written := write.GetBytes();
				if len(written) != nwrite {
					t.Errorf("%s: %d bytes written", context, len(written));
				}
				for l := 0; l < len(written); l++ {
					if written[i] != data[i] {
						t.Errorf("%s: wrong bytes written");
						t.Errorf("want=%s", data[0:len(written)]);
						t.Errorf("have=%s", written);
					}
				}
			}
		}
	}
}

