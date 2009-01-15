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

func StringToBytes(s string) []byte {
	b := make([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	return b
}

// Should be in language!
func Copy(p []byte, q []byte) {
	for i := 0; i < len(p); i++ {
		p[i] = q[i]
	}
}

// Reads from p.
type ByteReader struct {
	p []byte
}

func NewByteReader(p []byte) io.Read {
	b := new(ByteReader);
	b.p = p;
	return b
}

func (b *ByteReader) Read(p []byte) (int, *os.Error) {
	n := len(p);
	if n > len(b.p) {
		n = len(b.p)
	}
	Copy(p[0:n], b.p[0:n]);
	b.p = b.p[n:len(b.p)];
	return n, nil
}


// Reads from p but only returns half of what you asked for.
type HalfByteReader struct {
	p []byte
}

func NewHalfByteReader(p []byte) io.Read {
	b := new(HalfByteReader);
	b.p = p;
	return b
}

func (b *HalfByteReader) Read(p []byte) (int, *os.Error) {
	n := len(p)/2;
	if n == 0 && len(p) > 0 {
		n = 1
	}
	if n > len(b.p) {
		n = len(b.p)
	}
	Copy(p[0:n], b.p[0:n]);
	b.p = b.p[n:len(b.p)];
	return n, nil
}

// Reads from a reader and rot13s the result.
type Rot13Reader struct {
	r io.Read
}

func NewRot13Reader(r io.Read) *Rot13Reader {
	r13 := new(Rot13Reader);
	r13.r = r;
	return r13
}

func (r13 *Rot13Reader) Read(p []byte) (int, *os.Error) {
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

type Readmaker struct {
	name string;
	fn *([]byte) io.Read;
}
var readmakers = []Readmaker {
	Readmaker{ "full", func(p []byte) io.Read { return NewByteReader(p) } },
	Readmaker{ "half", func(p []byte) io.Read { return NewHalfByteReader(p) } },
}

// Call ReadLineString (which ends up calling everything else)
// to accumulate the text of a file.
func ReadLines(b *BufRead) string {
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
func ReadBytes(buf *BufRead) string {
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
func Reads(buf *BufRead, m int) string {
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

type Bufreader struct {
	name string;
	fn *(*BufRead) string;
}
var bufreaders = []Bufreader {
	Bufreader{ "1", func(b *BufRead) string { return Reads(b, 1) } },
	Bufreader{ "2", func(b *BufRead) string { return Reads(b, 2) } },
	Bufreader{ "3", func(b *BufRead) string { return Reads(b, 3) } },
	Bufreader{ "4", func(b *BufRead) string { return Reads(b, 4) } },
	Bufreader{ "5", func(b *BufRead) string { return Reads(b, 5) } },
	Bufreader{ "7", func(b *BufRead) string { return Reads(b, 7) } },
	Bufreader{ "bytes", &ReadBytes },
	Bufreader{ "lines", &ReadLines },
}

var bufsizes = []int {
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	23, 32, 46, 64, 93, 128, 1024, 4096
}

export func TestBufReadSimple(t *testing.T) {
	b, e := NewBufRead(NewByteReader(StringToBytes("hello world")));
	if s := ReadBytes(b); s != "hello world" {
		t.Errorf("simple hello world test failed: got %q", s);
	}

	b, e = NewBufRead(NewRot13Reader(NewByteReader(StringToBytes("hello world"))));
	if s := ReadBytes(b); s != "uryyb jbeyq" {
		t.Error("rot13 hello world test failed: got %q", s);
	}
}

export func TestBufRead(t *testing.T) {
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
		textbytes := StringToBytes(text);
		for i := 0; i < len(readmakers); i++ {
			for j := 0; j < len(bufreaders); j++ {
				for k := 0; k < len(bufsizes); k++ {
					readmaker := readmakers[i];
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

type WriteBuffer interface {
	Write(p []byte) (int, *os.Error);
	GetBytes() []byte
}

// Accumulates bytes into a byte array.
type ByteWriter struct {
	p []byte;
	n int
}

func NewByteWriter() WriteBuffer {
	return new(ByteWriter)
}

func (w *ByteWriter) Write(p []byte) (int, *os.Error) {
	if w.p == nil {
		w.p = make([]byte, len(p)+100)
	} else if w.n + len(p) >= len(w.p) {
		newp := make([]byte, len(w.p)*2 + len(p));
		Copy(newp[0:w.n], w.p[0:w.n]);
		w.p = newp
	}
	Copy(w.p[w.n:w.n+len(p)], p);
	w.n += len(p);
	return len(p), nil
}

func (w *ByteWriter) GetBytes() []byte {
	return w.p[0:w.n]
}

// Accumulates bytes written into a byte array
// but Write only takes half of what you give it.
// TODO: Could toss this -- Write() is not supposed to do that.
type HalfByteWriter struct {
	bw WriteBuffer
}

func NewHalfByteWriter() WriteBuffer {
	w := new(HalfByteWriter);
	w.bw = NewByteWriter();
	return w
}

func (w *HalfByteWriter) Write(p []byte) (int, *os.Error) {
	n := (len(p)+1) / 2;
	// BUG return w.bw.Write(p[0:n])
	r, e := w.bw.Write(p[0:n]);
	return r, e
}

func (w *HalfByteWriter) GetBytes() []byte {
	return w.bw.GetBytes()
}

type Writemaker struct {
	name string;
	fn *()WriteBuffer;
}
export func TestBufWrite(t *testing.T) {
	var data [8192]byte;

	var writers = []Writemaker {
		Writemaker{ "full", &NewByteWriter },
		Writemaker{ "half", &NewHalfByteWriter },
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

