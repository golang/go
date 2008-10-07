// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// $G $F.go && $L $F.$A && ./$A.out

package main

import (
	"os";
	"io";
	"bufio";
	"syscall";
	"rand"
)

func StringToBytes(s string) *[]byte {
	b := new([]byte, len(s));
	for i := 0; i < len(s); i++ {
		b[i] = s[i]
	}
	return b
}

// Should be in language!
func Copy(p *[]byte, q *[]byte) {
	for i := 0; i < len(p); i++ {
		p[i] = q[i]
	}
}

// Reads from p.
type ByteReader struct {
	p *[]byte
}

func NewByteReader(p *[]byte) io.Read {
	b := new(ByteReader);
	b.p = p;
	return b
}

func (b *ByteReader) Read(p *[]byte) (int, *os.Error) {
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
	p *[]byte
}

func NewHalfByteReader(p *[]byte) io.Read {
	b := new(HalfByteReader);
	b.p = p;
	return b
}

func (b *HalfByteReader) Read(p *[]byte) (int, *os.Error) {
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

func (r13 *Rot13Reader) Read(p *[]byte) (int, *os.Error) {
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

func MakeByteReader(p *[]byte) io.Read {
	return NewByteReader(p)
}
func MakeHalfByteReader(p *[]byte) io.Read {
	return NewHalfByteReader(p)
}

var readmakers = []*(p *[]byte) io.Read {
	&NewByteReader,
	&NewHalfByteReader
}


// Call ReadLineString (which ends up calling everything else)
// to accumulate the text of a file.
func ReadLines(b *bufio.BufRead) string {
	s := "";
	for {
		s1, e := b.ReadLineString('\n', true);
		if e == bufio.EndOfFile {
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
func ReadBytes(buf *bufio.BufRead) string {
	var b [1000]byte;
	nb := 0;
	for {
		c, e := buf.ReadByte();
		if e == bufio.EndOfFile {
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
func Reads(buf *bufio.BufRead, m int) string {
	var b [1000]byte;
	nb := 0;
	for {
		// BUG parens around (&b) should not be needed
		n, e := buf.Read((&b)[nb:nb+m]);
		nb += n;
		if e == bufio.EndOfFile {
			break
		}
	}
	// BUG 6g bug102 - out of bounds error on empty byte array -> string
	if nb == 0 { return "" }
	return string((&b)[0:nb])
}

func Read1(b *bufio.BufRead) string { return Reads(b, 1) }
func Read2(b *bufio.BufRead) string { return Reads(b, 2) }
func Read3(b *bufio.BufRead) string { return Reads(b, 3) }
func Read4(b *bufio.BufRead) string { return Reads(b, 4) }
func Read5(b *bufio.BufRead) string { return Reads(b, 5) }
func Read7(b *bufio.BufRead) string { return Reads(b, 7) }

var bufreaders = []*(b *bufio.BufRead) string {
	&Read1, &Read2, &Read3, &Read4, &Read5, &Read7,
	&ReadBytes, &ReadLines
}

var bufsizes = []int {
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	23, 32, 46, 64, 93, 128, 1024, 4096
}

func TestBufRead() {
	// work around 6g bug101
	readmakers[0] = &NewByteReader;
	readmakers[1] = &NewHalfByteReader;

	bufreaders[0] = &Read1;
	bufreaders[1] = &Read2;
	bufreaders[2] = &Read3;
	bufreaders[3] = &Read4;
	bufreaders[4] = &Read5;
	bufreaders[5] = &Read7;
	bufreaders[6] = &ReadBytes;
	bufreaders[7] = &ReadLines;

	bufsizes[0] = 1;
	bufsizes[1] = 2;
	bufsizes[2] = 3;
	bufsizes[3] = 4;
	bufsizes[4] = 5;
	bufsizes[5] = 6;
	bufsizes[6] = 7;
	bufsizes[7] = 8;
	bufsizes[8] = 9;
	bufsizes[9] = 10;
	bufsizes[10] = 23;
	bufsizes[11] = 32;
	bufsizes[12] = 46;
	bufsizes[13] = 64;
	bufsizes[14] = 93;
	bufsizes[15] = 128;
	bufsizes[16] = 1024;
	bufsizes[17] = 4096;

	var texts [31]string;
	str := "";
	all := "";
	for i := 0; i < len(texts)-1; i++ {
		texts[i] = str + "\n";
		all += texts[i];
		str += string(i%26+'a')
	}
	texts[len(texts)-1] = all;

	// BUG 6g should not need nbr temporary (bug099)
	nbr := NewByteReader(StringToBytes("hello world"));
	b, e := bufio.NewBufRead(nbr);
	if ReadBytes(b) != "hello world" { panic("hello world") }

	// BUG 6g should not need nbr nor nbr1 (bug009)
	nbr = NewByteReader(StringToBytes("hello world"));
	nbr1 := NewRot13Reader(nbr);
	b, e = bufio.NewBufRead(nbr1);
	if ReadBytes(b) != "uryyb jbeyq" { panic("hello world") }

	for h := 0; h < len(texts); h++ {
		text := texts[h];
		textbytes := StringToBytes(text);
		for i := 0; i < len(readmakers); i++ {
			readmaker := readmakers[i];
			for j := 0; j < len(bufreaders); j++ {
				bufreader := bufreaders[j];
				for k := 0; k < len(bufsizes); k++ {
					bufsize := bufsizes[k];
					read := readmaker(textbytes);
					buf, e := bufio.NewBufReadSize(read, bufsize);
					s := bufreader(buf);
					if s != text {
						print("Failed: ", h, " ", i, " ", j, " ", k, " ", len(s), " ", len(text), "\n");
						print("<", s, ">\nshould be <", text, ">\n");
						panic("bufio result")
					}
				}
			}
		}
	}
}


type WriteBuffer interface {
	Write(p *[]byte) (int, *os.Error);
	GetBytes() *[]byte
}

// Accumulates bytes into a byte array.
type ByteWriter struct {
	p *[]byte;
	n int
}

func NewByteWriter() WriteBuffer {
	return new(ByteWriter)
}

func (w *ByteWriter) Write(p *[]byte) (int, *os.Error) {
	if w.p == nil {
		w.p = new([]byte, len(p)+100)
	} else if w.n + len(p) >= len(w.p) {
		newp := new([]byte, len(w.p)*2 + len(p));
		Copy(newp[0:w.n], w.p[0:w.n]);
		w.p = newp
	}
	Copy(w.p[w.n:w.n+len(p)], p);
	w.n += len(p);
	return len(p), nil
}

func (w *ByteWriter) GetBytes() *[]byte {
	return w.p[0:w.n]
}

// Accumulates bytes written into a byte array
// but Write only takes half of what you give it.
type HalfByteWriter struct {
	bw WriteBuffer
}

func NewHalfByteWriter() WriteBuffer {
	w := new(HalfByteWriter);
	w.bw = NewByteWriter();
	return w
}

func (w *HalfByteWriter) Write(p *[]byte) (int, *os.Error) {
	n := (len(p)+1) / 2;
	// BUG return w.bw.Write(p[0:n])
	r, e := w.bw.Write(p[0:n]);
	return r, e
}

func (w *HalfByteWriter) GetBytes() *[]byte {
	return w.bw.GetBytes()
}

func TestBufWrite() {
	var data [8192]byte;

	var writers [2]*()WriteBuffer;
	writers[0] = &NewByteWriter;
	writers[1] = &NewHalfByteWriter;

	for i := 0; i < len(data); i++ {
		data[i] = byte(rand.rand())
	}
	for i := 0; i < len(bufsizes); i++ {
		for j := 0; j < len(bufsizes); j++ {
			for k := 0; k < len(writers); k++ {
				nwrite := bufsizes[i];
				bs := bufsizes[j];

				// Write nwrite bytes using buffer size bs.
				// Check that the right amount makes it out
				// and that the data is correct.

				write := writers[k]();
				buf, e := bufio.NewBufWriteSize(write, bs);
				if e != nil {
					panic("NewBufWriteSize error: "+e.String())
				}
				n, e1 := buf.Write((&data)[0:nwrite]);
				if e1 != nil {
					panic("buf.Write error "+e1.String())
				}
				if n != nwrite {
					panic("buf.Write wrong count")
				}
				e = buf.Flush();
				if e != nil {
					panic("buf.Flush error "+e.String())
				}

				written := write.GetBytes();
				if len(written) != nwrite {
					panic("wrong amount written")
				}
				for l := 0; l < len(written); l++ {
					if written[i] != data[i] {
						panic("wrong bytes written")
					}
				}
			}
		}
	}
}


func main() {
	TestBufRead();
	TestBufWrite()
}
