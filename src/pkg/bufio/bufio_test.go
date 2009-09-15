// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio

import (
	"bytes";
	"fmt";
	"io";
	"os";
	"strings";
	"testing";
	"testing/iotest";
)

// Reads from a reader and rot13s the result.
type rot13Reader struct {
	r io.Reader
}

func newRot13Reader(r io.Reader) *rot13Reader {
	r13 := new(rot13Reader);
	r13.r = r;
	return r13
}

func (r13 *rot13Reader) Read(p []byte) (int, os.Error) {
	n, e := r13.r.Read(p);
	if e != nil {
		return n, e
	}
	for i := 0; i < n; i++ {
		c := p[i] | 0x20;	// lowercase byte
		if 'a' <= c && c <= 'm' {
			p[i] += 13;
		} else if 'n' <= c && c <= 'z' {
			p[i] -= 13;
		}
	}
	return n, nil
}

// Call ReadByte to accumulate the text of a file
func readBytes(buf *Reader) string {
	var b [1000]byte;
	nb := 0;
	for {
		c, e := buf.ReadByte();
		if e == os.EOF {
			break
		}
		if e != nil {
			panic("Data: "+e.String())
		}
		b[nb] = c;
		nb++;
	}
	return string(b[0:nb])
}

func TestReaderSimple(t *testing.T) {
	data := strings.Bytes("hello world");
	b := NewReader(bytes.NewBuffer(data));
	if s := readBytes(b); s != "hello world" {
		t.Errorf("simple hello world test failed: got %q", s);
	}

	b = NewReader(newRot13Reader(bytes.NewBuffer(data)));
	if s := readBytes(b); s != "uryyb jbeyq" {
		t.Error("rot13 hello world test failed: got %q", s);
	}
}


type readMaker struct {
	name string;
	fn func(io.Reader) io.Reader;
}
var readMakers = []readMaker {
	readMaker{ "full", func(r io.Reader) io.Reader { return r } },
	readMaker{ "byte", iotest.OneByteReader },
	readMaker{ "half", iotest.HalfReader },
	readMaker{ "data+err", iotest.DataErrReader },
}

// Call ReadString (which ends up calling everything else)
// to accumulate the text of a file.
func readLines(b *Reader) string {
	s := "";
	for {
		s1, e := b.ReadString('\n');
		if e == os.EOF {
			break
		}
		if e != nil {
			panic("GetLines: "+e.String())
		}
		s += s1
	}
	return s
}

// Call Read to accumulate the text of a file
func reads(buf *Reader, m int) string {
	var b [1000]byte;
	nb := 0;
	for {
		n, e := buf.Read(b[nb:nb+m]);
		nb += n;
		if e == os.EOF {
			break
		}
	}
	return string(b[0:nb])
}

type bufReader struct {
	name string;
	fn func(*Reader) string;
}
var bufreaders = []bufReader {
	bufReader{ "1", func(b *Reader) string { return reads(b, 1) } },
	bufReader{ "2", func(b *Reader) string { return reads(b, 2) } },
	bufReader{ "3", func(b *Reader) string { return reads(b, 3) } },
	bufReader{ "4", func(b *Reader) string { return reads(b, 4) } },
	bufReader{ "5", func(b *Reader) string { return reads(b, 5) } },
	bufReader{ "7", func(b *Reader) string { return reads(b, 7) } },
	bufReader{ "bytes", readBytes },
	bufReader{ "lines", readLines },
}

var bufsizes = []int {
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	23, 32, 46, 64, 93, 128, 1024, 4096
}

func TestReader(t *testing.T) {
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
		textbytes := strings.Bytes(text);
		for i := 0; i < len(readMakers); i++ {
			for j := 0; j < len(bufreaders); j++ {
				for k := 0; k < len(bufsizes); k++ {
					readmaker := readMakers[i];
					bufreader := bufreaders[j];
					bufsize := bufsizes[k];
					read := readmaker.fn(bytes.NewBuffer(textbytes));
					buf, _ := NewReaderSize(read, bufsize);
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

// A StringReader delivers its data one string segment at a time via Read.
type StringReader struct {
	data []string;
	step int;
}

func (r *StringReader) Read (p []byte) (n int, err os.Error) {
	if r.step < len(r.data) {
		s := r.data[r.step];
		for i := 0; i < len(s); i++ {
			p[i] = s[i];
		}
		n = len(s);
		r.step++;
	} else {
		err = os.EOF;
	}
	return;
}

func readRuneSegments(t *testing.T, segments []string) {
	got := "";
	want := strings.Join(segments, "");
	r := NewReader(&StringReader{data: segments});
	for {
		rune, _, err := r.ReadRune();
		if err != nil {
			if err != os.EOF {
				return;
			}
			break;
		}
		got += string(rune);
	}
	if got != want {
		t.Errorf("segments=%v got=%s want=%s", segments, got, want);
	}
}

var segmentList = [][]string {
	[]string{},
	[]string{""},
	[]string{"日", "本語"},
	[]string{"\u65e5", "\u672c", "\u8a9e"},
	[]string{"\U000065e5, "", \U0000672c", "\U00008a9e"},
	[]string{"\xe6", "\x97\xa5\xe6", "\x9c\xac\xe8\xaa\x9e"},
	[]string{"Hello", ", ", "World", "!"},
	[]string{"Hello", ", ", "", "World", "!"},
}

func TestReadRune(t *testing.T) {
	for _, s := range segmentList {
		readRuneSegments(t, s);
	}
}

func TestWriter(t *testing.T) {
	var data [8192]byte;

	for i := 0; i < len(data); i++ {
		data[i] = byte(' '+ i%('~'-' '));
	}
	w := new(bytes.Buffer);
	for i := 0; i < len(bufsizes); i++ {
		for j := 0; j < len(bufsizes); j++ {
			nwrite := bufsizes[i];
			bs := bufsizes[j];

			// Write nwrite bytes using buffer size bs.
			// Check that the right amount makes it out
			// and that the data is correct.

			w.Reset();
			buf, e := NewWriterSize(w, bs);
			context := fmt.Sprintf("nwrite=%d bufsize=%d", nwrite, bs);
			if e != nil {
				t.Errorf("%s: NewWriterSize %d: %v", context, bs, e);
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

			written := w.Data();
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

// Check that write errors are returned properly.

type errorWriterTest struct {
	n, m int;
	err os.Error;
	expect os.Error;
}

func (w errorWriterTest) Write(p []byte) (int, os.Error) {
	return len(p)*w.n/w.m, w.err;
}

var errorWriterTests = []errorWriterTest {
	errorWriterTest{ 0, 1, nil, io.ErrShortWrite },
	errorWriterTest{ 1, 2, nil, io.ErrShortWrite },
	errorWriterTest{ 1, 1, nil, nil },
	errorWriterTest{ 0, 1, os.EPIPE, os.EPIPE },
	errorWriterTest{ 1, 2, os.EPIPE, os.EPIPE },
	errorWriterTest{ 1, 1, os.EPIPE, os.EPIPE },
}

func TestWriteErrors(t *testing.T) {
	for _, w := range errorWriterTests {
		buf := NewWriter(w);
		_, e := buf.Write(strings.Bytes("hello world"));
		if e != nil {
			t.Errorf("Write hello to %v: %v", w, e);
			continue;
		}
		e = buf.Flush();
		if e != w.expect {
			t.Errorf("Flush %v: got %v, wanted %v", w, e, w.expect);
		}
	}
}

func TestNewReaderSizeIdempotent(t *testing.T) {
	const BufSize = 1000;
	b, err := NewReaderSize(bytes.NewBuffer(strings.Bytes("hello world")), BufSize);
	if err != nil {
		t.Error("NewReaderSize create fail", err);
	}
	// Does it recognize itself?
	b1, err2 := NewReaderSize(b, BufSize);
	if err2 != nil {
		t.Error("NewReaderSize #2 create fail", err2);
	}
	if b1 != b {
		t.Error("NewReaderSize did not detect underlying Reader");
	}
	// Does it wrap if existing buffer is too small?
	b2, err3 := NewReaderSize(b, 2*BufSize);
	if err3 != nil {
		t.Error("NewReaderSize #3 create fail", err3);
	}
	if b2 == b {
		t.Error("NewReaderSize did not enlarge buffer");
	}
}

func TestNewWriterSizeIdempotent(t *testing.T) {
	const BufSize = 1000;
	b, err := NewWriterSize(new(bytes.Buffer), BufSize);
	if err != nil {
		t.Error("NewWriterSize create fail", err);
	}
	// Does it recognize itself?
	b1, err2 := NewWriterSize(b, BufSize);
	if err2 != nil {
		t.Error("NewWriterSize #2 create fail", err2);
	}
	if b1 != b {
		t.Error("NewWriterSize did not detect underlying Writer");
	}
	// Does it wrap if existing buffer is too small?
	b2, err3 := NewWriterSize(b, 2*BufSize);
	if err3 != nil {
		t.Error("NewWriterSize #3 create fail", err3);
	}
	if b2 == b {
		t.Error("NewWriterSize did not enlarge buffer");
	}
}

func TestWriteString(t *testing.T) {
	const BufSize = 8;
	buf := new(bytes.Buffer);
	b, err := NewWriterSize(buf, BufSize);
	if err != nil {
		t.Error("NewWriterSize create fail", err);
	}
	b.WriteString("0");	// easy
	b.WriteString("123456");	// still easy
	b.WriteString("7890");	// easy after flush
	b.WriteString("abcdefghijklmnopqrstuvwxy");	// hard
	b.WriteString("z");
	b.Flush();
	if b.err != nil {
		t.Error("WriteString", b.err);
	}
	s := "01234567890abcdefghijklmnopqrstuvwxyz";
	if string(buf.Data()) != s {
		t.Errorf("WriteString wants %q gets %q", s, string(buf.Data()))
	}
}
