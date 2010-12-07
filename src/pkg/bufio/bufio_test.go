// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"
	"testing/iotest"
	"utf8"
)

// Reads from a reader and rot13s the result.
type rot13Reader struct {
	r io.Reader
}

func newRot13Reader(r io.Reader) *rot13Reader {
	r13 := new(rot13Reader)
	r13.r = r
	return r13
}

func (r13 *rot13Reader) Read(p []byte) (int, os.Error) {
	n, e := r13.r.Read(p)
	if e != nil {
		return n, e
	}
	for i := 0; i < n; i++ {
		c := p[i] | 0x20 // lowercase byte
		if 'a' <= c && c <= 'm' {
			p[i] += 13
		} else if 'n' <= c && c <= 'z' {
			p[i] -= 13
		}
	}
	return n, nil
}

// Call ReadByte to accumulate the text of a file
func readBytes(buf *Reader) string {
	var b [1000]byte
	nb := 0
	for {
		c, e := buf.ReadByte()
		if e == os.EOF {
			break
		}
		if e != nil {
			panic("Data: " + e.String())
		}
		b[nb] = c
		nb++
	}
	return string(b[0:nb])
}

func TestReaderSimple(t *testing.T) {
	data := "hello world"
	b := NewReader(bytes.NewBufferString(data))
	if s := readBytes(b); s != "hello world" {
		t.Errorf("simple hello world test failed: got %q", s)
	}

	b = NewReader(newRot13Reader(bytes.NewBufferString(data)))
	if s := readBytes(b); s != "uryyb jbeyq" {
		t.Errorf("rot13 hello world test failed: got %q", s)
	}
}


type readMaker struct {
	name string
	fn   func(io.Reader) io.Reader
}

var readMakers = []readMaker{
	{"full", func(r io.Reader) io.Reader { return r }},
	{"byte", iotest.OneByteReader},
	{"half", iotest.HalfReader},
	{"data+err", iotest.DataErrReader},
}

// Call ReadString (which ends up calling everything else)
// to accumulate the text of a file.
func readLines(b *Reader) string {
	s := ""
	for {
		s1, e := b.ReadString('\n')
		if e == os.EOF {
			break
		}
		if e != nil {
			panic("GetLines: " + e.String())
		}
		s += s1
	}
	return s
}

// Call Read to accumulate the text of a file
func reads(buf *Reader, m int) string {
	var b [1000]byte
	nb := 0
	for {
		n, e := buf.Read(b[nb : nb+m])
		nb += n
		if e == os.EOF {
			break
		}
	}
	return string(b[0:nb])
}

type bufReader struct {
	name string
	fn   func(*Reader) string
}

var bufreaders = []bufReader{
	{"1", func(b *Reader) string { return reads(b, 1) }},
	{"2", func(b *Reader) string { return reads(b, 2) }},
	{"3", func(b *Reader) string { return reads(b, 3) }},
	{"4", func(b *Reader) string { return reads(b, 4) }},
	{"5", func(b *Reader) string { return reads(b, 5) }},
	{"7", func(b *Reader) string { return reads(b, 7) }},
	{"bytes", readBytes},
	{"lines", readLines},
}

var bufsizes = []int{
	1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
	23, 32, 46, 64, 93, 128, 1024, 4096,
}

func TestReader(t *testing.T) {
	var texts [31]string
	str := ""
	all := ""
	for i := 0; i < len(texts)-1; i++ {
		texts[i] = str + "\n"
		all += texts[i]
		str += string(i%26 + 'a')
	}
	texts[len(texts)-1] = all

	for h := 0; h < len(texts); h++ {
		text := texts[h]
		for i := 0; i < len(readMakers); i++ {
			for j := 0; j < len(bufreaders); j++ {
				for k := 0; k < len(bufsizes); k++ {
					readmaker := readMakers[i]
					bufreader := bufreaders[j]
					bufsize := bufsizes[k]
					read := readmaker.fn(bytes.NewBufferString(text))
					buf, _ := NewReaderSize(read, bufsize)
					s := bufreader.fn(buf)
					if s != text {
						t.Errorf("reader=%s fn=%s bufsize=%d want=%q got=%q",
							readmaker.name, bufreader.name, bufsize, text, s)
					}
				}
			}
		}
	}
}

// A StringReader delivers its data one string segment at a time via Read.
type StringReader struct {
	data []string
	step int
}

func (r *StringReader) Read(p []byte) (n int, err os.Error) {
	if r.step < len(r.data) {
		s := r.data[r.step]
		n = copy(p, s)
		r.step++
	} else {
		err = os.EOF
	}
	return
}

func readRuneSegments(t *testing.T, segments []string) {
	got := ""
	want := strings.Join(segments, "")
	r := NewReader(&StringReader{data: segments})
	for {
		rune, _, err := r.ReadRune()
		if err != nil {
			if err != os.EOF {
				return
			}
			break
		}
		got += string(rune)
	}
	if got != want {
		t.Errorf("segments=%v got=%s want=%s", segments, got, want)
	}
}

var segmentList = [][]string{
	{},
	{""},
	{"日", "本語"},
	{"\u65e5", "\u672c", "\u8a9e"},
	{"\U000065e5", "\U0000672c", "\U00008a9e"},
	{"\xe6", "\x97\xa5\xe6", "\x9c\xac\xe8\xaa\x9e"},
	{"Hello", ", ", "World", "!"},
	{"Hello", ", ", "", "World", "!"},
}

func TestReadRune(t *testing.T) {
	for _, s := range segmentList {
		readRuneSegments(t, s)
	}
}

func TestUnreadRune(t *testing.T) {
	got := ""
	segments := []string{"Hello, world:", "日本語"}
	data := strings.Join(segments, "")
	r := NewReader(&StringReader{data: segments})
	// Normal execution.
	for {
		rune, _, err := r.ReadRune()
		if err != nil {
			if err != os.EOF {
				t.Error("unexpected EOF")
			}
			break
		}
		got += string(rune)
		// Put it back and read it again
		if err = r.UnreadRune(); err != nil {
			t.Error("unexpected error on UnreadRune:", err)
		}
		rune1, _, err := r.ReadRune()
		if err != nil {
			t.Error("unexpected error reading after unreading:", err)
		}
		if rune != rune1 {
			t.Errorf("incorrect rune after unread: got %c wanted %c", rune1, rune)
		}
	}
	if got != data {
		t.Errorf("want=%q got=%q", data, got)
	}
}

// Test that UnreadRune fails if the preceding operation was not a ReadRune.
func TestUnreadRuneError(t *testing.T) {
	buf := make([]byte, 3) // All runes in this test are 3 bytes long
	r := NewReader(&StringReader{data: []string{"日本語日本語日本語"}})
	if r.UnreadRune() == nil {
		t.Error("expected error on UnreadRune from fresh buffer")
	}
	_, _, err := r.ReadRune()
	if err != nil {
		t.Error("unexpected error on ReadRune (1):", err)
	}
	if err = r.UnreadRune(); err != nil {
		t.Error("unexpected error on UnreadRune (1):", err)
	}
	if r.UnreadRune() == nil {
		t.Error("expected error after UnreadRune (1)")
	}
	// Test error after Read.
	_, _, err = r.ReadRune() // reset state
	if err != nil {
		t.Error("unexpected error on ReadRune (2):", err)
	}
	_, err = r.Read(buf)
	if err != nil {
		t.Error("unexpected error on Read (2):", err)
	}
	if r.UnreadRune() == nil {
		t.Error("expected error after Read (2)")
	}
	// Test error after ReadByte.
	_, _, err = r.ReadRune() // reset state
	if err != nil {
		t.Error("unexpected error on ReadRune (2):", err)
	}
	for _ = range buf {
		_, err = r.ReadByte()
		if err != nil {
			t.Error("unexpected error on ReadByte (2):", err)
		}
	}
	if r.UnreadRune() == nil {
		t.Error("expected error after ReadByte")
	}
	// Test error after UnreadByte.
	_, _, err = r.ReadRune() // reset state
	if err != nil {
		t.Error("unexpected error on ReadRune (3):", err)
	}
	_, err = r.ReadByte()
	if err != nil {
		t.Error("unexpected error on ReadByte (3):", err)
	}
	err = r.UnreadByte()
	if err != nil {
		t.Error("unexpected error on UnreadByte (3):", err)
	}
	if r.UnreadRune() == nil {
		t.Error("expected error after UnreadByte (3)")
	}
}

func TestUnreadRuneAtEOF(t *testing.T) {
	// UnreadRune/ReadRune should error at EOF (was a bug; used to panic)
	r := NewReader(strings.NewReader("x"))
	r.ReadRune()
	r.ReadRune()
	r.UnreadRune()
	_, _, err := r.ReadRune()
	if err == nil {
		t.Error("expected error at EOF")
	} else if err != os.EOF {
		t.Error("expected EOF; got", err)
	}
}

func TestReadWriteRune(t *testing.T) {
	const NRune = 1000
	byteBuf := new(bytes.Buffer)
	w := NewWriter(byteBuf)
	// Write the runes out using WriteRune
	buf := make([]byte, utf8.UTFMax)
	for rune := 0; rune < NRune; rune++ {
		size := utf8.EncodeRune(buf, rune)
		nbytes, err := w.WriteRune(rune)
		if err != nil {
			t.Fatalf("WriteRune(0x%x) error: %s", rune, err)
		}
		if nbytes != size {
			t.Fatalf("WriteRune(0x%x) expected %d, got %d", rune, size, nbytes)
		}
	}
	w.Flush()

	r := NewReader(byteBuf)
	// Read them back with ReadRune
	for rune := 0; rune < NRune; rune++ {
		size := utf8.EncodeRune(buf, rune)
		nr, nbytes, err := r.ReadRune()
		if nr != rune || nbytes != size || err != nil {
			t.Fatalf("ReadRune(0x%x) got 0x%x,%d not 0x%x,%d (err=%s)", r, nr, nbytes, r, size, err)
		}
	}
}

func TestWriter(t *testing.T) {
	var data [8192]byte

	for i := 0; i < len(data); i++ {
		data[i] = byte(' ' + i%('~'-' '))
	}
	w := new(bytes.Buffer)
	for i := 0; i < len(bufsizes); i++ {
		for j := 0; j < len(bufsizes); j++ {
			nwrite := bufsizes[i]
			bs := bufsizes[j]

			// Write nwrite bytes using buffer size bs.
			// Check that the right amount makes it out
			// and that the data is correct.

			w.Reset()
			buf, e := NewWriterSize(w, bs)
			context := fmt.Sprintf("nwrite=%d bufsize=%d", nwrite, bs)
			if e != nil {
				t.Errorf("%s: NewWriterSize %d: %v", context, bs, e)
				continue
			}
			n, e1 := buf.Write(data[0:nwrite])
			if e1 != nil || n != nwrite {
				t.Errorf("%s: buf.Write %d = %d, %v", context, nwrite, n, e1)
				continue
			}
			if e = buf.Flush(); e != nil {
				t.Errorf("%s: buf.Flush = %v", context, e)
			}

			written := w.Bytes()
			if len(written) != nwrite {
				t.Errorf("%s: %d bytes written", context, len(written))
			}
			for l := 0; l < len(written); l++ {
				if written[i] != data[i] {
					t.Errorf("wrong bytes written")
					t.Errorf("want=%q", data[0:len(written)])
					t.Errorf("have=%q", written)
				}
			}
		}
	}
}

// Check that write errors are returned properly.

type errorWriterTest struct {
	n, m   int
	err    os.Error
	expect os.Error
}

func (w errorWriterTest) Write(p []byte) (int, os.Error) {
	return len(p) * w.n / w.m, w.err
}

var errorWriterTests = []errorWriterTest{
	{0, 1, nil, io.ErrShortWrite},
	{1, 2, nil, io.ErrShortWrite},
	{1, 1, nil, nil},
	{0, 1, os.EPIPE, os.EPIPE},
	{1, 2, os.EPIPE, os.EPIPE},
	{1, 1, os.EPIPE, os.EPIPE},
}

func TestWriteErrors(t *testing.T) {
	for _, w := range errorWriterTests {
		buf := NewWriter(w)
		_, e := buf.Write([]byte("hello world"))
		if e != nil {
			t.Errorf("Write hello to %v: %v", w, e)
			continue
		}
		e = buf.Flush()
		if e != w.expect {
			t.Errorf("Flush %v: got %v, wanted %v", w, e, w.expect)
		}
	}
}

func TestNewReaderSizeIdempotent(t *testing.T) {
	const BufSize = 1000
	b, err := NewReaderSize(bytes.NewBufferString("hello world"), BufSize)
	if err != nil {
		t.Error("NewReaderSize create fail", err)
	}
	// Does it recognize itself?
	b1, err2 := NewReaderSize(b, BufSize)
	if err2 != nil {
		t.Error("NewReaderSize #2 create fail", err2)
	}
	if b1 != b {
		t.Error("NewReaderSize did not detect underlying Reader")
	}
	// Does it wrap if existing buffer is too small?
	b2, err3 := NewReaderSize(b, 2*BufSize)
	if err3 != nil {
		t.Error("NewReaderSize #3 create fail", err3)
	}
	if b2 == b {
		t.Error("NewReaderSize did not enlarge buffer")
	}
}

func TestNewWriterSizeIdempotent(t *testing.T) {
	const BufSize = 1000
	b, err := NewWriterSize(new(bytes.Buffer), BufSize)
	if err != nil {
		t.Error("NewWriterSize create fail", err)
	}
	// Does it recognize itself?
	b1, err2 := NewWriterSize(b, BufSize)
	if err2 != nil {
		t.Error("NewWriterSize #2 create fail", err2)
	}
	if b1 != b {
		t.Error("NewWriterSize did not detect underlying Writer")
	}
	// Does it wrap if existing buffer is too small?
	b2, err3 := NewWriterSize(b, 2*BufSize)
	if err3 != nil {
		t.Error("NewWriterSize #3 create fail", err3)
	}
	if b2 == b {
		t.Error("NewWriterSize did not enlarge buffer")
	}
}

func TestWriteString(t *testing.T) {
	const BufSize = 8
	buf := new(bytes.Buffer)
	b, err := NewWriterSize(buf, BufSize)
	if err != nil {
		t.Error("NewWriterSize create fail", err)
	}
	b.WriteString("0")                         // easy
	b.WriteString("123456")                    // still easy
	b.WriteString("7890")                      // easy after flush
	b.WriteString("abcdefghijklmnopqrstuvwxy") // hard
	b.WriteString("z")
	b.Flush()
	if b.err != nil {
		t.Error("WriteString", b.err)
	}
	s := "01234567890abcdefghijklmnopqrstuvwxyz"
	if string(buf.Bytes()) != s {
		t.Errorf("WriteString wants %q gets %q", s, string(buf.Bytes()))
	}
}

func TestBufferFull(t *testing.T) {
	buf, _ := NewReaderSize(strings.NewReader("hello, world"), 5)
	line, err := buf.ReadSlice(',')
	if string(line) != "hello" || err != ErrBufferFull {
		t.Errorf("first ReadSlice(,) = %q, %v", line, err)
	}
	line, err = buf.ReadSlice(',')
	if string(line) != "," || err != nil {
		t.Errorf("second ReadSlice(,) = %q, %v", line, err)
	}
}

func TestPeek(t *testing.T) {
	p := make([]byte, 10)
	buf, _ := NewReaderSize(strings.NewReader("abcdefghij"), 4)
	if s, err := buf.Peek(1); string(s) != "a" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "a", string(s), err)
	}
	if s, err := buf.Peek(4); string(s) != "abcd" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "abcd", string(s), err)
	}
	if _, err := buf.Peek(5); err != ErrBufferFull {
		t.Fatalf("want ErrBufFull got %v", err)
	}
	if _, err := buf.Read(p[0:3]); string(p[0:3]) != "abc" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "abc", string(p[0:3]), err)
	}
	if s, err := buf.Peek(1); string(s) != "d" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "d", string(s), err)
	}
	if s, err := buf.Peek(2); string(s) != "de" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "de", string(s), err)
	}
	if _, err := buf.Read(p[0:3]); string(p[0:3]) != "def" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "def", string(p[0:3]), err)
	}
	if s, err := buf.Peek(4); string(s) != "ghij" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "ghij", string(s), err)
	}
	if _, err := buf.Read(p[0:4]); string(p[0:4]) != "ghij" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "ghij", string(p[0:3]), err)
	}
	if s, err := buf.Peek(0); string(s) != "" || err != nil {
		t.Fatalf("want %q got %q, err=%v", "", string(s), err)
	}
	if _, err := buf.Peek(1); err != os.EOF {
		t.Fatalf("want EOF got %v", err)
	}
}

func TestPeekThenUnreadRune(t *testing.T) {
	// This sequence used to cause a crash.
	r := NewReader(strings.NewReader("x"))
	r.ReadRune()
	r.Peek(1)
	r.UnreadRune()
	r.ReadRune() // Used to panic here
}
