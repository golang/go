// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio_test

import (
	. "bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"testing/iotest"
	"unicode/utf8"
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

func (r13 *rot13Reader) Read(p []byte) (int, error) {
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
		if e == io.EOF {
			break
		}
		if e == nil {
			b[nb] = c
			nb++
		} else if e != iotest.ErrTimeout {
			panic("Data: " + e.Error())
		}
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
	{"timeout", iotest.TimeoutReader},
}

// Call ReadString (which ends up calling everything else)
// to accumulate the text of a file.
func readLines(b *Reader) string {
	s := ""
	for {
		s1, e := b.ReadString('\n')
		if e == io.EOF {
			break
		}
		if e != nil && e != iotest.ErrTimeout {
			panic("GetLines: " + e.Error())
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
		if e == io.EOF {
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
	2, 3, 4, 5, 6, 7, 8, 9, 10,
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

func (r *StringReader) Read(p []byte) (n int, err error) {
	if r.step < len(r.data) {
		s := r.data[r.step]
		n = copy(p, s)
		r.step++
	} else {
		err = io.EOF
	}
	return
}

func readRuneSegments(t *testing.T, segments []string) {
	got := ""
	want := strings.Join(segments, "")
	r := NewReader(&StringReader{data: segments})
	for {
		r, _, err := r.ReadRune()
		if err != nil {
			if err != io.EOF {
				return
			}
			break
		}
		got += string(r)
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
		r1, _, err := r.ReadRune()
		if err != nil {
			if err != io.EOF {
				t.Error("unexpected EOF")
			}
			break
		}
		got += string(r1)
		// Put it back and read it again
		if err = r.UnreadRune(); err != nil {
			t.Error("unexpected error on UnreadRune:", err)
		}
		r2, _, err := r.ReadRune()
		if err != nil {
			t.Error("unexpected error reading after unreading:", err)
		}
		if r1 != r2 {
			t.Errorf("incorrect rune after unread: got %c wanted %c", r1, r2)
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
	} else if err != io.EOF {
		t.Error("expected EOF; got", err)
	}
}

func TestReadWriteRune(t *testing.T) {
	const NRune = 1000
	byteBuf := new(bytes.Buffer)
	w := NewWriter(byteBuf)
	// Write the runes out using WriteRune
	buf := make([]byte, utf8.UTFMax)
	for r := rune(0); r < NRune; r++ {
		size := utf8.EncodeRune(buf, r)
		nbytes, err := w.WriteRune(r)
		if err != nil {
			t.Fatalf("WriteRune(0x%x) error: %s", r, err)
		}
		if nbytes != size {
			t.Fatalf("WriteRune(0x%x) expected %d, got %d", r, size, nbytes)
		}
	}
	w.Flush()

	r := NewReader(byteBuf)
	// Read them back with ReadRune
	for r1 := rune(0); r1 < NRune; r1++ {
		size := utf8.EncodeRune(buf, r1)
		nr, nbytes, err := r.ReadRune()
		if nr != r1 || nbytes != size || err != nil {
			t.Fatalf("ReadRune(0x%x) got 0x%x,%d not 0x%x,%d (err=%s)", r1, nr, nbytes, r1, size, err)
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
	err    error
	expect error
}

func (w errorWriterTest) Write(p []byte) (int, error) {
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
	if err := b.Flush(); err != nil {
		t.Error("WriteString", err)
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
	if _, err := buf.Peek(1); err != io.EOF {
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

var testOutput = []byte("0123456789abcdefghijklmnopqrstuvwxy")
var testInput = []byte("012\n345\n678\n9ab\ncde\nfgh\nijk\nlmn\nopq\nrst\nuvw\nxy")
var testInputrn = []byte("012\r\n345\r\n678\r\n9ab\r\ncde\r\nfgh\r\nijk\r\nlmn\r\nopq\r\nrst\r\nuvw\r\nxy\r\n\n\r\n")

// TestReader wraps a []byte and returns reads of a specific length.
type testReader struct {
	data   []byte
	stride int
}

func (t *testReader) Read(buf []byte) (n int, err error) {
	n = t.stride
	if n > len(t.data) {
		n = len(t.data)
	}
	if n > len(buf) {
		n = len(buf)
	}
	copy(buf, t.data)
	t.data = t.data[n:]
	if len(t.data) == 0 {
		err = io.EOF
	}
	return
}

func testReadLine(t *testing.T, input []byte) {
	//for stride := 1; stride < len(input); stride++ {
	for stride := 1; stride < 2; stride++ {
		done := 0
		reader := testReader{input, stride}
		l, _ := NewReaderSize(&reader, len(input)+1)
		for {
			line, isPrefix, err := l.ReadLine()
			if len(line) > 0 && err != nil {
				t.Errorf("ReadLine returned both data and error: %s", err)
			}
			if isPrefix {
				t.Errorf("ReadLine returned prefix")
			}
			if err != nil {
				if err != io.EOF {
					t.Fatalf("Got unknown error: %s", err)
				}
				break
			}
			if want := testOutput[done : done+len(line)]; !bytes.Equal(want, line) {
				t.Errorf("Bad line at stride %d: want: %x got: %x", stride, want, line)
			}
			done += len(line)
		}
		if done != len(testOutput) {
			t.Errorf("ReadLine didn't return everything: got: %d, want: %d (stride: %d)", done, len(testOutput), stride)
		}
	}
}

func TestReadLine(t *testing.T) {
	testReadLine(t, testInput)
	testReadLine(t, testInputrn)
}

func TestLineTooLong(t *testing.T) {
	buf := bytes.NewBuffer([]byte("aaabbbcc\n"))
	l, _ := NewReaderSize(buf, 3)
	line, isPrefix, err := l.ReadLine()
	if !isPrefix || !bytes.Equal(line, []byte("aaa")) || err != nil {
		t.Errorf("bad result for first line: %x %s", line, err)
	}
	line, isPrefix, err = l.ReadLine()
	if !isPrefix || !bytes.Equal(line, []byte("bbb")) || err != nil {
		t.Errorf("bad result for second line: %x", line)
	}
	line, isPrefix, err = l.ReadLine()
	if isPrefix || !bytes.Equal(line, []byte("cc")) || err != nil {
		t.Errorf("bad result for third line: %x", line)
	}
	line, isPrefix, err = l.ReadLine()
	if isPrefix || err == nil {
		t.Errorf("expected no more lines: %x %s", line, err)
	}
}

func TestReadAfterLines(t *testing.T) {
	line1 := "line1"
	restData := "line2\nline 3\n"
	inbuf := bytes.NewBuffer([]byte(line1 + "\n" + restData))
	outbuf := new(bytes.Buffer)
	maxLineLength := len(line1) + len(restData)/2
	l, _ := NewReaderSize(inbuf, maxLineLength)
	line, isPrefix, err := l.ReadLine()
	if isPrefix || err != nil || string(line) != line1 {
		t.Errorf("bad result for first line: isPrefix=%v err=%v line=%q", isPrefix, err, string(line))
	}
	n, err := io.Copy(outbuf, l)
	if int(n) != len(restData) || err != nil {
		t.Errorf("bad result for Read: n=%d err=%v", n, err)
	}
	if outbuf.String() != restData {
		t.Errorf("bad result for Read: got %q; expected %q", outbuf.String(), restData)
	}
}

func TestReadEmptyBuffer(t *testing.T) {
	l, _ := NewReaderSize(bytes.NewBuffer(nil), 10)
	line, isPrefix, err := l.ReadLine()
	if err != io.EOF {
		t.Errorf("expected EOF from ReadLine, got '%s' %t %s", line, isPrefix, err)
	}
}

func TestLinesAfterRead(t *testing.T) {
	l, _ := NewReaderSize(bytes.NewBuffer([]byte("foo")), 10)
	_, err := ioutil.ReadAll(l)
	if err != nil {
		t.Error(err)
		return
	}

	line, isPrefix, err := l.ReadLine()
	if err != io.EOF {
		t.Errorf("expected EOF from ReadLine, got '%s' %t %s", line, isPrefix, err)
	}
}

func TestReadLineNonNilLineOrError(t *testing.T) {
	r := NewReader(strings.NewReader("line 1\n"))
	for i := 0; i < 2; i++ {
		l, _, err := r.ReadLine()
		if l != nil && err != nil {
			t.Fatalf("on line %d/2; ReadLine=%#v, %v; want non-nil line or Error, but not both",
				i+1, l, err)
		}
	}
}

type readLineResult struct {
	line     []byte
	isPrefix bool
	err      error
}

var readLineNewlinesTests = []struct {
	input   string
	bufSize int
	expect  []readLineResult
}{
	{"h\r\nb\r\n", 2, []readLineResult{
		{[]byte("h"), true, nil},
		{nil, false, nil},
		{[]byte("b"), true, nil},
		{nil, false, nil},
		{nil, false, io.EOF},
	}},
	{"hello\r\nworld\r\n", 6, []readLineResult{
		{[]byte("hello"), true, nil},
		{nil, false, nil},
		{[]byte("world"), true, nil},
		{nil, false, nil},
		{nil, false, io.EOF},
	}},
	{"hello\rworld\r", 6, []readLineResult{
		{[]byte("hello"), true, nil},
		{[]byte("\rworld"), true, nil},
		{[]byte("\r"), false, nil},
		{nil, false, io.EOF},
	}},
	{"h\ri\r\n\r", 2, []readLineResult{
		{[]byte("h"), true, nil},
		{[]byte("\ri"), true, nil},
		{nil, false, nil},
		{[]byte("\r"), false, nil},
		{nil, false, io.EOF},
	}},
}

func TestReadLineNewlines(t *testing.T) {
	for _, e := range readLineNewlinesTests {
		testReadLineNewlines(t, e.input, e.bufSize, e.expect)
	}
}

func testReadLineNewlines(t *testing.T, input string, bufSize int, expect []readLineResult) {
	b, err := NewReaderSize(strings.NewReader(input), bufSize)
	if err != nil {
		t.Fatal(err)
	}
	for i, e := range expect {
		line, isPrefix, err := b.ReadLine()
		if bytes.Compare(line, e.line) != 0 {
			t.Errorf("%q call %d, line == %q, want %q", input, i, line, e.line)
			return
		}
		if isPrefix != e.isPrefix {
			t.Errorf("%q call %d, isPrefix == %v, want %v", input, i, isPrefix, e.isPrefix)
			return
		}
		if err != e.err {
			t.Errorf("%q call %d, err == %v, want %v", input, i, err, e.err)
			return
		}
	}
}
