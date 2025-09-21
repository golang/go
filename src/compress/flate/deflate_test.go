// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"sync"
	"testing"
)

type deflateTest struct {
	in    []byte
	level int
	out   []byte
}

type deflateInflateTest struct {
	in []byte
}

type reverseBitsTest struct {
	in       uint16
	bitCount uint8
	out      uint16
}

var deflateTests = []*deflateTest{
	0: {[]byte{}, 0, []byte{0x3, 0x0}},
	1: {[]byte{0x11}, BestCompression, []byte{0x12, 0x4, 0xc, 0x0}},
	2: {[]byte{0x11}, BestCompression, []byte{0x12, 0x4, 0xc, 0x0}},
	3: {[]byte{0x11}, BestCompression, []byte{0x12, 0x4, 0xc, 0x0}},

	4: {[]byte{0x11}, 0, []byte{0x0, 0x1, 0x0, 0xfe, 0xff, 0x11, 0x3, 0x0}},
	5: {[]byte{0x11, 0x12}, 0, []byte{0x0, 0x2, 0x0, 0xfd, 0xff, 0x11, 0x12, 0x3, 0x0}},
	6: {[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, 0,
		[]byte{0x0, 0x8, 0x0, 0xf7, 0xff, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x3, 0x0},
	},
	7:  {[]byte{}, 1, []byte{0x3, 0x0}},
	8:  {[]byte{0x11}, BestCompression, []byte{0x12, 0x4, 0xc, 0x0}},
	9:  {[]byte{0x11, 0x12}, BestCompression, []byte{0x12, 0x14, 0x2, 0xc, 0x0}},
	10: {[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, BestCompression, []byte{0x12, 0x84, 0x1, 0xc0, 0x0}},
	11: {[]byte{}, 9, []byte{0x3, 0x0}},
	12: {[]byte{0x11}, 9, []byte{0x12, 0x4, 0xc, 0x0}},
	13: {[]byte{0x11, 0x12}, 9, []byte{0x12, 0x14, 0x2, 0xc, 0x0}},
	14: {[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, 9, []byte{0x12, 0x84, 0x1, 0xc0, 0x0}},
}

var deflateInflateTests = []*deflateInflateTest{
	{[]byte{}},
	{[]byte{0x11}},
	{[]byte{0x11, 0x12}},
	{[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}},
	{[]byte{0x11, 0x10, 0x13, 0x41, 0x21, 0x21, 0x41, 0x13, 0x87, 0x78, 0x13}},
	{largeDataChunk()},
}

var reverseBitsTests = []*reverseBitsTest{
	{1, 1, 1},
	{1, 2, 2},
	{1, 3, 4},
	{1, 4, 8},
	{1, 5, 16},
	{17, 5, 17},
	{257, 9, 257},
	{29, 5, 23},
}

func largeDataChunk() []byte {
	result := make([]byte, 100000)
	for i := range result {
		result[i] = byte(i * i & 0xFF)
	}
	return result
}

func TestBulkHash4(t *testing.T) {
	for _, x := range deflateTests {
		y := x.out
		if len(y) >= minMatchLength {
			y = append(y, y...)
			for j := 4; j < len(y); j++ {
				y := y[:j]
				dst := make([]uint32, len(y)-minMatchLength+1)
				for i := range dst {
					dst[i] = uint32(i + 100)
				}
				bulkHash4(y, dst)
				for i, got := range dst {
					want := hash4(y[i:])
					if got != want && got == uint32(i)+100 {
						t.Errorf("Len:%d Index:%d, expected 0x%08x but not modified", len(y), i, want)
					} else if got != want {
						t.Errorf("Len:%d Index:%d, got 0x%08x expected:0x%08x", len(y), i, got, want)
					} else {
						//t.Logf("Len:%d Index:%d OK (0x%08x)", len(y), i, got)
					}
				}
			}
		}
	}
}

func TestDeflate(t *testing.T) {
	for i, h := range deflateTests {
		var buf bytes.Buffer
		w, err := NewWriter(&buf, h.level)
		if err != nil {
			t.Errorf("NewWriter: %v", err)
			continue
		}
		w.Write(h.in)
		w.Close()
		if !bytes.Equal(buf.Bytes(), h.out) {
			t.Errorf("%d: Deflate(%d, %x) got \n%#v, want \n%#v", i, h.level, h.in, buf.Bytes(), h.out)
		}
	}
}

// A sparseReader returns a stream consisting of 0s followed by 1<<16 1s.
// This tests missing hash references in a very large input.
type sparseReader struct {
	l   int64
	cur int64
}

func (r *sparseReader) Read(b []byte) (n int, err error) {
	if r.cur >= r.l {
		return 0, io.EOF
	}
	n = len(b)
	cur := r.cur + int64(n)
	if cur > r.l {
		n -= int(cur - r.l)
		cur = r.l
	}
	for i := range b[0:n] {
		if r.cur+int64(i) >= r.l-1<<16 {
			b[i] = 1
		} else {
			b[i] = 0
		}
	}
	r.cur = cur
	return
}

func TestVeryLongSparseChunk(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping sparse chunk during short test")
	}
	var buf bytes.Buffer
	w, err := NewWriter(&buf, 1)
	if err != nil {
		t.Errorf("NewWriter: %v", err)
		return
	}
	if _, err = io.Copy(w, &sparseReader{l: 23e8}); err != nil {
		t.Errorf("Compress failed: %v", err)
		return
	}
	t.Log("Length:", buf.Len())
}

type syncBuffer struct {
	buf    bytes.Buffer
	mu     sync.RWMutex
	closed bool
	ready  chan bool
}

func newSyncBuffer() *syncBuffer {
	return &syncBuffer{ready: make(chan bool, 1)}
}

func (b *syncBuffer) Read(p []byte) (n int, err error) {
	for {
		b.mu.RLock()
		n, err = b.buf.Read(p)
		b.mu.RUnlock()
		if n > 0 || b.closed {
			return
		}
		<-b.ready
	}
}

func (b *syncBuffer) signal() {
	select {
	case b.ready <- true:
	default:
	}
}

func (b *syncBuffer) Write(p []byte) (n int, err error) {
	n, err = b.buf.Write(p)
	b.signal()
	return
}

func (b *syncBuffer) WriteMode() {
	b.mu.Lock()
}

func (b *syncBuffer) ReadMode() {
	b.mu.Unlock()
	b.signal()
}

func (b *syncBuffer) Close() error {
	b.closed = true
	b.signal()
	return nil
}

func testSync(t *testing.T, level int, input []byte, name string) {
	if len(input) == 0 {
		return
	}

	t.Logf("--testSync %d, %d, %s", level, len(input), name)
	buf := newSyncBuffer()
	buf1 := new(bytes.Buffer)
	buf.WriteMode()
	w, err := NewWriter(io.MultiWriter(buf, buf1), level)
	if err != nil {
		t.Errorf("NewWriter: %v", err)
		return
	}
	r := NewReader(buf)

	// Write half the input and read back.
	for i := range 2 {
		var lo, hi int
		if i == 0 {
			lo, hi = 0, (len(input)+1)/2
		} else {
			lo, hi = (len(input)+1)/2, len(input)
		}
		t.Logf("#%d: write %d-%d", i, lo, hi)
		if _, err := w.Write(input[lo:hi]); err != nil {
			t.Errorf("testSync: write: %v", err)
			return
		}
		if i == 0 {
			if err := w.Flush(); err != nil {
				t.Errorf("testSync: flush: %v", err)
				return
			}
		} else {
			if err := w.Close(); err != nil {
				t.Errorf("testSync: close: %v", err)
			}
		}
		buf.ReadMode()
		out := make([]byte, hi-lo+1)
		m, err := io.ReadAtLeast(r, out, hi-lo)
		t.Logf("#%d: read %d", i, m)
		if m != hi-lo || err != nil {
			t.Errorf("testSync/%d (%d, %d, %s): read %d: %d, %v (%d left)", i, level, len(input), name, hi-lo, m, err, buf.buf.Len())
			return
		}
		if !bytes.Equal(input[lo:hi], out[:hi-lo]) {
			t.Errorf("testSync/%d: read wrong bytes: %x vs %x", i, input[lo:hi], out[:hi-lo])
			return
		}
		// This test originally checked that after reading
		// the first half of the input, there was nothing left
		// in the read buffer (buf.buf.Len() != 0) but that is
		// not necessarily the case: the write Flush may emit
		// some extra framing bits that are not necessary
		// to process to obtain the first half of the uncompressed
		// data. The test ran correctly most of the time, because
		// the background goroutine had usually read even
		// those extra bits by now, but it's not a useful thing to
		// check.
		buf.WriteMode()
	}
	buf.ReadMode()
	out := make([]byte, 10)
	if n, err := r.Read(out); n > 0 || err != io.EOF {
		t.Errorf("testSync (%d, %d, %s): final Read: %d, %v (hex: %x)", level, len(input), name, n, err, out[0:n])
	}
	if buf.buf.Len() != 0 {
		t.Errorf("testSync (%d, %d, %s): extra data at end", level, len(input), name)
	}
	r.Close()

	// stream should work for ordinary reader too
	r = NewReader(buf1)
	out, err = io.ReadAll(r)
	if err != nil {
		t.Errorf("testSync: read: %s", err)
		return
	}
	r.Close()
	if !bytes.Equal(input, out) {
		t.Errorf("testSync: decompress(compress(data)) != data: level=%d input=%s", level, name)
	}
}

func testToFromWithLevelAndLimit(t *testing.T, level int, input []byte, name string, limit int) {
	var buffer bytes.Buffer
	w, err := NewWriter(&buffer, level)
	if err != nil {
		t.Errorf("NewWriter: %v", err)
		return
	}
	w.Write(input)
	w.Close()
	if limit > 0 {
		t.Logf("level: %d - Size:%.2f%%, %d b\n", level, float64(buffer.Len()*100)/float64(limit), buffer.Len())
	}
	if limit > 0 && buffer.Len() > limit {
		t.Errorf("level: %d, len(compress(data)) = %d > limit = %d", level, buffer.Len(), limit)
	}

	r := NewReader(&buffer)
	out, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("read: %s", err)
		return
	}
	r.Close()
	if !bytes.Equal(input, out) {
		os.WriteFile("testdata/fails/"+t.Name()+".got", out, os.ModePerm)
		os.WriteFile("testdata/fails/"+t.Name()+".want", input, os.ModePerm)
		t.Errorf("decompress(compress(data)) != data: level=%d input=%s", level, name)
		return
	}
	testSync(t, level, input, name)
}

func testToFromWithLimit(t *testing.T, input []byte, name string, limit [11]int) {
	for i := range 10 {
		testToFromWithLevelAndLimit(t, i, input, name, limit[i])
	}
	testToFromWithLevelAndLimit(t, -2, input, name, limit[10])
}

func TestDeflateInflate(t *testing.T) {
	for i, h := range deflateInflateTests {
		testToFromWithLimit(t, h.in, fmt.Sprintf("#%d", i), [11]int{})
	}
}

func TestReverseBits(t *testing.T) {
	for _, h := range reverseBitsTests {
		if v := reverseBits(h.in, h.bitCount); v != h.out {
			t.Errorf("reverseBits(%v,%v) = %v, want %v",
				h.in, h.bitCount, v, h.out)
		}
	}
}

type deflateInflateStringTest struct {
	filename string
	label    string
	limit    [11]int // Number 11 is ConstantCompression
}

var deflateInflateStringTests = []deflateInflateStringTest{
	{
		"../testdata/e.txt",
		"2.718281828...",
		[...]int{100018, 67900, 50960, 51150, 50930, 50790, 50790, 50790, 50790, 50790, 43683 + 100},
	},
	{
		"../../testdata/Isaac.Newton-Opticks.txt",
		"Isaac.Newton-Opticks",
		[...]int{567248, 218338, 201354, 199101, 190627, 182587, 179765, 174982, 173422, 173422, 325240},
	},
}

func TestDeflateInflateString(t *testing.T) {
	for _, test := range deflateInflateStringTests {
		gold, err := os.ReadFile(test.filename)
		if err != nil {
			t.Error(err)
		}
		// Remove returns that may be present on Windows
		neutral := strings.Map(func(r rune) rune {
			if r != '\r' {
				return r
			}
			return -1
		}, string(gold))

		testToFromWithLimit(t, []byte(neutral), test.label, test.limit)

		if testing.Short() {
			break
		}
	}
}

func TestReaderDict(t *testing.T) {
	const (
		dict = "hello world"
		text = "hello again world"
	)
	var b bytes.Buffer
	w, err := NewWriter(&b, 5)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	w.Write([]byte(dict))
	w.Flush()
	b.Reset()
	w.Write([]byte(text))
	w.Close()

	r := NewReaderDict(&b, []byte(dict))
	data, err := io.ReadAll(r)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "hello again world" {
		t.Fatalf("read returned %q want %q", string(data), text)
	}
}

func TestWriterDict(t *testing.T) {
	const (
		dict = "hello world Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua."
		text = "hello world Lorem ipsum dolor sit amet"
	)
	// This test is sensitive to algorithm changes that skip
	// data in favour of speed. Higher levels are less prone to this
	// so we test level 4-9.
	for l := 4; l < 9; l++ {
		var b bytes.Buffer
		w, err := NewWriter(&b, l)
		if err != nil {
			t.Fatalf("level %d, NewWriter: %v", l, err)
		}
		w.Write([]byte(dict))
		w.Flush()
		b.Reset()
		w.Write([]byte(text))
		w.Close()

		var b1 bytes.Buffer
		w, _ = NewWriterDict(&b1, l, []byte(dict))
		w.Write([]byte(text))
		w.Close()

		if !bytes.Equal(b1.Bytes(), b.Bytes()) {
			t.Errorf("level %d, writer wrote\n%v\n want\n%v", l, b1.Bytes(), b.Bytes())
		}
	}
}

// See http://code.google.com/p/go/issues/detail?id=2508
func TestRegression2508(t *testing.T) {
	if testing.Short() {
		t.Logf("test disabled with -short")
		return
	}
	w, err := NewWriter(io.Discard, 1)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	buf := make([]byte, 1024)
	for range 131072 {
		if _, err := w.Write(buf); err != nil {
			t.Fatalf("writer failed: %v", err)
		}
	}
	w.Close()
}

func TestWriterReset(t *testing.T) {
	for level := -2; level <= 9; level++ {
		if level == -1 {
			level++
		}
		if testing.Short() && level > 1 {
			break
		}
		w, err := NewWriter(io.Discard, level)
		if err != nil {
			t.Fatalf("NewWriter: %v", err)
		}
		buf := []byte("hello world")
		for range 1024 {
			w.Write(buf)
		}
		w.Reset(io.Discard)

		wref, err := NewWriter(io.Discard, level)
		if err != nil {
			t.Fatalf("NewWriter: %v", err)
		}

		// DeepEqual doesn't compare functions.
		w.d.fill, wref.d.fill = nil, nil
		w.d.step, wref.d.step = nil, nil
		w.d.state, wref.d.state = nil, nil
		w.d.fast, wref.d.fast = nil, nil

		// hashMatch is always overwritten when used.
		if w.d.tokens.n != 0 {
			t.Errorf("level %d Writer not reset after Reset. %d tokens were present", level, w.d.tokens.n)
		}
		// As long as the length is 0, we don't care about the content.
		w.d.tokens = wref.d.tokens

		// We don't care if there are values in the window, as long as it is at d.index is 0
		w.d.window = wref.d.window
		if !reflect.DeepEqual(w, wref) {
			t.Errorf("level %d Writer not reset after Reset", level)
		}
	}

	for i := HuffmanOnly; i <= BestCompression; i++ {
		testResetOutput(t, fmt.Sprint("level-", i), func(w io.Writer) (*Writer, error) { return NewWriter(w, i) })
	}
	dict := []byte(strings.Repeat("we are the world - how are you?", 3))
	for i := HuffmanOnly; i <= BestCompression; i++ {
		testResetOutput(t, fmt.Sprint("dict-level-", i), func(w io.Writer) (*Writer, error) { return NewWriterDict(w, i, dict) })
	}
	for i := HuffmanOnly; i <= BestCompression; i++ {
		testResetOutput(t, fmt.Sprint("dict-reset-level-", i), func(w io.Writer) (*Writer, error) {
			w2, err := NewWriter(nil, i)
			if err != nil {
				return w2, err
			}
			w2.ResetDict(w, dict)
			return w2, nil
		})
	}
}

func testResetOutput(t *testing.T, name string, newWriter func(w io.Writer) (*Writer, error)) {
	t.Run(name, func(t *testing.T) {
		buf := new(bytes.Buffer)
		w, err := newWriter(buf)
		if err != nil {
			t.Fatalf("NewWriter: %v", err)
		}
		b := []byte("hello world - how are you doing?")
		for range 1024 {
			w.Write(b)
		}
		w.Close()
		out1 := buf.Bytes()

		buf2 := new(bytes.Buffer)
		w.Reset(buf2)
		for range 1024 {
			w.Write(b)
		}
		w.Close()
		out2 := buf2.Bytes()

		if len(out1) != len(out2) {
			t.Errorf("got %d, expected %d bytes", len(out2), len(out1))
		}
		if !bytes.Equal(out1, out2) {
			mm := 0
			for i, b := range out1[:len(out2)] {
				if b != out2[i] {
					t.Errorf("mismatch index %d: %02x, expected %02x", i, out2[i], b)
				}
				mm++
				if mm == 10 {
					t.Fatal("Stopping")
				}
			}
		}
		t.Logf("got %d bytes", len(out1))
	})
}

// TestBestSpeed tests that round-tripping through deflate and then inflate
// recovers the original input. The Write sizes are near the thresholds in the
// compressor.encSpeed method (0, 16, 128), as well as near maxStoreBlockSize
// (65535).
func TestBestSpeed(t *testing.T) {
	abc := make([]byte, 128)
	for i := range abc {
		abc[i] = byte(i)
	}
	abcabc := bytes.Repeat(abc, 131072/len(abc))
	var want []byte

	testCases := [][]int{
		{65536, 0},
		{65536, 1},
		{65536, 1, 256},
		{65536, 1, 65536},
		{65536, 14},
		{65536, 15},
		{65536, 16},
		{65536, 16, 256},
		{65536, 16, 65536},
		{65536, 127},
		{65536, 128},
		{65536, 128, 256},
		{65536, 128, 65536},
		{65536, 129},
		{65536, 65536, 256},
		{65536, 65536, 65536},
	}

	for i, tc := range testCases {
		if testing.Short() && i > 5 {
			t.Skip()
		}
		for _, firstN := range []int{1, 65534, 65535, 65536, 65537, 131072} {
			tc[0] = firstN
		outer:
			for _, flush := range []bool{false, true} {
				buf := new(bytes.Buffer)
				want = want[:0]

				w, err := NewWriter(buf, BestSpeed)
				if err != nil {
					t.Errorf("i=%d, firstN=%d, flush=%t: NewWriter: %v", i, firstN, flush, err)
					continue
				}
				for _, n := range tc {
					want = append(want, abcabc[:n]...)
					if _, err := w.Write(abcabc[:n]); err != nil {
						t.Errorf("i=%d, firstN=%d, flush=%t: Write: %v", i, firstN, flush, err)
						continue outer
					}
					if !flush {
						continue
					}
					if err := w.Flush(); err != nil {
						t.Errorf("i=%d, firstN=%d, flush=%t: Flush: %v", i, firstN, flush, err)
						continue outer
					}
				}
				if err := w.Close(); err != nil {
					t.Errorf("i=%d, firstN=%d, flush=%t: Close: %v", i, firstN, flush, err)
					continue
				}

				r := NewReader(buf)
				got, err := io.ReadAll(r)
				if err != nil {
					t.Errorf("i=%d, firstN=%d, flush=%t: ReadAll: %v", i, firstN, flush, err)
					continue
				}
				r.Close()

				if !bytes.Equal(got, want) {
					t.Errorf("i=%d, firstN=%d, flush=%t: corruption during deflate-then-inflate", i, firstN, flush)
					continue
				}
			}
		}
	}
}
