// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flate

import (
	"bytes"
	"errors"
	"fmt"
	"internal/testenv"
	"io"
	"math/rand"
	"os"
	"reflect"
	"runtime/debug"
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
	{[]byte{}, 0, []byte{1, 0, 0, 255, 255}},
	{[]byte{0x11}, -1, []byte{18, 4, 4, 0, 0, 255, 255}},
	{[]byte{0x11}, DefaultCompression, []byte{18, 4, 4, 0, 0, 255, 255}},
	{[]byte{0x11}, 4, []byte{18, 4, 4, 0, 0, 255, 255}},

	{[]byte{0x11}, 0, []byte{0, 1, 0, 254, 255, 17, 1, 0, 0, 255, 255}},
	{[]byte{0x11, 0x12}, 0, []byte{0, 2, 0, 253, 255, 17, 18, 1, 0, 0, 255, 255}},
	{[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, 0,
		[]byte{0, 8, 0, 247, 255, 17, 17, 17, 17, 17, 17, 17, 17, 1, 0, 0, 255, 255},
	},
	{[]byte{}, 2, []byte{1, 0, 0, 255, 255}},
	{[]byte{0x11}, 2, []byte{18, 4, 4, 0, 0, 255, 255}},
	{[]byte{0x11, 0x12}, 2, []byte{18, 20, 2, 4, 0, 0, 255, 255}},
	{[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, 2, []byte{18, 132, 2, 64, 0, 0, 0, 255, 255}},
	{[]byte{}, 9, []byte{1, 0, 0, 255, 255}},
	{[]byte{0x11}, 9, []byte{18, 4, 4, 0, 0, 255, 255}},
	{[]byte{0x11, 0x12}, 9, []byte{18, 20, 2, 4, 0, 0, 255, 255}},
	{[]byte{0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x11}, 9, []byte{18, 132, 2, 64, 0, 0, 0, 255, 255}},
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
		if len(y) < minMatchLength {
			continue
		}
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
					t.Errorf("Len:%d Index:%d, want 0x%08x but not modified", len(y), i, want)
				} else if got != want {
					t.Errorf("Len:%d Index:%d, got 0x%08x want:0x%08x", len(y), i, got, want)
				}
			}
		}
	}
}

func TestDeflate(t *testing.T) {
	for _, h := range deflateTests {
		var buf bytes.Buffer
		w, err := NewWriter(&buf, h.level)
		if err != nil {
			t.Errorf("NewWriter: %v", err)
			continue
		}
		w.Write(h.in)
		w.Close()
		if !bytes.Equal(buf.Bytes(), h.out) {
			t.Errorf("Deflate(%d, %x) = \n%#v, want \n%#v", h.level, h.in, buf.Bytes(), h.out)
		}
	}
}

func TestWriterClose(t *testing.T) {
	b := new(bytes.Buffer)
	zw, err := NewWriter(b, 6)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	if c, err := zw.Write([]byte("Test")); err != nil || c != 4 {
		t.Fatalf("Write to not closed writer: %s, %d", err, c)
	}

	if err := zw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	afterClose := b.Len()

	if c, err := zw.Write([]byte("Test")); err == nil || c != 0 {
		t.Fatalf("Write to closed writer: %v, %d", err, c)
	}

	if err := zw.Flush(); err == nil {
		t.Fatalf("Flush to closed writer: %s", err)
	}

	if err := zw.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	if afterClose != b.Len() {
		t.Fatalf("Writer wrote data after close. After close: %d. After writes on closed stream: %d", afterClose, b.Len())
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
	w, err := NewWriter(io.Discard, 1)
	if err != nil {
		t.Errorf("NewWriter: %v", err)
		return
	}
	if _, err = io.Copy(w, &sparseReader{l: 23e8}); err != nil {
		t.Errorf("Compress failed: %v", err)
		return
	}
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
	for i := 0; i < 2; i++ {
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
	if limit > 0 && buffer.Len() > limit {
		t.Errorf("level: %d, len(compress(data)) = %d > limit = %d", level, buffer.Len(), limit)
		return
	}
	if limit > 0 {
		t.Logf("level: %d, size:%.2f%%, %d b\n", level, float64(buffer.Len()*100)/float64(limit), buffer.Len())
	}
	r := NewReader(&buffer)
	out, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("read: %s", err)
		return
	}
	r.Close()
	if !bytes.Equal(input, out) {
		t.Errorf("decompress(compress(data)) != data: level=%d input=%s", level, name)
		return
	}
	testSync(t, level, input, name)
}

func testToFromWithLimit(t *testing.T, input []byte, name string, limit [11]int) {
	for i := 0; i < 10; i++ {
		testToFromWithLevelAndLimit(t, i, input, name, limit[i])
	}
	// Test HuffmanCompression
	testToFromWithLevelAndLimit(t, -2, input, name, limit[10])
}

func TestDeflateInflate(t *testing.T) {
	t.Parallel()
	for i, h := range deflateInflateTests {
		if testing.Short() && len(h.in) > 10000 {
			continue
		}
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
	limit    [11]int
}

var deflateInflateStringTests = []deflateInflateStringTest{
	{
		"../testdata/e.txt",
		"2.718281828...",
		[...]int{100018, 50650, 50960, 51150, 50930, 50790, 50790, 50790, 50790, 50790, 43683},
	},
	{
		"../../testdata/Isaac.Newton-Opticks.txt",
		"Isaac.Newton-Opticks",
		[...]int{567248, 218338, 198211, 193152, 181100, 175427, 175427, 173597, 173422, 173422, 325240},
	},
}

func TestDeflateInflateString(t *testing.T) {
	t.Parallel()
	if testing.Short() && testenv.Builder() == "" {
		t.Skip("skipping in short mode")
	}
	for _, test := range deflateInflateStringTests {
		gold, err := os.ReadFile(test.filename)
		if err != nil {
			t.Error(err)
		}
		testToFromWithLimit(t, gold, test.label, test.limit)
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

	var b1 bytes.Buffer
	w, _ = NewWriterDict(&b1, 5, []byte(dict))
	w.Write([]byte(text))
	w.Close()

	if !bytes.Equal(b1.Bytes(), b.Bytes()) {
		t.Fatalf("writer wrote %q want %q", b1.Bytes(), b.Bytes())
	}
}

// See https://golang.org/issue/2508
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
	for i := 0; i < 131072; i++ {
		if _, err := w.Write(buf); err != nil {
			t.Fatalf("writer failed: %v", err)
		}
	}
	w.Close()
}

func TestWriterReset(t *testing.T) {
	t.Parallel()
	for level := 0; level <= 9; level++ {
		if testing.Short() && level > 1 {
			break
		}
		w, err := NewWriter(io.Discard, level)
		if err != nil {
			t.Fatalf("NewWriter: %v", err)
		}
		buf := []byte("hello world")
		n := 1024
		if testing.Short() {
			n = 10
		}
		for i := 0; i < n; i++ {
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
		w.d.bulkHasher, wref.d.bulkHasher = nil, nil
		w.d.bestSpeed, wref.d.bestSpeed = nil, nil
		// hashMatch is always overwritten when used.
		copy(w.d.hashMatch[:], wref.d.hashMatch[:])
		if len(w.d.tokens) != 0 {
			t.Errorf("level %d Writer not reset after Reset. %d tokens were present", level, len(w.d.tokens))
		}
		// As long as the length is 0, we don't care about the content.
		w.d.tokens = wref.d.tokens

		// We don't care if there are values in the window, as long as it is at d.index is 0
		w.d.window = wref.d.window
		if !reflect.DeepEqual(w, wref) {
			t.Errorf("level %d Writer not reset after Reset", level)
		}
	}

	levels := []int{0, 1, 2, 5, 9}
	for _, level := range levels {
		t.Run(fmt.Sprint(level), func(t *testing.T) {
			testResetOutput(t, level, nil)
		})
	}

	t.Run("dict", func(t *testing.T) {
		for _, level := range levels {
			t.Run(fmt.Sprint(level), func(t *testing.T) {
				testResetOutput(t, level, nil)
			})
		}
	})
}

func testResetOutput(t *testing.T, level int, dict []byte) {
	writeData := func(w *Writer) {
		msg := []byte("now is the time for all good gophers")
		w.Write(msg)
		w.Flush()

		hello := []byte("hello world")
		for i := 0; i < 1024; i++ {
			w.Write(hello)
		}

		fill := bytes.Repeat([]byte("x"), 65000)
		w.Write(fill)
	}

	buf := new(bytes.Buffer)
	var w *Writer
	var err error
	if dict == nil {
		w, err = NewWriter(buf, level)
	} else {
		w, err = NewWriterDict(buf, level, dict)
	}
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	writeData(w)
	w.Close()
	out1 := buf.Bytes()

	buf2 := new(bytes.Buffer)
	w.Reset(buf2)
	writeData(w)
	w.Close()
	out2 := buf2.Bytes()

	if len(out1) != len(out2) {
		t.Errorf("got %d, expected %d bytes", len(out2), len(out1))
		return
	}
	if !bytes.Equal(out1, out2) {
		mm := 0
		for i, b := range out1[:len(out2)] {
			if b != out2[i] {
				t.Errorf("mismatch index %d: %#02x, expected %#02x", i, out2[i], b)
			}
			mm++
			if mm == 10 {
				t.Fatal("Stopping")
			}
		}
	}
	t.Logf("got %d bytes", len(out1))
}

// TestBestSpeed tests that round-tripping through deflate and then inflate
// recovers the original input. The Write sizes are near the thresholds in the
// compressor.encSpeed method (0, 16, 128), as well as near maxStoreBlockSize
// (65535).
func TestBestSpeed(t *testing.T) {
	t.Parallel()
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
		if i >= 3 && testing.Short() {
			break
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

var errIO = errors.New("IO error")

// failWriter fails with errIO exactly at the nth call to Write.
type failWriter struct{ n int }

func (w *failWriter) Write(b []byte) (int, error) {
	w.n--
	if w.n == -1 {
		return 0, errIO
	}
	return len(b), nil
}

func TestWriterPersistentWriteError(t *testing.T) {
	t.Parallel()
	d, err := os.ReadFile("../../testdata/Isaac.Newton-Opticks.txt")
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	d = d[:10000] // Keep this test short

	zw, err := NewWriter(nil, DefaultCompression)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}

	// Sweep over the threshold at which an error is returned.
	// The variable i makes it such that the ith call to failWriter.Write will
	// return errIO. Since failWriter errors are not persistent, we must ensure
	// that flate.Writer errors are persistent.
	for i := 0; i < 1000; i++ {
		fw := &failWriter{i}
		zw.Reset(fw)

		_, werr := zw.Write(d)
		cerr := zw.Close()
		ferr := zw.Flush()
		if werr != errIO && werr != nil {
			t.Errorf("test %d, mismatching Write error: got %v, want %v", i, werr, errIO)
		}
		if cerr != errIO && fw.n < 0 {
			t.Errorf("test %d, mismatching Close error: got %v, want %v", i, cerr, errIO)
		}
		if ferr != errIO && fw.n < 0 {
			t.Errorf("test %d, mismatching Flush error: got %v, want %v", i, ferr, errIO)
		}
		if fw.n >= 0 {
			// At this point, the failure threshold was sufficiently high enough
			// that we wrote the whole stream without any errors.
			return
		}
	}
}
func TestWriterPersistentFlushError(t *testing.T) {
	zw, err := NewWriter(&failWriter{0}, DefaultCompression)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	flushErr := zw.Flush()
	closeErr := zw.Close()
	_, writeErr := zw.Write([]byte("Test"))
	checkErrors([]error{closeErr, flushErr, writeErr}, errIO, t)
}

func TestWriterPersistentCloseError(t *testing.T) {
	// If underlying writer return error on closing stream we should persistent this error across all writer calls.
	zw, err := NewWriter(&failWriter{0}, DefaultCompression)
	if err != nil {
		t.Fatalf("NewWriter: %v", err)
	}
	closeErr := zw.Close()
	flushErr := zw.Flush()
	_, writeErr := zw.Write([]byte("Test"))
	checkErrors([]error{closeErr, flushErr, writeErr}, errIO, t)

	// After closing writer we should persistent "write after close" error across Flush and Write calls, but return nil
	// on next Close calls.
	var b bytes.Buffer
	zw.Reset(&b)
	err = zw.Close()
	if err != nil {
		t.Fatalf("First call to close returned error: %s", err)
	}
	err = zw.Close()
	if err != nil {
		t.Fatalf("Second call to close returned error: %s", err)
	}

	flushErr = zw.Flush()
	_, writeErr = zw.Write([]byte("Test"))
	checkErrors([]error{flushErr, writeErr}, errWriterClosed, t)
}

func checkErrors(got []error, want error, t *testing.T) {
	t.Helper()
	for _, err := range got {
		if err != want {
			t.Errorf("Error doesn't match\nWant: %s\nGot: %s", want, got)
		}
	}
}

func TestBestSpeedMatch(t *testing.T) {
	t.Parallel()
	cases := []struct {
		previous, current []byte
		t, s, want        int32
	}{{
		previous: []byte{0, 0, 0, 1, 2},
		current:  []byte{3, 4, 5, 0, 1, 2, 3, 4, 5},
		t:        -3,
		s:        3,
		want:     6,
	}, {
		previous: []byte{0, 0, 0, 1, 2},
		current:  []byte{2, 4, 5, 0, 1, 2, 3, 4, 5},
		t:        -3,
		s:        3,
		want:     3,
	}, {
		previous: []byte{0, 0, 0, 1, 1},
		current:  []byte{3, 4, 5, 0, 1, 2, 3, 4, 5},
		t:        -3,
		s:        3,
		want:     2,
	}, {
		previous: []byte{0, 0, 0, 1, 2},
		current:  []byte{2, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        -1,
		s:        0,
		want:     4,
	}, {
		previous: []byte{0, 0, 0, 1, 2, 3, 4, 5, 2, 2},
		current:  []byte{2, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        -7,
		s:        4,
		want:     5,
	}, {
		previous: []byte{9, 9, 9, 9, 9},
		current:  []byte{2, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        -1,
		s:        0,
		want:     0,
	}, {
		previous: []byte{9, 9, 9, 9, 9},
		current:  []byte{9, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        0,
		s:        1,
		want:     0,
	}, {
		previous: []byte{},
		current:  []byte{9, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        -5,
		s:        1,
		want:     0,
	}, {
		previous: []byte{},
		current:  []byte{9, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        -1,
		s:        1,
		want:     0,
	}, {
		previous: []byte{},
		current:  []byte{2, 2, 2, 2, 1, 2, 3, 4, 5},
		t:        0,
		s:        1,
		want:     3,
	}, {
		previous: []byte{3, 4, 5},
		current:  []byte{3, 4, 5},
		t:        -3,
		s:        0,
		want:     3,
	}, {
		previous: make([]byte, 1000),
		current:  make([]byte, 1000),
		t:        -1000,
		s:        0,
		want:     maxMatchLength - 4,
	}, {
		previous: make([]byte, 200),
		current:  make([]byte, 500),
		t:        -200,
		s:        0,
		want:     maxMatchLength - 4,
	}, {
		previous: make([]byte, 200),
		current:  make([]byte, 500),
		t:        0,
		s:        1,
		want:     maxMatchLength - 4,
	}, {
		previous: make([]byte, maxMatchLength-4),
		current:  make([]byte, 500),
		t:        -(maxMatchLength - 4),
		s:        0,
		want:     maxMatchLength - 4,
	}, {
		previous: make([]byte, 200),
		current:  make([]byte, 500),
		t:        -200,
		s:        400,
		want:     100,
	}, {
		previous: make([]byte, 10),
		current:  make([]byte, 500),
		t:        200,
		s:        400,
		want:     100,
	}}
	for i, c := range cases {
		e := deflateFast{prev: c.previous}
		got := e.matchLen(c.s, c.t, c.current)
		if got != c.want {
			t.Errorf("Test %d: match length, want %d, got %d", i, c.want, got)
		}
	}
}

func TestBestSpeedMaxMatchOffset(t *testing.T) {
	t.Parallel()
	const abc, xyz = "abcdefgh", "stuvwxyz"
	for _, matchBefore := range []bool{false, true} {
		for _, extra := range []int{0, inputMargin - 1, inputMargin, inputMargin + 1, 2 * inputMargin} {
			for offsetAdj := -5; offsetAdj <= +5; offsetAdj++ {
				report := func(desc string, err error) {
					t.Errorf("matchBefore=%t, extra=%d, offsetAdj=%d: %s%v",
						matchBefore, extra, offsetAdj, desc, err)
				}

				offset := maxMatchOffset + offsetAdj

				// Make src to be a []byte of the form
				//	"%s%s%s%s%s" % (abc, zeros0, xyzMaybe, abc, zeros1)
				// where:
				//	zeros0 is approximately maxMatchOffset zeros.
				//	xyzMaybe is either xyz or the empty string.
				//	zeros1 is between 0 and 30 zeros.
				// The difference between the two abc's will be offset, which
				// is maxMatchOffset plus or minus a small adjustment.
				src := make([]byte, offset+len(abc)+extra)
				copy(src, abc)
				if !matchBefore {
					copy(src[offset-len(xyz):], xyz)
				}
				copy(src[offset:], abc)

				buf := new(bytes.Buffer)
				w, err := NewWriter(buf, BestSpeed)
				if err != nil {
					report("NewWriter: ", err)
					continue
				}
				if _, err := w.Write(src); err != nil {
					report("Write: ", err)
					continue
				}
				if err := w.Close(); err != nil {
					report("Writer.Close: ", err)
					continue
				}

				r := NewReader(buf)
				dst, err := io.ReadAll(r)
				r.Close()
				if err != nil {
					report("ReadAll: ", err)
					continue
				}

				if !bytes.Equal(dst, src) {
					report("", fmt.Errorf("bytes differ after round-tripping"))
					continue
				}
			}
		}
	}
}

func TestBestSpeedShiftOffsets(t *testing.T) {
	// Test if shiftoffsets properly preserves matches and resets out-of-range matches
	// seen in https://github.com/golang/go/issues/4142
	enc := newDeflateFast()

	// testData may not generate internal matches.
	testData := make([]byte, 32)
	rng := rand.New(rand.NewSource(0))
	for i := range testData {
		testData[i] = byte(rng.Uint32())
	}

	// Encode the testdata with clean state.
	// Second part should pick up matches from the first block.
	wantFirstTokens := len(enc.encode(nil, testData))
	wantSecondTokens := len(enc.encode(nil, testData))

	if wantFirstTokens <= wantSecondTokens {
		t.Fatalf("test needs matches between inputs to be generated")
	}
	// Forward the current indicator to before wraparound.
	enc.cur = bufferReset - int32(len(testData))

	// Part 1 before wrap, should match clean state.
	got := len(enc.encode(nil, testData))
	if wantFirstTokens != got {
		t.Errorf("got %d, want %d tokens", got, wantFirstTokens)
	}

	// Verify we are about to wrap.
	if enc.cur != bufferReset {
		t.Errorf("got %d, want e.cur to be at bufferReset (%d)", enc.cur, bufferReset)
	}

	// Part 2 should match clean state as well even if wrapped.
	got = len(enc.encode(nil, testData))
	if wantSecondTokens != got {
		t.Errorf("got %d, want %d token", got, wantSecondTokens)
	}

	// Verify that we wrapped.
	if enc.cur >= bufferReset {
		t.Errorf("want e.cur to be < bufferReset (%d), got %d", bufferReset, enc.cur)
	}

	// Forward the current buffer, leaving the matches at the bottom.
	enc.cur = bufferReset
	enc.shiftOffsets()

	// Ensure that no matches were picked up.
	got = len(enc.encode(nil, testData))
	if wantFirstTokens != got {
		t.Errorf("got %d, want %d tokens", got, wantFirstTokens)
	}
}

func TestMaxStackSize(t *testing.T) {
	// This test must not run in parallel with other tests as debug.SetMaxStack
	// affects all goroutines.
	n := debug.SetMaxStack(1 << 16)
	defer debug.SetMaxStack(n)

	var wg sync.WaitGroup
	defer wg.Wait()

	b := make([]byte, 1<<20)
	for level := HuffmanOnly; level <= BestCompression; level++ {
		// Run in separate goroutine to increase probability of stack regrowth.
		wg.Add(1)
		go func(level int) {
			defer wg.Done()
			zw, err := NewWriter(io.Discard, level)
			if err != nil {
				t.Errorf("level %d, NewWriter() = %v, want nil", level, err)
			}
			if n, err := zw.Write(b); n != len(b) || err != nil {
				t.Errorf("level %d, Write() = (%d, %v), want (%d, nil)", level, n, err, len(b))
			}
			if err := zw.Close(); err != nil {
				t.Errorf("level %d, Close() = %v, want nil", level, err)
			}
			zw.Reset(io.Discard)
		}(level)
	}
}
