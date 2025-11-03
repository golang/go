// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package io_test

import (
	"bytes"
	"errors"
	"fmt"
	. "io"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
)

// A version of bytes.Buffer without ReadFrom and WriteTo
type Buffer struct {
	bytes.Buffer
	ReaderFrom // conflicts with and hides bytes.Buffer's ReaderFrom.
	WriterTo   // conflicts with and hides bytes.Buffer's WriterTo.
}

// Simple tests, primarily to verify the ReadFrom and WriteTo callouts inside Copy, CopyBuffer and CopyN.

func TestCopy(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyNegative(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello")
	Copy(wb, &LimitedReader{R: rb, N: -1})
	if wb.String() != "" {
		t.Errorf("Copy on LimitedReader with N<0 copied data")
	}

	CopyN(wb, rb, -1)
	if wb.String() != "" {
		t.Errorf("CopyN with N<0 copied data")
	}
}

func TestCopyBuffer(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	CopyBuffer(wb, rb, make([]byte, 1)) // Tiny buffer to keep it honest.
	if wb.String() != "hello, world." {
		t.Errorf("CopyBuffer did not work properly")
	}
}

func TestCopyBufferNil(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	CopyBuffer(wb, rb, nil) // Should allocate a buffer.
	if wb.String() != "hello, world." {
		t.Errorf("CopyBuffer did not work properly")
	}
}

func TestCopyReadFrom(t *testing.T) {
	rb := new(Buffer)
	wb := new(bytes.Buffer) // implements ReadFrom.
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

func TestCopyWriteTo(t *testing.T) {
	rb := new(bytes.Buffer) // implements WriteTo.
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	}
}

// Version of bytes.Buffer that checks whether WriteTo was called or not
type writeToChecker struct {
	bytes.Buffer
	writeToCalled bool
}

func (wt *writeToChecker) WriteTo(w Writer) (int64, error) {
	wt.writeToCalled = true
	return wt.Buffer.WriteTo(w)
}

// It's preferable to choose WriterTo over ReaderFrom, since a WriterTo can issue one large write,
// while the ReaderFrom must read until EOF, potentially allocating when running out of buffer.
// Make sure that we choose WriterTo when both are implemented.
func TestCopyPriority(t *testing.T) {
	rb := new(writeToChecker)
	wb := new(bytes.Buffer)
	rb.WriteString("hello, world.")
	Copy(wb, rb)
	if wb.String() != "hello, world." {
		t.Errorf("Copy did not work properly")
	} else if !rb.writeToCalled {
		t.Errorf("WriteTo was not prioritized over ReadFrom")
	}
}

type zeroErrReader struct {
	err error
}

func (r zeroErrReader) Read(p []byte) (int, error) {
	return copy(p, []byte{0}), r.err
}

type errWriter struct {
	err error
}

func (w errWriter) Write([]byte) (int, error) {
	return 0, w.err
}

// In case a Read results in an error with non-zero bytes read, and
// the subsequent Write also results in an error, the error from Write
// is returned, as it is the one that prevented progressing further.
func TestCopyReadErrWriteErr(t *testing.T) {
	er, ew := errors.New("readError"), errors.New("writeError")
	r, w := zeroErrReader{err: er}, errWriter{err: ew}
	n, err := Copy(w, r)
	if n != 0 || err != ew {
		t.Errorf("Copy(zeroErrReader, errWriter) = %d, %v; want 0, writeError", n, err)
	}
}

func TestCopyN(t *testing.T) {
	rb := new(Buffer)
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	CopyN(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("CopyN did not work properly")
	}
}

func TestCopyNReadFrom(t *testing.T) {
	rb := new(Buffer)
	wb := new(bytes.Buffer) // implements ReadFrom.
	rb.WriteString("hello")
	CopyN(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("CopyN did not work properly")
	}
}

func TestCopyNWriteTo(t *testing.T) {
	rb := new(bytes.Buffer) // implements WriteTo.
	wb := new(Buffer)
	rb.WriteString("hello, world.")
	CopyN(wb, rb, 5)
	if wb.String() != "hello" {
		t.Errorf("CopyN did not work properly")
	}
}

func BenchmarkCopyNSmall(b *testing.B) {
	bs := bytes.Repeat([]byte{0}, 512+1)
	rd := bytes.NewReader(bs)
	buf := new(Buffer)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		CopyN(buf, rd, 512)
		rd.Reset(bs)
	}
}

func BenchmarkCopyNLarge(b *testing.B) {
	bs := bytes.Repeat([]byte{0}, (32*1024)+1)
	rd := bytes.NewReader(bs)
	buf := new(Buffer)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		CopyN(buf, rd, 32*1024)
		rd.Reset(bs)
	}
}

type noReadFrom struct {
	w Writer
}

func (w *noReadFrom) Write(p []byte) (n int, err error) {
	return w.w.Write(p)
}

type wantedAndErrReader struct{}

func (wantedAndErrReader) Read(p []byte) (int, error) {
	return len(p), errors.New("wantedAndErrReader error")
}

func TestCopyNEOF(t *testing.T) {
	// Test that EOF behavior is the same regardless of whether
	// argument to CopyN has ReadFrom.

	b := new(bytes.Buffer)

	n, err := CopyN(&noReadFrom{b}, strings.NewReader("foo"), 3)
	if n != 3 || err != nil {
		t.Errorf("CopyN(noReadFrom, foo, 3) = %d, %v; want 3, nil", n, err)
	}

	n, err = CopyN(&noReadFrom{b}, strings.NewReader("foo"), 4)
	if n != 3 || err != EOF {
		t.Errorf("CopyN(noReadFrom, foo, 4) = %d, %v; want 3, EOF", n, err)
	}

	n, err = CopyN(b, strings.NewReader("foo"), 3) // b has read from
	if n != 3 || err != nil {
		t.Errorf("CopyN(bytes.Buffer, foo, 3) = %d, %v; want 3, nil", n, err)
	}

	n, err = CopyN(b, strings.NewReader("foo"), 4) // b has read from
	if n != 3 || err != EOF {
		t.Errorf("CopyN(bytes.Buffer, foo, 4) = %d, %v; want 3, EOF", n, err)
	}

	n, err = CopyN(b, wantedAndErrReader{}, 5)
	if n != 5 || err != nil {
		t.Errorf("CopyN(bytes.Buffer, wantedAndErrReader, 5) = %d, %v; want 5, nil", n, err)
	}

	n, err = CopyN(&noReadFrom{b}, wantedAndErrReader{}, 5)
	if n != 5 || err != nil {
		t.Errorf("CopyN(noReadFrom, wantedAndErrReader, 5) = %d, %v; want 5, nil", n, err)
	}
}

func TestReadAtLeast(t *testing.T) {
	var rb bytes.Buffer
	testReadAtLeast(t, &rb)
}

// A version of bytes.Buffer that returns n > 0, err on Read
// when the input is exhausted.
type dataAndErrorBuffer struct {
	err error
	bytes.Buffer
}

func (r *dataAndErrorBuffer) Read(p []byte) (n int, err error) {
	n, err = r.Buffer.Read(p)
	if n > 0 && r.Buffer.Len() == 0 && err == nil {
		err = r.err
	}
	return
}

func TestReadAtLeastWithDataAndEOF(t *testing.T) {
	var rb dataAndErrorBuffer
	rb.err = EOF
	testReadAtLeast(t, &rb)
}

func TestReadAtLeastWithDataAndError(t *testing.T) {
	var rb dataAndErrorBuffer
	rb.err = fmt.Errorf("fake error")
	testReadAtLeast(t, &rb)
}

func testReadAtLeast(t *testing.T, rb ReadWriter) {
	rb.Write([]byte("0123"))
	buf := make([]byte, 2)
	n, err := ReadAtLeast(rb, buf, 2)
	if err != nil {
		t.Error(err)
	}
	if n != 2 {
		t.Errorf("expected to have read 2 bytes, got %v", n)
	}
	n, err = ReadAtLeast(rb, buf, 4)
	if err != ErrShortBuffer {
		t.Errorf("expected ErrShortBuffer got %v", err)
	}
	if n != 0 {
		t.Errorf("expected to have read 0 bytes, got %v", n)
	}
	n, err = ReadAtLeast(rb, buf, 1)
	if err != nil {
		t.Error(err)
	}
	if n != 2 {
		t.Errorf("expected to have read 2 bytes, got %v", n)
	}
	n, err = ReadAtLeast(rb, buf, 2)
	if err != EOF {
		t.Errorf("expected EOF, got %v", err)
	}
	if n != 0 {
		t.Errorf("expected to have read 0 bytes, got %v", n)
	}
	rb.Write([]byte("4"))
	n, err = ReadAtLeast(rb, buf, 2)
	want := ErrUnexpectedEOF
	if rb, ok := rb.(*dataAndErrorBuffer); ok && rb.err != EOF {
		want = rb.err
	}
	if err != want {
		t.Errorf("expected %v, got %v", want, err)
	}
	if n != 1 {
		t.Errorf("expected to have read 1 bytes, got %v", n)
	}
}

func TestTeeReader(t *testing.T) {
	src := []byte("hello, world")
	dst := make([]byte, len(src))
	rb := bytes.NewBuffer(src)
	wb := new(bytes.Buffer)
	r := TeeReader(rb, wb)
	if n, err := ReadFull(r, dst); err != nil || n != len(src) {
		t.Fatalf("ReadFull(r, dst) = %d, %v; want %d, nil", n, err, len(src))
	}
	if !bytes.Equal(dst, src) {
		t.Errorf("bytes read = %q want %q", dst, src)
	}
	if !bytes.Equal(wb.Bytes(), src) {
		t.Errorf("bytes written = %q want %q", wb.Bytes(), src)
	}
	if n, err := r.Read(dst); n != 0 || err != EOF {
		t.Errorf("r.Read at EOF = %d, %v want 0, EOF", n, err)
	}
	rb = bytes.NewBuffer(src)
	pr, pw := Pipe()
	pr.Close()
	r = TeeReader(rb, pw)
	if n, err := ReadFull(r, dst); n != 0 || err != ErrClosedPipe {
		t.Errorf("closed tee: ReadFull(r, dst) = %d, %v; want 0, EPIPE", n, err)
	}
}

func TestSectionReader_ReadAt(t *testing.T) {
	dat := "a long sample data, 1234567890"
	tests := []struct {
		data   string
		off    int
		n      int
		bufLen int
		at     int
		exp    string
		err    error
	}{
		{data: "", off: 0, n: 10, bufLen: 2, at: 0, exp: "", err: EOF},
		{data: dat, off: 0, n: len(dat), bufLen: 0, at: 0, exp: "", err: nil},
		{data: dat, off: len(dat), n: 1, bufLen: 1, at: 0, exp: "", err: EOF},
		{data: dat, off: 0, n: len(dat) + 2, bufLen: len(dat), at: 0, exp: dat, err: nil},
		{data: dat, off: 0, n: len(dat), bufLen: len(dat) / 2, at: 0, exp: dat[:len(dat)/2], err: nil},
		{data: dat, off: 0, n: len(dat), bufLen: len(dat), at: 0, exp: dat, err: nil},
		{data: dat, off: 0, n: len(dat), bufLen: len(dat) / 2, at: 2, exp: dat[2 : 2+len(dat)/2], err: nil},
		{data: dat, off: 3, n: len(dat), bufLen: len(dat) / 2, at: 2, exp: dat[5 : 5+len(dat)/2], err: nil},
		{data: dat, off: 3, n: len(dat) / 2, bufLen: len(dat)/2 - 2, at: 2, exp: dat[5 : 5+len(dat)/2-2], err: nil},
		{data: dat, off: 3, n: len(dat) / 2, bufLen: len(dat)/2 + 2, at: 2, exp: dat[5 : 5+len(dat)/2-2], err: EOF},
		{data: dat, off: 0, n: 0, bufLen: 0, at: -1, exp: "", err: EOF},
		{data: dat, off: 0, n: 0, bufLen: 0, at: 1, exp: "", err: EOF},
	}
	for i, tt := range tests {
		r := strings.NewReader(tt.data)
		s := NewSectionReader(r, int64(tt.off), int64(tt.n))
		buf := make([]byte, tt.bufLen)
		if n, err := s.ReadAt(buf, int64(tt.at)); n != len(tt.exp) || string(buf[:n]) != tt.exp || err != tt.err {
			t.Fatalf("%d: ReadAt(%d) = %q, %v; expected %q, %v", i, tt.at, buf[:n], err, tt.exp, tt.err)
		}
		if _r, off, n := s.Outer(); _r != r || off != int64(tt.off) || n != int64(tt.n) {
			t.Fatalf("%d: Outer() = %v, %d, %d; expected %v, %d, %d", i, _r, off, n, r, tt.off, tt.n)
		}
	}
}

func TestSectionReader_Seek(t *testing.T) {
	// Verifies that NewSectionReader's Seeker behaves like bytes.NewReader (which is like strings.NewReader)
	br := bytes.NewReader([]byte("foo"))
	sr := NewSectionReader(br, 0, int64(len("foo")))

	for _, whence := range []int{SeekStart, SeekCurrent, SeekEnd} {
		for offset := int64(-3); offset <= 4; offset++ {
			brOff, brErr := br.Seek(offset, whence)
			srOff, srErr := sr.Seek(offset, whence)
			if (brErr != nil) != (srErr != nil) || brOff != srOff {
				t.Errorf("For whence %d, offset %d: bytes.Reader.Seek = (%v, %v) != SectionReader.Seek = (%v, %v)",
					whence, offset, brOff, brErr, srErr, srOff)
			}
		}
	}

	// And verify we can just seek past the end and get an EOF
	got, err := sr.Seek(100, SeekStart)
	if err != nil || got != 100 {
		t.Errorf("Seek = %v, %v; want 100, nil", got, err)
	}

	n, err := sr.Read(make([]byte, 10))
	if n != 0 || err != EOF {
		t.Errorf("Read = %v, %v; want 0, EOF", n, err)
	}
}

func TestSectionReader_Size(t *testing.T) {
	tests := []struct {
		data string
		want int64
	}{
		{"a long sample data, 1234567890", 30},
		{"", 0},
	}

	for _, tt := range tests {
		r := strings.NewReader(tt.data)
		sr := NewSectionReader(r, 0, int64(len(tt.data)))
		if got := sr.Size(); got != tt.want {
			t.Errorf("Size = %v; want %v", got, tt.want)
		}
	}
}

func TestSectionReader_Max(t *testing.T) {
	r := strings.NewReader("abcdef")
	const maxint64 = 1<<63 - 1
	sr := NewSectionReader(r, 3, maxint64)
	n, err := sr.Read(make([]byte, 3))
	if n != 3 || err != nil {
		t.Errorf("Read = %v %v, want 3, nil", n, err)
	}
	n, err = sr.Read(make([]byte, 3))
	if n != 0 || err != EOF {
		t.Errorf("Read = %v, %v, want 0, EOF", n, err)
	}
	if _r, off, n := sr.Outer(); _r != r || off != 3 || n != maxint64 {
		t.Fatalf("Outer = %v, %d, %d; expected %v, %d, %d", _r, off, n, r, 3, int64(maxint64))
	}
}

// largeWriter returns an invalid count that is larger than the number
// of bytes provided (issue 39978).
type largeWriter struct {
	err error
}

func (w largeWriter) Write(p []byte) (int, error) {
	return len(p) + 1, w.err
}

func TestCopyLargeWriter(t *testing.T) {
	want := ErrInvalidWrite
	rb := new(Buffer)
	wb := largeWriter{}
	rb.WriteString("hello, world.")
	if _, err := Copy(wb, rb); err != want {
		t.Errorf("Copy error: got %v, want %v", err, want)
	}

	want = errors.New("largeWriterError")
	rb = new(Buffer)
	wb = largeWriter{err: want}
	rb.WriteString("hello, world.")
	if _, err := Copy(wb, rb); err != want {
		t.Errorf("Copy error: got %v, want %v", err, want)
	}
}

func TestNopCloserWriterToForwarding(t *testing.T) {
	for _, tc := range [...]struct {
		Name string
		r    Reader
	}{
		{"not a WriterTo", Reader(nil)},
		{"a WriterTo", struct {
			Reader
			WriterTo
		}{}},
	} {
		nc := NopCloser(tc.r)

		_, expected := tc.r.(WriterTo)
		_, got := nc.(WriterTo)
		if expected != got {
			t.Errorf("NopCloser incorrectly forwards WriterTo for %s, got %t want %t", tc.Name, got, expected)
		}
	}
}

func TestOffsetWriter_Seek(t *testing.T) {
	tmpfilename := "TestOffsetWriter_Seek"
	tmpfile, err := os.CreateTemp(t.TempDir(), tmpfilename)
	if err != nil {
		t.Fatalf("CreateTemp(%s) failed: %v", tmpfilename, err)
	}
	defer tmpfile.Close()
	w := NewOffsetWriter(tmpfile, 0)

	// Should throw error errWhence if whence is not valid
	t.Run("errWhence", func(t *testing.T) {
		for _, whence := range []int{-3, -2, -1, 3, 4, 5} {
			var offset int64 = 0
			gotOff, gotErr := w.Seek(offset, whence)
			if gotOff != 0 || gotErr != ErrWhence {
				t.Errorf("For whence %d, offset %d, OffsetWriter.Seek got: (%d, %v), want: (%d, %v)",
					whence, offset, gotOff, gotErr, 0, ErrWhence)
			}
		}
	})

	// Should throw error errOffset if offset is negative
	t.Run("errOffset", func(t *testing.T) {
		for _, whence := range []int{SeekStart, SeekCurrent} {
			for offset := int64(-3); offset < 0; offset++ {
				gotOff, gotErr := w.Seek(offset, whence)
				if gotOff != 0 || gotErr != ErrOffset {
					t.Errorf("For whence %d, offset %d, OffsetWriter.Seek got: (%d, %v), want: (%d, %v)",
						whence, offset, gotOff, gotErr, 0, ErrOffset)
				}
			}
		}
	})

	// Normal tests
	t.Run("normal", func(t *testing.T) {
		tests := []struct {
			offset    int64
			whence    int
			returnOff int64
		}{
			// keep in order
			{whence: SeekStart, offset: 1, returnOff: 1},
			{whence: SeekStart, offset: 2, returnOff: 2},
			{whence: SeekStart, offset: 3, returnOff: 3},
			{whence: SeekCurrent, offset: 1, returnOff: 4},
			{whence: SeekCurrent, offset: 2, returnOff: 6},
			{whence: SeekCurrent, offset: 3, returnOff: 9},
		}
		for idx, tt := range tests {
			gotOff, gotErr := w.Seek(tt.offset, tt.whence)
			if gotOff != tt.returnOff || gotErr != nil {
				t.Errorf("%d:: For whence %d, offset %d, OffsetWriter.Seek got: (%d, %v), want: (%d, <nil>)",
					idx+1, tt.whence, tt.offset, gotOff, gotErr, tt.returnOff)
			}
		}
	})
}

func TestOffsetWriter_WriteAt(t *testing.T) {
	const content = "0123456789ABCDEF"
	contentSize := int64(len(content))
	tmpdir := t.TempDir()

	work := func(off, at int64) {
		position := fmt.Sprintf("off_%d_at_%d", off, at)
		tmpfile, err := os.CreateTemp(tmpdir, position)
		if err != nil {
			t.Fatalf("CreateTemp(%s) failed: %v", position, err)
		}
		defer tmpfile.Close()

		var writeN int64
		var wg sync.WaitGroup
		// Concurrent writes, one byte at a time
		for step, value := range []byte(content) {
			wg.Add(1)
			go func(wg *sync.WaitGroup, tmpfile *os.File, value byte, off, at int64, step int) {
				defer wg.Done()

				w := NewOffsetWriter(tmpfile, off)
				n, e := w.WriteAt([]byte{value}, at+int64(step))
				if e != nil {
					t.Errorf("WriteAt failed. off: %d, at: %d, step: %d\n error: %v", off, at, step, e)
				}
				atomic.AddInt64(&writeN, int64(n))
			}(&wg, tmpfile, value, off, at, step)
		}
		wg.Wait()

		// Read one more byte to reach EOF
		buf := make([]byte, contentSize+1)
		readN, err := tmpfile.ReadAt(buf, off+at)
		if err != EOF {
			t.Fatalf("ReadAt failed: %v", err)
		}
		readContent := string(buf[:contentSize])
		if writeN != int64(readN) || writeN != contentSize || readContent != content {
			t.Fatalf("%s:: WriteAt(%s, %d) error. \ngot n: %v, content: %s \nexpected n: %v, content: %v",
				position, content, at, readN, readContent, contentSize, content)
		}
	}
	for off := int64(0); off < 2; off++ {
		for at := int64(0); at < 2; at++ {
			work(off, at)
		}
	}
}

func TestWriteAt_PositionPriorToBase(t *testing.T) {
	tmpdir := t.TempDir()
	tmpfilename := "TestOffsetWriter_WriteAt"
	tmpfile, err := os.CreateTemp(tmpdir, tmpfilename)
	if err != nil {
		t.Fatalf("CreateTemp(%s) failed: %v", tmpfilename, err)
	}
	defer tmpfile.Close()

	// start writing position in OffsetWriter
	offset := int64(10)
	// position we want to write to the tmpfile
	at := int64(-1)
	w := NewOffsetWriter(tmpfile, offset)
	_, e := w.WriteAt([]byte("hello"), at)
	if e == nil {
		t.Errorf("error expected to be not nil")
	}
}

func TestOffsetWriter_Write(t *testing.T) {
	const content = "0123456789ABCDEF"
	contentSize := len(content)
	tmpdir := t.TempDir()

	makeOffsetWriter := func(name string) (*OffsetWriter, *os.File) {
		tmpfilename := "TestOffsetWriter_Write_" + name
		tmpfile, err := os.CreateTemp(tmpdir, tmpfilename)
		if err != nil {
			t.Fatalf("CreateTemp(%s) failed: %v", tmpfilename, err)
		}
		return NewOffsetWriter(tmpfile, 0), tmpfile
	}
	checkContent := func(name string, f *os.File) {
		// Read one more byte to reach EOF
		buf := make([]byte, contentSize+1)
		readN, err := f.ReadAt(buf, 0)
		if err != EOF {
			t.Fatalf("ReadAt failed, err: %v", err)
		}
		readContent := string(buf[:contentSize])
		if readN != contentSize || readContent != content {
			t.Fatalf("%s error. \ngot n: %v, content: %s \nexpected n: %v, content: %v",
				name, readN, readContent, contentSize, content)
		}
	}

	var name string
	name = "Write"
	t.Run(name, func(t *testing.T) {
		// Write directly (off: 0, at: 0)
		// Write content to file
		w, f := makeOffsetWriter(name)
		defer f.Close()
		for _, value := range []byte(content) {
			n, err := w.Write([]byte{value})
			if err != nil {
				t.Fatalf("Write failed, n: %d, err: %v", n, err)
			}
		}
		checkContent(name, f)

		// Copy -> Write
		// Copy file f to file f2
		name = "Copy"
		w2, f2 := makeOffsetWriter(name)
		defer f2.Close()
		Copy(w2, f)
		checkContent(name, f2)
	})

	// Copy -> WriteTo -> Write
	// Note: strings.Reader implements the io.WriterTo interface.
	name = "Write_Of_Copy_WriteTo"
	t.Run(name, func(t *testing.T) {
		w, f := makeOffsetWriter(name)
		defer f.Close()
		Copy(w, strings.NewReader(content))
		checkContent(name, f)
	})
}

var errLimit = errors.New("limit exceeded")

func TestLimitedReader(t *testing.T) {
	src := strings.NewReader("abc")
	r := LimitReader(src, 5)
	lr, ok := r.(*LimitedReader)
	if !ok {
		t.Fatalf("LimitReader should return *LimitedReader, got %T", r)
	}
	if lr.R != src || lr.N != 5 || lr.Err != nil {
		t.Fatalf("LimitReader() = {R: %v, N: %d, Err: %v}, want {R: %v, N: 5, Err: nil}", lr.R, lr.N, lr.Err, src)
	}

	t.Run("WithoutCustomErr", func(t *testing.T) {
		tests := []struct {
			name   string
			data   string
			limit  int64
			want1N int
			want1E error
			want2E error
		}{
			{"UnderLimit", "hello", 10, 5, nil, EOF},
			{"ExactLimit", "hello", 5, 5, nil, EOF},
			{"OverLimit", "hello world", 5, 5, nil, EOF},
			{"ZeroLimit", "hello", 0, 0, EOF, EOF},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				lr := &LimitedReader{R: strings.NewReader(tt.data), N: tt.limit}
				buf := make([]byte, 10)

				n, err := lr.Read(buf)
				if n != tt.want1N || err != tt.want1E {
					t.Errorf("first Read() = (%d, %v), want (%d, %v)", n, err, tt.want1N, tt.want1E)
				}

				n, err = lr.Read(buf)
				if n != 0 || err != tt.want2E {
					t.Errorf("second Read() = (%d, %v), want (0, %v)", n, err, tt.want2E)
				}
			})
		}
	})

	t.Run("WithCustomErr", func(t *testing.T) {
		tests := []struct {
			name      string
			data      string
			limit     int64
			err       error
			wantFirst string
			wantErr1  error
			wantErr2  error
		}{
			{"ExactLimit", "hello", 5, errLimit, "hello", nil, EOF},
			{"OverLimit", "hello world", 5, errLimit, "hello", nil, errLimit},
			{"UnderLimit", "hi", 5, errLimit, "hi", nil, EOF},
			{"ZeroLimitEmpty", "", 0, errLimit, "", EOF, EOF},
			{"ZeroLimitNonEmpty", "hello", 0, errLimit, "", errLimit, errLimit},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				lr := &LimitedReader{R: strings.NewReader(tt.data), N: tt.limit, Err: tt.err}
				buf := make([]byte, 10)

				n, err := lr.Read(buf)
				if n != len(tt.wantFirst) || string(buf[:n]) != tt.wantFirst || err != tt.wantErr1 {
					t.Errorf("first Read() = (%d, %q, %v), want (%d, %q, %v)", n, buf[:n], err, len(tt.wantFirst), tt.wantFirst, tt.wantErr1)
				}

				n, err = lr.Read(buf)
				if n != 0 || err != tt.wantErr2 {
					t.Errorf("second Read() = (%d, %v), want (0, %v)", n, err, tt.wantErr2)
				}
			})
		}
	})

	t.Run("CustomErrPersists", func(t *testing.T) {
		lr := &LimitedReader{R: strings.NewReader("hello world"), N: 5, Err: errLimit}
		buf := make([]byte, 10)

		n, err := lr.Read(buf)
		if n != 5 || err != nil || string(buf[:5]) != "hello" {
			t.Errorf("Read() = (%d, %v, %q), want (5, nil, \"hello\")", n, err, buf[:5])
		}

		n, err = lr.Read(buf)
		if n != 0 || err != errLimit {
			t.Errorf("Read() = (%d, %v), want (0, errLimit)", n, err)
		}

		n, err = lr.Read(buf)
		if n != 0 || err != errLimit {
			t.Errorf("Read() = (%d, %v), want (0, errLimit)", n, err)
		}
	})

	t.Run("ErrEOF", func(t *testing.T) {
		lr := &LimitedReader{R: strings.NewReader("hello world"), N: 5, Err: EOF}
		buf := make([]byte, 10)

		n, err := lr.Read(buf)
		if n != 5 || err != nil {
			t.Errorf("Read() = (%d, %v), want (5, nil)", n, err)
		}

		n, err = lr.Read(buf)
		if n != 0 || err != EOF {
			t.Errorf("Read() = (%d, %v), want (0, EOF)", n, err)
		}
	})

	t.Run("NoSideEffects", func(t *testing.T) {
		lr := &LimitedReader{R: strings.NewReader("hello"), N: 5, Err: errLimit}
		buf := make([]byte, 0)

		for i := 0; i < 3; i++ {
			n, err := lr.Read(buf)
			if n != 0 || err != nil {
				t.Errorf("zero-length read #%d = (%d, %v), want (0, nil)", i+1, n, err)
			}
			if lr.N != 5 {
				t.Errorf("N after zero-length read #%d = %d, want 5", i+1, lr.N)
			}
		}

		buf = make([]byte, 10)
		n, err := lr.Read(buf)
		if n != 5 || string(buf[:5]) != "hello" || err != nil {
			t.Errorf("normal Read() = (%d, %q, %v), want (5, \"hello\", nil)", n, buf[:5], err)
		}
	})
}

type errorReader struct {
	data []byte
	pos  int
	err  error
}

func (r *errorReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, r.err
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

func TestLimitedReaderErrors(t *testing.T) {
	t.Run("UnderlyingError", func(t *testing.T) {
		underlyingErr := errors.New("boom")
		lr := &LimitedReader{R: &errorReader{data: []byte("hello"), err: underlyingErr}, N: 10}
		buf := make([]byte, 10)

		n, err := lr.Read(buf)
		if n != 5 || string(buf[:5]) != "hello" || err != nil {
			t.Errorf("first Read() = (%d, %q, %v), want (5, \"hello\", nil)", n, buf[:5], err)
		}

		n, err = lr.Read(buf)
		if n != 0 || err != underlyingErr {
			t.Errorf("second Read() = (%d, %v), want (0, %v)", n, err, underlyingErr)
		}
	})

	t.Run("SentinelMasksProbeError", func(t *testing.T) {
		probeErr := errors.New("probe failed")
		lr := &LimitedReader{R: &errorReader{data: []byte("hello"), err: probeErr}, N: 5, Err: errLimit}
		buf := make([]byte, 10)

		n, err := lr.Read(buf)
		if n != 5 || string(buf[:5]) != "hello" || err != nil {
			t.Errorf("first Read() = (%d, %q, %v), want (5, \"hello\", nil)", n, buf[:5], err)
		}

		n, err = lr.Read(buf)
		if n != 0 || err != errLimit {
			t.Errorf("second Read() = (%d, %v), want (0, errLimit)", n, err)
		}
	})
}

func TestLimitedReaderCopy(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		limit   int64
		wantN   int64
		wantErr error
	}{
		{"Exact", "hello", 5, 5, nil},
		{"Under", "hi", 5, 2, nil},
		{"Over", "hello world", 5, 5, errLimit},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			lr := &LimitedReader{R: strings.NewReader(tt.input), N: tt.limit, Err: errLimit}
			var dst Buffer
			n, err := Copy(&dst, lr)
			if n != tt.wantN || err != tt.wantErr {
				t.Errorf("Copy() = (%d, %v), want (%d, %v)", n, err, tt.wantN, tt.wantErr)
			}
		})
	}
}
