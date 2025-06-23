// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bytes_test

import (
	. "bytes"
	"fmt"
	"io"
	"sync"
	"testing"
)

func TestReader(t *testing.T) {
	r := NewReader([]byte("0123456789"))
	tests := []struct {
		off     int64
		seek    int
		n       int
		want    string
		wantpos int64
		readerr error
		seekerr string
	}{
		{seek: io.SeekStart, off: 0, n: 20, want: "0123456789"},
		{seek: io.SeekStart, off: 1, n: 1, want: "1"},
		{seek: io.SeekCurrent, off: 1, wantpos: 3, n: 2, want: "34"},
		{seek: io.SeekStart, off: -1, seekerr: "bytes.Reader.Seek: negative position"},
		{seek: io.SeekStart, off: 1 << 33, wantpos: 1 << 33, readerr: io.EOF},
		{seek: io.SeekCurrent, off: 1, wantpos: 1<<33 + 1, readerr: io.EOF},
		{seek: io.SeekStart, n: 5, want: "01234"},
		{seek: io.SeekCurrent, n: 5, want: "56789"},
		{seek: io.SeekEnd, off: -1, n: 1, wantpos: 9, want: "9"},
	}

	for i, tt := range tests {
		pos, err := r.Seek(tt.off, tt.seek)
		if err == nil && tt.seekerr != "" {
			t.Errorf("%d. want seek error %q", i, tt.seekerr)
			continue
		}
		if err != nil && err.Error() != tt.seekerr {
			t.Errorf("%d. seek error = %q; want %q", i, err.Error(), tt.seekerr)
			continue
		}
		if tt.wantpos != 0 && tt.wantpos != pos {
			t.Errorf("%d. pos = %d, want %d", i, pos, tt.wantpos)
		}
		buf := make([]byte, tt.n)
		n, err := r.Read(buf)
		if err != tt.readerr {
			t.Errorf("%d. read = %v; want %v", i, err, tt.readerr)
			continue
		}
		got := string(buf[:n])
		if got != tt.want {
			t.Errorf("%d. got %q; want %q", i, got, tt.want)
		}
	}
}

func TestReadAfterBigSeek(t *testing.T) {
	r := NewReader([]byte("0123456789"))
	if _, err := r.Seek(1<<31+5, io.SeekStart); err != nil {
		t.Fatal(err)
	}
	if n, err := r.Read(make([]byte, 10)); n != 0 || err != io.EOF {
		t.Errorf("Read = %d, %v; want 0, EOF", n, err)
	}
}

func TestReaderAt(t *testing.T) {
	r := NewReader([]byte("0123456789"))
	tests := []struct {
		off     int64
		n       int
		want    string
		wanterr any
	}{
		{0, 10, "0123456789", nil},
		{1, 10, "123456789", io.EOF},
		{1, 9, "123456789", nil},
		{11, 10, "", io.EOF},
		{0, 0, "", nil},
		{-1, 0, "", "bytes.Reader.ReadAt: negative offset"},
	}
	for i, tt := range tests {
		b := make([]byte, tt.n)
		rn, err := r.ReadAt(b, tt.off)
		got := string(b[:rn])
		if got != tt.want {
			t.Errorf("%d. got %q; want %q", i, got, tt.want)
		}
		if fmt.Sprintf("%v", err) != fmt.Sprintf("%v", tt.wanterr) {
			t.Errorf("%d. got error = %v; want %v", i, err, tt.wanterr)
		}
	}
}

func TestReaderAtConcurrent(t *testing.T) {
	// Test for the race detector, to verify ReadAt doesn't mutate
	// any state.
	r := NewReader([]byte("0123456789"))
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			var buf [1]byte
			r.ReadAt(buf[:], int64(i))
		}(i)
	}
	wg.Wait()
}

func TestEmptyReaderConcurrent(t *testing.T) {
	// Test for the race detector, to verify a Read that doesn't yield any bytes
	// is okay to use from multiple goroutines. This was our historic behavior.
	// See golang.org/issue/7856
	r := NewReader([]byte{})
	var wg sync.WaitGroup
	for i := 0; i < 5; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			var buf [1]byte
			r.Read(buf[:])
		}()
		go func() {
			defer wg.Done()
			r.Read(nil)
		}()
	}
	wg.Wait()
}

func TestReaderWriteTo(t *testing.T) {
	for i := 0; i < 30; i += 3 {
		var l int
		if i > 0 {
			l = len(testString) / i
		}
		s := testString[:l]
		r := NewReader(testBytes[:l])
		var b Buffer
		n, err := r.WriteTo(&b)
		if expect := int64(len(s)); n != expect {
			t.Errorf("got %v; want %v", n, expect)
		}
		if err != nil {
			t.Errorf("for length %d: got error = %v; want nil", l, err)
		}
		if b.String() != s {
			t.Errorf("got string %q; want %q", b.String(), s)
		}
		if r.Len() != 0 {
			t.Errorf("reader contains %v bytes; want 0", r.Len())
		}
	}
}

func TestReaderLen(t *testing.T) {
	const data = "hello world"
	r := NewReader([]byte(data))
	if got, want := r.Len(), 11; got != want {
		t.Errorf("r.Len(): got %d, want %d", got, want)
	}
	if n, err := r.Read(make([]byte, 10)); err != nil || n != 10 {
		t.Errorf("Read failed: read %d %v", n, err)
	}
	if got, want := r.Len(), 1; got != want {
		t.Errorf("r.Len(): got %d, want %d", got, want)
	}
	if n, err := r.Read(make([]byte, 1)); err != nil || n != 1 {
		t.Errorf("Read failed: read %d %v; want 1, nil", n, err)
	}
	if got, want := r.Len(), 0; got != want {
		t.Errorf("r.Len(): got %d, want %d", got, want)
	}
}

var UnreadRuneErrorTests = []struct {
	name string
	f    func(*Reader)
}{
	{"Read", func(r *Reader) { r.Read([]byte{0}) }},
	{"ReadByte", func(r *Reader) { r.ReadByte() }},
	{"UnreadRune", func(r *Reader) { r.UnreadRune() }},
	{"Seek", func(r *Reader) { r.Seek(0, io.SeekCurrent) }},
	{"WriteTo", func(r *Reader) { r.WriteTo(&Buffer{}) }},
}

func TestUnreadRuneError(t *testing.T) {
	for _, tt := range UnreadRuneErrorTests {
		reader := NewReader([]byte("0123456789"))
		if _, _, err := reader.ReadRune(); err != nil {
			// should not happen
			t.Fatal(err)
		}
		tt.f(reader)
		err := reader.UnreadRune()
		if err == nil {
			t.Errorf("Unreading after %s: expected error", tt.name)
		}
	}
}

func TestReaderDoubleUnreadRune(t *testing.T) {
	buf := NewBuffer([]byte("groucho"))
	if _, _, err := buf.ReadRune(); err != nil {
		// should not happen
		t.Fatal(err)
	}
	if err := buf.UnreadByte(); err != nil {
		// should not happen
		t.Fatal(err)
	}
	if err := buf.UnreadByte(); err == nil {
		t.Fatal("UnreadByte: expected error, got nil")
	}
}

// verify that copying from an empty reader always has the same results,
// regardless of the presence of a WriteTo method.
func TestReaderCopyNothing(t *testing.T) {
	type nErr struct {
		n   int64
		err error
	}
	type justReader struct {
		io.Reader
	}
	type justWriter struct {
		io.Writer
	}
	discard := justWriter{io.Discard} // hide ReadFrom

	var with, withOut nErr
	with.n, with.err = io.Copy(discard, NewReader(nil))
	withOut.n, withOut.err = io.Copy(discard, justReader{NewReader(nil)})
	if with != withOut {
		t.Errorf("behavior differs: with = %#v; without: %#v", with, withOut)
	}
}

// tests that Len is affected by reads, but Size is not.
func TestReaderLenSize(t *testing.T) {
	r := NewReader([]byte("abc"))
	io.CopyN(io.Discard, r, 1)
	if r.Len() != 2 {
		t.Errorf("Len = %d; want 2", r.Len())
	}
	if r.Size() != 3 {
		t.Errorf("Size = %d; want 3", r.Size())
	}
}

func TestReaderReset(t *testing.T) {
	r := NewReader([]byte("世界"))
	if _, _, err := r.ReadRune(); err != nil {
		t.Errorf("ReadRune: unexpected error: %v", err)
	}

	const want = "abcdef"
	r.Reset([]byte(want))
	if err := r.UnreadRune(); err == nil {
		t.Errorf("UnreadRune: expected error, got nil")
	}
	buf, err := io.ReadAll(r)
	if err != nil {
		t.Errorf("ReadAll: unexpected error: %v", err)
	}
	if got := string(buf); got != want {
		t.Errorf("ReadAll: got %q, want %q", got, want)
	}
}

func TestReaderZero(t *testing.T) {
	if l := (&Reader{}).Len(); l != 0 {
		t.Errorf("Len: got %d, want 0", l)
	}

	if n, err := (&Reader{}).Read(nil); n != 0 || err != io.EOF {
		t.Errorf("Read: got %d, %v; want 0, io.EOF", n, err)
	}

	if n, err := (&Reader{}).ReadAt(nil, 11); n != 0 || err != io.EOF {
		t.Errorf("ReadAt: got %d, %v; want 0, io.EOF", n, err)
	}

	if b, err := (&Reader{}).ReadByte(); b != 0 || err != io.EOF {
		t.Errorf("ReadByte: got %d, %v; want 0, io.EOF", b, err)
	}

	if ch, size, err := (&Reader{}).ReadRune(); ch != 0 || size != 0 || err != io.EOF {
		t.Errorf("ReadRune: got %d, %d, %v; want 0, 0, io.EOF", ch, size, err)
	}

	if offset, err := (&Reader{}).Seek(11, io.SeekStart); offset != 11 || err != nil {
		t.Errorf("Seek: got %d, %v; want 11, nil", offset, err)
	}

	if s := (&Reader{}).Size(); s != 0 {
		t.Errorf("Size: got %d, want 0", s)
	}

	if (&Reader{}).UnreadByte() == nil {
		t.Errorf("UnreadByte: got nil, want error")
	}

	if (&Reader{}).UnreadRune() == nil {
		t.Errorf("UnreadRune: got nil, want error")
	}

	if n, err := (&Reader{}).WriteTo(io.Discard); n != 0 || err != nil {
		t.Errorf("WriteTo: got %d, %v; want 0, nil", n, err)
	}
}
