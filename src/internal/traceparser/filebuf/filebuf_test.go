// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package filebuf

import (
	"bytes"
	"io"
	"io/ioutil"
	"log"
	"os"
	"testing"
)

var (
	inited       bool
	small, large string // files
	dir          string // in this dir
	contents     []byte // contents of the large file
)

func TestMain(m *testing.M) {
	create()
	n := m.Run()

	os.RemoveAll(dir)
	os.Exit(n)
}

func create() {
	if inited {
		return
	}
	log.SetFlags(log.Lshortfile)
	d, erra := ioutil.TempDir("", "filebuf")
	s, errb := ioutil.TempFile(dir, "small")
	l, errc := ioutil.TempFile(dir, "large")
	if erra != nil || errb != nil || errc != nil {
		log.Fatal(erra, errb, errc)
	}
	dir, small, large = d, s.Name(), l.Name()
	buf := make([]byte, 2*Buflen+3)
	for i := 0; i < len(buf); i++ {
		buf[i] = byte(i)
	}
	err := ioutil.WriteFile(small, buf[:7], 0666)
	if err != nil {
		log.Fatal(err)
	}
	err = ioutil.WriteFile(large, buf, 0666)
	if err != nil {
		log.Fatal(err)
	}
	contents = buf
	inited = true
}

func get(n int) io.Reader {
	if n <= len(contents) {
		return bytes.NewReader(contents[:n])
	}
	return bytes.NewReader(contents)
}

func TestSmall(t *testing.T) {
	var f Buf
	var err error
	f, err = New(small)
	small := func(t *testing.T) {
		if err != nil {
			t.Fatal(err)
		}
		buf := make([]byte, 23)
		n, err := f.Read(buf)
		if n != 7 || err != io.EOF {
			t.Errorf("got %d, expected 7, %v", n, err)
		}
		m, err := f.Seek(0, io.SeekCurrent)
		if m != 7 || err != nil {
			t.Errorf("got %d, expected 7, %v", m, err)
		}
		m, err = f.Seek(1, io.SeekStart)
		if m != 1 || err != nil {
			t.Errorf("got %d expected 1, %v", m, err)
		}
		n, err = f.Read(buf)
		if n != 6 || err != io.EOF {
			t.Errorf("got %d, expected 6, %v", n, err)
		}
		for i := 0; i < 6; i++ {
			if buf[i] != byte(i+1) {
				t.Fatalf("byte %d is %d, not %d, %v", i, buf[i], i+1, buf)
			}
		}
	}
	t.Run("New", small)
	f, err = FromReader(get(7))
	t.Run("Rdr", small)
}

func TestLarge(t *testing.T) {
	var f Buf
	var err error
	big := func(t *testing.T) {
		if err != nil {
			t.Fatal(err)
		}
		x := Buflen - 7
		n, err := f.Seek(int64(x), io.SeekStart)
		if n != Buflen-7 || err != nil {
			t.Fatalf("expected %d, got %d, %v", x, n, err)
		}
		buf := make([]byte, 23)
		m, err := f.Read(buf)
		if m != len(buf) || err != nil {
			t.Fatalf("expected %d, got %d, %v", len(buf), m, err)
		}
		for i := 0; i < 23; i++ {
			if buf[i] != byte(x+i) {
				t.Fatalf("byte %d, got %d, wanted %d", i, buf[i],
					byte(x+i))
			}
		}
		m, err = f.Read(buf)
		if m != len(buf) || err != nil {
			t.Fatalf("got %d, expected %d, %v", m, len(buf), err)
		}
		x += len(buf)
		for i := 0; i < 23; i++ {
			if buf[i] != byte(x+i) {
				t.Fatalf("byte %d, got %d, wanted %d", i, buf[i],
					byte(x+i))
			}
		}
	}
	f, err = New(large)
	t.Run("New", big)
	f, err = FromReader(get(1 << 30))
	t.Run("Rdr", big)
}

func TestMore(t *testing.T) {
	f, err := New(large)
	if err != nil {
		t.Fatal(err)
	}
	var a, b [4]byte
	f.Seek(16, 0)
	f.Read(a[:])
	f.Seek(16, 0)
	f.Read(b[:])
	if a != b {
		t.Errorf("oops %v %v", a, b)
	}
}

func TestSeek(t *testing.T) {
	f, err := New(small)
	if err != nil {
		log.Fatal(err)
	}
	n, err := f.Seek(f.Size(), 0)
	if n != f.Size() || err != nil {
		t.Errorf("seek got %d, expected %d, %v", n, f.Size(), err)
	}
	n, err = f.Seek(1, io.SeekCurrent)
	if n != f.Size() || err != io.EOF {
		t.Errorf("n=%d, expected 0. %v", n, err)
	}
	n, err = f.Seek(f.Size(), 0)
	if n != f.Size() || err != nil {
		t.Errorf("seek got %d, expected %d, %v", n, f.Size(), err)
	}
}

func TestReread(t *testing.T) {
	f, err := New(small)
	if err != nil {
		t.Fatal(err)
	}
	var buf [1]byte
	f.Seek(0, 0)
	for i := 0; i < int(f.Size()); i++ {
		n, err := f.Read(buf[:])
		if n != 1 || err != nil {
			t.Fatalf("n=%d, err=%v", n, err)
		}
	}
	stats := f.Stats()
	if stats.Bytes != f.Size() || stats.Reads != 1 || stats.Seeks != 1 {
		t.Errorf("%v %d %d", stats, f.(*fbuf).bufloc, f.(*fbuf).bufpos)
	}
	n, err := f.Read(buf[:])
	if n != 0 || err != io.EOF {
		t.Fatalf("expected 0 and io.EOF, got %d %v", n, err)
	}
	f.Seek(0, 0)
	xstats := f.Stats()
	if xstats.Bytes != f.Size() || xstats.Reads != 1 || xstats.Seeks != 2 {
		t.Errorf("%v %v %d %d", stats, xstats, f.(*fbuf).bufloc, f.(*fbuf).bufpos)
	}
	f.Close()
}
