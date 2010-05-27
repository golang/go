// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zlib

import (
	"io"
	"io/ioutil"
	"os"
	"testing"
)

var filenames = []string{
	"testdata/e.txt",
	"testdata/pi.txt",
}

// Tests that compressing and then decompressing the given file at the given compression level
// yields equivalent bytes to the original file.
func testFileLevel(t *testing.T, fn string, level int) {
	// Read the file, as golden output.
	golden, err := os.Open(fn, os.O_RDONLY, 0444)
	if err != nil {
		t.Errorf("%s (level=%d): %v", fn, level, err)
		return
	}
	defer golden.Close()

	// Read the file again, and push it through a pipe that compresses at the write end, and decompresses at the read end.
	raw, err := os.Open(fn, os.O_RDONLY, 0444)
	if err != nil {
		t.Errorf("%s (level=%d): %v", fn, level, err)
		return
	}
	piper, pipew := io.Pipe()
	defer piper.Close()
	go func() {
		defer raw.Close()
		defer pipew.Close()
		zlibw, err := NewWriterLevel(pipew, level)
		if err != nil {
			t.Errorf("%s (level=%d): %v", fn, level, err)
			return
		}
		defer zlibw.Close()
		var b [1024]byte
		for {
			n, err0 := raw.Read(b[0:])
			if err0 != nil && err0 != os.EOF {
				t.Errorf("%s (level=%d): %v", fn, level, err0)
				return
			}
			_, err1 := zlibw.Write(b[0:n])
			if err1 == os.EPIPE {
				// Fail, but do not report the error, as some other (presumably reportable) error broke the pipe.
				return
			}
			if err1 != nil {
				t.Errorf("%s (level=%d): %v", fn, level, err1)
				return
			}
			if err0 == os.EOF {
				break
			}
		}
	}()
	zlibr, err := NewReader(piper)
	if err != nil {
		t.Errorf("%s (level=%d): %v", fn, level, err)
		return
	}
	defer zlibr.Close()

	// Compare the two.
	b0, err0 := ioutil.ReadAll(golden)
	b1, err1 := ioutil.ReadAll(zlibr)
	if err0 != nil {
		t.Errorf("%s (level=%d): %v", fn, level, err0)
		return
	}
	if err1 != nil {
		t.Errorf("%s (level=%d): %v", fn, level, err1)
		return
	}
	if len(b0) != len(b1) {
		t.Errorf("%s (level=%d): length mismatch %d versus %d", fn, level, len(b0), len(b1))
		return
	}
	for i := 0; i < len(b0); i++ {
		if b0[i] != b1[i] {
			t.Errorf("%s (level=%d): mismatch at %d, 0x%02x versus 0x%02x\n", fn, level, i, b0[i], b1[i])
			return
		}
	}
}

func TestWriter(t *testing.T) {
	for _, fn := range filenames {
		testFileLevel(t, fn, DefaultCompression)
		testFileLevel(t, fn, NoCompression)
		for level := BestSpeed; level <= BestCompression; level++ {
			testFileLevel(t, fn, level)
		}
	}
}
