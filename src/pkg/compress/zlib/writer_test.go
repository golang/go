// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zlib

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"testing"
)

var filenames = []string{
	"../testdata/e.txt",
	"../testdata/pi.txt",
}

var data = []string{
	"test a reasonable sized string that can be compressed",
}

// Tests that compressing and then decompressing the given file at the given compression level and dictionary
// yields equivalent bytes to the original file.
func testFileLevelDict(t *testing.T, fn string, level int, d string) {
	// Read the file, as golden output.
	golden, err := os.Open(fn)
	if err != nil {
		t.Errorf("%s (level=%d, dict=%q): %v", fn, level, d, err)
		return
	}
	defer golden.Close()
	b0, err0 := ioutil.ReadAll(golden)
	if err0 != nil {
		t.Errorf("%s (level=%d, dict=%q): %v", fn, level, d, err0)
		return
	}
	testLevelDict(t, fn, b0, level, d)
}

func testLevelDict(t *testing.T, fn string, b0 []byte, level int, d string) {
	// Make dictionary, if given.
	var dict []byte
	if d != "" {
		dict = []byte(d)
	}

	// Push data through a pipe that compresses at the write end, and decompresses at the read end.
	piper, pipew := io.Pipe()
	defer piper.Close()
	go func() {
		defer pipew.Close()
		zlibw, err := NewWriterDict(pipew, level, dict)
		if err != nil {
			t.Errorf("%s (level=%d, dict=%q): %v", fn, level, d, err)
			return
		}
		defer zlibw.Close()
		_, err = zlibw.Write(b0)
		if err != nil {
			t.Errorf("%s (level=%d, dict=%q): %v", fn, level, d, err)
			return
		}
	}()
	zlibr, err := NewReaderDict(piper, dict)
	if err != nil {
		t.Errorf("%s (level=%d, dict=%q): %v", fn, level, d, err)
		return
	}
	defer zlibr.Close()

	// Compare the decompressed data.
	b1, err1 := ioutil.ReadAll(zlibr)
	if err1 != nil {
		t.Errorf("%s (level=%d, dict=%q): %v", fn, level, d, err1)
		return
	}
	if len(b0) != len(b1) {
		t.Errorf("%s (level=%d, dict=%q): length mismatch %d versus %d", fn, level, d, len(b0), len(b1))
		return
	}
	for i := 0; i < len(b0); i++ {
		if b0[i] != b1[i] {
			t.Errorf("%s (level=%d, dict=%q): mismatch at %d, 0x%02x versus 0x%02x\n", fn, level, d, i, b0[i], b1[i])
			return
		}
	}
}

func TestWriter(t *testing.T) {
	for i, s := range data {
		b := []byte(s)
		tag := fmt.Sprintf("#%d", i)
		testLevelDict(t, tag, b, DefaultCompression, "")
		testLevelDict(t, tag, b, NoCompression, "")
		for level := BestSpeed; level <= BestCompression; level++ {
			testLevelDict(t, tag, b, level, "")
		}
	}
}

func TestWriterBig(t *testing.T) {
	for _, fn := range filenames {
		testFileLevelDict(t, fn, DefaultCompression, "")
		testFileLevelDict(t, fn, NoCompression, "")
		for level := BestSpeed; level <= BestCompression; level++ {
			testFileLevelDict(t, fn, level, "")
		}
	}
}

func TestWriterDict(t *testing.T) {
	const dictionary = "0123456789."
	for _, fn := range filenames {
		testFileLevelDict(t, fn, DefaultCompression, dictionary)
		testFileLevelDict(t, fn, NoCompression, dictionary)
		for level := BestSpeed; level <= BestCompression; level++ {
			testFileLevelDict(t, fn, level, dictionary)
		}
	}
}

func TestWriterDictIsUsed(t *testing.T) {
	var input = []byte("Lorem ipsum dolor sit amet, consectetur adipisicing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
	buf := bytes.NewBuffer(nil)
	compressor, err := NewWriterDict(buf, BestCompression, input)
	if err != nil {
		t.Errorf("error in NewWriterDict: %s", err)
		return
	}
	compressor.Write(input)
	compressor.Close()
	const expectedMaxSize = 25
	output := buf.Bytes()
	if len(output) > expectedMaxSize {
		t.Errorf("result too large (got %d, want <= %d bytes). Is the dictionary being used?", len(output), expectedMaxSize)
	}
}
