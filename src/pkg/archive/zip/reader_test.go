// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"encoding/binary"
	"io"
	"io/ioutil"
	"os"
	"testing"
)

type ZipTest struct {
	Name    string
	Comment string
	File    []ZipTestFile
	Error   os.Error // the error that Opening this file should return
}

type ZipTestFile struct {
	Name    string
	Content []byte // if blank, will attempt to compare against File
	File    string // name of file to compare to (relative to testdata/)
}

var tests = []ZipTest{
	{
		Name:    "test.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:    "test.txt",
				Content: []byte("This is a test text file.\n"),
			},
			{
				Name: "gophercolor16x16.png",
				File: "gophercolor16x16.png",
			},
		},
	},
	{
		Name: "r.zip",
		File: []ZipTestFile{
			{
				Name: "r/r.zip",
				File: "r.zip",
			},
		},
	},
	{Name: "readme.zip"},
	{Name: "readme.notzip", Error: FormatError},
}

func TestReader(t *testing.T) {
	for _, zt := range tests {
		readTestZip(t, zt)
	}
}

func readTestZip(t *testing.T, zt ZipTest) {
	z, err := OpenReader("testdata/" + zt.Name)
	if err != zt.Error {
		t.Errorf("error=%v, want %v", err, zt.Error)
		return
	}

	// bail here if no Files expected to be tested
	// (there may actually be files in the zip, but we don't care)
	if zt.File == nil {
		return
	}

	if z.Comment != zt.Comment {
		t.Errorf("%s: comment=%q, want %q", zt.Name, z.Comment, zt.Comment)
	}
	if len(z.File) != len(zt.File) {
		t.Errorf("%s: file count=%d, want %d", zt.Name, len(z.File), len(zt.File))
	}

	// test read of each file
	for i, ft := range zt.File {
		readTestFile(t, ft, z.File[i])
	}

	// test simultaneous reads
	n := 0
	done := make(chan bool)
	for i := 0; i < 5; i++ {
		for j, ft := range zt.File {
			go func() {
				readTestFile(t, ft, z.File[j])
				done <- true
			}()
			n++
		}
	}
	for ; n > 0; n-- {
		<-done
	}

	// test invalid checksum
	z.File[0].CRC32++ // invalidate
	r, err := z.File[0].Open()
	if err != nil {
		t.Error(err)
		return
	}
	var b bytes.Buffer
	_, err = io.Copy(&b, r)
	if err != ChecksumError {
		t.Errorf("%s: copy error=%v, want %v", z.File[0].Name, err, ChecksumError)
	}
}

func readTestFile(t *testing.T, ft ZipTestFile, f *File) {
	if f.Name != ft.Name {
		t.Errorf("name=%q, want %q", f.Name, ft.Name)
	}
	var b bytes.Buffer
	r, err := f.Open()
	if err != nil {
		t.Error(err)
		return
	}
	_, err = io.Copy(&b, r)
	if err != nil {
		t.Error(err)
		return
	}
	r.Close()
	var c []byte
	if len(ft.Content) != 0 {
		c = ft.Content
	} else if c, err = ioutil.ReadFile("testdata/" + ft.File); err != nil {
		t.Error(err)
		return
	}
	if b.Len() != len(c) {
		t.Errorf("%s: len=%d, want %d", f.Name, b.Len(), len(c))
		return
	}
	for i, b := range b.Bytes() {
		if b != c[i] {
			t.Errorf("%s: content[%d]=%q want %q", f.Name, i, b, c[i])
			return
		}
	}
}

func TestInvalidFiles(t *testing.T) {
	const size = 1024 * 70 // 70kb
	b := make([]byte, size)

	// zeroes
	_, err := NewReader(sliceReaderAt(b), size)
	if err != FormatError {
		t.Errorf("zeroes: error=%v, want %v", err, FormatError)
	}

	// repeated directoryEndSignatures
	sig := make([]byte, 4)
	binary.LittleEndian.PutUint32(sig, directoryEndSignature)
	for i := 0; i < size-4; i += 4 {
		copy(b[i:i+4], sig)
	}
	_, err = NewReader(sliceReaderAt(b), size)
	if err != FormatError {
		t.Errorf("sigs: error=%v, want %v", err, FormatError)
	}
}

type sliceReaderAt []byte

func (r sliceReaderAt) ReadAt(b []byte, off int64) (int, os.Error) {
	copy(b, r[int(off):int(off)+len(b)])
	return len(b), nil
}
