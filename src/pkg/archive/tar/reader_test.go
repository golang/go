// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"io"
	"os"
	"reflect"
	"testing"
	"time"
)

type untarTest struct {
	file    string
	headers []*Header
	cksums  []string
}

var gnuTarTest = &untarTest{
	file: "testdata/gnu.tar",
	headers: []*Header{
		&Header{
			Name:     "small.txt",
			Mode:     0640,
			Uid:      73025,
			Gid:      5000,
			Size:     5,
			ModTime:  time.Unix(1244428340, 0),
			Typeflag: '0',
			Uname:    "dsymonds",
			Gname:    "eng",
		},
		&Header{
			Name:     "small2.txt",
			Mode:     0640,
			Uid:      73025,
			Gid:      5000,
			Size:     11,
			ModTime:  time.Unix(1244436044, 0),
			Typeflag: '0',
			Uname:    "dsymonds",
			Gname:    "eng",
		},
	},
	cksums: []string{
		"e38b27eaccb4391bdec553a7f3ae6b2f",
		"c65bd2e50a56a2138bf1716f2fd56fe9",
	},
}

var untarTests = []*untarTest{
	gnuTarTest,
	&untarTest{
		file: "testdata/star.tar",
		headers: []*Header{
			&Header{
				Name:       "small.txt",
				Mode:       0640,
				Uid:        73025,
				Gid:        5000,
				Size:       5,
				ModTime:    time.Unix(1244592783, 0),
				Typeflag:   '0',
				Uname:      "dsymonds",
				Gname:      "eng",
				AccessTime: time.Unix(1244592783, 0),
				ChangeTime: time.Unix(1244592783, 0),
			},
			&Header{
				Name:       "small2.txt",
				Mode:       0640,
				Uid:        73025,
				Gid:        5000,
				Size:       11,
				ModTime:    time.Unix(1244592783, 0),
				Typeflag:   '0',
				Uname:      "dsymonds",
				Gname:      "eng",
				AccessTime: time.Unix(1244592783, 0),
				ChangeTime: time.Unix(1244592783, 0),
			},
		},
	},
	&untarTest{
		file: "testdata/v7.tar",
		headers: []*Header{
			&Header{
				Name:     "small.txt",
				Mode:     0444,
				Uid:      73025,
				Gid:      5000,
				Size:     5,
				ModTime:  time.Unix(1244593104, 0),
				Typeflag: '\x00',
			},
			&Header{
				Name:     "small2.txt",
				Mode:     0444,
				Uid:      73025,
				Gid:      5000,
				Size:     11,
				ModTime:  time.Unix(1244593104, 0),
				Typeflag: '\x00',
			},
		},
	},
}

func TestReader(t *testing.T) {
testLoop:
	for i, test := range untarTests {
		f, err := os.Open(test.file)
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err)
			continue
		}
		tr := NewReader(f)
		for j, header := range test.headers {
			hdr, err := tr.Next()
			if err != nil || hdr == nil {
				t.Errorf("test %d, entry %d: Didn't get entry: %v", i, j, err)
				f.Close()
				continue testLoop
			}
			if !reflect.DeepEqual(hdr, header) {
				t.Errorf("test %d, entry %d: Incorrect header:\nhave %+v\nwant %+v",
					i, j, *hdr, *header)
			}
		}
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if hdr != nil || err != nil {
			t.Errorf("test %d: Unexpected entry or error: hdr=%v err=%v", i, hdr, err)
		}
		f.Close()
	}
}

func TestPartialRead(t *testing.T) {
	f, err := os.Open("testdata/gnu.tar")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer f.Close()

	tr := NewReader(f)

	// Read the first four bytes; Next() should skip the last byte.
	hdr, err := tr.Next()
	if err != nil || hdr == nil {
		t.Fatalf("Didn't get first file: %v", err)
	}
	buf := make([]byte, 4)
	if _, err := io.ReadFull(tr, buf); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if expected := []byte("Kilt"); !bytes.Equal(buf, expected) {
		t.Errorf("Contents = %v, want %v", buf, expected)
	}

	// Second file
	hdr, err = tr.Next()
	if err != nil || hdr == nil {
		t.Fatalf("Didn't get second file: %v", err)
	}
	buf = make([]byte, 6)
	if _, err := io.ReadFull(tr, buf); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if expected := []byte("Google"); !bytes.Equal(buf, expected) {
		t.Errorf("Contents = %v, want %v", buf, expected)
	}
}

func TestIncrementalRead(t *testing.T) {
	test := gnuTarTest
	f, err := os.Open(test.file)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer f.Close()

	tr := NewReader(f)

	headers := test.headers
	cksums := test.cksums
	nread := 0

	// loop over all files
	for ; ; nread++ {
		hdr, err := tr.Next()
		if hdr == nil || err == io.EOF {
			break
		}

		// check the header
		if !reflect.DeepEqual(hdr, headers[nread]) {
			t.Errorf("Incorrect header:\nhave %+v\nwant %+v",
				*hdr, headers[nread])
		}

		// read file contents in little chunks EOF,
		// checksumming all the way
		h := md5.New()
		rdbuf := make([]uint8, 8)
		for {
			nr, err := tr.Read(rdbuf)
			if err == io.EOF {
				break
			}
			if err != nil {
				t.Errorf("Read: unexpected error %v\n", err)
				break
			}
			h.Write(rdbuf[0:nr])
		}
		// verify checksum
		have := fmt.Sprintf("%x", h.Sum(nil))
		want := cksums[nread]
		if want != have {
			t.Errorf("Bad checksum on file %s:\nhave %+v\nwant %+v", hdr.Name, have, want)
		}
	}
	if nread != len(headers) {
		t.Errorf("Didn't process all files\nexpected: %d\nprocessed %d\n", len(headers), nread)
	}
}

func TestNonSeekable(t *testing.T) {
	test := gnuTarTest
	f, err := os.Open(test.file)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer f.Close()

	// pipe the data in
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("Unexpected error %s", err)
	}
	go func() {
		rdbuf := make([]uint8, 1<<16)
		for {
			nr, err := f.Read(rdbuf)
			w.Write(rdbuf[0:nr])
			if err == io.EOF {
				break
			}
		}
		w.Close()
	}()

	tr := NewReader(r)
	nread := 0

	for ; ; nread++ {
		hdr, err := tr.Next()
		if hdr == nil || err == io.EOF {
			break
		}
	}

	if nread != len(test.headers) {
		t.Errorf("Didn't process all files\nexpected: %d\nprocessed %d\n", len(test.headers), nread)
	}
}
