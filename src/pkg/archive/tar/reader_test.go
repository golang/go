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
	"strings"
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
		{
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
		{
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
	{
		file: "testdata/star.tar",
		headers: []*Header{
			{
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
			{
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
	{
		file: "testdata/v7.tar",
		headers: []*Header{
			{
				Name:     "small.txt",
				Mode:     0444,
				Uid:      73025,
				Gid:      5000,
				Size:     5,
				ModTime:  time.Unix(1244593104, 0),
				Typeflag: '\x00',
			},
			{
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
	{
		file: "testdata/pax.tar",
		headers: []*Header{
			{
				Name:       "a/123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100",
				Mode:       0664,
				Uid:        1000,
				Gid:        1000,
				Uname:      "shane",
				Gname:      "shane",
				Size:       7,
				ModTime:    time.Unix(1350244992, 23960108),
				ChangeTime: time.Unix(1350244992, 23960108),
				AccessTime: time.Unix(1350244992, 23960108),
				Typeflag:   TypeReg,
			},
			{
				Name:       "a/b",
				Mode:       0777,
				Uid:        1000,
				Gid:        1000,
				Uname:      "shane",
				Gname:      "shane",
				Size:       0,
				ModTime:    time.Unix(1350266320, 910238425),
				ChangeTime: time.Unix(1350266320, 910238425),
				AccessTime: time.Unix(1350266320, 910238425),
				Typeflag:   TypeSymlink,
				Linkname:   "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100",
			},
		},
	},
	{
		file: "testdata/nil-uid.tar", // golang.org/issue/5290
		headers: []*Header{
			{
				Name:     "P1050238.JPG.log",
				Mode:     0664,
				Uid:      0,
				Gid:      0,
				Size:     14,
				ModTime:  time.Unix(1365454838, 0),
				Typeflag: TypeReg,
				Linkname: "",
				Uname:    "eyefi",
				Gname:    "eyefi",
				Devmajor: 0,
				Devminor: 0,
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
		defer f.Close()
		tr := NewReader(f)
		for j, header := range test.headers {
			hdr, err := tr.Next()
			if err != nil || hdr == nil {
				t.Errorf("test %d, entry %d: Didn't get entry: %v", i, j, err)
				f.Close()
				continue testLoop
			}
			if *hdr != *header {
				t.Errorf("test %d, entry %d: Incorrect header:\nhave %+v\nwant %+v",
					i, j, *hdr, *header)
			}
		}
		hdr, err := tr.Next()
		if err == io.EOF {
			continue testLoop
		}
		if hdr != nil || err != nil {
			t.Errorf("test %d: Unexpected entry or error: hdr=%v err=%v", i, hdr, err)
		}
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
		if *hdr != *headers[nread] {
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

	type readerOnly struct {
		io.Reader
	}
	tr := NewReader(readerOnly{f})
	nread := 0

	for ; ; nread++ {
		_, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
	}

	if nread != len(test.headers) {
		t.Errorf("Didn't process all files\nexpected: %d\nprocessed %d\n", len(test.headers), nread)
	}
}

func TestParsePAXHeader(t *testing.T) {
	paxTests := [][3]string{
		{"a", "a=name", "10 a=name\n"}, // Test case involving multiple acceptable lengths
		{"a", "a=name", "9 a=name\n"},  // Test case involving multiple acceptable length
		{"mtime", "mtime=1350244992.023960108", "30 mtime=1350244992.023960108\n"}}
	for _, test := range paxTests {
		key, expected, raw := test[0], test[1], test[2]
		reader := bytes.NewBuffer([]byte(raw))
		headers, err := parsePAX(reader)
		if err != nil {
			t.Errorf("Couldn't parse correctly formatted headers: %v", err)
			continue
		}
		if strings.EqualFold(headers[key], expected) {
			t.Errorf("mtime header incorrectly parsed: got %s, wanted %s", headers[key], expected)
			continue
		}
		trailer := make([]byte, 100)
		n, err := reader.Read(trailer)
		if err != io.EOF || n != 0 {
			t.Error("Buffer wasn't consumed")
		}
	}
	badHeader := bytes.NewBuffer([]byte("3 somelongkey="))
	if _, err := parsePAX(badHeader); err != ErrHeader {
		t.Fatal("Unexpected success when parsing bad header")
	}
}

func TestParsePAXTime(t *testing.T) {
	// Some valid PAX time values
	timestamps := map[string]time.Time{
		"1350244992.023960108":  time.Unix(1350244992, 23960108), // The commoon case
		"1350244992.02396010":   time.Unix(1350244992, 23960100), // Lower precision value
		"1350244992.0239601089": time.Unix(1350244992, 23960108), // Higher precision value
		"1350244992":            time.Unix(1350244992, 0),        // Low precision value
	}
	for input, expected := range timestamps {
		ts, err := parsePAXTime(input)
		if err != nil {
			t.Fatal(err)
		}
		if !ts.Equal(expected) {
			t.Fatalf("Time parsing failure %s %s", ts, expected)
		}
	}
}

func TestMergePAX(t *testing.T) {
	hdr := new(Header)
	// Test a string, integer, and time based value.
	headers := map[string]string{
		"path":  "a/b/c",
		"uid":   "1000",
		"mtime": "1350244992.023960108",
	}
	err := mergePAX(hdr, headers)
	if err != nil {
		t.Fatal(err)
	}
	want := &Header{
		Name:    "a/b/c",
		Uid:     1000,
		ModTime: time.Unix(1350244992, 23960108),
	}
	if !reflect.DeepEqual(hdr, want) {
		t.Errorf("incorrect merge: got %+v, want %+v", hdr, want)
	}
}
