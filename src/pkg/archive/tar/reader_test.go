// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes";
	"io";
	"os";
	"reflect";
	"strings";
	"testing";
)

type untarTest struct {
	file	string;
	headers	[]*Header;
}

var untarTests = []*untarTest{
	&untarTest{
		file: "testdata/gnu.tar",
		headers: []*Header{
			&Header{
				Name: "small.txt",
				Mode: 0640,
				Uid: 73025,
				Gid: 5000,
				Size: 5,
				Mtime: 1244428340,
				Typeflag: '0',
				Uname: "dsymonds",
				Gname: "eng",
			},
			&Header{
				Name: "small2.txt",
				Mode: 0640,
				Uid: 73025,
				Gid: 5000,
				Size: 11,
				Mtime: 1244436044,
				Typeflag: '0',
				Uname: "dsymonds",
				Gname: "eng",
			},
		},
	},
	&untarTest{
		file: "testdata/star.tar",
		headers: []*Header{
			&Header{
				Name: "small.txt",
				Mode: 0640,
				Uid: 73025,
				Gid: 5000,
				Size: 5,
				Mtime: 1244592783,
				Typeflag: '0',
				Uname: "dsymonds",
				Gname: "eng",
				Atime: 1244592783,
				Ctime: 1244592783,
			},
			&Header{
				Name: "small2.txt",
				Mode: 0640,
				Uid: 73025,
				Gid: 5000,
				Size: 11,
				Mtime: 1244592783,
				Typeflag: '0',
				Uname: "dsymonds",
				Gname: "eng",
				Atime: 1244592783,
				Ctime: 1244592783,
			},
		},
	},
	&untarTest{
		file: "testdata/v7.tar",
		headers: []*Header{
			&Header{
				Name: "small.txt",
				Mode: 0444,
				Uid: 73025,
				Gid: 5000,
				Size: 5,
				Mtime: 1244593104,
				Typeflag: '\x00',
			},
			&Header{
				Name: "small2.txt",
				Mode: 0444,
				Uid: 73025,
				Gid: 5000,
				Size: 11,
				Mtime: 1244593104,
				Typeflag: '\x00',
			},
		},
	},
}

func TestReader(t *testing.T) {
testLoop:
	for i, test := range untarTests {
		f, err := os.Open(test.file, os.O_RDONLY, 0444);
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err);
			continue;
		}
		tr := NewReader(f);
		for j, header := range test.headers {
			hdr, err := tr.Next();
			if err != nil || hdr == nil {
				t.Errorf("test %d, entry %d: Didn't get entry: %v", i, j, err);
				f.Close();
				continue testLoop;
			}
			if !reflect.DeepEqual(hdr, header) {
				t.Errorf("test %d, entry %d: Incorrect header:\nhave %+v\nwant %+v",
					i, j, *hdr, *header)
			}
		}
		hdr, err := tr.Next();
		if hdr != nil || err != nil {
			t.Errorf("test %d: Unexpected entry or error: hdr=%v err=%v", i, err)
		}
		f.Close();
	}
}

func TestPartialRead(t *testing.T) {
	f, err := os.Open("testdata/gnu.tar", os.O_RDONLY, 0444);
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer f.Close();

	tr := NewReader(f);

	// Read the first four bytes; Next() should skip the last byte.
	hdr, err := tr.Next();
	if err != nil || hdr == nil {
		t.Fatalf("Didn't get first file: %v", err)
	}
	buf := make([]byte, 4);
	if _, err := io.ReadFull(tr, buf); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if expected := strings.Bytes("Kilt"); !bytes.Equal(buf, expected) {
		t.Errorf("Contents = %v, want %v", buf, expected)
	}

	// Second file
	hdr, err = tr.Next();
	if err != nil || hdr == nil {
		t.Fatalf("Didn't get second file: %v", err)
	}
	buf = make([]byte, 6);
	if _, err := io.ReadFull(tr, buf); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if expected := strings.Bytes("Google"); !bytes.Equal(buf, expected) {
		t.Errorf("Contents = %v, want %v", buf, expected)
	}
}
