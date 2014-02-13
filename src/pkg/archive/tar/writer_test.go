// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

type writerTestEntry struct {
	header   *Header
	contents string
}

type writerTest struct {
	file    string // filename of expected output
	entries []*writerTestEntry
}

var writerTests = []*writerTest{
	// The writer test file was produced with this command:
	// tar (GNU tar) 1.26
	//   ln -s small.txt link.txt
	//   tar -b 1 --format=ustar -c -f writer.tar small.txt small2.txt link.txt
	{
		file: "testdata/writer.tar",
		entries: []*writerTestEntry{
			{
				header: &Header{
					Name:     "small.txt",
					Mode:     0640,
					Uid:      73025,
					Gid:      5000,
					Size:     5,
					ModTime:  time.Unix(1246508266, 0),
					Typeflag: '0',
					Uname:    "dsymonds",
					Gname:    "eng",
				},
				contents: "Kilts",
			},
			{
				header: &Header{
					Name:     "small2.txt",
					Mode:     0640,
					Uid:      73025,
					Gid:      5000,
					Size:     11,
					ModTime:  time.Unix(1245217492, 0),
					Typeflag: '0',
					Uname:    "dsymonds",
					Gname:    "eng",
				},
				contents: "Google.com\n",
			},
			{
				header: &Header{
					Name:     "link.txt",
					Mode:     0777,
					Uid:      1000,
					Gid:      1000,
					Size:     0,
					ModTime:  time.Unix(1314603082, 0),
					Typeflag: '2',
					Linkname: "small.txt",
					Uname:    "strings",
					Gname:    "strings",
				},
				// no contents
			},
		},
	},
	// The truncated test file was produced using these commands:
	//   dd if=/dev/zero bs=1048576 count=16384 > /tmp/16gig.txt
	//   tar -b 1 -c -f- /tmp/16gig.txt | dd bs=512 count=8 > writer-big.tar
	{
		file: "testdata/writer-big.tar",
		entries: []*writerTestEntry{
			{
				header: &Header{
					Name:     "tmp/16gig.txt",
					Mode:     0640,
					Uid:      73025,
					Gid:      5000,
					Size:     16 << 30,
					ModTime:  time.Unix(1254699560, 0),
					Typeflag: '0',
					Uname:    "dsymonds",
					Gname:    "eng",
				},
				// fake contents
				contents: strings.Repeat("\x00", 4<<10),
			},
		},
	},
	// This file was produced using gnu tar 1.17
	// gnutar  -b 4 --format=ustar (longname/)*15 + file.txt
	{
		file: "testdata/ustar.tar",
		entries: []*writerTestEntry{
			{
				header: &Header{
					Name:     strings.Repeat("longname/", 15) + "file.txt",
					Mode:     0644,
					Uid:      0765,
					Gid:      024,
					Size:     06,
					ModTime:  time.Unix(1360135598, 0),
					Typeflag: '0',
					Uname:    "shane",
					Gname:    "staff",
				},
				contents: "hello\n",
			},
		},
	},
}

// Render byte array in a two-character hexadecimal string, spaced for easy visual inspection.
func bytestr(offset int, b []byte) string {
	const rowLen = 32
	s := fmt.Sprintf("%04x ", offset)
	for _, ch := range b {
		switch {
		case '0' <= ch && ch <= '9', 'A' <= ch && ch <= 'Z', 'a' <= ch && ch <= 'z':
			s += fmt.Sprintf("  %c", ch)
		default:
			s += fmt.Sprintf(" %02x", ch)
		}
	}
	return s
}

// Render a pseudo-diff between two blocks of bytes.
func bytediff(a []byte, b []byte) string {
	const rowLen = 32
	s := fmt.Sprintf("(%d bytes vs. %d bytes)\n", len(a), len(b))
	for offset := 0; len(a)+len(b) > 0; offset += rowLen {
		na, nb := rowLen, rowLen
		if na > len(a) {
			na = len(a)
		}
		if nb > len(b) {
			nb = len(b)
		}
		sa := bytestr(offset, a[0:na])
		sb := bytestr(offset, b[0:nb])
		if sa != sb {
			s += fmt.Sprintf("-%v\n+%v\n", sa, sb)
		}
		a = a[na:]
		b = b[nb:]
	}
	return s
}

func TestWriter(t *testing.T) {
testLoop:
	for i, test := range writerTests {
		expected, err := ioutil.ReadFile(test.file)
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err)
			continue
		}

		buf := new(bytes.Buffer)
		tw := NewWriter(iotest.TruncateWriter(buf, 4<<10)) // only catch the first 4 KB
		big := false
		for j, entry := range test.entries {
			big = big || entry.header.Size > 1<<10
			if err := tw.WriteHeader(entry.header); err != nil {
				t.Errorf("test %d, entry %d: Failed writing header: %v", i, j, err)
				continue testLoop
			}
			if _, err := io.WriteString(tw, entry.contents); err != nil {
				t.Errorf("test %d, entry %d: Failed writing contents: %v", i, j, err)
				continue testLoop
			}
		}
		// Only interested in Close failures for the small tests.
		if err := tw.Close(); err != nil && !big {
			t.Errorf("test %d: Failed closing archive: %v", i, err)
			continue testLoop
		}

		actual := buf.Bytes()
		if !bytes.Equal(expected, actual) {
			t.Errorf("test %d: Incorrect result: (-=expected, +=actual)\n%v",
				i, bytediff(expected, actual))
		}
		if testing.Short() { // The second test is expensive.
			break
		}
	}
}

func TestPax(t *testing.T) {
	// Create an archive with a large name
	fileinfo, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := FileInfoHeader(fileinfo, "")
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	// Force a PAX long name to be written
	longName := strings.Repeat("ab", 100)
	contents := strings.Repeat(" ", int(hdr.Size))
	hdr.Name = longName
	var buf bytes.Buffer
	writer := NewWriter(&buf)
	if err := writer.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err = writer.Write([]byte(contents)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	// Simple test to make sure PAX extensions are in effect
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.")) {
		t.Fatal("Expected at least one PAX header to be written.")
	}
	// Test that we can get a long name back out of the archive.
	reader := NewReader(&buf)
	hdr, err = reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Name != longName {
		t.Fatal("Couldn't recover long file name")
	}
}

func TestPaxSymlink(t *testing.T) {
	// Create an archive with a large linkname
	fileinfo, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := FileInfoHeader(fileinfo, "")
	hdr.Typeflag = TypeSymlink
	if err != nil {
		t.Fatalf("os.Stat:1 %v", err)
	}
	// Force a PAX long linkname to be written
	longLinkname := strings.Repeat("1234567890/1234567890", 10)
	hdr.Linkname = longLinkname

	hdr.Size = 0
	var buf bytes.Buffer
	writer := NewWriter(&buf)
	if err := writer.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	// Simple test to make sure PAX extensions are in effect
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.")) {
		t.Fatal("Expected at least one PAX header to be written.")
	}
	// Test that we can get a long name back out of the archive.
	reader := NewReader(&buf)
	hdr, err = reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Linkname != longLinkname {
		t.Fatal("Couldn't recover long link name")
	}
}

func TestPaxNonAscii(t *testing.T) {
	// Create an archive with non ascii. These should trigger a pax header
	// because pax headers have a defined utf-8 encoding.
	fileinfo, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}

	hdr, err := FileInfoHeader(fileinfo, "")
	if err != nil {
		t.Fatalf("os.Stat:1 %v", err)
	}

	// some sample data
	chineseFilename := "文件名"
	chineseGroupname := "組"
	chineseUsername := "用戶名"

	hdr.Name = chineseFilename
	hdr.Gname = chineseGroupname
	hdr.Uname = chineseUsername

	contents := strings.Repeat(" ", int(hdr.Size))

	var buf bytes.Buffer
	writer := NewWriter(&buf)
	if err := writer.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err = writer.Write([]byte(contents)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	// Simple test to make sure PAX extensions are in effect
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.")) {
		t.Fatal("Expected at least one PAX header to be written.")
	}
	// Test that we can get a long name back out of the archive.
	reader := NewReader(&buf)
	hdr, err = reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Name != chineseFilename {
		t.Fatal("Couldn't recover unicode name")
	}
	if hdr.Gname != chineseGroupname {
		t.Fatal("Couldn't recover unicode group")
	}
	if hdr.Uname != chineseUsername {
		t.Fatal("Couldn't recover unicode user")
	}
}

func TestPaxXattrs(t *testing.T) {
	xattrs := map[string]string{
		"user.key": "value",
	}

	// Create an archive with an xattr
	fileinfo, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := FileInfoHeader(fileinfo, "")
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	contents := "Kilts"
	hdr.Xattrs = xattrs
	var buf bytes.Buffer
	writer := NewWriter(&buf)
	if err := writer.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if _, err = writer.Write([]byte(contents)); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	// Test that we can get the xattrs back out of the archive.
	reader := NewReader(&buf)
	hdr, err = reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(hdr.Xattrs, xattrs) {
		t.Fatalf("xattrs did not survive round trip: got %+v, want %+v",
			hdr.Xattrs, xattrs)
	}
}

func TestPAXHeader(t *testing.T) {
	medName := strings.Repeat("CD", 50)
	longName := strings.Repeat("AB", 100)
	paxTests := [][2]string{
		{paxPath + "=/etc/hosts", "19 path=/etc/hosts\n"},
		{"a=b", "6 a=b\n"},          // Single digit length
		{"a=names", "11 a=names\n"}, // Test case involving carries
		{paxPath + "=" + longName, fmt.Sprintf("210 path=%s\n", longName)},
		{paxPath + "=" + medName, fmt.Sprintf("110 path=%s\n", medName)}}

	for _, test := range paxTests {
		key, expected := test[0], test[1]
		if result := paxHeader(key); result != expected {
			t.Fatalf("paxHeader: got %s, expected %s", result, expected)
		}
	}
}

func TestUSTARLongName(t *testing.T) {
	// Create an archive with a path that failed to split with USTAR extension in previous versions.
	fileinfo, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := FileInfoHeader(fileinfo, "")
	hdr.Typeflag = TypeDir
	if err != nil {
		t.Fatalf("os.Stat:1 %v", err)
	}
	// Force a PAX long name to be written. The name was taken from a practical example
	// that fails and replaced ever char through numbers to anonymize the sample.
	longName := "/0000_0000000/00000-000000000/0000_0000000/00000-0000000000000/0000_0000000/00000-0000000-00000000/0000_0000000/00000000/0000_0000000/000/0000_0000000/00000000v00/0000_0000000/000000/0000_0000000/0000000/0000_0000000/00000y-00/0000/0000/00000000/0x000000/"
	hdr.Name = longName

	hdr.Size = 0
	var buf bytes.Buffer
	writer := NewWriter(&buf)
	if err := writer.WriteHeader(hdr); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	// Test that we can get a long name back out of the archive.
	reader := NewReader(&buf)
	hdr, err = reader.Next()
	if err != nil {
		t.Fatal(err)
	}
	if hdr.Name != longName {
		t.Fatal("Couldn't recover long name")
	}
}
