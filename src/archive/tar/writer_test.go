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
	"sort"
	"strings"
	"testing"
	"testing/iotest"
	"time"
)

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
	type entry struct {
		header   *Header
		contents string
	}

	vectors := []struct {
		file    string // filename of expected output
		entries []*entry
	}{{
		// The writer test file was produced with this command:
		// tar (GNU tar) 1.26
		//   ln -s small.txt link.txt
		//   tar -b 1 --format=ustar -c -f writer.tar small.txt small2.txt link.txt
		file: "testdata/writer.tar",
		entries: []*entry{{
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
		}, {
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
		}, {
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
		}},
	}, {
		// The truncated test file was produced using these commands:
		//   dd if=/dev/zero bs=1048576 count=16384 > /tmp/16gig.txt
		//   tar -b 1 -c -f- /tmp/16gig.txt | dd bs=512 count=8 > writer-big.tar
		file: "testdata/writer-big.tar",
		entries: []*entry{{
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
		}},
	}, {
		// This truncated file was produced using this library.
		// It was verified to work with GNU tar 1.27.1 and BSD tar 3.1.2.
		//  dd if=/dev/zero bs=1G count=16 >> writer-big-long.tar
		//  gnutar -xvf writer-big-long.tar
		//  bsdtar -xvf writer-big-long.tar
		//
		// This file is in PAX format.
		file: "testdata/writer-big-long.tar",
		entries: []*entry{{
			header: &Header{
				Name:     strings.Repeat("longname/", 15) + "16gig.txt",
				Mode:     0644,
				Uid:      1000,
				Gid:      1000,
				Size:     16 << 30,
				ModTime:  time.Unix(1399583047, 0),
				Typeflag: '0',
				Uname:    "guillaume",
				Gname:    "guillaume",
			},
			// fake contents
			contents: strings.Repeat("\x00", 4<<10),
		}},
	}, {
		// TODO(dsnet): The Writer output should match the following file.
		// To fix an issue (see https://golang.org/issue/12594), we disabled
		// prefix support, which alters the generated output.
		/*
			// This file was produced using gnu tar 1.17
			// gnutar  -b 4 --format=ustar (longname/)*15 + file.txt
			file: "testdata/ustar.tar"
		*/
		file: "testdata/ustar.issue12594.tar", // This is a valid tar file, but not expected
		entries: []*entry{{
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
		}},
	}, {
		// This file was produced using gnu tar 1.26
		// echo "Slartibartfast" > file.txt
		// ln file.txt hard.txt
		// tar -b 1 --format=ustar -c -f hardlink.tar file.txt hard.txt
		file: "testdata/hardlink.tar",
		entries: []*entry{{
			header: &Header{
				Name:     "file.txt",
				Mode:     0644,
				Uid:      1000,
				Gid:      100,
				Size:     15,
				ModTime:  time.Unix(1425484303, 0),
				Typeflag: '0',
				Uname:    "vbatts",
				Gname:    "users",
			},
			contents: "Slartibartfast\n",
		}, {
			header: &Header{
				Name:     "hard.txt",
				Mode:     0644,
				Uid:      1000,
				Gid:      100,
				Size:     0,
				ModTime:  time.Unix(1425484303, 0),
				Typeflag: '1',
				Linkname: "file.txt",
				Uname:    "vbatts",
				Gname:    "users",
			},
			// no contents
		}},
	}}

testLoop:
	for i, v := range vectors {
		expected, err := ioutil.ReadFile(v.file)
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err)
			continue
		}

		buf := new(bytes.Buffer)
		tw := NewWriter(iotest.TruncateWriter(buf, 4<<10)) // only catch the first 4 KB
		big := false
		for j, entry := range v.entries {
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
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.0")) {
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
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.0")) {
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
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.0")) {
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

func TestPaxHeadersSorted(t *testing.T) {
	fileinfo, err := os.Stat("testdata/small.txt")
	if err != nil {
		t.Fatal(err)
	}
	hdr, err := FileInfoHeader(fileinfo, "")
	if err != nil {
		t.Fatalf("os.Stat: %v", err)
	}
	contents := strings.Repeat(" ", int(hdr.Size))

	hdr.Xattrs = map[string]string{
		"foo": "foo",
		"bar": "bar",
		"baz": "baz",
		"qux": "qux",
	}

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
	if !bytes.Contains(buf.Bytes(), []byte("PaxHeaders.0")) {
		t.Fatal("Expected at least one PAX header to be written.")
	}

	// xattr bar should always appear before others
	indices := []int{
		bytes.Index(buf.Bytes(), []byte("bar=bar")),
		bytes.Index(buf.Bytes(), []byte("baz=baz")),
		bytes.Index(buf.Bytes(), []byte("foo=foo")),
		bytes.Index(buf.Bytes(), []byte("qux=qux")),
	}
	if !sort.IntsAreSorted(indices) {
		t.Fatal("PAX headers are not sorted")
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

func TestValidTypeflagWithPAXHeader(t *testing.T) {
	var buffer bytes.Buffer
	tw := NewWriter(&buffer)

	fileName := strings.Repeat("ab", 100)

	hdr := &Header{
		Name:     fileName,
		Size:     4,
		Typeflag: 0,
	}
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatalf("Failed to write header: %s", err)
	}
	if _, err := tw.Write([]byte("fooo")); err != nil {
		t.Fatalf("Failed to write the file's data: %s", err)
	}
	tw.Close()

	tr := NewReader(&buffer)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("Failed to read header: %s", err)
		}
		if header.Typeflag != 0 {
			t.Fatalf("Typeflag should've been 0, found %d", header.Typeflag)
		}
	}
}

func TestWriteAfterClose(t *testing.T) {
	var buffer bytes.Buffer
	tw := NewWriter(&buffer)

	hdr := &Header{
		Name: "small.txt",
		Size: 5,
	}
	if err := tw.WriteHeader(hdr); err != nil {
		t.Fatalf("Failed to write header: %s", err)
	}
	tw.Close()
	if _, err := tw.Write([]byte("Kilts")); err != ErrWriteAfterClose {
		t.Fatalf("Write: got %v; want ErrWriteAfterClose", err)
	}
}

func TestSplitUSTARPath(t *testing.T) {
	sr := strings.Repeat

	vectors := []struct {
		input  string // Input path
		prefix string // Expected output prefix
		suffix string // Expected output suffix
		ok     bool   // Split success?
	}{
		{"", "", "", false},
		{"abc", "", "", false},
		{"用戶名", "", "", false},
		{sr("a", nameSize), "", "", false},
		{sr("a", nameSize) + "/", "", "", false},
		{sr("a", nameSize) + "/a", sr("a", nameSize), "a", true},
		{sr("a", prefixSize) + "/", "", "", false},
		{sr("a", prefixSize) + "/a", sr("a", prefixSize), "a", true},
		{sr("a", nameSize+1), "", "", false},
		{sr("/", nameSize+1), sr("/", nameSize-1), "/", true},
		{sr("a", prefixSize) + "/" + sr("b", nameSize),
			sr("a", prefixSize), sr("b", nameSize), true},
		{sr("a", prefixSize) + "//" + sr("b", nameSize), "", "", false},
		{sr("a/", nameSize), sr("a/", 77) + "a", sr("a/", 22), true},
	}

	for _, v := range vectors {
		prefix, suffix, ok := splitUSTARPath(v.input)
		if prefix != v.prefix || suffix != v.suffix || ok != v.ok {
			t.Errorf("splitUSTARPath(%q):\ngot  (%q, %q, %v)\nwant (%q, %q, %v)",
				v.input, prefix, suffix, ok, v.prefix, v.suffix, v.ok)
		}
	}
}

// TestIssue12594 tests that the Writer does not attempt to populate the prefix
// field when encoding a header in the GNU format. The prefix field is valid
// in USTAR and PAX, but not GNU.
func TestIssue12594(t *testing.T) {
	names := []string{
		"0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/file.txt",
		"0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/file.txt",
		"0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/333/file.txt",
		"0/1/2/3/4/5/6/7/8/9/10/11/12/13/14/15/16/17/18/19/20/21/22/23/24/25/26/27/28/29/30/31/32/33/34/35/36/37/38/39/40/file.txt",
		"0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000/file.txt",
		"/home/support/.openoffice.org/3/user/uno_packages/cache/registry/com.sun.star.comp.deployment.executable.PackageRegistryBackend",
	}

	for i, name := range names {
		var b bytes.Buffer

		tw := NewWriter(&b)
		if err := tw.WriteHeader(&Header{
			Name: name,
			Uid:  1 << 25, // Prevent USTAR format
		}); err != nil {
			t.Errorf("test %d, unexpected WriteHeader error: %v", i, err)
		}
		if err := tw.Close(); err != nil {
			t.Errorf("test %d, unexpected Close error: %v", i, err)
		}

		// The prefix field should never appear in the GNU format.
		var blk block
		copy(blk[:], b.Bytes())
		prefix := string(blk.USTAR().Prefix())
		if i := strings.IndexByte(prefix, 0); i >= 0 {
			prefix = prefix[:i] // Truncate at the NUL terminator
		}
		if blk.GetFormat() == formatGNU && len(prefix) > 0 && strings.HasPrefix(name, prefix) {
			t.Errorf("test %d, found prefix in GNU format: %s", i, prefix)
		}

		tr := NewReader(&b)
		hdr, err := tr.Next()
		if err != nil {
			t.Errorf("test %d, unexpected Next error: %v", i, err)
		}
		if hdr.Name != name {
			t.Errorf("test %d, hdr.Name = %s, want %s", i, hdr.Name, name)
		}
	}
}
