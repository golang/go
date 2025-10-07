// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"encoding/hex"
	"errors"
	"io"
	"io/fs"
	"maps"
	"os"
	"path"
	"slices"
	"strings"
	"testing"
	"testing/fstest"
	"testing/iotest"
	"time"
)

func bytediff(a, b []byte) string {
	const (
		uniqueA  = "-  "
		uniqueB  = "+  "
		identity = "   "
	)
	var ss []string
	sa := strings.Split(strings.TrimSpace(hex.Dump(a)), "\n")
	sb := strings.Split(strings.TrimSpace(hex.Dump(b)), "\n")
	for len(sa) > 0 && len(sb) > 0 {
		if sa[0] == sb[0] {
			ss = append(ss, identity+sa[0])
		} else {
			ss = append(ss, uniqueA+sa[0])
			ss = append(ss, uniqueB+sb[0])
		}
		sa, sb = sa[1:], sb[1:]
	}
	for len(sa) > 0 {
		ss = append(ss, uniqueA+sa[0])
		sa = sa[1:]
	}
	for len(sb) > 0 {
		ss = append(ss, uniqueB+sb[0])
		sb = sb[1:]
	}
	return strings.Join(ss, "\n")
}

func TestWriter(t *testing.T) {
	type (
		testHeader struct { // WriteHeader(hdr) == wantErr
			hdr     Header
			wantErr error
		}
		testWrite struct { // Write(str) == (wantCnt, wantErr)
			str     string
			wantCnt int
			wantErr error
		}
		testReadFrom struct { // ReadFrom(testFile{ops}) == (wantCnt, wantErr)
			ops     fileOps
			wantCnt int64
			wantErr error
		}
		testClose struct { // Close() == wantErr
			wantErr error
		}
		testFnc any // testHeader | testWrite | testReadFrom | testClose
	)

	vectors := []struct {
		file  string // Optional filename of expected output
		tests []testFnc
	}{{
		// The writer test file was produced with this command:
		// tar (GNU tar) 1.26
		//   ln -s small.txt link.txt
		//   tar -b 1 --format=ustar -c -f writer.tar small.txt small2.txt link.txt
		file: "testdata/writer.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "small.txt",
				Size:     5,
				Mode:     0640,
				Uid:      73025,
				Gid:      5000,
				Uname:    "dsymonds",
				Gname:    "eng",
				ModTime:  time.Unix(1246508266, 0),
			}, nil},
			testWrite{"Kilts", 5, nil},

			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "small2.txt",
				Size:     11,
				Mode:     0640,
				Uid:      73025,
				Uname:    "dsymonds",
				Gname:    "eng",
				Gid:      5000,
				ModTime:  time.Unix(1245217492, 0),
			}, nil},
			testWrite{"Google.com\n", 11, nil},

			testHeader{Header{
				Typeflag: TypeSymlink,
				Name:     "link.txt",
				Linkname: "small.txt",
				Mode:     0777,
				Uid:      1000,
				Gid:      1000,
				Uname:    "strings",
				Gname:    "strings",
				ModTime:  time.Unix(1314603082, 0),
			}, nil},
			testWrite{"", 0, nil},

			testClose{nil},
		},
	}, {
		// The truncated test file was produced using these commands:
		//   dd if=/dev/zero bs=1048576 count=16384 > /tmp/16gig.txt
		//   tar -b 1 -c -f- /tmp/16gig.txt | dd bs=512 count=8 > writer-big.tar
		file: "testdata/writer-big.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "tmp/16gig.txt",
				Size:     16 << 30,
				Mode:     0640,
				Uid:      73025,
				Gid:      5000,
				Uname:    "dsymonds",
				Gname:    "eng",
				ModTime:  time.Unix(1254699560, 0),
				Format:   FormatGNU,
			}, nil},
		},
	}, {
		// This truncated file was produced using this library.
		// It was verified to work with GNU tar 1.27.1 and BSD tar 3.1.2.
		//  dd if=/dev/zero bs=1G count=16 >> writer-big-long.tar
		//  gnutar -xvf writer-big-long.tar
		//  bsdtar -xvf writer-big-long.tar
		//
		// This file is in PAX format.
		file: "testdata/writer-big-long.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     strings.Repeat("longname/", 15) + "16gig.txt",
				Size:     16 << 30,
				Mode:     0644,
				Uid:      1000,
				Gid:      1000,
				Uname:    "guillaume",
				Gname:    "guillaume",
				ModTime:  time.Unix(1399583047, 0),
			}, nil},
		},
	}, {
		// This file was produced using GNU tar v1.17.
		//	gnutar -b 4 --format=ustar (longname/)*15 + file.txt
		file: "testdata/ustar.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     strings.Repeat("longname/", 15) + "file.txt",
				Size:     6,
				Mode:     0644,
				Uid:      501,
				Gid:      20,
				Uname:    "shane",
				Gname:    "staff",
				ModTime:  time.Unix(1360135598, 0),
			}, nil},
			testWrite{"hello\n", 6, nil},
			testClose{nil},
		},
	}, {
		// This file was produced using GNU tar v1.26:
		//	echo "Slartibartfast" > file.txt
		//	ln file.txt hard.txt
		//	tar -b 1 --format=ustar -c -f hardlink.tar file.txt hard.txt
		file: "testdata/hardlink.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "file.txt",
				Size:     15,
				Mode:     0644,
				Uid:      1000,
				Gid:      100,
				Uname:    "vbatts",
				Gname:    "users",
				ModTime:  time.Unix(1425484303, 0),
			}, nil},
			testWrite{"Slartibartfast\n", 15, nil},

			testHeader{Header{
				Typeflag: TypeLink,
				Name:     "hard.txt",
				Linkname: "file.txt",
				Mode:     0644,
				Uid:      1000,
				Gid:      100,
				Uname:    "vbatts",
				Gname:    "users",
				ModTime:  time.Unix(1425484303, 0),
			}, nil},
			testWrite{"", 0, nil},

			testClose{nil},
		},
	}, {
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "bad-null.txt",
				Xattrs:   map[string]string{"null\x00null\x00": "fizzbuzz"},
			}, headerError{}},
		},
	}, {
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "null\x00.txt",
			}, headerError{}},
		},
	}, {
		file: "testdata/pax-records.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "file",
				Uname:    strings.Repeat("long", 10),
				PAXRecords: map[string]string{
					"path":           "FILE", // Should be ignored
					"GNU.sparse.map": "0,0",  // Should be ignored
					"comment":        "Hello, 世界",
					"GOLANG.pkg":     "tar",
				},
			}, nil},
			testClose{nil},
		},
	}, {
		// Craft a theoretically valid PAX archive with global headers.
		// The GNU and BSD tar tools do not parse these the same way.
		//
		// BSD tar v3.1.2 parses and ignores all global headers;
		// the behavior is verified by researching the source code.
		//
		//	$ bsdtar -tvf pax-global-records.tar
		//	----------  0 0      0           0 Dec 31  1969 file1
		//	----------  0 0      0           0 Dec 31  1969 file2
		//	----------  0 0      0           0 Dec 31  1969 file3
		//	----------  0 0      0           0 May 13  2014 file4
		//
		// GNU tar v1.27.1 applies global headers to subsequent records,
		// but does not do the following properly:
		//	* It does not treat an empty record as deletion.
		//	* It does not use subsequent global headers to update previous ones.
		//
		//	$ gnutar -tvf pax-global-records.tar
		//	---------- 0/0               0 2017-07-13 19:40 global1
		//	---------- 0/0               0 2017-07-13 19:40 file2
		//	gnutar: Substituting `.' for empty member name
		//	---------- 0/0               0 1969-12-31 16:00
		//	gnutar: Substituting `.' for empty member name
		//	---------- 0/0               0 2014-05-13 09:53
		//
		// According to the PAX specification, this should have been the result:
		//	---------- 0/0               0 2017-07-13 19:40 global1
		//	---------- 0/0               0 2017-07-13 19:40 file2
		//	---------- 0/0               0 2017-07-13 19:40 file3
		//	---------- 0/0               0 2014-05-13 09:53 file4
		file: "testdata/pax-global-records.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag:   TypeXGlobalHeader,
				PAXRecords: map[string]string{"path": "global1", "mtime": "1500000000.0"},
			}, nil},
			testHeader{Header{
				Typeflag: TypeReg, Name: "file1",
			}, nil},
			testHeader{Header{
				Typeflag:   TypeReg,
				Name:       "file2",
				PAXRecords: map[string]string{"path": "file2"},
			}, nil},
			testHeader{Header{
				Typeflag:   TypeXGlobalHeader,
				PAXRecords: map[string]string{"path": ""}, // Should delete "path", but keep "mtime"
			}, nil},
			testHeader{Header{
				Typeflag: TypeReg, Name: "file3",
			}, nil},
			testHeader{Header{
				Typeflag:   TypeReg,
				Name:       "file4",
				ModTime:    time.Unix(1400000000, 0),
				PAXRecords: map[string]string{"mtime": "1400000000"},
			}, nil},
			testClose{nil},
		},
	}, {
		file: "testdata/gnu-utf8.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹",
				Mode:     0644,
				Uid:      1000, Gid: 1000,
				Uname:   "☺",
				Gname:   "⚹",
				ModTime: time.Unix(0, 0),
				Format:  FormatGNU,
			}, nil},
			testClose{nil},
		},
	}, {
		file: "testdata/gnu-not-utf8.tar",
		tests: []testFnc{
			testHeader{Header{
				Typeflag: TypeReg,
				Name:     "hi\x80\x81\x82\x83bye",
				Mode:     0644,
				Uid:      1000,
				Gid:      1000,
				Uname:    "rawr",
				Gname:    "dsnet",
				ModTime:  time.Unix(0, 0),
				Format:   FormatGNU,
			}, nil},
			testClose{nil},
		},
		// TODO(dsnet): Re-enable this test when adding sparse support.
		// See https://golang.org/issue/22735
		/*
			}, {
				file: "testdata/gnu-nil-sparse-data.tar",
				tests: []testFnc{
					testHeader{Header{
						Typeflag:    TypeGNUSparse,
						Name:        "sparse.db",
						Size:        1000,
						SparseHoles: []sparseEntry{{Offset: 1000, Length: 0}},
					}, nil},
					testWrite{strings.Repeat("0123456789", 100), 1000, nil},
					testClose{},
				},
			}, {
				file: "testdata/gnu-nil-sparse-hole.tar",
				tests: []testFnc{
					testHeader{Header{
						Typeflag:    TypeGNUSparse,
						Name:        "sparse.db",
						Size:        1000,
						SparseHoles: []sparseEntry{{Offset: 0, Length: 1000}},
					}, nil},
					testWrite{strings.Repeat("\x00", 1000), 1000, nil},
					testClose{},
				},
			}, {
				file: "testdata/pax-nil-sparse-data.tar",
				tests: []testFnc{
					testHeader{Header{
						Typeflag:    TypeReg,
						Name:        "sparse.db",
						Size:        1000,
						SparseHoles: []sparseEntry{{Offset: 1000, Length: 0}},
					}, nil},
					testWrite{strings.Repeat("0123456789", 100), 1000, nil},
					testClose{},
				},
			}, {
				file: "testdata/pax-nil-sparse-hole.tar",
				tests: []testFnc{
					testHeader{Header{
						Typeflag:    TypeReg,
						Name:        "sparse.db",
						Size:        1000,
						SparseHoles: []sparseEntry{{Offset: 0, Length: 1000}},
					}, nil},
					testWrite{strings.Repeat("\x00", 1000), 1000, nil},
					testClose{},
				},
			}, {
				file: "testdata/gnu-sparse-big.tar",
				tests: []testFnc{
					testHeader{Header{
						Typeflag: TypeGNUSparse,
						Name:     "gnu-sparse",
						Size:     6e10,
						SparseHoles: []sparseEntry{
							{Offset: 0e10, Length: 1e10 - 100},
							{Offset: 1e10, Length: 1e10 - 100},
							{Offset: 2e10, Length: 1e10 - 100},
							{Offset: 3e10, Length: 1e10 - 100},
							{Offset: 4e10, Length: 1e10 - 100},
							{Offset: 5e10, Length: 1e10 - 100},
						},
					}, nil},
					testReadFrom{fileOps{
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
					}, 6e10, nil},
					testClose{nil},
				},
			}, {
				file: "testdata/pax-sparse-big.tar",
				tests: []testFnc{
					testHeader{Header{
						Typeflag: TypeReg,
						Name:     "pax-sparse",
						Size:     6e10,
						SparseHoles: []sparseEntry{
							{Offset: 0e10, Length: 1e10 - 100},
							{Offset: 1e10, Length: 1e10 - 100},
							{Offset: 2e10, Length: 1e10 - 100},
							{Offset: 3e10, Length: 1e10 - 100},
							{Offset: 4e10, Length: 1e10 - 100},
							{Offset: 5e10, Length: 1e10 - 100},
						},
					}, nil},
					testReadFrom{fileOps{
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
						int64(1e10 - blockSize),
						strings.Repeat("\x00", blockSize-100) + strings.Repeat("0123456789", 10),
					}, 6e10, nil},
					testClose{nil},
				},
		*/
	}, {
		file: "testdata/trailing-slash.tar",
		tests: []testFnc{
			testHeader{Header{Name: strings.Repeat("123456789/", 30)}, nil},
			testClose{nil},
		},
	}, {
		// Automatically promote zero value of Typeflag depending on the name.
		file: "testdata/file-and-dir.tar",
		tests: []testFnc{
			testHeader{Header{Name: "small.txt", Size: 5}, nil},
			testWrite{"Kilts", 5, nil},
			testHeader{Header{Name: "dir/"}, nil},
			testClose{nil},
		},
	}}

	equalError := func(x, y error) bool {
		_, ok1 := x.(headerError)
		_, ok2 := y.(headerError)
		if ok1 || ok2 {
			return ok1 && ok2
		}
		return x == y
	}
	for _, v := range vectors {
		t.Run(path.Base(v.file), func(t *testing.T) {
			const maxSize = 10 << 10 // 10KiB
			buf := new(bytes.Buffer)
			tw := NewWriter(iotest.TruncateWriter(buf, maxSize))

			for i, tf := range v.tests {
				switch tf := tf.(type) {
				case testHeader:
					err := tw.WriteHeader(&tf.hdr)
					if !equalError(err, tf.wantErr) {
						t.Fatalf("test %d, WriteHeader() = %v, want %v", i, err, tf.wantErr)
					}
				case testWrite:
					got, err := tw.Write([]byte(tf.str))
					if got != tf.wantCnt || !equalError(err, tf.wantErr) {
						t.Fatalf("test %d, Write() = (%d, %v), want (%d, %v)", i, got, err, tf.wantCnt, tf.wantErr)
					}
				case testReadFrom:
					f := &testFile{ops: tf.ops}
					got, err := tw.readFrom(f)
					if _, ok := err.(testError); ok {
						t.Errorf("test %d, ReadFrom(): %v", i, err)
					} else if got != tf.wantCnt || !equalError(err, tf.wantErr) {
						t.Errorf("test %d, ReadFrom() = (%d, %v), want (%d, %v)", i, got, err, tf.wantCnt, tf.wantErr)
					}
					if len(f.ops) > 0 {
						t.Errorf("test %d, expected %d more operations", i, len(f.ops))
					}
				case testClose:
					err := tw.Close()
					if !equalError(err, tf.wantErr) {
						t.Fatalf("test %d, Close() = %v, want %v", i, err, tf.wantErr)
					}
				default:
					t.Fatalf("test %d, unknown test operation: %T", i, tf)
				}
			}

			if v.file != "" {
				want, err := os.ReadFile(v.file)
				if err != nil {
					t.Fatalf("ReadFile() = %v, want nil", err)
				}
				got := buf.Bytes()
				if !bytes.Equal(want, got) {
					t.Fatalf("incorrect result: (-got +want)\n%v", bytediff(got, want))
				}
			}
		})
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
	if err != nil {
		t.Fatalf("os.Stat:1 %v", err)
	}
	hdr.Typeflag = TypeSymlink
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
	if !maps.Equal(hdr.Xattrs, xattrs) {
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
	if !slices.IsSorted(indices) {
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
	if err != nil {
		t.Fatalf("os.Stat:1 %v", err)
	}
	hdr.Typeflag = TypeDir
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
	if err != nil && err != ErrInsecurePath {
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
		if header.Typeflag != TypeReg {
			t.Fatalf("Typeflag should've been %d, found %d", TypeReg, header.Typeflag)
		}
	}
}

// failOnceWriter fails exactly once and then always reports success.
type failOnceWriter bool

func (w *failOnceWriter) Write(b []byte) (int, error) {
	if !*w {
		return 0, io.ErrShortWrite
	}
	*w = true
	return len(b), nil
}

func TestWriterErrors(t *testing.T) {
	t.Run("HeaderOnly", func(t *testing.T) {
		tw := NewWriter(new(bytes.Buffer))
		hdr := &Header{Name: "dir/", Typeflag: TypeDir}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("WriteHeader() = %v, want nil", err)
		}
		if _, err := tw.Write([]byte{0x00}); err != ErrWriteTooLong {
			t.Fatalf("Write() = %v, want %v", err, ErrWriteTooLong)
		}
	})

	t.Run("NegativeSize", func(t *testing.T) {
		tw := NewWriter(new(bytes.Buffer))
		hdr := &Header{Name: "small.txt", Size: -1}
		if err := tw.WriteHeader(hdr); err == nil {
			t.Fatalf("WriteHeader() = nil, want non-nil error")
		}
	})

	t.Run("BeforeHeader", func(t *testing.T) {
		tw := NewWriter(new(bytes.Buffer))
		if _, err := tw.Write([]byte("Kilts")); err != ErrWriteTooLong {
			t.Fatalf("Write() = %v, want %v", err, ErrWriteTooLong)
		}
	})

	t.Run("AfterClose", func(t *testing.T) {
		tw := NewWriter(new(bytes.Buffer))
		hdr := &Header{Name: "small.txt"}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("WriteHeader() = %v, want nil", err)
		}
		if err := tw.Close(); err != nil {
			t.Fatalf("Close() = %v, want nil", err)
		}
		if _, err := tw.Write([]byte("Kilts")); err != ErrWriteAfterClose {
			t.Fatalf("Write() = %v, want %v", err, ErrWriteAfterClose)
		}
		if err := tw.Flush(); err != ErrWriteAfterClose {
			t.Fatalf("Flush() = %v, want %v", err, ErrWriteAfterClose)
		}
		if err := tw.Close(); err != nil {
			t.Fatalf("Close() = %v, want nil", err)
		}
	})

	t.Run("PrematureFlush", func(t *testing.T) {
		tw := NewWriter(new(bytes.Buffer))
		hdr := &Header{Name: "small.txt", Size: 5}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("WriteHeader() = %v, want nil", err)
		}
		if err := tw.Flush(); err == nil {
			t.Fatalf("Flush() = %v, want non-nil error", err)
		}
	})

	t.Run("PrematureClose", func(t *testing.T) {
		tw := NewWriter(new(bytes.Buffer))
		hdr := &Header{Name: "small.txt", Size: 5}
		if err := tw.WriteHeader(hdr); err != nil {
			t.Fatalf("WriteHeader() = %v, want nil", err)
		}
		if err := tw.Close(); err == nil {
			t.Fatalf("Close() = %v, want non-nil error", err)
		}
	})

	t.Run("Persistence", func(t *testing.T) {
		tw := NewWriter(new(failOnceWriter))
		if err := tw.WriteHeader(&Header{}); err != io.ErrShortWrite {
			t.Fatalf("WriteHeader() = %v, want %v", err, io.ErrShortWrite)
		}
		if err := tw.WriteHeader(&Header{Name: "small.txt"}); err == nil {
			t.Errorf("WriteHeader() = got %v, want non-nil error", err)
		}
		if _, err := tw.Write(nil); err == nil {
			t.Errorf("Write() = %v, want non-nil error", err)
		}
		if err := tw.Flush(); err == nil {
			t.Errorf("Flush() = %v, want non-nil error", err)
		}
		if err := tw.Close(); err == nil {
			t.Errorf("Close() = %v, want non-nil error", err)
		}
	})
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
		prefix := string(blk.toUSTAR().prefix())
		prefix, _, _ = strings.Cut(prefix, "\x00") // Truncate at the NUL terminator
		if blk.getFormat() == FormatGNU && len(prefix) > 0 && strings.HasPrefix(name, prefix) {
			t.Errorf("test %d, found prefix in GNU format: %s", i, prefix)
		}

		tr := NewReader(&b)
		hdr, err := tr.Next()
		if err != nil && err != ErrInsecurePath {
			t.Errorf("test %d, unexpected Next error: %v", i, err)
		}
		if hdr.Name != name {
			t.Errorf("test %d, hdr.Name = %s, want %s", i, hdr.Name, name)
		}
	}
}

func TestWriteLongHeader(t *testing.T) {
	for _, test := range []struct {
		name string
		h    *Header
	}{{
		name: "name too long",
		h:    &Header{Name: strings.Repeat("a", maxSpecialFileSize)},
	}, {
		name: "linkname too long",
		h:    &Header{Linkname: strings.Repeat("a", maxSpecialFileSize)},
	}, {
		name: "uname too long",
		h:    &Header{Uname: strings.Repeat("a", maxSpecialFileSize)},
	}, {
		name: "gname too long",
		h:    &Header{Gname: strings.Repeat("a", maxSpecialFileSize)},
	}, {
		name: "PAX header too long",
		h:    &Header{PAXRecords: map[string]string{"GOLANG.x": strings.Repeat("a", maxSpecialFileSize)}},
	}} {
		w := NewWriter(io.Discard)
		if err := w.WriteHeader(test.h); err != ErrFieldTooLong {
			t.Errorf("%v: w.WriteHeader() = %v, want ErrFieldTooLong", test.name, err)
		}
	}
}

// testNonEmptyWriter wraps an io.Writer and ensures that
// Write is never called with an empty buffer.
type testNonEmptyWriter struct{ io.Writer }

func (w testNonEmptyWriter) Write(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, errors.New("unexpected empty Write call")
	}
	return w.Writer.Write(b)
}

func TestFileWriter(t *testing.T) {
	type (
		testWrite struct { // Write(str) == (wantCnt, wantErr)
			str     string
			wantCnt int
			wantErr error
		}
		testReadFrom struct { // ReadFrom(testFile{ops}) == (wantCnt, wantErr)
			ops     fileOps
			wantCnt int64
			wantErr error
		}
		testRemaining struct { // logicalRemaining() == wantLCnt, physicalRemaining() == wantPCnt
			wantLCnt int64
			wantPCnt int64
		}
		testFnc any // testWrite | testReadFrom | testRemaining
	)

	type (
		makeReg struct {
			size    int64
			wantStr string
		}
		makeSparse struct {
			makeReg makeReg
			sph     sparseHoles
			size    int64
		}
		fileMaker any // makeReg | makeSparse
	)

	vectors := []struct {
		maker fileMaker
		tests []testFnc
	}{{
		maker: makeReg{0, ""},
		tests: []testFnc{
			testRemaining{0, 0},
			testWrite{"", 0, nil},
			testWrite{"a", 0, ErrWriteTooLong},
			testReadFrom{fileOps{""}, 0, nil},
			testReadFrom{fileOps{"a"}, 0, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{1, "a"},
		tests: []testFnc{
			testRemaining{1, 1},
			testWrite{"", 0, nil},
			testWrite{"a", 1, nil},
			testWrite{"bcde", 0, ErrWriteTooLong},
			testWrite{"", 0, nil},
			testReadFrom{fileOps{""}, 0, nil},
			testReadFrom{fileOps{"a"}, 0, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{5, "hello"},
		tests: []testFnc{
			testRemaining{5, 5},
			testWrite{"hello", 5, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{5, "\x00\x00\x00\x00\x00"},
		tests: []testFnc{
			testRemaining{5, 5},
			testReadFrom{fileOps{"\x00\x00\x00\x00\x00"}, 5, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{5, "\x00\x00\x00\x00\x00"},
		tests: []testFnc{
			testRemaining{5, 5},
			testReadFrom{fileOps{"\x00\x00\x00\x00\x00extra"}, 5, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{5, "abc\x00\x00"},
		tests: []testFnc{
			testRemaining{5, 5},
			testWrite{"abc", 3, nil},
			testRemaining{2, 2},
			testReadFrom{fileOps{"\x00\x00"}, 2, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{5, "\x00\x00abc"},
		tests: []testFnc{
			testRemaining{5, 5},
			testWrite{"\x00\x00", 2, nil},
			testRemaining{3, 3},
			testWrite{"abc", 3, nil},
			testReadFrom{fileOps{"z"}, 0, ErrWriteTooLong},
			testWrite{"z", 0, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{5, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testRemaining{8, 5},
			testWrite{"ab\x00\x00\x00cde", 8, nil},
			testWrite{"a", 0, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{5, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testWrite{"ab\x00\x00\x00cdez", 8, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{5, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testWrite{"ab\x00", 3, nil},
			testRemaining{5, 3},
			testWrite{"\x00\x00cde", 5, nil},
			testWrite{"a", 0, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{5, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testWrite{"ab", 2, nil},
			testRemaining{6, 3},
			testReadFrom{fileOps{int64(3), "cde"}, 6, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{5, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testReadFrom{fileOps{"ab", int64(3), "cde"}, 8, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{5, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testReadFrom{fileOps{"ab", int64(3), "cdeX"}, 8, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{4, "abcd"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testReadFrom{fileOps{"ab", int64(3), "cd"}, 7, io.ErrUnexpectedEOF},
			testRemaining{1, 0},
		},
	}, {
		maker: makeSparse{makeReg{4, "abcd"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testReadFrom{fileOps{"ab", int64(3), "cde"}, 7, errMissData},
			testRemaining{1, 0},
		},
	}, {
		maker: makeSparse{makeReg{6, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testReadFrom{fileOps{"ab", int64(3), "cde"}, 8, errUnrefData},
			testRemaining{0, 1},
		},
	}, {
		maker: makeSparse{makeReg{4, "abcd"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testWrite{"ab", 2, nil},
			testRemaining{6, 2},
			testWrite{"\x00\x00\x00", 3, nil},
			testRemaining{3, 2},
			testWrite{"cde", 2, errMissData},
			testRemaining{1, 0},
		},
	}, {
		maker: makeSparse{makeReg{6, "abcde"}, sparseHoles{{2, 3}}, 8},
		tests: []testFnc{
			testWrite{"ab", 2, nil},
			testRemaining{6, 4},
			testWrite{"\x00\x00\x00", 3, nil},
			testRemaining{3, 4},
			testWrite{"cde", 3, errUnrefData},
			testRemaining{0, 1},
		},
	}, {
		maker: makeSparse{makeReg{3, "abc"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testRemaining{7, 3},
			testWrite{"\x00\x00abc\x00\x00", 7, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{3, "abc"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testRemaining{7, 3},
			testReadFrom{fileOps{int64(2), "abc", int64(1), "\x00"}, 7, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{3, ""}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testWrite{"abcdefg", 0, errWriteHole},
		},
	}, {
		maker: makeSparse{makeReg{3, "abc"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testWrite{"\x00\x00abcde", 5, errWriteHole},
		},
	}, {
		maker: makeSparse{makeReg{3, "abc"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testWrite{"\x00\x00abc\x00\x00z", 7, ErrWriteTooLong},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{3, "abc"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testWrite{"\x00\x00", 2, nil},
			testRemaining{5, 3},
			testWrite{"abc", 3, nil},
			testRemaining{2, 0},
			testWrite{"\x00\x00", 2, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{2, "ab"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testWrite{"\x00\x00", 2, nil},
			testWrite{"abc", 2, errMissData},
			testWrite{"\x00\x00", 0, errMissData},
		},
	}, {
		maker: makeSparse{makeReg{4, "abc"}, sparseHoles{{0, 2}, {5, 2}}, 7},
		tests: []testFnc{
			testWrite{"\x00\x00", 2, nil},
			testWrite{"abc", 3, nil},
			testWrite{"\x00\x00", 2, errUnrefData},
		},
	}}

	for i, v := range vectors {
		var wantStr string
		bb := new(strings.Builder)
		w := testNonEmptyWriter{bb}
		var fw fileWriter
		switch maker := v.maker.(type) {
		case makeReg:
			fw = &regFileWriter{w, maker.size}
			wantStr = maker.wantStr
		case makeSparse:
			if !validateSparseEntries(maker.sph, maker.size) {
				t.Fatalf("invalid sparse map: %v", maker.sph)
			}
			spd := invertSparseEntries(maker.sph, maker.size)
			fw = &regFileWriter{w, maker.makeReg.size}
			fw = &sparseFileWriter{fw, spd, 0}
			wantStr = maker.makeReg.wantStr
		default:
			t.Fatalf("test %d, unknown make operation: %T", i, maker)
		}

		for j, tf := range v.tests {
			switch tf := tf.(type) {
			case testWrite:
				got, err := fw.Write([]byte(tf.str))
				if got != tf.wantCnt || err != tf.wantErr {
					t.Errorf("test %d.%d, Write(%s):\ngot  (%d, %v)\nwant (%d, %v)", i, j, tf.str, got, err, tf.wantCnt, tf.wantErr)
				}
			case testReadFrom:
				f := &testFile{ops: tf.ops}
				got, err := fw.ReadFrom(f)
				if _, ok := err.(testError); ok {
					t.Errorf("test %d.%d, ReadFrom(): %v", i, j, err)
				} else if got != tf.wantCnt || err != tf.wantErr {
					t.Errorf("test %d.%d, ReadFrom() = (%d, %v), want (%d, %v)", i, j, got, err, tf.wantCnt, tf.wantErr)
				}
				if len(f.ops) > 0 {
					t.Errorf("test %d.%d, expected %d more operations", i, j, len(f.ops))
				}
			case testRemaining:
				if got := fw.logicalRemaining(); got != tf.wantLCnt {
					t.Errorf("test %d.%d, logicalRemaining() = %d, want %d", i, j, got, tf.wantLCnt)
				}
				if got := fw.physicalRemaining(); got != tf.wantPCnt {
					t.Errorf("test %d.%d, physicalRemaining() = %d, want %d", i, j, got, tf.wantPCnt)
				}
			default:
				t.Fatalf("test %d.%d, unknown test operation: %T", i, j, tf)
			}
		}

		if got := bb.String(); got != wantStr {
			t.Fatalf("test %d, String() = %q, want %q", i, got, wantStr)
		}
	}
}

func TestWriterAddFS(t *testing.T) {
	fsys := fstest.MapFS{
		"emptyfolder":          {Mode: 0o755 | os.ModeDir},
		"file.go":              {Data: []byte("hello")},
		"subfolder/another.go": {Data: []byte("world")},
		"symlink.go":           {Mode: 0o777 | os.ModeSymlink, Data: []byte("file.go")},
		// Notably missing here is the "subfolder" directory. This makes sure even
		// if we don't have a subfolder directory listed.
	}
	var buf bytes.Buffer
	tw := NewWriter(&buf)
	if err := tw.AddFS(fsys); err != nil {
		t.Fatal(err)
	}
	if err := tw.Close(); err != nil {
		t.Fatal(err)
	}

	// Add subfolder into fsys to match what we'll read from the tar.
	fsys["subfolder"] = &fstest.MapFile{Mode: 0o555 | os.ModeDir}

	// Test that we can get the files back from the archive
	tr := NewReader(&buf)

	names := make([]string, 0, len(fsys))
	for name := range fsys {
		names = append(names, name)
	}
	slices.Sort(names)

	entriesLeft := len(fsys)
	for _, name := range names {
		entriesLeft--

		entryInfo, err := fsys.Lstat(name)
		if err != nil {
			t.Fatalf("getting entry info error: %v", err)
		}
		hdr, err := tr.Next()
		if err == io.EOF {
			break // End of archive
		}
		if err != nil {
			t.Fatal(err)
		}

		tmpName := name
		if entryInfo.IsDir() {
			tmpName += "/"
		}
		if hdr.Name != tmpName {
			t.Errorf("test fs has filename %v; archive header has %v",
				name, hdr.Name)
		}

		if entryInfo.Mode() != hdr.FileInfo().Mode() {
			t.Errorf("%s: test fs has mode %v; archive header has %v",
				name, entryInfo.Mode(), hdr.FileInfo().Mode())
		}

		switch entryInfo.Mode().Type() {
		case fs.ModeDir:
			// No additional checks necessary.
		case fs.ModeSymlink:
			origtarget := string(fsys[name].Data)
			if hdr.Linkname != origtarget {
				t.Fatalf("test fs has link content %s; archive header %v", origtarget, hdr.Linkname)
			}
		default:
			data, err := io.ReadAll(tr)
			if err != nil {
				t.Fatal(err)
			}
			origdata := fsys[name].Data
			if string(data) != string(origdata) {
				t.Fatalf("test fs has file content %v; archive header has %v", origdata, data)
			}
		}
	}
	if entriesLeft > 0 {
		t.Fatalf("not all entries are in the archive")
	}
}

func TestWriterAddFSNonRegularFiles(t *testing.T) {
	fsys := fstest.MapFS{
		"device":  {Data: []byte("hello"), Mode: 0755 | fs.ModeDevice},
		"symlink": {Data: []byte("world"), Mode: 0755 | fs.ModeSymlink},
	}
	var buf bytes.Buffer
	tw := NewWriter(&buf)
	if err := tw.AddFS(fsys); err == nil {
		t.Fatal("expected error, got nil")
	}
}
