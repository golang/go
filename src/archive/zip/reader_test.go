// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
	"time"
)

type ZipTest struct {
	Name    string
	Source  func() (r io.ReaderAt, size int64) // if non-nil, used instead of testdata/<Name> file
	Comment string
	File    []ZipTestFile
	Error   error // the error that Opening this file should return
}

type ZipTestFile struct {
	Name       string
	Content    []byte // if blank, will attempt to compare against File
	ContentErr error
	File       string // name of file to compare to (relative to testdata/)
	Mtime      string // modified time in format "mm-dd-yy hh:mm:ss"
	Mode       os.FileMode
}

// Caution: The Mtime values found for the test files should correspond to
//          the values listed with unzip -l <zipfile>. However, the values
//          listed by unzip appear to be off by some hours. When creating
//          fresh test files and testing them, this issue is not present.
//          The test files were created in Sydney, so there might be a time
//          zone issue. The time zone information does have to be encoded
//          somewhere, because otherwise unzip -l could not provide a different
//          time from what the archive/zip package provides, but there appears
//          to be no documentation about this.

var tests = []ZipTest{
	{
		Name:    "test.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:    "test.txt",
				Content: []byte("This is a test text file.\n"),
				Mtime:   "09-05-10 12:12:02",
				Mode:    0644,
			},
			{
				Name:  "gophercolor16x16.png",
				File:  "gophercolor16x16.png",
				Mtime: "09-05-10 15:52:58",
				Mode:  0644,
			},
		},
	},
	{
		Name:    "test-trailing-junk.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:    "test.txt",
				Content: []byte("This is a test text file.\n"),
				Mtime:   "09-05-10 12:12:02",
				Mode:    0644,
			},
			{
				Name:  "gophercolor16x16.png",
				File:  "gophercolor16x16.png",
				Mtime: "09-05-10 15:52:58",
				Mode:  0644,
			},
		},
	},
	{
		Name:   "r.zip",
		Source: returnRecursiveZip,
		File: []ZipTestFile{
			{
				Name:    "r/r.zip",
				Content: rZipBytes(),
				Mtime:   "03-04-10 00:24:16",
				Mode:    0666,
			},
		},
	},
	{
		Name: "symlink.zip",
		File: []ZipTestFile{
			{
				Name:    "symlink",
				Content: []byte("../target"),
				Mode:    0777 | os.ModeSymlink,
			},
		},
	},
	{
		Name: "readme.zip",
	},
	{
		Name:  "readme.notzip",
		Error: ErrFormat,
	},
	{
		Name: "dd.zip",
		File: []ZipTestFile{
			{
				Name:    "filename",
				Content: []byte("This is a test textfile.\n"),
				Mtime:   "02-02-11 13:06:20",
				Mode:    0666,
			},
		},
	},
	{
		// created in windows XP file manager.
		Name: "winxp.zip",
		File: crossPlatform,
	},
	{
		// created by Zip 3.0 under Linux
		Name: "unix.zip",
		File: crossPlatform,
	},
	{
		// created by Go, before we wrote the "optional" data
		// descriptor signatures (which are required by OS X)
		Name: "go-no-datadesc-sig.zip",
		File: []ZipTestFile{
			{
				Name:    "foo.txt",
				Content: []byte("foo\n"),
				Mtime:   "03-08-12 16:59:10",
				Mode:    0644,
			},
			{
				Name:    "bar.txt",
				Content: []byte("bar\n"),
				Mtime:   "03-08-12 16:59:12",
				Mode:    0644,
			},
		},
	},
	{
		// created by Go, after we wrote the "optional" data
		// descriptor signatures (which are required by OS X)
		Name: "go-with-datadesc-sig.zip",
		File: []ZipTestFile{
			{
				Name:    "foo.txt",
				Content: []byte("foo\n"),
				Mode:    0666,
			},
			{
				Name:    "bar.txt",
				Content: []byte("bar\n"),
				Mode:    0666,
			},
		},
	},
	{
		Name:   "Bad-CRC32-in-data-descriptor",
		Source: returnCorruptCRC32Zip,
		File: []ZipTestFile{
			{
				Name:       "foo.txt",
				Content:    []byte("foo\n"),
				Mode:       0666,
				ContentErr: ErrChecksum,
			},
			{
				Name:    "bar.txt",
				Content: []byte("bar\n"),
				Mode:    0666,
			},
		},
	},
	// Tests that we verify (and accept valid) crc32s on files
	// with crc32s in their file header (not in data descriptors)
	{
		Name: "crc32-not-streamed.zip",
		File: []ZipTestFile{
			{
				Name:    "foo.txt",
				Content: []byte("foo\n"),
				Mtime:   "03-08-12 16:59:10",
				Mode:    0644,
			},
			{
				Name:    "bar.txt",
				Content: []byte("bar\n"),
				Mtime:   "03-08-12 16:59:12",
				Mode:    0644,
			},
		},
	},
	// Tests that we verify (and reject invalid) crc32s on files
	// with crc32s in their file header (not in data descriptors)
	{
		Name:   "crc32-not-streamed.zip",
		Source: returnCorruptNotStreamedZip,
		File: []ZipTestFile{
			{
				Name:       "foo.txt",
				Content:    []byte("foo\n"),
				Mtime:      "03-08-12 16:59:10",
				Mode:       0644,
				ContentErr: ErrChecksum,
			},
			{
				Name:    "bar.txt",
				Content: []byte("bar\n"),
				Mtime:   "03-08-12 16:59:12",
				Mode:    0644,
			},
		},
	},
	{
		Name: "zip64.zip",
		File: []ZipTestFile{
			{
				Name:    "README",
				Content: []byte("This small file is in ZIP64 format.\n"),
				Mtime:   "08-10-12 14:33:32",
				Mode:    0644,
			},
		},
	},
	// Another zip64 file with different Extras fields. (golang.org/issue/7069)
	{
		Name: "zip64-2.zip",
		File: []ZipTestFile{
			{
				Name:    "README",
				Content: []byte("This small file is in ZIP64 format.\n"),
				Mtime:   "08-10-12 14:33:32",
				Mode:    0644,
			},
		},
	},
}

var crossPlatform = []ZipTestFile{
	{
		Name:    "hello",
		Content: []byte("world \r\n"),
		Mode:    0666,
	},
	{
		Name:    "dir/bar",
		Content: []byte("foo \r\n"),
		Mode:    0666,
	},
	{
		Name:    "dir/empty/",
		Content: []byte{},
		Mode:    os.ModeDir | 0777,
	},
	{
		Name:    "readonly",
		Content: []byte("important \r\n"),
		Mode:    0444,
	},
}

func TestReader(t *testing.T) {
	for _, zt := range tests {
		readTestZip(t, zt)
	}
}

func readTestZip(t *testing.T, zt ZipTest) {
	var z *Reader
	var err error
	if zt.Source != nil {
		rat, size := zt.Source()
		z, err = NewReader(rat, size)
	} else {
		var rc *ReadCloser
		rc, err = OpenReader(filepath.Join("testdata", zt.Name))
		if err == nil {
			defer rc.Close()
			z = &rc.Reader
		}
	}
	if err != zt.Error {
		t.Errorf("%s: error=%v, want %v", zt.Name, err, zt.Error)
		return
	}

	// bail if file is not zip
	if err == ErrFormat {
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
		t.Fatalf("%s: file count=%d, want %d", zt.Name, len(z.File), len(zt.File))
	}

	// test read of each file
	for i, ft := range zt.File {
		readTestFile(t, zt, ft, z.File[i])
	}

	// test simultaneous reads
	n := 0
	done := make(chan bool)
	for i := 0; i < 5; i++ {
		for j, ft := range zt.File {
			go func(j int, ft ZipTestFile) {
				readTestFile(t, zt, ft, z.File[j])
				done <- true
			}(j, ft)
			n++
		}
	}
	for ; n > 0; n-- {
		<-done
	}
}

func readTestFile(t *testing.T, zt ZipTest, ft ZipTestFile, f *File) {
	if f.Name != ft.Name {
		t.Errorf("%s: name=%q, want %q", zt.Name, f.Name, ft.Name)
	}

	if ft.Mtime != "" {
		mtime, err := time.Parse("01-02-06 15:04:05", ft.Mtime)
		if err != nil {
			t.Error(err)
			return
		}
		if ft := f.ModTime(); !ft.Equal(mtime) {
			t.Errorf("%s: %s: mtime=%s, want %s", zt.Name, f.Name, ft, mtime)
		}
	}

	testFileMode(t, zt.Name, f, ft.Mode)

	var b bytes.Buffer
	r, err := f.Open()
	if err != nil {
		t.Errorf("%s: %v", zt.Name, err)
		return
	}

	_, err = io.Copy(&b, r)
	if err != ft.ContentErr {
		t.Errorf("%s: copying contents: %v (want %v)", zt.Name, err, ft.ContentErr)
	}
	if err != nil {
		return
	}
	r.Close()

	size := uint64(f.UncompressedSize)
	if size == uint32max {
		size = f.UncompressedSize64
	}
	if g := uint64(b.Len()); g != size {
		t.Errorf("%v: read %v bytes but f.UncompressedSize == %v", f.Name, g, size)
	}

	var c []byte
	if ft.Content != nil {
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

func testFileMode(t *testing.T, zipName string, f *File, want os.FileMode) {
	mode := f.Mode()
	if want == 0 {
		t.Errorf("%s: %s mode: got %v, want none", zipName, f.Name, mode)
	} else if mode != want {
		t.Errorf("%s: %s mode: want %v, got %v", zipName, f.Name, want, mode)
	}
}

func TestInvalidFiles(t *testing.T) {
	const size = 1024 * 70 // 70kb
	b := make([]byte, size)

	// zeroes
	_, err := NewReader(bytes.NewReader(b), size)
	if err != ErrFormat {
		t.Errorf("zeroes: error=%v, want %v", err, ErrFormat)
	}

	// repeated directoryEndSignatures
	sig := make([]byte, 4)
	binary.LittleEndian.PutUint32(sig, directoryEndSignature)
	for i := 0; i < size-4; i += 4 {
		copy(b[i:i+4], sig)
	}
	_, err = NewReader(bytes.NewReader(b), size)
	if err != ErrFormat {
		t.Errorf("sigs: error=%v, want %v", err, ErrFormat)
	}
}

func messWith(fileName string, corrupter func(b []byte)) (r io.ReaderAt, size int64) {
	data, err := ioutil.ReadFile(filepath.Join("testdata", fileName))
	if err != nil {
		panic("Error reading " + fileName + ": " + err.Error())
	}
	corrupter(data)
	return bytes.NewReader(data), int64(len(data))
}

func returnCorruptCRC32Zip() (r io.ReaderAt, size int64) {
	return messWith("go-with-datadesc-sig.zip", func(b []byte) {
		// Corrupt one of the CRC32s in the data descriptor:
		b[0x2d]++
	})
}

func returnCorruptNotStreamedZip() (r io.ReaderAt, size int64) {
	return messWith("crc32-not-streamed.zip", func(b []byte) {
		// Corrupt foo.txt's final crc32 byte, in both
		// the file header and TOC. (0x7e -> 0x7f)
		b[0x11]++
		b[0x9d]++

		// TODO(bradfitz): add a new test that only corrupts
		// one of these values, and verify that that's also an
		// error. Currently, the reader code doesn't verify the
		// fileheader and TOC's crc32 match if they're both
		// non-zero and only the second line above, the TOC,
		// is what matters.
	})
}

// rZipBytes returns the bytes of a recursive zip file, without
// putting it on disk and triggering certain virus scanners.
func rZipBytes() []byte {
	s := `
0000000 50 4b 03 04 14 00 00 00 08 00 08 03 64 3c f9 f4
0000010 89 64 48 01 00 00 b8 01 00 00 07 00 00 00 72 2f
0000020 72 2e 7a 69 70 00 25 00 da ff 50 4b 03 04 14 00
0000030 00 00 08 00 08 03 64 3c f9 f4 89 64 48 01 00 00
0000040 b8 01 00 00 07 00 00 00 72 2f 72 2e 7a 69 70 00
0000050 2f 00 d0 ff 00 25 00 da ff 50 4b 03 04 14 00 00
0000060 00 08 00 08 03 64 3c f9 f4 89 64 48 01 00 00 b8
0000070 01 00 00 07 00 00 00 72 2f 72 2e 7a 69 70 00 2f
0000080 00 d0 ff c2 54 8e 57 39 00 05 00 fa ff c2 54 8e
0000090 57 39 00 05 00 fa ff 00 05 00 fa ff 00 14 00 eb
00000a0 ff c2 54 8e 57 39 00 05 00 fa ff 00 05 00 fa ff
00000b0 00 14 00 eb ff 42 88 21 c4 00 00 14 00 eb ff 42
00000c0 88 21 c4 00 00 14 00 eb ff 42 88 21 c4 00 00 14
00000d0 00 eb ff 42 88 21 c4 00 00 14 00 eb ff 42 88 21
00000e0 c4 00 00 00 00 ff ff 00 00 00 ff ff 00 34 00 cb
00000f0 ff 42 88 21 c4 00 00 00 00 ff ff 00 00 00 ff ff
0000100 00 34 00 cb ff 42 e8 21 5e 0f 00 00 00 ff ff 0a
0000110 f0 66 64 12 61 c0 15 dc e8 a0 48 bf 48 af 2a b3
0000120 20 c0 9b 95 0d c4 67 04 42 53 06 06 06 40 00 06
0000130 00 f9 ff 6d 01 00 00 00 00 42 e8 21 5e 0f 00 00
0000140 00 ff ff 0a f0 66 64 12 61 c0 15 dc e8 a0 48 bf
0000150 48 af 2a b3 20 c0 9b 95 0d c4 67 04 42 53 06 06
0000160 06 40 00 06 00 f9 ff 6d 01 00 00 00 00 50 4b 01
0000170 02 14 00 14 00 00 00 08 00 08 03 64 3c f9 f4 89
0000180 64 48 01 00 00 b8 01 00 00 07 00 00 00 00 00 00
0000190 00 00 00 00 00 00 00 00 00 00 00 72 2f 72 2e 7a
00001a0 69 70 50 4b 05 06 00 00 00 00 01 00 01 00 35 00
00001b0 00 00 6d 01 00 00 00 00`
	s = regexp.MustCompile(`[0-9a-f]{7}`).ReplaceAllString(s, "")
	s = regexp.MustCompile(`\s+`).ReplaceAllString(s, "")
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func returnRecursiveZip() (r io.ReaderAt, size int64) {
	b := rZipBytes()
	return bytes.NewReader(b), int64(len(b))
}

func TestIssue8186(t *testing.T) {
	// Directory headers & data found in the TOC of a JAR file.
	dirEnts := []string{
		"PK\x01\x02\n\x00\n\x00\x00\b\x00\x004\x9d3?\xaa\x1b\x06\xf0\x81\x02\x00\x00\x81\x02\x00\x00-\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00res/drawable-xhdpi-v4/ic_actionbar_accept.png\xfe\xca\x00\x00\x00",
		"PK\x01\x02\n\x00\n\x00\x00\b\x00\x004\x9d3?\x90K\x89\xc7t\n\x00\x00t\n\x00\x00\x0e\x00\x03\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xd1\x02\x00\x00resources.arsc\x00\x00\x00",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?\xff$\x18\xed3\x03\x00\x00\xb4\b\x00\x00\x13\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00t\r\x00\x00AndroidManifest.xml",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?\x14\xc5K\xab\x192\x02\x00\xc8\xcd\x04\x00\v\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xe8\x10\x00\x00classes.dex",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?E\x96\nD\xac\x01\x00\x00P\x03\x00\x00&\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:C\x02\x00res/layout/actionbar_set_wallpaper.xml",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?Ļ\x14\xe3\xd8\x01\x00\x00\xd8\x03\x00\x00 \x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00:E\x02\x00res/layout/wallpaper_cropper.xml",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?}\xc1\x15\x9eZ\x01\x00\x00!\x02\x00\x00\x14\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00`G\x02\x00META-INF/MANIFEST.MF",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?\xe6\x98Ьo\x01\x00\x00\x84\x02\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xfcH\x02\x00META-INF/CERT.SF",
		"PK\x01\x02\x14\x00\x14\x00\b\b\b\x004\x9d3?\xbfP\x96b\x86\x04\x00\x00\xb2\x06\x00\x00\x11\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xa9J\x02\x00META-INF/CERT.RSA",
	}
	for i, s := range dirEnts {
		var f File
		err := readDirectoryHeader(&f, strings.NewReader(s))
		if err != nil {
			t.Errorf("error reading #%d: %v", i, err)
		}
	}
}

// Verify we return ErrUnexpectedEOF when length is short.
func TestIssue10957(t *testing.T) {
	data := []byte("PK\x03\x040000000PK\x01\x0200000" +
		"0000000000000000000\x00" +
		"\x00\x00\x00\x00\x00000000000000PK\x01" +
		"\x020000000000000000000" +
		"00000\v\x00\x00\x00\x00\x00000000000" +
		"00000000000000PK\x01\x0200" +
		"00000000000000000000" +
		"00\v\x00\x00\x00\x00\x00000000000000" +
		"00000000000PK\x01\x020000<" +
		"0\x00\x0000000000000000\v\x00\v" +
		"\x00\x00\x00\x00\x0000000000\x00\x00\x00\x00000" +
		"00000000PK\x01\x0200000000" +
		"0000000000000000\v\x00\x00\x00" +
		"\x00\x0000PK\x05\x06000000\x05\x000000" +
		"\v\x00\x00\x00\x00\x00")
	z, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	for i, f := range z.File {
		r, err := f.Open()
		if err != nil {
			continue
		}
		if f.UncompressedSize64 < 1e6 {
			n, err := io.Copy(ioutil.Discard, r)
			if i == 3 && err != io.ErrUnexpectedEOF {
				t.Errorf("File[3] error = %v; want io.ErrUnexpectedEOF", err)
			}
			if err == nil && uint64(n) != f.UncompressedSize64 {
				t.Errorf("file %d: bad size: copied=%d; want=%d", i, n, f.UncompressedSize64)
			}
		}
		r.Close()
	}
}

// Verify the number of files is sane.
func TestIssue10956(t *testing.T) {
	data := []byte("PK\x06\x06PK\x06\a0000\x00\x00\x00\x00\x00\x00\x00\x00" +
		"0000PK\x05\x06000000000000" +
		"0000\v\x00000\x00\x00\x00\x00\x00\x00\x000")
	_, err := NewReader(bytes.NewReader(data), int64(len(data)))
	const want = "TOC declares impossible 3472328296227680304 files in 57 byte"
	if err == nil && !strings.Contains(err.Error(), want) {
		t.Errorf("error = %v; want %q", err, want)
	}
}

// Verify we return ErrUnexpectedEOF when reading truncated data descriptor.
func TestIssue11146(t *testing.T) {
	data := []byte("PK\x03\x040000000000000000" +
		"000000\x01\x00\x00\x000\x01\x00\x00\xff\xff0000" +
		"0000000000000000PK\x01\x02" +
		"0000\b0\b\x00000000000000" +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000PK\x05\x06\x00\x00" +
		"\x00\x0000\x01\x0000008\x00\x00\x00\x00\x00")
	z, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	r, err := z.File[0].Open()
	if err != nil {
		t.Fatal(err)
	}
	_, err = ioutil.ReadAll(r)
	if err != io.ErrUnexpectedEOF {
		t.Errorf("File[0] error = %v; want io.ErrUnexpectedEOF", err)
	}
	r.Close()
}

// Verify we do not treat non-zip64 archives as zip64
func TestIssue12449(t *testing.T) {
	data := []byte{
		0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x08, 0x00,
		0x00, 0x00, 0x6b, 0xb4, 0xba, 0x46, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x03, 0x00, 0x18, 0x00, 0xca, 0x64,
		0x55, 0x75, 0x78, 0x0b, 0x00, 0x50, 0x4b, 0x05,
		0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01,
		0x00, 0x49, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
		0x00, 0x31, 0x31, 0x31, 0x32, 0x32, 0x32, 0x0a,
		0x50, 0x4b, 0x07, 0x08, 0x1d, 0x88, 0x77, 0xb0,
		0x07, 0x00, 0x00, 0x00, 0x07, 0x00, 0x00, 0x00,
		0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x14, 0x00,
		0x08, 0x00, 0x00, 0x00, 0x6b, 0xb4, 0xba, 0x46,
		0x1d, 0x88, 0x77, 0xb0, 0x07, 0x00, 0x00, 0x00,
		0x07, 0x00, 0x00, 0x00, 0x03, 0x00, 0x18, 0x00,
		0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0xa0, 0x81, 0x00, 0x00, 0x00, 0x00, 0xca, 0x64,
		0x55, 0x75, 0x78, 0x0b, 0x00, 0x50, 0x4b, 0x05,
		0x06, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01,
		0x00, 0x49, 0x00, 0x00, 0x00, 0x44, 0x00, 0x00,
		0x00, 0x97, 0x2b, 0x49, 0x23, 0x05, 0xc5, 0x0b,
		0xa7, 0xd1, 0x52, 0xa2, 0x9c, 0x50, 0x4b, 0x06,
		0x07, 0xc8, 0x19, 0xc1, 0xaf, 0x94, 0x9c, 0x61,
		0x44, 0xbe, 0x94, 0x19, 0x42, 0x58, 0x12, 0xc6,
		0x5b, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00,
		0x00, 0x01, 0x00, 0x01, 0x00, 0x69, 0x00, 0x00,
		0x00, 0x50, 0x00, 0x00, 0x00, 0x00, 0x00,
	}
	// Read in the archive.
	_, err := NewReader(bytes.NewReader([]byte(data)), int64(len(data)))
	if err != nil {
		t.Errorf("Error reading the archive: %v", err)
	}
}
