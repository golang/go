// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zip

import (
	"bytes"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"internal/obscuretestdata"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"testing"
	"testing/fstest"
	"time"
)

type ZipTest struct {
	Name     string
	Source   func() (r io.ReaderAt, size int64) // if non-nil, used instead of testdata/<Name> file
	Comment  string
	File     []ZipTestFile
	Obscured bool  // needed for Apple notarization (golang.org/issue/34986)
	Error    error // the error that Opening this file should return
}

type ZipTestFile struct {
	Name     string
	Mode     fs.FileMode
	NonUTF8  bool
	ModTime  time.Time
	Modified time.Time

	// Information describing expected zip file content.
	// First, reading the entire content should produce the error ContentErr.
	// Second, if ContentErr==nil, the content should match Content.
	// If content is large, an alternative to setting Content is to set File,
	// which names a file in the testdata/ directory containing the
	// uncompressed expected content.
	// If content is very large, an alternative to setting Content or File
	// is to set Size, which will then be checked against the header-reported size
	// but will bypass the decompressing of the actual data.
	// This last option is used for testing very large (multi-GB) compressed files.
	ContentErr error
	Content    []byte
	File       string
	Size       uint64
}

var tests = []ZipTest{
	{
		Name:    "test.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte("This is a test text file.\n"),
				Modified: time.Date(2010, 9, 5, 12, 12, 1, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "gophercolor16x16.png",
				File:     "gophercolor16x16.png",
				Modified: time.Date(2010, 9, 5, 15, 52, 58, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name:    "test-trailing-junk.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte("This is a test text file.\n"),
				Modified: time.Date(2010, 9, 5, 12, 12, 1, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "gophercolor16x16.png",
				File:     "gophercolor16x16.png",
				Modified: time.Date(2010, 9, 5, 15, 52, 58, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name:    "test-prefix.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte("This is a test text file.\n"),
				Modified: time.Date(2010, 9, 5, 12, 12, 1, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "gophercolor16x16.png",
				File:     "gophercolor16x16.png",
				Modified: time.Date(2010, 9, 5, 15, 52, 58, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name:    "test-baddirsz.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte("This is a test text file.\n"),
				Modified: time.Date(2010, 9, 5, 12, 12, 1, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "gophercolor16x16.png",
				File:     "gophercolor16x16.png",
				Modified: time.Date(2010, 9, 5, 15, 52, 58, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name:    "test-badbase.zip",
		Comment: "This is a zipfile comment.",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte("This is a test text file.\n"),
				Modified: time.Date(2010, 9, 5, 12, 12, 1, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "gophercolor16x16.png",
				File:     "gophercolor16x16.png",
				Modified: time.Date(2010, 9, 5, 15, 52, 58, 0, timeZone(+10*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name:   "r.zip",
		Source: returnRecursiveZip,
		File: []ZipTestFile{
			{
				Name:     "r/r.zip",
				Content:  rZipBytes(),
				Modified: time.Date(2010, 3, 4, 0, 24, 16, 0, time.UTC),
				Mode:     0666,
			},
		},
	},
	{
		Name: "symlink.zip",
		File: []ZipTestFile{
			{
				Name:     "symlink",
				Content:  []byte("../target"),
				Modified: time.Date(2012, 2, 3, 19, 56, 48, 0, timeZone(-2*time.Hour)),
				Mode:     0777 | fs.ModeSymlink,
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
				Name:     "filename",
				Content:  []byte("This is a test textfile.\n"),
				Modified: time.Date(2011, 2, 2, 13, 6, 20, 0, time.UTC),
				Mode:     0666,
			},
		},
	},
	{
		// created in windows XP file manager.
		Name: "winxp.zip",
		File: []ZipTestFile{
			{
				Name:     "hello",
				Content:  []byte("world \r\n"),
				Modified: time.Date(2011, 12, 8, 10, 4, 24, 0, time.UTC),
				Mode:     0666,
			},
			{
				Name:     "dir/bar",
				Content:  []byte("foo \r\n"),
				Modified: time.Date(2011, 12, 8, 10, 4, 50, 0, time.UTC),
				Mode:     0666,
			},
			{
				Name:     "dir/empty/",
				Content:  []byte{},
				Modified: time.Date(2011, 12, 8, 10, 8, 6, 0, time.UTC),
				Mode:     fs.ModeDir | 0777,
			},
			{
				Name:     "readonly",
				Content:  []byte("important \r\n"),
				Modified: time.Date(2011, 12, 8, 10, 6, 8, 0, time.UTC),
				Mode:     0444,
			},
		},
	},
	{
		// created by Zip 3.0 under Linux
		Name: "unix.zip",
		File: []ZipTestFile{
			{
				Name:     "hello",
				Content:  []byte("world \r\n"),
				Modified: time.Date(2011, 12, 8, 10, 4, 24, 0, timeZone(0)),
				Mode:     0666,
			},
			{
				Name:     "dir/bar",
				Content:  []byte("foo \r\n"),
				Modified: time.Date(2011, 12, 8, 10, 4, 50, 0, timeZone(0)),
				Mode:     0666,
			},
			{
				Name:     "dir/empty/",
				Content:  []byte{},
				Modified: time.Date(2011, 12, 8, 10, 8, 6, 0, timeZone(0)),
				Mode:     fs.ModeDir | 0777,
			},
			{
				Name:     "readonly",
				Content:  []byte("important \r\n"),
				Modified: time.Date(2011, 12, 8, 10, 6, 8, 0, timeZone(0)),
				Mode:     0444,
			},
		},
	},
	{
		// created by Go, before we wrote the "optional" data
		// descriptor signatures (which are required by macOS).
		// Use obscured file to avoid Apple’s notarization service
		// rejecting the toolchain due to an inability to unzip this archive.
		// See golang.org/issue/34986
		Name:     "go-no-datadesc-sig.zip.base64",
		Obscured: true,
		File: []ZipTestFile{
			{
				Name:     "foo.txt",
				Content:  []byte("foo\n"),
				Modified: time.Date(2012, 3, 8, 16, 59, 10, 0, timeZone(-8*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "bar.txt",
				Content:  []byte("bar\n"),
				Modified: time.Date(2012, 3, 8, 16, 59, 12, 0, timeZone(-8*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		// created by Go, after we wrote the "optional" data
		// descriptor signatures (which are required by macOS)
		Name: "go-with-datadesc-sig.zip",
		File: []ZipTestFile{
			{
				Name:     "foo.txt",
				Content:  []byte("foo\n"),
				Modified: time.Date(1979, 11, 30, 0, 0, 0, 0, time.UTC),
				Mode:     0666,
			},
			{
				Name:     "bar.txt",
				Content:  []byte("bar\n"),
				Modified: time.Date(1979, 11, 30, 0, 0, 0, 0, time.UTC),
				Mode:     0666,
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
				Modified:   time.Date(1979, 11, 30, 0, 0, 0, 0, time.UTC),
				Mode:       0666,
				ContentErr: ErrChecksum,
			},
			{
				Name:     "bar.txt",
				Content:  []byte("bar\n"),
				Modified: time.Date(1979, 11, 30, 0, 0, 0, 0, time.UTC),
				Mode:     0666,
			},
		},
	},
	// Tests that we verify (and accept valid) crc32s on files
	// with crc32s in their file header (not in data descriptors)
	{
		Name: "crc32-not-streamed.zip",
		File: []ZipTestFile{
			{
				Name:     "foo.txt",
				Content:  []byte("foo\n"),
				Modified: time.Date(2012, 3, 8, 16, 59, 10, 0, timeZone(-8*time.Hour)),
				Mode:     0644,
			},
			{
				Name:     "bar.txt",
				Content:  []byte("bar\n"),
				Modified: time.Date(2012, 3, 8, 16, 59, 12, 0, timeZone(-8*time.Hour)),
				Mode:     0644,
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
				Modified:   time.Date(2012, 3, 8, 16, 59, 10, 0, timeZone(-8*time.Hour)),
				Mode:       0644,
				ContentErr: ErrChecksum,
			},
			{
				Name:     "bar.txt",
				Content:  []byte("bar\n"),
				Modified: time.Date(2012, 3, 8, 16, 59, 12, 0, timeZone(-8*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name: "zip64.zip",
		File: []ZipTestFile{
			{
				Name:     "README",
				Content:  []byte("This small file is in ZIP64 format.\n"),
				Modified: time.Date(2012, 8, 10, 14, 33, 32, 0, time.UTC),
				Mode:     0644,
			},
		},
	},
	// Another zip64 file with different Extras fields. (golang.org/issue/7069)
	{
		Name: "zip64-2.zip",
		File: []ZipTestFile{
			{
				Name:     "README",
				Content:  []byte("This small file is in ZIP64 format.\n"),
				Modified: time.Date(2012, 8, 10, 14, 33, 32, 0, timeZone(-4*time.Hour)),
				Mode:     0644,
			},
		},
	},
	// Largest possible non-zip64 file, with no zip64 header.
	{
		Name:   "big.zip",
		Source: returnBigZipBytes,
		File: []ZipTestFile{
			{
				Name:     "big.file",
				Content:  nil,
				Size:     1<<32 - 1,
				Modified: time.Date(1979, 11, 30, 0, 0, 0, 0, time.UTC),
				Mode:     0666,
			},
		},
	},
	{
		Name: "utf8-7zip.zip",
		File: []ZipTestFile{
			{
				Name:     "世界",
				Content:  []byte{},
				Mode:     0666,
				Modified: time.Date(2017, 11, 6, 13, 9, 27, 867862500, timeZone(-8*time.Hour)),
			},
		},
	},
	{
		Name: "utf8-infozip.zip",
		File: []ZipTestFile{
			{
				Name:    "世界",
				Content: []byte{},
				Mode:    0644,
				// Name is valid UTF-8, but format does not have UTF-8 flag set.
				// We don't do UTF-8 detection for multi-byte runes due to
				// false-positives with other encodings (e.g., Shift-JIS).
				// Format says encoding is not UTF-8, so we trust it.
				NonUTF8:  true,
				Modified: time.Date(2017, 11, 6, 13, 9, 27, 0, timeZone(-8*time.Hour)),
			},
		},
	},
	{
		Name: "utf8-osx.zip",
		File: []ZipTestFile{
			{
				Name:    "世界",
				Content: []byte{},
				Mode:    0644,
				// Name is valid UTF-8, but format does not have UTF-8 set.
				NonUTF8:  true,
				Modified: time.Date(2017, 11, 6, 13, 9, 27, 0, timeZone(-8*time.Hour)),
			},
		},
	},
	{
		Name: "utf8-winrar.zip",
		File: []ZipTestFile{
			{
				Name:     "世界",
				Content:  []byte{},
				Mode:     0666,
				Modified: time.Date(2017, 11, 6, 13, 9, 27, 867862500, timeZone(-8*time.Hour)),
			},
		},
	},
	{
		Name: "utf8-winzip.zip",
		File: []ZipTestFile{
			{
				Name:     "世界",
				Content:  []byte{},
				Mode:     0666,
				Modified: time.Date(2017, 11, 6, 13, 9, 27, 867000000, timeZone(-8*time.Hour)),
			},
		},
	},
	{
		Name: "time-7zip.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 57, 244817900, timeZone(-7*time.Hour)),
				Mode:     0666,
			},
		},
	},
	{
		Name: "time-infozip.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 57, 0, timeZone(-7*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name: "time-osx.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 57, 0, timeZone(-7*time.Hour)),
				Mode:     0644,
			},
		},
	},
	{
		Name: "time-win7.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 58, 0, time.UTC),
				Mode:     0666,
			},
		},
	},
	{
		Name: "time-winrar.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 57, 244817900, timeZone(-7*time.Hour)),
				Mode:     0666,
			},
		},
	},
	{
		Name: "time-winzip.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 57, 244000000, timeZone(-7*time.Hour)),
				Mode:     0666,
			},
		},
	},
	{
		Name: "time-go.zip",
		File: []ZipTestFile{
			{
				Name:     "test.txt",
				Content:  []byte{},
				Size:     1<<32 - 1,
				Modified: time.Date(2017, 10, 31, 21, 11, 57, 0, timeZone(-7*time.Hour)),
				Mode:     0666,
			},
		},
	},
	{
		Name: "time-22738.zip",
		File: []ZipTestFile{
			{
				Name:     "file",
				Content:  []byte{},
				Mode:     0666,
				Modified: time.Date(1999, 12, 31, 19, 0, 0, 0, timeZone(-5*time.Hour)),
				ModTime:  time.Date(1999, 12, 31, 19, 0, 0, 0, time.UTC),
			},
		},
	},
	{
		Name: "dupdir.zip",
		File: []ZipTestFile{
			{
				Name:     "a/",
				Content:  []byte{},
				Mode:     fs.ModeDir | 0666,
				Modified: time.Date(2021, 12, 29, 0, 0, 0, 0, timeZone(0)),
			},
			{
				Name:     "a/b",
				Content:  []byte{},
				Mode:     0666,
				Modified: time.Date(2021, 12, 29, 0, 0, 0, 0, timeZone(0)),
			},
			{
				Name:     "a/b/",
				Content:  []byte{},
				Mode:     fs.ModeDir | 0666,
				Modified: time.Date(2021, 12, 29, 0, 0, 0, 0, timeZone(0)),
			},
			{
				Name:     "a/b/c",
				Content:  []byte{},
				Mode:     0666,
				Modified: time.Date(2021, 12, 29, 0, 0, 0, 0, timeZone(0)),
			},
		},
	},
	// Issue 66869: Don't skip over an EOCDR with a truncated comment.
	// The test file sneakily hides a second EOCDR before the first one;
	// previously we would extract one file ("file") from this archive,
	// while most other tools would reject the file or extract a different one ("FILE").
	{
		Name:  "comment-truncated.zip",
		Error: ErrFormat,
	},
}

func TestReader(t *testing.T) {
	for _, zt := range tests {
		t.Run(zt.Name, func(t *testing.T) {
			readTestZip(t, zt)
		})
	}
}

func readTestZip(t *testing.T, zt ZipTest) {
	var z *Reader
	var err error
	var raw []byte
	if zt.Source != nil {
		rat, size := zt.Source()
		z, err = NewReader(rat, size)
		raw = make([]byte, size)
		if _, err := rat.ReadAt(raw, 0); err != nil {
			t.Errorf("ReadAt error=%v", err)
			return
		}
	} else {
		path := filepath.Join("testdata", zt.Name)
		if zt.Obscured {
			tf, err := obscuretestdata.DecodeToTempFile(path)
			if err != nil {
				t.Errorf("obscuretestdata.DecodeToTempFile(%s): %v", path, err)
				return
			}
			defer os.Remove(tf)
			path = tf
		}
		var rc *ReadCloser
		rc, err = OpenReader(path)
		if err == nil {
			defer rc.Close()
			z = &rc.Reader
		}
		var err2 error
		raw, err2 = os.ReadFile(path)
		if err2 != nil {
			t.Errorf("ReadFile(%s) error=%v", path, err2)
			return
		}
	}
	if err != zt.Error {
		t.Errorf("error=%v, want %v", err, zt.Error)
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
		t.Errorf("comment=%q, want %q", z.Comment, zt.Comment)
	}
	if len(z.File) != len(zt.File) {
		t.Fatalf("file count=%d, want %d", len(z.File), len(zt.File))
	}

	// test read of each file
	for i, ft := range zt.File {
		readTestFile(t, zt, ft, z.File[i], raw)
	}
	if t.Failed() {
		return
	}

	// test simultaneous reads
	n := 0
	done := make(chan bool)
	for i := 0; i < 5; i++ {
		for j, ft := range zt.File {
			go func(j int, ft ZipTestFile) {
				readTestFile(t, zt, ft, z.File[j], raw)
				done <- true
			}(j, ft)
			n++
		}
	}
	for ; n > 0; n-- {
		<-done
	}
}

func equalTimeAndZone(t1, t2 time.Time) bool {
	name1, offset1 := t1.Zone()
	name2, offset2 := t2.Zone()
	return t1.Equal(t2) && name1 == name2 && offset1 == offset2
}

func readTestFile(t *testing.T, zt ZipTest, ft ZipTestFile, f *File, raw []byte) {
	if f.Name != ft.Name {
		t.Errorf("name=%q, want %q", f.Name, ft.Name)
	}
	if !ft.Modified.IsZero() && !equalTimeAndZone(f.Modified, ft.Modified) {
		t.Errorf("%s: Modified=%s, want %s", f.Name, f.Modified, ft.Modified)
	}
	if !ft.ModTime.IsZero() && !equalTimeAndZone(f.ModTime(), ft.ModTime) {
		t.Errorf("%s: ModTime=%s, want %s", f.Name, f.ModTime(), ft.ModTime)
	}

	testFileMode(t, f, ft.Mode)

	size := uint64(f.UncompressedSize)
	if size == uint32max {
		size = f.UncompressedSize64
	} else if size != f.UncompressedSize64 {
		t.Errorf("%v: UncompressedSize=%#x does not match UncompressedSize64=%#x", f.Name, size, f.UncompressedSize64)
	}

	// Check that OpenRaw returns the correct byte segment
	rw, err := f.OpenRaw()
	if err != nil {
		t.Errorf("%v: OpenRaw error=%v", f.Name, err)
		return
	}
	start, err := f.DataOffset()
	if err != nil {
		t.Errorf("%v: DataOffset error=%v", f.Name, err)
		return
	}
	got, err := io.ReadAll(rw)
	if err != nil {
		t.Errorf("%v: OpenRaw ReadAll error=%v", f.Name, err)
		return
	}
	end := uint64(start) + f.CompressedSize64
	want := raw[start:end]
	if !bytes.Equal(got, want) {
		t.Logf("got %q", got)
		t.Logf("want %q", want)
		t.Errorf("%v: OpenRaw returned unexpected bytes", f.Name)
		return
	}

	r, err := f.Open()
	if err != nil {
		t.Errorf("%v", err)
		return
	}

	// For very large files, just check that the size is correct.
	// The content is expected to be all zeros.
	// Don't bother uncompressing: too big.
	if ft.Content == nil && ft.File == "" && ft.Size > 0 {
		if size != ft.Size {
			t.Errorf("%v: uncompressed size %#x, want %#x", ft.Name, size, ft.Size)
		}
		r.Close()
		return
	}

	var b bytes.Buffer
	_, err = io.Copy(&b, r)
	if err != ft.ContentErr {
		t.Errorf("copying contents: %v (want %v)", err, ft.ContentErr)
	}
	if err != nil {
		return
	}
	r.Close()

	if g := uint64(b.Len()); g != size {
		t.Errorf("%v: read %v bytes but f.UncompressedSize == %v", f.Name, g, size)
	}

	var c []byte
	if ft.Content != nil {
		c = ft.Content
	} else if c, err = os.ReadFile("testdata/" + ft.File); err != nil {
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

func testFileMode(t *testing.T, f *File, want fs.FileMode) {
	mode := f.Mode()
	if want == 0 {
		t.Errorf("%s mode: got %v, want none", f.Name, mode)
	} else if mode != want {
		t.Errorf("%s mode: want %v, got %v", f.Name, want, mode)
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

	// negative size
	_, err = NewReader(bytes.NewReader([]byte("foobar")), -1)
	if err == nil {
		t.Errorf("archive/zip.NewReader: expected error when negative size is passed")
	}
}

func messWith(fileName string, corrupter func(b []byte)) (r io.ReaderAt, size int64) {
	data, err := os.ReadFile(filepath.Join("testdata", fileName))
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

// biggestZipBytes returns the bytes of a zip file biggest.zip
// that contains a zip file bigger.zip that contains a zip file
// big.zip that contains big.file, which contains 2³²-1 zeros.
// The big.zip file is interesting because it has no zip64 header,
// much like the innermost zip files in the well-known 42.zip.
//
// biggest.zip was generated by changing isZip64 to use > uint32max
// instead of >= uint32max and then running this program:
//
//	package main
//
//	import (
//		"archive/zip"
//		"bytes"
//		"io"
//		"log"
//		"os"
//	)
//
//	type zeros struct{}
//
//	func (zeros) Read(b []byte) (int, error) {
//		clear(b)
//		return len(b), nil
//	}
//
//	func main() {
//		bigZip := makeZip("big.file", io.LimitReader(zeros{}, 1<<32-1))
//		if err := os.WriteFile("/tmp/big.zip", bigZip, 0666); err != nil {
//			log.Fatal(err)
//		}
//
//		biggerZip := makeZip("big.zip", bytes.NewReader(bigZip))
//		if err := os.WriteFile("/tmp/bigger.zip", biggerZip, 0666); err != nil {
//			log.Fatal(err)
//		}
//
//		biggestZip := makeZip("bigger.zip", bytes.NewReader(biggerZip))
//		if err := os.WriteFile("/tmp/biggest.zip", biggestZip, 0666); err != nil {
//			log.Fatal(err)
//		}
//	}
//
//	func makeZip(name string, r io.Reader) []byte {
//		var buf bytes.Buffer
//		w := zip.NewWriter(&buf)
//		wf, err := w.Create(name)
//		if err != nil {
//			log.Fatal(err)
//		}
//		if _, err = io.Copy(wf, r); err != nil {
//			log.Fatal(err)
//		}
//		if err := w.Close(); err != nil {
//			log.Fatal(err)
//		}
//		return buf.Bytes()
//	}
//
// The 4 GB of zeros compresses to 4 MB, which compresses to 20 kB,
// which compresses to 1252 bytes (in the hex dump below).
//
// It's here in hex for the same reason as rZipBytes above: to avoid
// problems with on-disk virus scanners or other zip processors.
func biggestZipBytes() []byte {
	s := `
0000000 50 4b 03 04 14 00 08 00 08 00 00 00 00 00 00 00
0000010 00 00 00 00 00 00 00 00 00 00 0a 00 00 00 62 69
0000020 67 67 65 72 2e 7a 69 70 ec dc 6b 4c 53 67 18 07
0000030 f0 16 c5 ca 65 2e cb b8 94 20 61 1f 44 33 c7 cd
0000040 c0 86 4a b5 c0 62 8a 61 05 c6 cd 91 b2 54 8c 1b
0000050 63 8b 03 9c 1b 95 52 5a e3 a0 19 6c b2 05 59 44
0000060 64 9d 73 83 71 11 46 61 14 b9 1d 14 09 4a c3 60
0000070 2e 4c 6e a5 60 45 02 62 81 95 b6 94 9e 9e 77 e7
0000080 d0 43 b6 f8 71 df 96 3c e7 a4 69 ce bf cf e9 79
0000090 ce ef 79 3f bf f1 31 db b6 bb 31 76 92 e7 f3 07
00000a0 8b fc 9c ca cc 08 cc cb cc 5e d2 1c 88 d9 7e bb
00000b0 4f bb 3a 3f 75 f1 5d 7f 8f c2 68 67 77 8f 25 ff
00000c0 84 e2 93 2d ef a4 95 3d 71 4e 2c b9 b0 87 c3 be
00000d0 3d f8 a7 60 24 61 c5 ef ae 9e c8 6c 6d 4e 69 c8
00000e0 67 65 34 f8 37 76 2d 76 5c 54 f3 95 65 49 c7 0f
00000f0 18 71 4b 7e 5b 6a d1 79 47 61 41 b0 4e 2a 74 45
0000100 43 58 12 b2 5a a5 c6 7d 68 55 88 d4 98 75 18 6d
0000110 08 d1 1f 8f 5a 9e 96 ee 45 cf a4 84 4e 4b e8 50
0000120 a7 13 d9 06 de 52 81 97 36 b2 d7 b8 fc 2b 5f 55
0000130 23 1f 32 59 cf 30 27 fb e2 8a b9 de 45 dd 63 9c
0000140 4b b5 8b 96 4c 7a 62 62 cc a1 a7 cf fa f1 fe dd
0000150 54 62 11 bf 36 78 b3 c7 b1 b5 f2 61 4d 4e dd 66
0000160 32 2e e6 70 34 5f f4 c9 e6 6c 43 6f da 6b c6 c3
0000170 09 2c ce 09 57 7f d2 7e b4 23 ba 7c 1b 99 bc 22
0000180 3e f1 de 91 2f e3 9c 1b 82 cc c2 84 39 aa e6 de
0000190 b4 69 fc cc cb 72 a6 61 45 f0 d3 1d 26 19 7c 8d
00001a0 29 c8 66 02 be 77 6a f9 3d 34 79 17 19 c8 96 24
00001b0 a3 ac e4 dd 3b 1a 8e c6 fe 96 38 6b bf 67 5a 23
00001c0 f4 16 f4 e6 8a b4 fc c2 cd bf 95 66 1d bb 35 aa
00001d0 92 7d 66 d8 08 8d a5 1f 54 2a af 09 cf 61 ff d2
00001e0 85 9d 8f b6 d7 88 07 4a 86 03 db 64 f3 d9 92 73
00001f0 df ec a7 fc 23 4c 8d 83 79 63 2a d9 fd 8d b3 c8
0000200 8f 7e d4 19 85 e6 8d 1c 76 f0 8b 58 32 fd 9a d6
0000210 85 e2 48 ad c3 d5 60 6f 7e 22 dd ef 09 49 7c 7f
0000220 3a 45 c3 71 b7 df f3 4c 63 fb b5 d9 31 5f 6e d6
0000230 24 1d a4 4a fe 32 a7 5c 16 48 5c 3e 08 6b 8a d3
0000240 25 1d a2 12 a5 59 24 ea 20 5f 52 6d ad 94 db 6b
0000250 94 b9 5d eb 4b a7 5c 44 bb 1e f2 3c 6b cf 52 c9
0000260 e9 e5 ba 06 b9 c4 e5 0a d0 00 0d d0 00 0d d0 00
0000270 0d d0 00 0d d0 00 0d d0 00 0d d0 00 0d d0 00 0d
0000280 d0 00 0d d0 00 0d d0 00 0d d0 00 0d d0 00 0d d0
0000290 00 0d d0 00 0d d0 00 0d d0 00 0d d0 00 0d d0 00
00002a0 0d d0 00 cd ff 9e 46 86 fa a7 7d 3a 43 d7 8e 10
00002b0 52 e9 be e6 6e cf eb 9e 85 4d 65 ce cc 30 c1 44
00002c0 c0 4e af bc 9c 6c 4b a0 d7 54 ff 1d d5 5c 89 fb
00002d0 b5 34 7e c4 c2 9e f5 a0 f6 5b 7e 6e ca 73 c7 ef
00002e0 5d be de f9 e8 81 eb a5 0a a5 63 54 2c d7 1c d1
00002f0 89 17 85 f8 16 94 f2 8a b2 a3 f5 b6 6d df 75 cd
0000300 90 dd 64 bd 5d 55 4e f2 55 19 1b b7 cc ef 1b ea
0000310 2e 05 9c f4 aa 1e a8 cd a6 82 c7 59 0f 5e 9d e0
0000320 bb fc 6c d6 99 23 eb 36 ad c6 c5 e1 d8 e1 e2 3e
0000330 d9 90 5a f7 91 5d 6f bc 33 6d 98 47 d2 7c 2e 2f
0000340 99 a4 25 72 85 49 2c be 0b 5b af 8f e5 6e 81 a6
0000350 a3 5a 6f 39 53 3a ab 7a 8b 1e 26 f7 46 6c 7d 26
0000360 53 b3 22 31 94 d3 83 f2 18 4d f5 92 33 27 53 97
0000370 0f d3 e6 55 9c a6 c5 31 87 6f d3 f3 ae 39 6f 56
0000380 10 7b ab 7e d0 b4 ca f2 b8 05 be 3f 0e 6e 5a 75
0000390 ab 0c f5 37 0e ba 8e 75 71 7a aa ed 7a dd 6a 63
00003a0 be 9b a0 97 27 6a 6f e7 d3 8b c4 7c ec d3 91 56
00003b0 d9 ac 5e bf 16 42 2f 00 1f 93 a2 23 87 bd e2 59
00003c0 a0 de 1a 66 c8 62 eb 55 8f 91 17 b4 61 42 7a 50
00003d0 40 03 34 40 03 34 40 03 34 40 03 34 40 03 34 40
00003e0 03 34 40 03 34 40 03 34 40 03 34 40 03 34 40 03
00003f0 34 40 03 34 40 03 34 ff 85 86 90 8b ea 67 90 0d
0000400 e1 42 1b d2 61 d6 79 ec fd 3e 44 28 a4 51 6c 5c
0000410 fc d2 72 ca ba 82 18 46 16 61 cd 93 a9 0f d1 24
0000420 17 99 e2 2c 71 16 84 0c c8 7a 13 0f 9a 5e c5 f0
0000430 79 64 e2 12 4d c8 82 a1 81 19 2d aa 44 6d 87 54
0000440 84 71 c1 f6 d4 ca 25 8c 77 b9 08 c7 c8 5e 10 8a
0000450 8f 61 ed 8c ba 30 1f 79 9a c7 60 34 2b b9 8c f8
0000460 18 a6 83 1b e3 9f ad 79 fe fd 1b 8b f1 fc 41 6f
0000470 d4 13 1f e3 b8 83 ba 64 92 e7 eb e4 77 05 8f ba
0000480 fa 3b 00 00 ff ff 50 4b 07 08 a6 18 b1 91 5e 04
0000490 00 00 e4 47 00 00 50 4b 01 02 14 00 14 00 08 00
00004a0 08 00 00 00 00 00 a6 18 b1 91 5e 04 00 00 e4 47
00004b0 00 00 0a 00 00 00 00 00 00 00 00 00 00 00 00 00
00004c0 00 00 00 00 62 69 67 67 65 72 2e 7a 69 70 50 4b
00004d0 05 06 00 00 00 00 01 00 01 00 38 00 00 00 96 04
00004e0 00 00 00 00`
	s = regexp.MustCompile(`[0-9a-f]{7}`).ReplaceAllString(s, "")
	s = regexp.MustCompile(`\s+`).ReplaceAllString(s, "")
	b, err := hex.DecodeString(s)
	if err != nil {
		panic(err)
	}
	return b
}

func returnBigZipBytes() (r io.ReaderAt, size int64) {
	b := biggestZipBytes()
	for i := 0; i < 2; i++ {
		r, err := NewReader(bytes.NewReader(b), int64(len(b)))
		if err != nil {
			panic(err)
		}
		f, err := r.File[0].Open()
		if err != nil {
			panic(err)
		}
		b, err = io.ReadAll(f)
		if err != nil {
			panic(err)
		}
	}
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
		"\x00\x0000PK\x05\x06000000\x05\x00\xfd\x00\x00\x00" +
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
			n, err := io.Copy(io.Discard, r)
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

// Verify that this particular malformed zip file is rejected.
func TestIssue10956(t *testing.T) {
	data := []byte("PK\x06\x06PK\x06\a0000\x00\x00\x00\x00\x00\x00\x00\x00" +
		"0000PK\x05\x06000000000000" +
		"0000\v\x00000\x00\x00\x00\x00\x00\x00\x000")
	r, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err == nil {
		t.Errorf("got nil error, want ErrFormat")
	}
	if r != nil {
		t.Errorf("got non-nil Reader, want nil")
	}
}

// Verify we return ErrUnexpectedEOF when reading truncated data descriptor.
func TestIssue11146(t *testing.T) {
	data := []byte("PK\x03\x040000000000000000" +
		"000000\x01\x00\x00\x000\x01\x00\x00\xff\xff0000" +
		"0000000000000000PK\x01\x02" +
		"0000\b0\b\x00000000000000" +
		"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x000000PK\x05\x06\x00\x00" +
		"\x00\x0000\x01\x00\x26\x00\x00\x008\x00\x00\x00\x00\x00")
	z, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatal(err)
	}
	r, err := z.File[0].Open()
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.ReadAll(r)
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
	_, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Errorf("Error reading the archive: %v", err)
	}
}

func TestFS(t *testing.T) {
	for _, test := range []struct {
		file string
		want []string
	}{
		{
			"testdata/unix.zip",
			[]string{"hello", "dir/bar", "readonly"},
		},
		{
			"testdata/subdir.zip",
			[]string{"a/b/c"},
		},
	} {
		t.Run(test.file, func(t *testing.T) {
			t.Parallel()
			z, err := OpenReader(test.file)
			if err != nil {
				t.Fatal(err)
			}
			defer z.Close()
			if err := fstest.TestFS(z, test.want...); err != nil {
				t.Error(err)
			}
		})
	}
}

func TestFSWalk(t *testing.T) {
	for _, test := range []struct {
		file    string
		want    []string
		wantErr bool
	}{
		{
			file: "testdata/unix.zip",
			want: []string{".", "dir", "dir/bar", "dir/empty", "hello", "readonly"},
		},
		{
			file: "testdata/subdir.zip",
			want: []string{".", "a", "a/b", "a/b/c"},
		},
		{
			file:    "testdata/dupdir.zip",
			wantErr: true,
		},
	} {
		t.Run(test.file, func(t *testing.T) {
			t.Parallel()
			z, err := OpenReader(test.file)
			if err != nil {
				t.Fatal(err)
			}
			var files []string
			sawErr := false
			err = fs.WalkDir(z, ".", func(path string, d fs.DirEntry, err error) error {
				if err != nil {
					if !test.wantErr {
						t.Errorf("%s: %v", path, err)
					}
					sawErr = true
					return nil
				}
				files = append(files, path)
				return nil
			})
			if err != nil {
				t.Errorf("fs.WalkDir error: %v", err)
			}
			if test.wantErr && !sawErr {
				t.Error("succeeded but want error")
			} else if !test.wantErr && sawErr {
				t.Error("unexpected error")
			}
			if test.want != nil && !slices.Equal(files, test.want) {
				t.Errorf("got %v want %v", files, test.want)
			}
		})
	}
}

func TestFSWalkBadFile(t *testing.T) {
	t.Parallel()

	var buf bytes.Buffer
	zw := NewWriter(&buf)
	hdr := &FileHeader{Name: "."}
	hdr.SetMode(fs.ModeDir | 0o755)
	w, err := zw.CreateHeader(hdr)
	if err != nil {
		t.Fatalf("create zip header: %v", err)
	}
	_, err = w.Write([]byte("some data"))
	if err != nil {
		t.Fatalf("write zip contents: %v", err)

	}
	err = zw.Close()
	if err != nil {
		t.Fatalf("close zip writer: %v", err)

	}

	zr, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("create zip reader: %v", err)

	}
	var count int
	var errRepeat = errors.New("repeated call to path")
	err = fs.WalkDir(zr, ".", func(p string, d fs.DirEntry, err error) error {
		count++
		if count > 2 { // once for directory read, once for the error
			return errRepeat
		}
		return err
	})
	if err == nil {
		t.Fatalf("expected error from invalid file name")
	} else if errors.Is(err, errRepeat) {
		t.Fatal(err)
	}
}

func TestFSModTime(t *testing.T) {
	t.Parallel()
	z, err := OpenReader("testdata/subdir.zip")
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()

	for _, test := range []struct {
		name string
		want time.Time
	}{
		{
			"a",
			time.Date(2021, 4, 19, 12, 29, 56, 0, timeZone(-7*time.Hour)).UTC(),
		},
		{
			"a/b/c",
			time.Date(2021, 4, 19, 12, 29, 59, 0, timeZone(-7*time.Hour)).UTC(),
		},
	} {
		fi, err := fs.Stat(z, test.name)
		if err != nil {
			t.Errorf("%s: %v", test.name, err)
			continue
		}
		if got := fi.ModTime(); !got.Equal(test.want) {
			t.Errorf("%s: got modtime %v, want %v", test.name, got, test.want)
		}
	}
}

func TestCVE202127919(t *testing.T) {
	t.Setenv("GODEBUG", "zipinsecurepath=0")
	// Archive containing only the file "../test.txt"
	data := []byte{
		0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x08, 0x00,
		0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x2e, 0x2e,
		0x2f, 0x74, 0x65, 0x73, 0x74, 0x2e, 0x74, 0x78,
		0x74, 0x0a, 0xc9, 0xc8, 0x2c, 0x56, 0xc8, 0x2c,
		0x56, 0x48, 0x54, 0x28, 0x49, 0x2d, 0x2e, 0x51,
		0x28, 0x49, 0xad, 0x28, 0x51, 0x48, 0xcb, 0xcc,
		0x49, 0xd5, 0xe3, 0x02, 0x04, 0x00, 0x00, 0xff,
		0xff, 0x50, 0x4b, 0x07, 0x08, 0xc0, 0xd7, 0xed,
		0xc3, 0x20, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00,
		0x00, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x00, 0x14,
		0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xd7, 0xed, 0xc3, 0x20, 0x00, 0x00,
		0x00, 0x1a, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e,
		0x2e, 0x2f, 0x74, 0x65, 0x73, 0x74, 0x2e, 0x74,
		0x78, 0x74, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00,
		0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x39, 0x00,
		0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00,
	}
	r, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != ErrInsecurePath {
		t.Fatalf("Error reading the archive: %v", err)
	}
	_, err = r.Open("test.txt")
	if err != nil {
		t.Errorf("Error reading file: %v", err)
	}
	if len(r.File) != 1 {
		t.Fatalf("No entries in the file list")
	}
	if r.File[0].Name != "../test.txt" {
		t.Errorf("Unexpected entry name: %s", r.File[0].Name)
	}
	if _, err := r.File[0].Open(); err != nil {
		t.Errorf("Error opening file: %v", err)
	}
}

func TestOpenReaderInsecurePath(t *testing.T) {
	t.Setenv("GODEBUG", "zipinsecurepath=0")
	// Archive containing only the file "../test.txt"
	data := []byte{
		0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x08, 0x00,
		0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x2e, 0x2e,
		0x2f, 0x74, 0x65, 0x73, 0x74, 0x2e, 0x74, 0x78,
		0x74, 0x0a, 0xc9, 0xc8, 0x2c, 0x56, 0xc8, 0x2c,
		0x56, 0x48, 0x54, 0x28, 0x49, 0x2d, 0x2e, 0x51,
		0x28, 0x49, 0xad, 0x28, 0x51, 0x48, 0xcb, 0xcc,
		0x49, 0xd5, 0xe3, 0x02, 0x04, 0x00, 0x00, 0xff,
		0xff, 0x50, 0x4b, 0x07, 0x08, 0xc0, 0xd7, 0xed,
		0xc3, 0x20, 0x00, 0x00, 0x00, 0x1a, 0x00, 0x00,
		0x00, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x00, 0x14,
		0x00, 0x08, 0x00, 0x08, 0x00, 0x00, 0x00, 0x00,
		0x00, 0xc0, 0xd7, 0xed, 0xc3, 0x20, 0x00, 0x00,
		0x00, 0x1a, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2e,
		0x2e, 0x2f, 0x74, 0x65, 0x73, 0x74, 0x2e, 0x74,
		0x78, 0x74, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00,
		0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x39, 0x00,
		0x00, 0x00, 0x59, 0x00, 0x00, 0x00, 0x00, 0x00,
	}

	// Read in the archive with the OpenReader interface
	name := filepath.Join(t.TempDir(), "test.zip")
	err := os.WriteFile(name, data, 0644)
	if err != nil {
		t.Fatalf("Unable to write out the bugos zip entry")
	}
	r, err := OpenReader(name)
	if r != nil {
		defer r.Close()
	}

	if err != ErrInsecurePath {
		t.Fatalf("Error reading the archive, we expected ErrInsecurePath but got: %v", err)
	}
	_, err = r.Open("test.txt")
	if err != nil {
		t.Errorf("Error reading file: %v", err)
	}
	if len(r.File) != 1 {
		t.Fatalf("No entries in the file list")
	}
	if r.File[0].Name != "../test.txt" {
		t.Errorf("Unexpected entry name: %s", r.File[0].Name)
	}
	if _, err := r.File[0].Open(); err != nil {
		t.Errorf("Error opening file: %v", err)
	}
}

func TestCVE202133196(t *testing.T) {
	// Archive that indicates it has 1 << 128 -1 files,
	// this would previously cause a panic due to attempting
	// to allocate a slice with 1 << 128 -1 elements.
	data := []byte{
		0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x08, 0x08,
		0x08, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x02,
		0x03, 0x62, 0x61, 0x65, 0x03, 0x04, 0x00, 0x00,
		0xff, 0xff, 0x50, 0x4b, 0x07, 0x08, 0xbe, 0x20,
		0x5c, 0x6c, 0x09, 0x00, 0x00, 0x00, 0x03, 0x00,
		0x00, 0x00, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x00,
		0x14, 0x00, 0x08, 0x08, 0x08, 0x00, 0x00, 0x00,
		0x00, 0x00, 0xbe, 0x20, 0x5c, 0x6c, 0x09, 0x00,
		0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x01, 0x02, 0x03, 0x50, 0x4b, 0x06, 0x06, 0x2c,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x2d,
		0x00, 0x2d, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0x31, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x3a, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x50, 0x4b, 0x06, 0x07, 0x00,
		0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x50,
		0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0x00, 0x00,
	}
	_, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != ErrFormat {
		t.Fatalf("unexpected error, got: %v, want: %v", err, ErrFormat)
	}

	// Also check that an archive containing a handful of empty
	// files doesn't cause an issue
	b := bytes.NewBuffer(nil)
	w := NewWriter(b)
	for i := 0; i < 5; i++ {
		_, err := w.Create("")
		if err != nil {
			t.Fatalf("Writer.Create failed: %s", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Writer.Close failed: %s", err)
	}
	r, err := NewReader(bytes.NewReader(b.Bytes()), int64(b.Len()))
	if err != nil {
		t.Fatalf("NewReader failed: %s", err)
	}
	if len(r.File) != 5 {
		t.Errorf("Archive has unexpected number of files, got %d, want 5", len(r.File))
	}
}

func TestCVE202139293(t *testing.T) {
	// directory size is so large, that the check in Reader.init
	// overflows when subtracting from the archive size, causing
	// the pre-allocation check to be bypassed.
	data := []byte{
		0x50, 0x4b, 0x06, 0x06, 0x05, 0x06, 0x31, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x4b,
		0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x1a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x4b,
		0x06, 0x07, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x00, 0x00, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x31, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff,
		0xff, 0x50, 0xfe, 0x00, 0xff, 0x00, 0x3a, 0x00, 0x00, 0x00, 0xff,
	}
	_, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != ErrFormat {
		t.Fatalf("unexpected error, got: %v, want: %v", err, ErrFormat)
	}
}

func TestCVE202141772(t *testing.T) {
	t.Setenv("GODEBUG", "zipinsecurepath=0")
	// Archive contains a file whose name is exclusively made up of '/', '\'
	// characters, or "../", "..\" paths, which would previously cause a panic.
	//
	//  Length   Method    Size  Cmpr    Date    Time   CRC-32   Name
	// --------  ------  ------- ---- ---------- ----- --------  ----
	//        0  Stored        0   0% 08-05-2021 18:32 00000000  /
	//        0  Stored        0   0% 09-14-2021 12:59 00000000  //
	//        0  Stored        0   0% 09-14-2021 12:59 00000000  \
	//       11  Stored       11   0% 09-14-2021 13:04 0d4a1185  /test.txt
	// --------          -------  ---                            -------
	//       11               11   0%                            4 files
	data := []byte{
		0x50, 0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x08,
		0x00, 0x00, 0x06, 0x94, 0x05, 0x53, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x2f, 0x50,
		0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0x67, 0x2e, 0x53, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x02, 0x00, 0x00, 0x00, 0x2f, 0x2f, 0x50,
		0x4b, 0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x78, 0x67, 0x2e, 0x53, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x01, 0x00, 0x00, 0x00, 0x5c, 0x50, 0x4b,
		0x03, 0x04, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x91, 0x68, 0x2e, 0x53, 0x85, 0x11, 0x4a, 0x0d,
		0x0b, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00,
		0x09, 0x00, 0x00, 0x00, 0x2f, 0x74, 0x65, 0x73,
		0x74, 0x2e, 0x74, 0x78, 0x74, 0x68, 0x65, 0x6c,
		0x6c, 0x6f, 0x20, 0x77, 0x6f, 0x72, 0x6c, 0x64,
		0x50, 0x4b, 0x01, 0x02, 0x14, 0x03, 0x0a, 0x00,
		0x00, 0x08, 0x00, 0x00, 0x06, 0x94, 0x05, 0x53,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00,
		0xed, 0x41, 0x00, 0x00, 0x00, 0x00, 0x2f, 0x50,
		0x4b, 0x01, 0x02, 0x3f, 0x00, 0x0a, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x78, 0x67, 0x2e, 0x53, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x02, 0x00, 0x24, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00,
		0x00, 0x1f, 0x00, 0x00, 0x00, 0x2f, 0x2f, 0x0a,
		0x00, 0x20, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x18, 0x00, 0x93, 0x98, 0x25, 0x57, 0x25,
		0xa9, 0xd7, 0x01, 0x93, 0x98, 0x25, 0x57, 0x25,
		0xa9, 0xd7, 0x01, 0x93, 0x98, 0x25, 0x57, 0x25,
		0xa9, 0xd7, 0x01, 0x50, 0x4b, 0x01, 0x02, 0x3f,
		0x00, 0x0a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x78,
		0x67, 0x2e, 0x53, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01,
		0x00, 0x24, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x20, 0x00, 0x00, 0x00, 0x3f, 0x00, 0x00,
		0x00, 0x5c, 0x0a, 0x00, 0x20, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x01, 0x00, 0x18, 0x00, 0x93, 0x98,
		0x25, 0x57, 0x25, 0xa9, 0xd7, 0x01, 0x93, 0x98,
		0x25, 0x57, 0x25, 0xa9, 0xd7, 0x01, 0x93, 0x98,
		0x25, 0x57, 0x25, 0xa9, 0xd7, 0x01, 0x50, 0x4b,
		0x01, 0x02, 0x3f, 0x00, 0x0a, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x91, 0x68, 0x2e, 0x53, 0x85, 0x11,
		0x4a, 0x0d, 0x0b, 0x00, 0x00, 0x00, 0x0b, 0x00,
		0x00, 0x00, 0x09, 0x00, 0x24, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x20, 0x00, 0x00, 0x00,
		0x5e, 0x00, 0x00, 0x00, 0x2f, 0x74, 0x65, 0x73,
		0x74, 0x2e, 0x74, 0x78, 0x74, 0x0a, 0x00, 0x20,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x18,
		0x00, 0xa9, 0x80, 0x51, 0x01, 0x26, 0xa9, 0xd7,
		0x01, 0x31, 0xd1, 0x57, 0x01, 0x26, 0xa9, 0xd7,
		0x01, 0xdf, 0x48, 0x85, 0xf9, 0x25, 0xa9, 0xd7,
		0x01, 0x50, 0x4b, 0x05, 0x06, 0x00, 0x00, 0x00,
		0x00, 0x04, 0x00, 0x04, 0x00, 0x31, 0x01, 0x00,
		0x00, 0x90, 0x00, 0x00, 0x00, 0x00, 0x00,
	}
	r, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != ErrInsecurePath {
		t.Fatalf("Error reading the archive: %v", err)
	}
	entryNames := []string{`/`, `//`, `\`, `/test.txt`}
	var names []string
	for _, f := range r.File {
		names = append(names, f.Name)
		if _, err := f.Open(); err != nil {
			t.Errorf("Error opening %q: %v", f.Name, err)
		}
		if _, err := r.Open(f.Name); err == nil {
			t.Errorf("Opening %q with fs.FS API succeeded", f.Name)
		}
	}
	if !slices.Equal(names, entryNames) {
		t.Errorf("Unexpected file entries: %q", names)
	}
	if _, err := r.Open(""); err == nil {
		t.Errorf("Opening %q with fs.FS API succeeded", "")
	}
	if _, err := r.Open("test.txt"); err != nil {
		t.Errorf("Error opening %q with fs.FS API: %v", "test.txt", err)
	}
	dirEntries, err := fs.ReadDir(r, ".")
	if err != nil {
		t.Fatalf("Error reading the root directory: %v", err)
	}
	if len(dirEntries) != 1 || dirEntries[0].Name() != "test.txt" {
		t.Errorf("Unexpected directory entries")
		for _, dirEntry := range dirEntries {
			_, err := r.Open(dirEntry.Name())
			t.Logf("%q (Open error: %v)", dirEntry.Name(), err)
		}
		t.FailNow()
	}
	info, err := dirEntries[0].Info()
	if err != nil {
		t.Fatalf("Error reading info entry: %v", err)
	}
	if name := info.Name(); name != "test.txt" {
		t.Errorf("Inconsistent name in info entry: %v", name)
	}
}

func TestUnderSize(t *testing.T) {
	z, err := OpenReader("testdata/readme.zip")
	if err != nil {
		t.Fatal(err)
	}
	defer z.Close()

	for _, f := range z.File {
		f.UncompressedSize64 = 1
	}

	for _, f := range z.File {
		t.Run(f.Name, func(t *testing.T) {
			rd, err := f.Open()
			if err != nil {
				t.Fatal(err)
			}
			defer rd.Close()

			_, err = io.Copy(io.Discard, rd)
			if err != ErrFormat {
				t.Fatalf("Error mismatch\n\tGot:  %v\n\tWant: %v", err, ErrFormat)
			}
		})
	}
}

func TestIssue54801(t *testing.T) {
	for _, input := range []string{"testdata/readme.zip", "testdata/dd.zip"} {
		z, err := OpenReader(input)
		if err != nil {
			t.Fatal(err)
		}
		defer z.Close()

		for _, f := range z.File {
			// Make file a directory
			f.Name += "/"

			t.Run(f.Name, func(t *testing.T) {
				t.Logf("CompressedSize64: %d, Flags: %#x", f.CompressedSize64, f.Flags)

				rd, err := f.Open()
				if err != nil {
					t.Fatal(err)
				}
				defer rd.Close()

				n, got := io.Copy(io.Discard, rd)
				if n != 0 || got != ErrFormat {
					t.Fatalf("Error mismatch, got: %d, %v, want: %v", n, got, ErrFormat)
				}
			})
		}
	}
}

func TestInsecurePaths(t *testing.T) {
	t.Setenv("GODEBUG", "zipinsecurepath=0")
	for _, path := range []string{
		"../foo",
		"/foo",
		"a/b/../../../c",
		`a\b`,
	} {
		var buf bytes.Buffer
		zw := NewWriter(&buf)
		_, err := zw.Create(path)
		if err != nil {
			t.Errorf("zw.Create(%q) = %v", path, err)
			continue
		}
		zw.Close()

		zr, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
		if err != ErrInsecurePath {
			t.Errorf("NewReader for archive with file %q: got err %v, want ErrInsecurePath", path, err)
			continue
		}
		var gotPaths []string
		for _, f := range zr.File {
			gotPaths = append(gotPaths, f.Name)
		}
		if !slices.Equal(gotPaths, []string{path}) {
			t.Errorf("NewReader for archive with file %q: got files %q", path, gotPaths)
			continue
		}
	}
}

func TestDisableInsecurePathCheck(t *testing.T) {
	t.Setenv("GODEBUG", "zipinsecurepath=1")
	var buf bytes.Buffer
	zw := NewWriter(&buf)
	const name = "/foo"
	_, err := zw.Create(name)
	if err != nil {
		t.Fatalf("zw.Create(%q) = %v", name, err)
	}
	zw.Close()
	zr, err := NewReader(bytes.NewReader(buf.Bytes()), int64(buf.Len()))
	if err != nil {
		t.Fatalf("NewReader with zipinsecurepath=1: got err %v, want nil", err)
	}
	var gotPaths []string
	for _, f := range zr.File {
		gotPaths = append(gotPaths, f.Name)
	}
	if want := []string{name}; !slices.Equal(gotPaths, want) {
		t.Errorf("NewReader with zipinsecurepath=1: got files %q, want %q", gotPaths, want)
	}
}

func TestCompressedDirectory(t *testing.T) {
	// Empty Java JAR, with a compressed directory with uncompressed size 0
	// which should not fail.
	//
	// Length   Method    Size  Cmpr    Date    Time   CRC-32   Name
	// --------  ------  ------- ---- ---------- ----- --------  ----
	//        0  Defl:N        2   0% 12-01-2022 16:50 00000000  META-INF/
	//       60  Defl:N       59   2% 12-01-2022 16:50 af937e93  META-INF/MANIFEST.MF
	// --------          -------  ---                            -------
	//       60               61  -2%                            2 files
	data := []byte{
		0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x08, 0x08,
		0x08, 0x00, 0x49, 0x86, 0x81, 0x55, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x09, 0x00, 0x04, 0x00, 0x4d, 0x45,
		0x54, 0x41, 0x2d, 0x49, 0x4e, 0x46, 0x2f, 0xfe,
		0xca, 0x00, 0x00, 0x03, 0x00, 0x50, 0x4b, 0x07,
		0x08, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x50, 0x4b, 0x03,
		0x04, 0x14, 0x00, 0x08, 0x08, 0x08, 0x00, 0x49,
		0x86, 0x81, 0x55, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x14,
		0x00, 0x00, 0x00, 0x4d, 0x45, 0x54, 0x41, 0x2d,
		0x49, 0x4e, 0x46, 0x2f, 0x4d, 0x41, 0x4e, 0x49,
		0x46, 0x45, 0x53, 0x54, 0x2e, 0x4d, 0x46, 0xf3,
		0x4d, 0xcc, 0xcb, 0x4c, 0x4b, 0x2d, 0x2e, 0xd1,
		0x0d, 0x4b, 0x2d, 0x2a, 0xce, 0xcc, 0xcf, 0xb3,
		0x52, 0x30, 0xd4, 0x33, 0xe0, 0xe5, 0x72, 0x2e,
		0x4a, 0x4d, 0x2c, 0x49, 0x4d, 0xd1, 0x75, 0xaa,
		0x04, 0x0a, 0x00, 0x45, 0xf4, 0x0c, 0x8d, 0x15,
		0x34, 0xdc, 0xf3, 0xf3, 0xd3, 0x73, 0x52, 0x15,
		0x3c, 0xf3, 0x92, 0xf5, 0x34, 0x79, 0xb9, 0x78,
		0xb9, 0x00, 0x50, 0x4b, 0x07, 0x08, 0x93, 0x7e,
		0x93, 0xaf, 0x3b, 0x00, 0x00, 0x00, 0x3c, 0x00,
		0x00, 0x00, 0x50, 0x4b, 0x01, 0x02, 0x14, 0x00,
		0x14, 0x00, 0x08, 0x08, 0x08, 0x00, 0x49, 0x86,
		0x81, 0x55, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x09, 0x00,
		0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x4d, 0x45, 0x54, 0x41, 0x2d, 0x49, 0x4e, 0x46,
		0x2f, 0xfe, 0xca, 0x00, 0x00, 0x50, 0x4b, 0x01,
		0x02, 0x14, 0x00, 0x14, 0x00, 0x08, 0x08, 0x08,
		0x00, 0x49, 0x86, 0x81, 0x55, 0x93, 0x7e, 0x93,
		0xaf, 0x3b, 0x00, 0x00, 0x00, 0x3c, 0x00, 0x00,
		0x00, 0x14, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3d,
		0x00, 0x00, 0x00, 0x4d, 0x45, 0x54, 0x41, 0x2d,
		0x49, 0x4e, 0x46, 0x2f, 0x4d, 0x41, 0x4e, 0x49,
		0x46, 0x45, 0x53, 0x54, 0x2e, 0x4d, 0x46, 0x50,
		0x4b, 0x05, 0x06, 0x00, 0x00, 0x00, 0x00, 0x02,
		0x00, 0x02, 0x00, 0x7d, 0x00, 0x00, 0x00, 0xba,
		0x00, 0x00, 0x00, 0x00, 0x00,
	}
	r, err := NewReader(bytes.NewReader(data), int64(len(data)))
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, f := range r.File {
		r, err := f.Open()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if _, err := io.Copy(io.Discard, r); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}

func TestBaseOffsetPlusOverflow(t *testing.T) {
	// directoryOffset > maxInt64 && size-directoryOffset < 0
	data := []byte{
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0xff, 0xff, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x50, 0x4b, 0x06, 0x06, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20, 0x20,
		0x20, 0xff, 0xff, 0x20, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x20, 0x08, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x80, 0x50, 0x4b, 0x06, 0x07, 0x00,
		0x00, 0x00, 0x00, 0x6b, 0x00, 0x00, 0x00, 0x00,
		0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x50,
		0x4b, 0x05, 0x06, 0x20, 0x20, 0x20, 0x20, 0xff,
		0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
		0xff, 0xff, 0xff, 0x20, 0x00,
	}
	defer func() {
		if r := recover(); r != nil {
			t.Fatalf("NewReader panicked: %s", r)
		}
	}()
	// Previously, this would trigger a panic as we attempt to read from
	// an io.SectionReader which would access a slice at a negative offset
	// as the section reader offset & size were < 0.
	NewReader(bytes.NewReader(data), int64(len(data))+1875)
}
