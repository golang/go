// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"crypto/md5"
	"fmt"
	"io"
	"io/ioutil"
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

var sparseTarTest = &untarTest{
	file: "testdata/sparse-formats.tar",
	headers: []*Header{
		{
			Name:     "sparse-gnu",
			Mode:     420,
			Uid:      1000,
			Gid:      1000,
			Size:     200,
			ModTime:  time.Unix(1392395740, 0),
			Typeflag: 0x53,
			Linkname: "",
			Uname:    "david",
			Gname:    "david",
			Devmajor: 0,
			Devminor: 0,
		},
		{
			Name:     "sparse-posix-0.0",
			Mode:     420,
			Uid:      1000,
			Gid:      1000,
			Size:     200,
			ModTime:  time.Unix(1392342187, 0),
			Typeflag: 0x30,
			Linkname: "",
			Uname:    "david",
			Gname:    "david",
			Devmajor: 0,
			Devminor: 0,
		},
		{
			Name:     "sparse-posix-0.1",
			Mode:     420,
			Uid:      1000,
			Gid:      1000,
			Size:     200,
			ModTime:  time.Unix(1392340456, 0),
			Typeflag: 0x30,
			Linkname: "",
			Uname:    "david",
			Gname:    "david",
			Devmajor: 0,
			Devminor: 0,
		},
		{
			Name:     "sparse-posix-1.0",
			Mode:     420,
			Uid:      1000,
			Gid:      1000,
			Size:     200,
			ModTime:  time.Unix(1392337404, 0),
			Typeflag: 0x30,
			Linkname: "",
			Uname:    "david",
			Gname:    "david",
			Devmajor: 0,
			Devminor: 0,
		},
		{
			Name:     "end",
			Mode:     420,
			Uid:      1000,
			Gid:      1000,
			Size:     4,
			ModTime:  time.Unix(1392398319, 0),
			Typeflag: 0x30,
			Linkname: "",
			Uname:    "david",
			Gname:    "david",
			Devmajor: 0,
			Devminor: 0,
		},
	},
	cksums: []string{
		"6f53234398c2449fe67c1812d993012f",
		"6f53234398c2449fe67c1812d993012f",
		"6f53234398c2449fe67c1812d993012f",
		"6f53234398c2449fe67c1812d993012f",
		"b0061974914468de549a2af8ced10316",
	},
}

var untarTests = []*untarTest{
	gnuTarTest,
	sparseTarTest,
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
	{
		file: "testdata/xattrs.tar",
		headers: []*Header{
			{
				Name:       "small.txt",
				Mode:       0644,
				Uid:        1000,
				Gid:        10,
				Size:       5,
				ModTime:    time.Unix(1386065770, 448252320),
				Typeflag:   '0',
				Uname:      "alex",
				Gname:      "wheel",
				AccessTime: time.Unix(1389782991, 419875220),
				ChangeTime: time.Unix(1389782956, 794414986),
				Xattrs: map[string]string{
					"user.key":  "value",
					"user.key2": "value2",
					// Interestingly, selinux encodes the terminating null inside the xattr
					"security.selinux": "unconfined_u:object_r:default_t:s0\x00",
				},
			},
			{
				Name:       "small2.txt",
				Mode:       0644,
				Uid:        1000,
				Gid:        10,
				Size:       11,
				ModTime:    time.Unix(1386065770, 449252304),
				Typeflag:   '0',
				Uname:      "alex",
				Gname:      "wheel",
				AccessTime: time.Unix(1389782991, 419875220),
				ChangeTime: time.Unix(1386065770, 449252304),
				Xattrs: map[string]string{
					"security.selinux": "unconfined_u:object_r:default_t:s0\x00",
				},
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
			if !reflect.DeepEqual(*hdr, *header) {
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
		if !reflect.DeepEqual(*hdr, *headers[nread]) {
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
		reader := bytes.NewReader([]byte(raw))
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
	badHeaderTests := [][]byte{
		[]byte("3 somelongkey=\n"),
		[]byte("50 tooshort=\n"),
	}
	for _, test := range badHeaderTests {
		if _, err := parsePAX(bytes.NewReader(test)); err != ErrHeader {
			t.Fatal("Unexpected success when parsing bad header")
		}
	}
}

func TestParsePAXTime(t *testing.T) {
	// Some valid PAX time values
	timestamps := map[string]time.Time{
		"1350244992.023960108":  time.Unix(1350244992, 23960108), // The common case
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

func TestSparseEndToEnd(t *testing.T) {
	test := sparseTarTest
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
		if !reflect.DeepEqual(*hdr, *headers[nread]) {
			t.Errorf("Incorrect header:\nhave %+v\nwant %+v",
				*hdr, headers[nread])
		}

		// read and checksum the file data
		h := md5.New()
		_, err = io.Copy(h, tr)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
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

type sparseFileReadTest struct {
	sparseData []byte
	sparseMap  []sparseEntry
	realSize   int64
	expected   []byte
}

var sparseFileReadTests = []sparseFileReadTest{
	{
		sparseData: []byte("abcde"),
		sparseMap: []sparseEntry{
			{offset: 0, numBytes: 2},
			{offset: 5, numBytes: 3},
		},
		realSize: 8,
		expected: []byte("ab\x00\x00\x00cde"),
	},
	{
		sparseData: []byte("abcde"),
		sparseMap: []sparseEntry{
			{offset: 0, numBytes: 2},
			{offset: 5, numBytes: 3},
		},
		realSize: 10,
		expected: []byte("ab\x00\x00\x00cde\x00\x00"),
	},
	{
		sparseData: []byte("abcde"),
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 2},
		},
		realSize: 8,
		expected: []byte("\x00abc\x00\x00de"),
	},
	{
		sparseData: []byte("abcde"),
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 2},
		},
		realSize: 10,
		expected: []byte("\x00abc\x00\x00de\x00\x00"),
	},
	{
		sparseData: []byte(""),
		sparseMap:  nil,
		realSize:   2,
		expected:   []byte("\x00\x00"),
	},
}

func TestSparseFileReader(t *testing.T) {
	for i, test := range sparseFileReadTests {
		r := bytes.NewReader(test.sparseData)
		nb := int64(r.Len())
		sfr := &sparseFileReader{
			rfr: &regFileReader{r: r, nb: nb},
			sp:  test.sparseMap,
			pos: 0,
			tot: test.realSize,
		}
		if sfr.numBytes() != nb {
			t.Errorf("test %d: Before reading, sfr.numBytes() = %d, want %d", i, sfr.numBytes(), nb)
		}
		buf, err := ioutil.ReadAll(sfr)
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err)
		}
		if e := test.expected; !bytes.Equal(buf, e) {
			t.Errorf("test %d: Contents = %v, want %v", i, buf, e)
		}
		if sfr.numBytes() != 0 {
			t.Errorf("test %d: After draining the reader, numBytes() was nonzero", i)
		}
	}
}

func TestSparseIncrementalRead(t *testing.T) {
	sparseMap := []sparseEntry{{10, 2}}
	sparseData := []byte("Go")
	expected := "\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00Go\x00\x00\x00\x00\x00\x00\x00\x00"

	r := bytes.NewReader(sparseData)
	nb := int64(r.Len())
	sfr := &sparseFileReader{
		rfr: &regFileReader{r: r, nb: nb},
		sp:  sparseMap,
		pos: 0,
		tot: int64(len(expected)),
	}

	// We'll read the data 6 bytes at a time, with a hole of size 10 at
	// the beginning and one of size 8 at the end.
	var outputBuf bytes.Buffer
	buf := make([]byte, 6)
	for {
		n, err := sfr.Read(buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Errorf("Read: unexpected error %v\n", err)
		}
		if n > 0 {
			_, err := outputBuf.Write(buf[:n])
			if err != nil {
				t.Errorf("Write: unexpected error %v\n", err)
			}
		}
	}
	got := outputBuf.String()
	if got != expected {
		t.Errorf("Contents = %v, want %v", got, expected)
	}
}

func TestReadGNUSparseMap0x1(t *testing.T) {
	headers := map[string]string{
		paxGNUSparseNumBlocks: "4",
		paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
	}
	expected := []sparseEntry{
		{offset: 0, numBytes: 5},
		{offset: 10, numBytes: 5},
		{offset: 20, numBytes: 5},
		{offset: 30, numBytes: 5},
	}

	sp, err := readGNUSparseMap0x1(headers)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(sp, expected) {
		t.Errorf("Incorrect sparse map: got %v, wanted %v", sp, expected)
	}
}

func TestReadGNUSparseMap1x0(t *testing.T) {
	// This test uses lots of holes so the sparse header takes up more than two blocks
	numEntries := 100
	expected := make([]sparseEntry, 0, numEntries)
	sparseMap := new(bytes.Buffer)

	fmt.Fprintf(sparseMap, "%d\n", numEntries)
	for i := 0; i < numEntries; i++ {
		offset := int64(2048 * i)
		numBytes := int64(1024)
		expected = append(expected, sparseEntry{offset: offset, numBytes: numBytes})
		fmt.Fprintf(sparseMap, "%d\n%d\n", offset, numBytes)
	}

	// Make the header the smallest multiple of blockSize that fits the sparseMap
	headerBlocks := (sparseMap.Len() + blockSize - 1) / blockSize
	bufLen := blockSize * headerBlocks
	buf := make([]byte, bufLen)
	copy(buf, sparseMap.Bytes())

	// Get an reader to read the sparse map
	r := bytes.NewReader(buf)

	// Read the sparse map
	sp, err := readGNUSparseMap1x0(r)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !reflect.DeepEqual(sp, expected) {
		t.Errorf("Incorrect sparse map: got %v, wanted %v", sp, expected)
	}
}

func TestUninitializedRead(t *testing.T) {
	test := gnuTarTest
	f, err := os.Open(test.file)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer f.Close()

	tr := NewReader(f)
	_, err = tr.Read([]byte{})
	if err == nil || err != io.EOF {
		t.Errorf("Unexpected error: %v, wanted %v", err, io.EOF)
	}

}

// Negative header size should not cause panic.
// Issues 10959 and 10960.
func TestNegativeHdrSize(t *testing.T) {
	f, err := os.Open("testdata/neg-size.tar")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := NewReader(f)
	_, err = r.Next()
	if err != ErrHeader {
		t.Error("want ErrHeader, got", err)
	}
	io.Copy(ioutil.Discard, r)
}

// This used to hang in (*sparseFileReader).readHole due to missing
// verification of sparse offsets against file size.
func TestIssue10968(t *testing.T) {
	f, err := os.Open("testdata/issue10968.tar")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := NewReader(f)
	_, err = r.Next()
	if err != nil {
		t.Fatal(err)
	}
	_, err = io.Copy(ioutil.Discard, r)
	if err != io.ErrUnexpectedEOF {
		t.Fatalf("expected %q, got %q", io.ErrUnexpectedEOF, err)
	}
}

// Do not panic if there are errors in header blocks after the pax header.
// Issue 11169
func TestIssue11169(t *testing.T) {
	f, err := os.Open("testdata/issue11169.tar")
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	r := NewReader(f)
	_, err = r.Next()
	if err == nil {
		t.Fatal("Unexpected success")
	}
}
