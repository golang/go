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
	"math"
	"os"
	"reflect"
	"strings"
	"testing"
	"time"
)

type untarTest struct {
	file    string    // Test input file
	headers []*Header // Expected output headers
	chksums []string  // MD5 checksum of files, leave as nil if not checked
	err     error     // Expected error to occur
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
	chksums: []string{
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
	chksums: []string{
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
	{
		file: "testdata/neg-size.tar",
		err:  ErrHeader,
	},
	{
		file: "testdata/issue10968.tar",
		err:  ErrHeader,
	},
	{
		file: "testdata/issue11169.tar",
		// TODO(dsnet): Currently the library does not detect that this file is
		// malformed. Instead it incorrectly believes that file just ends.
		// At least the library doesn't crash anymore.
		// err:  ErrHeader,
	},
	{
		file: "testdata/issue12435.tar",
		// TODO(dsnet): Currently the library does not detect that this file is
		// malformed. Instead, it incorrectly believes that file just ends.
		// At least the library doesn't crash anymore.
		// err:  ErrHeader,
	},
}

func TestReader(t *testing.T) {
	for i, v := range untarTests {
		f, err := os.Open(v.file)
		if err != nil {
			t.Errorf("file %s, test %d: unexpected error: %v", v.file, i, err)
			continue
		}
		defer f.Close()

		// Capture all headers and checksums.
		var (
			tr      = NewReader(f)
			hdrs    []*Header
			chksums []string
			rdbuf   = make([]byte, 8)
		)
		for {
			var hdr *Header
			hdr, err = tr.Next()
			if err != nil {
				if err == io.EOF {
					err = nil // Expected error
				}
				break
			}
			hdrs = append(hdrs, hdr)

			if v.chksums == nil {
				continue
			}
			h := md5.New()
			_, err = io.CopyBuffer(h, tr, rdbuf) // Effectively an incremental read
			if err != nil {
				break
			}
			chksums = append(chksums, fmt.Sprintf("%x", h.Sum(nil)))
		}

		for j, hdr := range hdrs {
			if j >= len(v.headers) {
				t.Errorf("file %s, test %d, entry %d: unexpected header:\ngot %+v",
					v.file, i, j, *hdr)
				continue
			}
			if !reflect.DeepEqual(*hdr, *v.headers[j]) {
				t.Errorf("file %s, test %d, entry %d: incorrect header:\ngot  %+v\nwant %+v",
					v.file, i, j, *hdr, *v.headers[j])
			}
		}
		if len(hdrs) != len(v.headers) {
			t.Errorf("file %s, test %d: got %d headers, want %d headers",
				v.file, i, len(hdrs), len(v.headers))
		}

		for j, sum := range chksums {
			if j >= len(v.chksums) {
				t.Errorf("file %s, test %d, entry %d: unexpected sum: got %s",
					v.file, i, j, sum)
				continue
			}
			if sum != v.chksums[j] {
				t.Errorf("file %s, test %d, entry %d: incorrect checksum: got %s, want %s",
					v.file, i, j, sum, v.chksums[j])
			}
		}

		if err != v.err {
			t.Errorf("file %s, test %d: unexpected error: got %v, want %v",
				v.file, i, err, v.err)
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

func TestSparseFileReader(t *testing.T) {
	var vectors = []struct {
		realSize   int64         // Real size of the output file
		sparseMap  []sparseEntry // Input sparse map
		sparseData string        // Input compact data
		expected   string        // Expected output data
		err        error         // Expected error outcome
	}{{
		realSize: 8,
		sparseMap: []sparseEntry{
			{offset: 0, numBytes: 2},
			{offset: 5, numBytes: 3},
		},
		sparseData: "abcde",
		expected:   "ab\x00\x00\x00cde",
	}, {
		realSize: 10,
		sparseMap: []sparseEntry{
			{offset: 0, numBytes: 2},
			{offset: 5, numBytes: 3},
		},
		sparseData: "abcde",
		expected:   "ab\x00\x00\x00cde\x00\x00",
	}, {
		realSize: 8,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 2},
		},
		sparseData: "abcde",
		expected:   "\x00abc\x00\x00de",
	}, {
		realSize: 8,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 0},
			{offset: 6, numBytes: 0},
			{offset: 6, numBytes: 2},
		},
		sparseData: "abcde",
		expected:   "\x00abc\x00\x00de",
	}, {
		realSize: 10,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 2},
		},
		sparseData: "abcde",
		expected:   "\x00abc\x00\x00de\x00\x00",
	}, {
		realSize: 10,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 2},
			{offset: 8, numBytes: 0},
			{offset: 8, numBytes: 0},
			{offset: 8, numBytes: 0},
			{offset: 8, numBytes: 0},
		},
		sparseData: "abcde",
		expected:   "\x00abc\x00\x00de\x00\x00",
	}, {
		realSize:   2,
		sparseMap:  []sparseEntry{},
		sparseData: "",
		expected:   "\x00\x00",
	}, {
		realSize:  -2,
		sparseMap: []sparseEntry{},
		err:       ErrHeader,
	}, {
		realSize: -10,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 2},
		},
		sparseData: "abcde",
		err:        ErrHeader,
	}, {
		realSize: 10,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 5},
		},
		sparseData: "abcde",
		err:        ErrHeader,
	}, {
		realSize: 35,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: 5},
		},
		sparseData: "abcde",
		err:        io.ErrUnexpectedEOF,
	}, {
		realSize: 35,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 6, numBytes: -5},
		},
		sparseData: "abcde",
		err:        ErrHeader,
	}, {
		realSize: 35,
		sparseMap: []sparseEntry{
			{offset: math.MaxInt64, numBytes: 3},
			{offset: 6, numBytes: -5},
		},
		sparseData: "abcde",
		err:        ErrHeader,
	}, {
		realSize: 10,
		sparseMap: []sparseEntry{
			{offset: 1, numBytes: 3},
			{offset: 2, numBytes: 2},
		},
		sparseData: "abcde",
		err:        ErrHeader,
	}}

	for i, v := range vectors {
		r := bytes.NewReader([]byte(v.sparseData))
		rfr := &regFileReader{r: r, nb: int64(len(v.sparseData))}

		var sfr *sparseFileReader
		var err error
		var buf []byte

		sfr, err = newSparseFileReader(rfr, v.sparseMap, v.realSize)
		if err != nil {
			goto fail
		}
		if sfr.numBytes() != int64(len(v.sparseData)) {
			t.Errorf("test %d, numBytes() before reading: got %d, want %d", i, sfr.numBytes(), len(v.sparseData))
		}
		buf, err = ioutil.ReadAll(sfr)
		if err != nil {
			goto fail
		}
		if string(buf) != v.expected {
			t.Errorf("test %d, ReadAll(): got %q, want %q", i, string(buf), v.expected)
		}
		if sfr.numBytes() != 0 {
			t.Errorf("test %d, numBytes() after reading: got %d, want %d", i, sfr.numBytes(), 0)
		}

	fail:
		if err != v.err {
			t.Errorf("test %d, unexpected error: got %v, want %v", i, err, v.err)
		}
	}
}

func TestReadGNUSparseMap0x1(t *testing.T) {
	const (
		maxUint = ^uint(0)
		maxInt  = int(maxUint >> 1)
	)
	var (
		big1 = fmt.Sprintf("%d", int64(maxInt))
		big2 = fmt.Sprintf("%d", (int64(maxInt)/2)+1)
		big3 = fmt.Sprintf("%d", (int64(maxInt) / 3))
	)

	var vectors = []struct {
		extHdrs   map[string]string // Input data
		sparseMap []sparseEntry     // Expected sparse entries to be outputted
		err       error             // Expected errors that may be raised
	}{{
		extHdrs: map[string]string{paxGNUSparseNumBlocks: "-4"},
		err:     ErrHeader,
	}, {
		extHdrs: map[string]string{paxGNUSparseNumBlocks: "fee "},
		err:     ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: big1,
			paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
		},
		err: ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: big2,
			paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
		},
		err: ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: big3,
			paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
		},
		err: ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0.5,5,10,5,20,5,30,5",
		},
		err: ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0,5.5,10,5,20,5,30,5",
		},
		err: ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0,fewafewa.5,fewafw,5,20,5,30,5",
		},
		err: ErrHeader,
	}, {
		extHdrs: map[string]string{
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
		},
		sparseMap: []sparseEntry{{0, 5}, {10, 5}, {20, 5}, {30, 5}},
	}}

	for i, v := range vectors {
		sp, err := readGNUSparseMap0x1(v.extHdrs)
		if !reflect.DeepEqual(sp, v.sparseMap) && !(len(sp) == 0 && len(v.sparseMap) == 0) {
			t.Errorf("test %d, readGNUSparseMap0x1(...): got %v, want %v", i, sp, v.sparseMap)
		}
		if err != v.err {
			t.Errorf("test %d, unexpected error: got %v, want %v", i, err, v.err)
		}
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

type reader struct{ io.Reader }
type readSeeker struct{ io.ReadSeeker }
type readBadSeeker struct{ io.ReadSeeker }

func (rbs *readBadSeeker) Seek(int64, int) (int64, error) { return 0, fmt.Errorf("illegal seek") }

// TestReadTruncation test the ending condition on various truncated files and
// that truncated files are still detected even if the underlying io.Reader
// satisfies io.Seeker.
func TestReadTruncation(t *testing.T) {
	var ss []string
	for _, p := range []string{
		"testdata/gnu.tar",
		"testdata/ustar-file-reg.tar",
		"testdata/pax-path-hdr.tar",
		"testdata/sparse-formats.tar",
	} {
		buf, err := ioutil.ReadFile(p)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		ss = append(ss, string(buf))
	}

	data1, data2, pax, sparse := ss[0], ss[1], ss[2], ss[3]
	data2 += strings.Repeat("\x00", 10*512)
	trash := strings.Repeat("garbage ", 64) // Exactly 512 bytes

	var vectors = []struct {
		input string // Input stream
		cnt   int    // Expected number of headers read
		err   error  // Expected error outcome
	}{
		{"", 0, io.EOF}, // Empty file is a "valid" tar file
		{data1[:511], 0, io.ErrUnexpectedEOF},
		{data1[:512], 1, io.ErrUnexpectedEOF},
		{data1[:1024], 1, io.EOF},
		{data1[:1536], 2, io.ErrUnexpectedEOF},
		{data1[:2048], 2, io.EOF},
		{data1, 2, io.EOF},
		{data1[:2048] + data2[:1536], 3, io.EOF},
		{data2[:511], 0, io.ErrUnexpectedEOF},
		{data2[:512], 1, io.ErrUnexpectedEOF},
		{data2[:1195], 1, io.ErrUnexpectedEOF},
		{data2[:1196], 1, io.EOF}, // Exact end of data and start of padding
		{data2[:1200], 1, io.EOF},
		{data2[:1535], 1, io.EOF},
		{data2[:1536], 1, io.EOF}, // Exact end of padding
		{data2[:1536] + trash[:1], 1, io.ErrUnexpectedEOF},
		{data2[:1536] + trash[:511], 1, io.ErrUnexpectedEOF},
		{data2[:1536] + trash, 1, ErrHeader},
		{data2[:2048], 1, io.EOF}, // Exactly 1 empty block
		{data2[:2048] + trash[:1], 1, io.ErrUnexpectedEOF},
		{data2[:2048] + trash[:511], 1, io.ErrUnexpectedEOF},
		{data2[:2048] + trash, 1, ErrHeader},
		{data2[:2560], 1, io.EOF}, // Exactly 2 empty blocks (normal end-of-stream)
		{data2[:2560] + trash[:1], 1, io.EOF},
		{data2[:2560] + trash[:511], 1, io.EOF},
		{data2[:2560] + trash, 1, io.EOF},
		{data2[:3072], 1, io.EOF},
		{pax, 0, io.EOF}, // PAX header without data is a "valid" tar file
		{pax + trash[:1], 0, io.ErrUnexpectedEOF},
		{pax + trash[:511], 0, io.ErrUnexpectedEOF},
		{sparse[:511], 0, io.ErrUnexpectedEOF},
		// TODO(dsnet): This should pass, but currently fails.
		// {sparse[:512], 0, io.ErrUnexpectedEOF},
		{sparse[:3584], 1, io.EOF},
		{sparse[:9200], 1, io.EOF}, // Terminate in padding of sparse header
		{sparse[:9216], 1, io.EOF},
		{sparse[:9728], 2, io.ErrUnexpectedEOF},
		{sparse[:10240], 2, io.EOF},
		{sparse[:11264], 2, io.ErrUnexpectedEOF},
		{sparse, 5, io.EOF},
		{sparse + trash, 5, io.EOF},
	}

	for i, v := range vectors {
		for j := 0; j < 6; j++ {
			var tr *Reader
			var s1, s2 string

			switch j {
			case 0:
				tr = NewReader(&reader{strings.NewReader(v.input)})
				s1, s2 = "io.Reader", "auto"
			case 1:
				tr = NewReader(&reader{strings.NewReader(v.input)})
				s1, s2 = "io.Reader", "manual"
			case 2:
				tr = NewReader(&readSeeker{strings.NewReader(v.input)})
				s1, s2 = "io.ReadSeeker", "auto"
			case 3:
				tr = NewReader(&readSeeker{strings.NewReader(v.input)})
				s1, s2 = "io.ReadSeeker", "manual"
			case 4:
				tr = NewReader(&readBadSeeker{strings.NewReader(v.input)})
				s1, s2 = "ReadBadSeeker", "auto"
			case 5:
				tr = NewReader(&readBadSeeker{strings.NewReader(v.input)})
				s1, s2 = "ReadBadSeeker", "manual"
			}

			var cnt int
			var err error
			for {
				if _, err = tr.Next(); err != nil {
					break
				}
				cnt++
				if s2 == "manual" {
					if _, err = io.Copy(ioutil.Discard, tr); err != nil {
						break
					}
				}
			}
			if err != v.err {
				t.Errorf("test %d, NewReader(%s(...)) with %s discard: got %v, want %v",
					i, s1, s2, err, v.err)
			}
			if cnt != v.cnt {
				t.Errorf("test %d, NewReader(%s(...)) with %s discard: got %d headers, want %d headers",
					i, s1, s2, cnt, v.cnt)
			}
		}
	}
}
