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
	"path"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"
)

func TestReader(t *testing.T) {
	vectors := []struct {
		file    string    // Test input file
		headers []*Header // Expected output headers
		chksums []string  // MD5 checksum of files, leave as nil if not checked
		err     error     // Expected error to occur
	}{{
		file: "testdata/gnu.tar",
		headers: []*Header{{
			Name:     "small.txt",
			Mode:     0640,
			Uid:      73025,
			Gid:      5000,
			Size:     5,
			ModTime:  time.Unix(1244428340, 0),
			Typeflag: '0',
			Uname:    "dsymonds",
			Gname:    "eng",
		}, {
			Name:     "small2.txt",
			Mode:     0640,
			Uid:      73025,
			Gid:      5000,
			Size:     11,
			ModTime:  time.Unix(1244436044, 0),
			Typeflag: '0',
			Uname:    "dsymonds",
			Gname:    "eng",
		}},
		chksums: []string{
			"e38b27eaccb4391bdec553a7f3ae6b2f",
			"c65bd2e50a56a2138bf1716f2fd56fe9",
		},
	}, {
		file: "testdata/sparse-formats.tar",
		headers: []*Header{{
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
			SparseHoles: []SparseEntry{
				{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}, {12, 1}, {14, 1},
				{16, 1}, {18, 1}, {20, 1}, {22, 1}, {24, 1}, {26, 1}, {28, 1},
				{30, 1}, {32, 1}, {34, 1}, {36, 1}, {38, 1}, {40, 1}, {42, 1},
				{44, 1}, {46, 1}, {48, 1}, {50, 1}, {52, 1}, {54, 1}, {56, 1},
				{58, 1}, {60, 1}, {62, 1}, {64, 1}, {66, 1}, {68, 1}, {70, 1},
				{72, 1}, {74, 1}, {76, 1}, {78, 1}, {80, 1}, {82, 1}, {84, 1},
				{86, 1}, {88, 1}, {90, 1}, {92, 1}, {94, 1}, {96, 1}, {98, 1},
				{100, 1}, {102, 1}, {104, 1}, {106, 1}, {108, 1}, {110, 1},
				{112, 1}, {114, 1}, {116, 1}, {118, 1}, {120, 1}, {122, 1},
				{124, 1}, {126, 1}, {128, 1}, {130, 1}, {132, 1}, {134, 1},
				{136, 1}, {138, 1}, {140, 1}, {142, 1}, {144, 1}, {146, 1},
				{148, 1}, {150, 1}, {152, 1}, {154, 1}, {156, 1}, {158, 1},
				{160, 1}, {162, 1}, {164, 1}, {166, 1}, {168, 1}, {170, 1},
				{172, 1}, {174, 1}, {176, 1}, {178, 1}, {180, 1}, {182, 1},
				{184, 1}, {186, 1}, {188, 1}, {190, 10},
			},
		}, {
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
			SparseHoles: []SparseEntry{
				{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}, {12, 1}, {14, 1},
				{16, 1}, {18, 1}, {20, 1}, {22, 1}, {24, 1}, {26, 1}, {28, 1},
				{30, 1}, {32, 1}, {34, 1}, {36, 1}, {38, 1}, {40, 1}, {42, 1},
				{44, 1}, {46, 1}, {48, 1}, {50, 1}, {52, 1}, {54, 1}, {56, 1},
				{58, 1}, {60, 1}, {62, 1}, {64, 1}, {66, 1}, {68, 1}, {70, 1},
				{72, 1}, {74, 1}, {76, 1}, {78, 1}, {80, 1}, {82, 1}, {84, 1},
				{86, 1}, {88, 1}, {90, 1}, {92, 1}, {94, 1}, {96, 1}, {98, 1},
				{100, 1}, {102, 1}, {104, 1}, {106, 1}, {108, 1}, {110, 1},
				{112, 1}, {114, 1}, {116, 1}, {118, 1}, {120, 1}, {122, 1},
				{124, 1}, {126, 1}, {128, 1}, {130, 1}, {132, 1}, {134, 1},
				{136, 1}, {138, 1}, {140, 1}, {142, 1}, {144, 1}, {146, 1},
				{148, 1}, {150, 1}, {152, 1}, {154, 1}, {156, 1}, {158, 1},
				{160, 1}, {162, 1}, {164, 1}, {166, 1}, {168, 1}, {170, 1},
				{172, 1}, {174, 1}, {176, 1}, {178, 1}, {180, 1}, {182, 1},
				{184, 1}, {186, 1}, {188, 1}, {190, 10},
			},
		}, {
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
			SparseHoles: []SparseEntry{
				{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}, {12, 1}, {14, 1},
				{16, 1}, {18, 1}, {20, 1}, {22, 1}, {24, 1}, {26, 1}, {28, 1},
				{30, 1}, {32, 1}, {34, 1}, {36, 1}, {38, 1}, {40, 1}, {42, 1},
				{44, 1}, {46, 1}, {48, 1}, {50, 1}, {52, 1}, {54, 1}, {56, 1},
				{58, 1}, {60, 1}, {62, 1}, {64, 1}, {66, 1}, {68, 1}, {70, 1},
				{72, 1}, {74, 1}, {76, 1}, {78, 1}, {80, 1}, {82, 1}, {84, 1},
				{86, 1}, {88, 1}, {90, 1}, {92, 1}, {94, 1}, {96, 1}, {98, 1},
				{100, 1}, {102, 1}, {104, 1}, {106, 1}, {108, 1}, {110, 1},
				{112, 1}, {114, 1}, {116, 1}, {118, 1}, {120, 1}, {122, 1},
				{124, 1}, {126, 1}, {128, 1}, {130, 1}, {132, 1}, {134, 1},
				{136, 1}, {138, 1}, {140, 1}, {142, 1}, {144, 1}, {146, 1},
				{148, 1}, {150, 1}, {152, 1}, {154, 1}, {156, 1}, {158, 1},
				{160, 1}, {162, 1}, {164, 1}, {166, 1}, {168, 1}, {170, 1},
				{172, 1}, {174, 1}, {176, 1}, {178, 1}, {180, 1}, {182, 1},
				{184, 1}, {186, 1}, {188, 1}, {190, 10},
			},
		}, {
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
			SparseHoles: []SparseEntry{
				{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}, {12, 1}, {14, 1},
				{16, 1}, {18, 1}, {20, 1}, {22, 1}, {24, 1}, {26, 1}, {28, 1},
				{30, 1}, {32, 1}, {34, 1}, {36, 1}, {38, 1}, {40, 1}, {42, 1},
				{44, 1}, {46, 1}, {48, 1}, {50, 1}, {52, 1}, {54, 1}, {56, 1},
				{58, 1}, {60, 1}, {62, 1}, {64, 1}, {66, 1}, {68, 1}, {70, 1},
				{72, 1}, {74, 1}, {76, 1}, {78, 1}, {80, 1}, {82, 1}, {84, 1},
				{86, 1}, {88, 1}, {90, 1}, {92, 1}, {94, 1}, {96, 1}, {98, 1},
				{100, 1}, {102, 1}, {104, 1}, {106, 1}, {108, 1}, {110, 1},
				{112, 1}, {114, 1}, {116, 1}, {118, 1}, {120, 1}, {122, 1},
				{124, 1}, {126, 1}, {128, 1}, {130, 1}, {132, 1}, {134, 1},
				{136, 1}, {138, 1}, {140, 1}, {142, 1}, {144, 1}, {146, 1},
				{148, 1}, {150, 1}, {152, 1}, {154, 1}, {156, 1}, {158, 1},
				{160, 1}, {162, 1}, {164, 1}, {166, 1}, {168, 1}, {170, 1},
				{172, 1}, {174, 1}, {176, 1}, {178, 1}, {180, 1}, {182, 1},
				{184, 1}, {186, 1}, {188, 1}, {190, 10},
			},
		}, {
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
		}},
		chksums: []string{
			"6f53234398c2449fe67c1812d993012f",
			"6f53234398c2449fe67c1812d993012f",
			"6f53234398c2449fe67c1812d993012f",
			"6f53234398c2449fe67c1812d993012f",
			"b0061974914468de549a2af8ced10316",
		},
	}, {
		file: "testdata/star.tar",
		headers: []*Header{{
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
		}, {
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
		}},
	}, {
		file: "testdata/v7.tar",
		headers: []*Header{{
			Name:     "small.txt",
			Mode:     0444,
			Uid:      73025,
			Gid:      5000,
			Size:     5,
			ModTime:  time.Unix(1244593104, 0),
			Typeflag: '\x00',
		}, {
			Name:     "small2.txt",
			Mode:     0444,
			Uid:      73025,
			Gid:      5000,
			Size:     11,
			ModTime:  time.Unix(1244593104, 0),
			Typeflag: '\x00',
		}},
	}, {
		file: "testdata/pax.tar",
		headers: []*Header{{
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
		}, {
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
		}},
	}, {
		file: "testdata/pax-bad-hdr-file.tar",
		err:  ErrHeader,
	}, {
		file: "testdata/pax-bad-mtime-file.tar",
		err:  ErrHeader,
	}, {
		file: "testdata/pax-pos-size-file.tar",
		headers: []*Header{{
			Name:     "foo",
			Mode:     0640,
			Uid:      319973,
			Gid:      5000,
			Size:     999,
			ModTime:  time.Unix(1442282516, 0),
			Typeflag: '0',
			Uname:    "joetsai",
			Gname:    "eng",
		}},
		chksums: []string{
			"0afb597b283fe61b5d4879669a350556",
		},
	}, {
		file: "testdata/nil-uid.tar", // golang.org/issue/5290
		headers: []*Header{{
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
		}},
	}, {
		file: "testdata/xattrs.tar",
		headers: []*Header{{
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
		}, {
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
		}},
	}, {
		// Matches the behavior of GNU, BSD, and STAR tar utilities.
		file: "testdata/gnu-multi-hdrs.tar",
		headers: []*Header{{
			Name:     "GNU2/GNU2/long-path-name",
			Linkname: "GNU4/GNU4/long-linkpath-name",
			ModTime:  time.Unix(0, 0),
			Typeflag: '2',
		}},
	}, {
		// GNU tar file with atime and ctime fields set.
		// Created with the GNU tar v1.27.1.
		//	tar --incremental -S -cvf gnu-incremental.tar test2
		file: "testdata/gnu-incremental.tar",
		headers: []*Header{{
			Name:       "test2/",
			Mode:       16877,
			Uid:        1000,
			Gid:        1000,
			Size:       14,
			ModTime:    time.Unix(1441973427, 0),
			Typeflag:   'D',
			Uname:      "rawr",
			Gname:      "dsnet",
			AccessTime: time.Unix(1441974501, 0),
			ChangeTime: time.Unix(1441973436, 0),
		}, {
			Name:       "test2/foo",
			Mode:       33188,
			Uid:        1000,
			Gid:        1000,
			Size:       64,
			ModTime:    time.Unix(1441973363, 0),
			Typeflag:   '0',
			Uname:      "rawr",
			Gname:      "dsnet",
			AccessTime: time.Unix(1441974501, 0),
			ChangeTime: time.Unix(1441973436, 0),
		}, {
			Name:        "test2/sparse",
			Mode:        33188,
			Uid:         1000,
			Gid:         1000,
			Size:        536870912,
			ModTime:     time.Unix(1441973427, 0),
			Typeflag:    'S',
			Uname:       "rawr",
			Gname:       "dsnet",
			AccessTime:  time.Unix(1441991948, 0),
			ChangeTime:  time.Unix(1441973436, 0),
			SparseHoles: []SparseEntry{{0, 536870912}},
		}},
	}, {
		// Matches the behavior of GNU and BSD tar utilities.
		file: "testdata/pax-multi-hdrs.tar",
		headers: []*Header{{
			Name:     "bar",
			Linkname: "PAX4/PAX4/long-linkpath-name",
			ModTime:  time.Unix(0, 0),
			Typeflag: '2',
		}},
	}, {
		// Both BSD and GNU tar truncate long names at first NUL even
		// if there is data following that NUL character.
		// This is reasonable as GNU long names are C-strings.
		file: "testdata/gnu-long-nul.tar",
		headers: []*Header{{
			Name:     "0123456789",
			Mode:     0644,
			Uid:      1000,
			Gid:      1000,
			ModTime:  time.Unix(1486082191, 0),
			Typeflag: '0',
			Uname:    "rawr",
			Gname:    "dsnet",
		}},
	}, {
		// This archive was generated by Writer but is readable by both
		// GNU and BSD tar utilities.
		// The archive generated by GNU is nearly byte-for-byte identical
		// to the Go version except the Go version sets a negative Devminor
		// just to force the GNU format.
		file: "testdata/gnu-utf8.tar",
		headers: []*Header{{
			Name: "☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹☺☻☹",
			Mode: 0644,
			Uid:  1000, Gid: 1000,
			ModTime:  time.Unix(0, 0),
			Typeflag: '0',
			Uname:    "☺",
			Gname:    "⚹",
			Devminor: -1,
		}},
	}, {
		// This archive was generated by Writer but is readable by both
		// GNU and BSD tar utilities.
		// The archive generated by GNU is nearly byte-for-byte identical
		// to the Go version except the Go version sets a negative Devminor
		// just to force the GNU format.
		file: "testdata/gnu-not-utf8.tar",
		headers: []*Header{{
			Name:     "hi\x80\x81\x82\x83bye",
			Mode:     0644,
			Uid:      1000,
			Gid:      1000,
			ModTime:  time.Unix(0, 0),
			Typeflag: '0',
			Uname:    "rawr",
			Gname:    "dsnet",
			Devminor: -1,
		}},
	}, {
		// BSD tar v3.1.2 and GNU tar v1.27.1 both rejects PAX records
		// with NULs in the key.
		file: "testdata/pax-nul-xattrs.tar",
		err:  ErrHeader,
	}, {
		// BSD tar v3.1.2 rejects a PAX path with NUL in the value, while
		// GNU tar v1.27.1 simply truncates at first NUL.
		// We emulate the behavior of BSD since it is strange doing NUL
		// truncations since PAX records are length-prefix strings instead
		// of NUL-terminated C-strings.
		file: "testdata/pax-nul-path.tar",
		err:  ErrHeader,
	}, {
		file: "testdata/neg-size.tar",
		err:  ErrHeader,
	}, {
		file: "testdata/issue10968.tar",
		err:  ErrHeader,
	}, {
		file: "testdata/issue11169.tar",
		err:  ErrHeader,
	}, {
		file: "testdata/issue12435.tar",
		err:  ErrHeader,
	}, {
		// Ensure that we can read back the original Header as written with
		// a buggy pre-Go1.8 tar.Writer.
		file: "testdata/invalid-go17.tar",
		headers: []*Header{{
			Name:    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/foo",
			Uid:     010000000,
			ModTime: time.Unix(0, 0),
		}},
	}, {
		// USTAR archive with a regular entry with non-zero device numbers.
		file: "testdata/ustar-file-devs.tar",
		headers: []*Header{{
			Name:     "file",
			Mode:     0644,
			Typeflag: '0',
			ModTime:  time.Unix(0, 0),
			Devmajor: 1,
			Devminor: 1,
		}},
	}}

	for _, v := range vectors {
		t.Run(path.Base(v.file), func(t *testing.T) {
			f, err := os.Open(v.file)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
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

			for i, hdr := range hdrs {
				if i >= len(v.headers) {
					t.Fatalf("entry %d: unexpected header:\ngot %+v", i, *hdr)
					continue
				}
				if !reflect.DeepEqual(*hdr, *v.headers[i]) {
					t.Fatalf("entry %d: incorrect header:\ngot  %+v\nwant %+v", i, *hdr, *v.headers[i])
				}
			}
			if len(hdrs) != len(v.headers) {
				t.Fatalf("got %d headers, want %d headers", len(hdrs), len(v.headers))
			}

			for i, sum := range chksums {
				if i >= len(v.chksums) {
					t.Fatalf("entry %d: unexpected sum: got %s", i, sum)
					continue
				}
				if sum != v.chksums[i] {
					t.Fatalf("entry %d: incorrect checksum: got %s, want %s", i, sum, v.chksums[i])
				}
			}

			if err != v.err {
				t.Fatalf("unexpected error: got %v, want %v", err, v.err)
			}
			f.Close()
		})
	}
}

func TestPartialRead(t *testing.T) {
	type testCase struct {
		cnt    int    // Number of bytes to read
		output string // Expected value of string read
	}
	vectors := []struct {
		file  string
		cases []testCase
	}{{
		file: "testdata/gnu.tar",
		cases: []testCase{
			{4, "Kilt"},
			{6, "Google"},
		},
	}, {
		file: "testdata/sparse-formats.tar",
		cases: []testCase{
			{2, "\x00G"},
			{4, "\x00G\x00o"},
			{6, "\x00G\x00o\x00G"},
			{8, "\x00G\x00o\x00G\x00o"},
			{4, "end\n"},
		},
	}}

	for _, v := range vectors {
		t.Run(path.Base(v.file), func(t *testing.T) {
			f, err := os.Open(v.file)
			if err != nil {
				t.Fatalf("Open() error: %v", err)
			}
			defer f.Close()

			tr := NewReader(f)
			for i, tc := range v.cases {
				hdr, err := tr.Next()
				if err != nil || hdr == nil {
					t.Fatalf("entry %d, Next(): got %v, want %v", i, err, nil)
				}
				buf := make([]byte, tc.cnt)
				if _, err := io.ReadFull(tr, buf); err != nil {
					t.Fatalf("entry %d, ReadFull(): got %v, want %v", i, err, nil)
				}
				if string(buf) != tc.output {
					t.Fatalf("entry %d, ReadFull(): got %q, want %q", i, string(buf), tc.output)
				}
			}

			if _, err := tr.Next(); err != io.EOF {
				t.Fatalf("Next(): got %v, want EOF", err)
			}
		})
	}
}

func TestUninitializedRead(t *testing.T) {
	f, err := os.Open("testdata/gnu.tar")
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

	vectors := []struct {
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
		{sparse[:512], 0, io.ErrUnexpectedEOF},
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

// TestReadHeaderOnly tests that Reader does not attempt to read special
// header-only files.
func TestReadHeaderOnly(t *testing.T) {
	f, err := os.Open("testdata/hdr-only.tar")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	defer f.Close()

	var hdrs []*Header
	tr := NewReader(f)
	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Errorf("Next(): got %v, want %v", err, nil)
			continue
		}
		hdrs = append(hdrs, hdr)

		// If a special flag, we should read nothing.
		cnt, _ := io.ReadFull(tr, []byte{0})
		if cnt > 0 && hdr.Typeflag != TypeReg {
			t.Errorf("ReadFull(...): got %d bytes, want 0 bytes", cnt)
		}
	}

	// File is crafted with 16 entries. The later 8 are identical to the first
	// 8 except that the size is set.
	if len(hdrs) != 16 {
		t.Fatalf("len(hdrs): got %d, want %d", len(hdrs), 16)
	}
	for i := 0; i < 8; i++ {
		hdr1, hdr2 := hdrs[i+0], hdrs[i+8]
		hdr1.Size, hdr2.Size = 0, 0
		if !reflect.DeepEqual(*hdr1, *hdr2) {
			t.Errorf("incorrect header:\ngot  %+v\nwant %+v", *hdr1, *hdr2)
		}
	}
}

func TestMergePAX(t *testing.T) {
	vectors := []struct {
		in   map[string]string
		want *Header
		ok   bool
	}{{
		in: map[string]string{
			"path":  "a/b/c",
			"uid":   "1000",
			"mtime": "1350244992.023960108",
		},
		want: &Header{
			Name:    "a/b/c",
			Uid:     1000,
			ModTime: time.Unix(1350244992, 23960108),
		},
		ok: true,
	}, {
		in: map[string]string{
			"gid": "gtgergergersagersgers",
		},
	}, {
		in: map[string]string{
			"missing":          "missing",
			"SCHILY.xattr.key": "value",
		},
		want: &Header{
			Xattrs: map[string]string{"key": "value"},
		},
		ok: true,
	}}

	for i, v := range vectors {
		got := new(Header)
		err := mergePAX(got, v.in)
		if v.ok && !reflect.DeepEqual(*got, *v.want) {
			t.Errorf("test %d, mergePAX(...):\ngot  %+v\nwant %+v", i, *got, *v.want)
		}
		if ok := err == nil; ok != v.ok {
			t.Errorf("test %d, mergePAX(...): got %v, want %v", i, ok, v.ok)
		}
	}
}

func TestParsePAX(t *testing.T) {
	vectors := []struct {
		in   string
		want map[string]string
		ok   bool
	}{
		{"", nil, true},
		{"6 k=1\n", map[string]string{"k": "1"}, true},
		{"10 a=name\n", map[string]string{"a": "name"}, true},
		{"9 a=name\n", map[string]string{"a": "name"}, true},
		{"30 mtime=1350244992.023960108\n", map[string]string{"mtime": "1350244992.023960108"}, true},
		{"3 somelongkey=\n", nil, false},
		{"50 tooshort=\n", nil, false},
		{"13 key1=haha\n13 key2=nana\n13 key3=kaka\n",
			map[string]string{"key1": "haha", "key2": "nana", "key3": "kaka"}, true},
		{"13 key1=val1\n13 key2=val2\n8 key1=\n",
			map[string]string{"key2": "val2"}, true},
		{"22 GNU.sparse.size=10\n26 GNU.sparse.numblocks=2\n" +
			"23 GNU.sparse.offset=1\n25 GNU.sparse.numbytes=2\n" +
			"23 GNU.sparse.offset=3\n25 GNU.sparse.numbytes=4\n",
			map[string]string{paxGNUSparseSize: "10", paxGNUSparseNumBlocks: "2", paxGNUSparseMap: "1,2,3,4"}, true},
		{"22 GNU.sparse.size=10\n26 GNU.sparse.numblocks=1\n" +
			"25 GNU.sparse.numbytes=2\n23 GNU.sparse.offset=1\n",
			nil, false},
		{"22 GNU.sparse.size=10\n26 GNU.sparse.numblocks=1\n" +
			"25 GNU.sparse.offset=1,2\n25 GNU.sparse.numbytes=2\n",
			nil, false},
	}

	for i, v := range vectors {
		r := strings.NewReader(v.in)
		got, err := parsePAX(r)
		if !reflect.DeepEqual(got, v.want) && !(len(got) == 0 && len(v.want) == 0) {
			t.Errorf("test %d, parsePAX(...):\ngot  %v\nwant %v", i, got, v.want)
		}
		if ok := err == nil; ok != v.ok {
			t.Errorf("test %d, parsePAX(...): got %v, want %v", i, ok, v.ok)
		}
	}
}

func TestReadOldGNUSparseMap(t *testing.T) {
	populateSparseMap := func(sa sparseArray, sps []string) []string {
		for i := 0; len(sps) > 0 && i < sa.MaxEntries(); i++ {
			copy(sa.Entry(i), sps[0])
			sps = sps[1:]
		}
		if len(sps) > 0 {
			copy(sa.IsExtended(), "\x80")
		}
		return sps
	}

	makeInput := func(format int, size string, sps ...string) (out []byte) {
		// Write the initial GNU header.
		var blk block
		gnu := blk.GNU()
		sparse := gnu.Sparse()
		copy(gnu.RealSize(), size)
		sps = populateSparseMap(sparse, sps)
		if format != formatUnknown {
			blk.SetFormat(format)
		}
		out = append(out, blk[:]...)

		// Write extended sparse blocks.
		for len(sps) > 0 {
			var blk block
			sps = populateSparseMap(blk.Sparse(), sps)
			out = append(out, blk[:]...)
		}
		return out
	}

	makeSparseStrings := func(sp []SparseEntry) (out []string) {
		var f formatter
		for _, s := range sp {
			var b [24]byte
			f.formatNumeric(b[:12], s.Offset)
			f.formatNumeric(b[12:], s.Length)
			out = append(out, string(b[:]))
		}
		return out
	}

	vectors := []struct {
		input    []byte
		wantMap  sparseDatas
		wantSize int64
		wantErr  error
	}{{
		input:   makeInput(formatUnknown, ""),
		wantErr: ErrHeader,
	}, {
		input:    makeInput(formatGNU, "1234", "fewa"),
		wantSize: 01234,
		wantErr:  ErrHeader,
	}, {
		input:    makeInput(formatGNU, "0031"),
		wantSize: 031,
	}, {
		input:   makeInput(formatGNU, "80"),
		wantErr: ErrHeader,
	}, {
		input: makeInput(formatGNU, "1234",
			makeSparseStrings(sparseDatas{{0, 0}, {1, 1}})...),
		wantMap:  sparseDatas{{0, 0}, {1, 1}},
		wantSize: 01234,
	}, {
		input: makeInput(formatGNU, "1234",
			append(makeSparseStrings(sparseDatas{{0, 0}, {1, 1}}), []string{"", "blah"}...)...),
		wantMap:  sparseDatas{{0, 0}, {1, 1}},
		wantSize: 01234,
	}, {
		input: makeInput(formatGNU, "3333",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}})...),
		wantMap:  sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}},
		wantSize: 03333,
	}, {
		input: makeInput(formatGNU, "",
			append(append(
				makeSparseStrings(sparseDatas{{0, 1}, {2, 1}}),
				[]string{"", ""}...),
				makeSparseStrings(sparseDatas{{4, 1}, {6, 1}})...)...),
		wantMap: sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}},
	}, {
		input: makeInput(formatGNU, "",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}})...)[:blockSize],
		wantErr: io.ErrUnexpectedEOF,
	}, {
		input: makeInput(formatGNU, "",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}})...)[:3*blockSize/2],
		wantErr: io.ErrUnexpectedEOF,
	}, {
		input: makeInput(formatGNU, "",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}})...),
		wantMap: sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}},
	}, {
		input: makeInput(formatGNU, "",
			makeSparseStrings(sparseDatas{{10 << 30, 512}, {20 << 30, 512}})...),
		wantMap: sparseDatas{{10 << 30, 512}, {20 << 30, 512}},
	}}

	for i, v := range vectors {
		var blk block
		var hdr Header
		v.input = v.input[copy(blk[:], v.input):]
		tr := Reader{r: bytes.NewReader(v.input)}
		got, err := tr.readOldGNUSparseMap(&hdr, &blk)
		if !equalSparseEntries(got, v.wantMap) {
			t.Errorf("test %d, readOldGNUSparseMap(): got %v, want %v", i, got, v.wantMap)
		}
		if err != v.wantErr {
			t.Errorf("test %d, readOldGNUSparseMap() = %v, want %v", i, err, v.wantErr)
		}
		if hdr.Size != v.wantSize {
			t.Errorf("test %d, Header.Size = %d, want %d", i, hdr.Size, v.wantSize)
		}
	}
}

func TestReadGNUSparsePAXHeaders(t *testing.T) {
	padInput := func(s string) string {
		return s + string(zeroBlock[:blockPadding(int64(len(s)))])
	}

	vectors := []struct {
		inputData string
		inputHdrs map[string]string
		wantMap   sparseDatas
		wantSize  int64
		wantName  string
		wantErr   error
	}{{
		inputHdrs: nil,
		wantErr:   nil,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: strconv.FormatInt(math.MaxInt64, 10),
			paxGNUSparseMap:       "0,1,2,3",
		},
		wantErr: ErrHeader,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: "4\x00",
			paxGNUSparseMap:       "0,1,2,3",
		},
		wantErr: ErrHeader,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0,1,2,3",
		},
		wantErr: ErrHeader,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: "2",
			paxGNUSparseMap:       "0,1,2,3",
		},
		wantMap: sparseDatas{{0, 1}, {2, 3}},
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: "2",
			paxGNUSparseMap:       "0, 1,2,3",
		},
		wantErr: ErrHeader,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: "2",
			paxGNUSparseMap:       "0,1,02,3",
			paxGNUSparseRealSize:  "4321",
		},
		wantMap:  sparseDatas{{0, 1}, {2, 3}},
		wantSize: 4321,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseNumBlocks: "2",
			paxGNUSparseMap:       "0,one1,2,3",
		},
		wantErr: ErrHeader,
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseMajor:     "0",
			paxGNUSparseMinor:     "0",
			paxGNUSparseNumBlocks: "2",
			paxGNUSparseMap:       "0,1,2,3",
			paxGNUSparseSize:      "1234",
			paxGNUSparseRealSize:  "4321",
			paxGNUSparseName:      "realname",
		},
		wantMap:  sparseDatas{{0, 1}, {2, 3}},
		wantSize: 1234,
		wantName: "realname",
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseMajor:     "0",
			paxGNUSparseMinor:     "0",
			paxGNUSparseNumBlocks: "1",
			paxGNUSparseMap:       "10737418240,512",
			paxGNUSparseSize:      "10737418240",
			paxGNUSparseName:      "realname",
		},
		wantMap:  sparseDatas{{10737418240, 512}},
		wantSize: 10737418240,
		wantName: "realname",
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseMajor:     "0",
			paxGNUSparseMinor:     "0",
			paxGNUSparseNumBlocks: "0",
			paxGNUSparseMap:       "",
		},
		wantMap: sparseDatas{},
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseMajor:     "0",
			paxGNUSparseMinor:     "1",
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
		},
		wantMap: sparseDatas{{0, 5}, {10, 5}, {20, 5}, {30, 5}},
	}, {
		inputHdrs: map[string]string{
			paxGNUSparseMajor:     "1",
			paxGNUSparseMinor:     "0",
			paxGNUSparseNumBlocks: "4",
			paxGNUSparseMap:       "0,5,10,5,20,5,30,5",
		},
		wantErr: io.ErrUnexpectedEOF,
	}, {
		inputData: padInput("0\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap:   sparseDatas{},
	}, {
		inputData: padInput("0\n")[:blockSize-1] + "#",
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap:   sparseDatas{},
	}, {
		inputData: padInput("0"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantErr:   io.ErrUnexpectedEOF,
	}, {
		inputData: padInput("ab\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantErr:   ErrHeader,
	}, {
		inputData: padInput("1\n2\n3\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap:   sparseDatas{{2, 3}},
	}, {
		inputData: padInput("1\n2\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantErr:   io.ErrUnexpectedEOF,
	}, {
		inputData: padInput("1\n2\n\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantErr:   ErrHeader,
	}, {
		inputData: string(zeroBlock[:]) + padInput("0\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantErr:   ErrHeader,
	}, {
		inputData: strings.Repeat("0", blockSize) + padInput("1\n5\n1\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap:   sparseDatas{{5, 1}},
	}, {
		inputData: padInput(fmt.Sprintf("%d\n", int64(math.MaxInt64))),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantErr:   ErrHeader,
	}, {
		inputData: padInput(strings.Repeat("0", 300) + "1\n" + strings.Repeat("0", 1000) + "5\n" + strings.Repeat("0", 800) + "2\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap:   sparseDatas{{5, 2}},
	}, {
		inputData: padInput("2\n10737418240\n512\n21474836480\n512\n"),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap:   sparseDatas{{10737418240, 512}, {21474836480, 512}},
	}, {
		inputData: padInput("100\n" + func() string {
			var ss []string
			for i := 0; i < 100; i++ {
				ss = append(ss, fmt.Sprintf("%d\n%d\n", int64(i)<<30, 512))
			}
			return strings.Join(ss, "")
		}()),
		inputHdrs: map[string]string{paxGNUSparseMajor: "1", paxGNUSparseMinor: "0"},
		wantMap: func() (spd sparseDatas) {
			for i := 0; i < 100; i++ {
				spd = append(spd, SparseEntry{int64(i) << 30, 512})
			}
			return spd
		}(),
	}}

	for i, v := range vectors {
		var hdr Header
		r := strings.NewReader(v.inputData + "#") // Add canary byte
		tr := Reader{curr: &regFileReader{r, int64(r.Len())}}
		got, err := tr.readGNUSparsePAXHeaders(&hdr, v.inputHdrs)
		if !equalSparseEntries(got, v.wantMap) {
			t.Errorf("test %d, readGNUSparsePAXHeaders(): got %v, want %v", i, got, v.wantMap)
		}
		if err != v.wantErr {
			t.Errorf("test %d, readGNUSparsePAXHeaders() = %v, want %v", i, err, v.wantErr)
		}
		if hdr.Size != v.wantSize {
			t.Errorf("test %d, Header.Size = %d, want %d", i, hdr.Size, v.wantSize)
		}
		if hdr.Name != v.wantName {
			t.Errorf("test %d, Header.Name = %s, want %s", i, hdr.Name, v.wantName)
		}
		if v.wantErr == nil && r.Len() == 0 {
			t.Errorf("test %d, canary byte unexpectedly consumed", i)
		}
	}
}

func TestFileReader(t *testing.T) {
	type (
		testRead struct { // ReadN(cnt) == (wantStr, wantErr)
			cnt     int
			wantStr string
			wantErr error
		}
		testDiscard struct { // Discard(cnt) == (wantCnt, wantErr)
			cnt     int64
			wantCnt int64
			wantErr error
		}
		testRemaining struct { // Remaining() == wantCnt
			wantCnt int64
		}
		testFnc interface{} // testRead | testDiscard | testRemaining
	)

	makeReg := func(s string, n int) fileReader {
		return &regFileReader{strings.NewReader(s), int64(n)}
	}
	makeSparse := func(fr fileReader, spd sparseDatas, size int64) fileReader {
		if !validateSparseEntries(spd, size) {
			t.Fatalf("invalid sparse map: %v", spd)
		}
		sph := invertSparseEntries(append([]SparseEntry{}, spd...), size)
		return &sparseFileReader{fr, sph, 0}
	}

	vectors := []struct {
		fr    fileReader
		tests []testFnc
	}{{
		fr: makeReg("", 0),
		tests: []testFnc{
			testRemaining{0},
			testRead{0, "", io.EOF},
			testRead{1, "", io.EOF},
			testDiscard{0, 0, nil},
			testDiscard{1, 0, io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeReg("", 1),
		tests: []testFnc{
			testRemaining{1},
			testRead{0, "", io.ErrUnexpectedEOF},
			testRead{5, "", io.ErrUnexpectedEOF},
			testDiscard{0, 0, nil},
			testDiscard{1, 0, io.ErrUnexpectedEOF},
			testRemaining{1},
		},
	}, {
		fr: makeReg("hello", 5),
		tests: []testFnc{
			testRemaining{5},
			testRead{5, "hello", io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeReg("hello, world", 50),
		tests: []testFnc{
			testRemaining{50},
			testDiscard{7, 7, nil},
			testRemaining{43},
			testRead{5, "world", nil},
			testRemaining{38},
			testDiscard{1, 0, io.ErrUnexpectedEOF},
			testRead{1, "", io.ErrUnexpectedEOF},
			testRemaining{38},
		},
	}, {
		fr: makeReg("hello, world", 5),
		tests: []testFnc{
			testRemaining{5},
			testRead{0, "", nil},
			testRead{4, "hell", nil},
			testRemaining{1},
			testDiscard{5, 1, io.EOF},
			testRemaining{0},
			testDiscard{5, 0, io.EOF},
			testRead{0, "", io.EOF},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{0, 2}, {5, 3}}, 8),
		tests: []testFnc{
			testRemaining{8},
			testRead{3, "ab\x00", nil},
			testRead{10, "\x00\x00cde", io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{0, 2}, {5, 3}}, 8),
		tests: []testFnc{
			testRemaining{8},
			testDiscard{100, 8, io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{0, 2}, {5, 3}}, 10),
		tests: []testFnc{
			testRemaining{10},
			testRead{100, "ab\x00\x00\x00cde\x00\x00", io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeSparse(makeReg("abc", 5), sparseDatas{{0, 2}, {5, 3}}, 10),
		tests: []testFnc{
			testRemaining{10},
			testRead{100, "ab\x00\x00\x00c", io.ErrUnexpectedEOF},
			testRemaining{4},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{1, 3}, {6, 2}}, 8),
		tests: []testFnc{
			testRemaining{8},
			testRead{8, "\x00abc\x00\x00de", io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{1, 3}, {6, 0}, {6, 0}, {6, 2}}, 8),
		tests: []testFnc{
			testRemaining{8},
			testRead{8, "\x00abc\x00\x00de", io.EOF},
			testRemaining{0},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{1, 3}, {6, 2}}, 10),
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de\x00\x00", io.EOF},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{1, 3}, {6, 2}, {8, 0}, {8, 0}, {8, 0}, {8, 0}}, 10),
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de\x00\x00", io.EOF},
		},
	}, {
		fr: makeSparse(makeReg("", 0), sparseDatas{}, 2),
		tests: []testFnc{
			testRead{100, "\x00\x00", io.EOF},
		},
	}, {
		fr: makeSparse(makeReg("", 8), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00", io.ErrUnexpectedEOF},
		},
	}, {
		fr: makeSparse(makeReg("ab", 2), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00ab", errMissData},
		},
	}, {
		fr: makeSparse(makeReg("ab", 8), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00ab", io.ErrUnexpectedEOF},
		},
	}, {
		fr: makeSparse(makeReg("abc", 3), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00", errMissData},
		},
	}, {
		fr: makeSparse(makeReg("abc", 8), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00", io.ErrUnexpectedEOF},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 5), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de", errMissData},
		},
	}, {
		fr: makeSparse(makeReg("abcde", 8), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de", io.ErrUnexpectedEOF},
		},
	}, {
		fr: makeSparse(makeReg("abcdefghEXTRA", 13), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRemaining{15},
			testRead{100, "\x00abc\x00\x00defgh\x00\x00\x00\x00", errUnrefData},
			testDiscard{100, 0, errUnrefData},
			testRemaining{0},
		},
	}, {
		fr: makeSparse(makeReg("abcdefghEXTRA", 13), sparseDatas{{1, 3}, {6, 5}}, 15),
		tests: []testFnc{
			testRemaining{15},
			testDiscard{100, 15, errUnrefData},
			testRead{100, "", errUnrefData},
			testRemaining{0},
		},
	}}

	for i, v := range vectors {
		for j, tf := range v.tests {
			switch tf := tf.(type) {
			case testRead:
				b := make([]byte, tf.cnt)
				n, err := v.fr.Read(b)
				if got := string(b[:n]); got != tf.wantStr || err != tf.wantErr {
					t.Errorf("test %d.%d, Read(%d):\ngot  (%q, %v)\nwant (%q, %v)", i, j, tf.cnt, got, err, tf.wantStr, tf.wantErr)
				}
			case testDiscard:
				got, err := v.fr.Discard(tf.cnt)
				if got != tf.wantCnt || err != tf.wantErr {
					t.Errorf("test %d.%d, Discard(%d) = (%d, %v), want (%d, %v)", i, j, tf.cnt, got, err, tf.wantCnt, tf.wantErr)
				}
			case testRemaining:
				got := v.fr.Remaining()
				if got != tf.wantCnt {
					t.Errorf("test %d.%d, Remaining() = %d, want %d", i, j, got, tf.wantCnt)
				}
			default:
				t.Fatalf("test %d.%d, unknown test operation: %T", i, j, tf)
			}
		}
	}
}
