// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"crypto/md5"
	"errors"
	"fmt"
	"io"
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
			Format:   FormatGNU,
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
			Format:   FormatGNU,
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
			Format:   FormatGNU,
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
			PAXRecords: map[string]string{
				"GNU.sparse.size":      "200",
				"GNU.sparse.numblocks": "95",
				"GNU.sparse.map":       "1,1,3,1,5,1,7,1,9,1,11,1,13,1,15,1,17,1,19,1,21,1,23,1,25,1,27,1,29,1,31,1,33,1,35,1,37,1,39,1,41,1,43,1,45,1,47,1,49,1,51,1,53,1,55,1,57,1,59,1,61,1,63,1,65,1,67,1,69,1,71,1,73,1,75,1,77,1,79,1,81,1,83,1,85,1,87,1,89,1,91,1,93,1,95,1,97,1,99,1,101,1,103,1,105,1,107,1,109,1,111,1,113,1,115,1,117,1,119,1,121,1,123,1,125,1,127,1,129,1,131,1,133,1,135,1,137,1,139,1,141,1,143,1,145,1,147,1,149,1,151,1,153,1,155,1,157,1,159,1,161,1,163,1,165,1,167,1,169,1,171,1,173,1,175,1,177,1,179,1,181,1,183,1,185,1,187,1,189,1",
			},
			Format: FormatPAX,
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
			PAXRecords: map[string]string{
				"GNU.sparse.size":      "200",
				"GNU.sparse.numblocks": "95",
				"GNU.sparse.map":       "1,1,3,1,5,1,7,1,9,1,11,1,13,1,15,1,17,1,19,1,21,1,23,1,25,1,27,1,29,1,31,1,33,1,35,1,37,1,39,1,41,1,43,1,45,1,47,1,49,1,51,1,53,1,55,1,57,1,59,1,61,1,63,1,65,1,67,1,69,1,71,1,73,1,75,1,77,1,79,1,81,1,83,1,85,1,87,1,89,1,91,1,93,1,95,1,97,1,99,1,101,1,103,1,105,1,107,1,109,1,111,1,113,1,115,1,117,1,119,1,121,1,123,1,125,1,127,1,129,1,131,1,133,1,135,1,137,1,139,1,141,1,143,1,145,1,147,1,149,1,151,1,153,1,155,1,157,1,159,1,161,1,163,1,165,1,167,1,169,1,171,1,173,1,175,1,177,1,179,1,181,1,183,1,185,1,187,1,189,1",
				"GNU.sparse.name":      "sparse-posix-0.1",
			},
			Format: FormatPAX,
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
			PAXRecords: map[string]string{
				"GNU.sparse.major":    "1",
				"GNU.sparse.minor":    "0",
				"GNU.sparse.realsize": "200",
				"GNU.sparse.name":     "sparse-posix-1.0",
			},
			Format: FormatPAX,
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
			Format:   FormatGNU,
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
			Typeflag: '0',
		}, {
			Name:     "small2.txt",
			Mode:     0444,
			Uid:      73025,
			Gid:      5000,
			Size:     11,
			ModTime:  time.Unix(1244593104, 0),
			Typeflag: '0',
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
			PAXRecords: map[string]string{
				"path":  "a/123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100",
				"mtime": "1350244992.023960108",
				"atime": "1350244992.023960108",
				"ctime": "1350244992.023960108",
			},
			Format: FormatPAX,
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
			PAXRecords: map[string]string{
				"linkpath": "123456789101112131415161718192021222324252627282930313233343536373839404142434445464748495051525354555657585960616263646566676869707172737475767778798081828384858687888990919293949596979899100",
				"mtime":    "1350266320.910238425",
				"atime":    "1350266320.910238425",
				"ctime":    "1350266320.910238425",
			},
			Format: FormatPAX,
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
			PAXRecords: map[string]string{
				"size": "000000000000000000000999",
			},
			Format: FormatPAX,
		}},
		chksums: []string{
			"0afb597b283fe61b5d4879669a350556",
		},
	}, {
		file: "testdata/pax-records.tar",
		headers: []*Header{{
			Typeflag: TypeReg,
			Name:     "file",
			Uname:    strings.Repeat("long", 10),
			ModTime:  time.Unix(0, 0),
			PAXRecords: map[string]string{
				"GOLANG.pkg": "tar",
				"comment":    "Hello, 世界",
				"uname":      strings.Repeat("long", 10),
			},
			Format: FormatPAX,
		}},
	}, {
		file: "testdata/pax-global-records.tar",
		headers: []*Header{{
			Typeflag:   TypeXGlobalHeader,
			Name:       "global1",
			PAXRecords: map[string]string{"path": "global1", "mtime": "1500000000.0"},
			Format:     FormatPAX,
		}, {
			Typeflag: TypeReg,
			Name:     "file1",
			ModTime:  time.Unix(0, 0),
			Format:   FormatUSTAR,
		}, {
			Typeflag:   TypeReg,
			Name:       "file2",
			PAXRecords: map[string]string{"path": "file2"},
			ModTime:    time.Unix(0, 0),
			Format:     FormatPAX,
		}, {
			Typeflag:   TypeXGlobalHeader,
			Name:       "GlobalHead.0.0",
			PAXRecords: map[string]string{"path": ""},
			Format:     FormatPAX,
		}, {
			Typeflag: TypeReg,
			Name:     "file3",
			ModTime:  time.Unix(0, 0),
			Format:   FormatUSTAR,
		}, {
			Typeflag:   TypeReg,
			Name:       "file4",
			ModTime:    time.Unix(1400000000, 0),
			PAXRecords: map[string]string{"mtime": "1400000000"},
			Format:     FormatPAX,
		}},
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
			Format:   FormatGNU,
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
			PAXRecords: map[string]string{
				"mtime":                         "1386065770.44825232",
				"atime":                         "1389782991.41987522",
				"ctime":                         "1389782956.794414986",
				"SCHILY.xattr.user.key":         "value",
				"SCHILY.xattr.user.key2":        "value2",
				"SCHILY.xattr.security.selinux": "unconfined_u:object_r:default_t:s0\x00",
			},
			Format: FormatPAX,
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
			PAXRecords: map[string]string{
				"mtime":                         "1386065770.449252304",
				"atime":                         "1389782991.41987522",
				"ctime":                         "1386065770.449252304",
				"SCHILY.xattr.security.selinux": "unconfined_u:object_r:default_t:s0\x00",
			},
			Format: FormatPAX,
		}},
	}, {
		// Matches the behavior of GNU, BSD, and STAR tar utilities.
		file: "testdata/gnu-multi-hdrs.tar",
		headers: []*Header{{
			Name:     "GNU2/GNU2/long-path-name",
			Linkname: "GNU4/GNU4/long-linkpath-name",
			ModTime:  time.Unix(0, 0),
			Typeflag: '2',
			Format:   FormatGNU,
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
			Format:     FormatGNU,
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
			Format:     FormatGNU,
		}, {
			Name:       "test2/sparse",
			Mode:       33188,
			Uid:        1000,
			Gid:        1000,
			Size:       536870912,
			ModTime:    time.Unix(1441973427, 0),
			Typeflag:   'S',
			Uname:      "rawr",
			Gname:      "dsnet",
			AccessTime: time.Unix(1441991948, 0),
			ChangeTime: time.Unix(1441973436, 0),
			Format:     FormatGNU,
		}},
	}, {
		// Matches the behavior of GNU and BSD tar utilities.
		file: "testdata/pax-multi-hdrs.tar",
		headers: []*Header{{
			Name:     "bar",
			Linkname: "PAX4/PAX4/long-linkpath-name",
			ModTime:  time.Unix(0, 0),
			Typeflag: '2',
			PAXRecords: map[string]string{
				"linkpath": "PAX4/PAX4/long-linkpath-name",
			},
			Format: FormatPAX,
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
			Format:   FormatGNU,
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
			Format:   FormatGNU,
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
			Format:   FormatGNU,
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
			Name:     "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/foo",
			Uid:      010000000,
			ModTime:  time.Unix(0, 0),
			Typeflag: '0',
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
			Format:   FormatUSTAR,
		}},
	}, {
		// Generated by Go, works on BSD tar v3.1.2 and GNU tar v.1.27.1.
		file: "testdata/gnu-nil-sparse-data.tar",
		headers: []*Header{{
			Name:     "sparse.db",
			Typeflag: TypeGNUSparse,
			Size:     1000,
			ModTime:  time.Unix(0, 0),
			Format:   FormatGNU,
		}},
	}, {
		// Generated by Go, works on BSD tar v3.1.2 and GNU tar v.1.27.1.
		file: "testdata/gnu-nil-sparse-hole.tar",
		headers: []*Header{{
			Name:     "sparse.db",
			Typeflag: TypeGNUSparse,
			Size:     1000,
			ModTime:  time.Unix(0, 0),
			Format:   FormatGNU,
		}},
	}, {
		// Generated by Go, works on BSD tar v3.1.2 and GNU tar v.1.27.1.
		file: "testdata/pax-nil-sparse-data.tar",
		headers: []*Header{{
			Name:     "sparse.db",
			Typeflag: TypeReg,
			Size:     1000,
			ModTime:  time.Unix(0, 0),
			PAXRecords: map[string]string{
				"size":                "1512",
				"GNU.sparse.major":    "1",
				"GNU.sparse.minor":    "0",
				"GNU.sparse.realsize": "1000",
				"GNU.sparse.name":     "sparse.db",
			},
			Format: FormatPAX,
		}},
	}, {
		// Generated by Go, works on BSD tar v3.1.2 and GNU tar v.1.27.1.
		file: "testdata/pax-nil-sparse-hole.tar",
		headers: []*Header{{
			Name:     "sparse.db",
			Typeflag: TypeReg,
			Size:     1000,
			ModTime:  time.Unix(0, 0),
			PAXRecords: map[string]string{
				"size":                "512",
				"GNU.sparse.major":    "1",
				"GNU.sparse.minor":    "0",
				"GNU.sparse.realsize": "1000",
				"GNU.sparse.name":     "sparse.db",
			},
			Format: FormatPAX,
		}},
	}, {
		file: "testdata/trailing-slash.tar",
		headers: []*Header{{
			Typeflag: TypeDir,
			Name:     strings.Repeat("123456789/", 30),
			ModTime:  time.Unix(0, 0),
			PAXRecords: map[string]string{
				"path": strings.Repeat("123456789/", 30),
			},
			Format: FormatPAX,
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
		buf, err := os.ReadFile(p)
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
					if _, err = tr.writeTo(io.Discard); err != nil {
						break
					}
				}
			}
			if err != v.err {
				t.Errorf("test %d, NewReader(%s) with %s discard: got %v, want %v",
					i, s1, s2, err, v.err)
			}
			if cnt != v.cnt {
				t.Errorf("test %d, NewReader(%s) with %s discard: got %d headers, want %d headers",
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
			PAXRecords: map[string]string{
				"path":  "a/b/c",
				"uid":   "1000",
				"mtime": "1350244992.023960108",
			},
		},
		ok: true,
	}, {
		in: map[string]string{
			"gid": "gtgergergersagersgers",
		},
		ok: false,
	}, {
		in: map[string]string{
			"missing":          "missing",
			"SCHILY.xattr.key": "value",
		},
		want: &Header{
			Xattrs: map[string]string{"key": "value"},
			PAXRecords: map[string]string{
				"missing":          "missing",
				"SCHILY.xattr.key": "value",
			},
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
			map[string]string{"key1": "", "key2": "val2"}, true},
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
			t.Errorf("test %d, parsePAX():\ngot  %v\nwant %v", i, got, v.want)
		}
		if ok := err == nil; ok != v.ok {
			t.Errorf("test %d, parsePAX(): got %v, want %v", i, ok, v.ok)
		}
	}
}

func TestReadOldGNUSparseMap(t *testing.T) {
	populateSparseMap := func(sa sparseArray, sps []string) []string {
		for i := 0; len(sps) > 0 && i < sa.maxEntries(); i++ {
			copy(sa.entry(i), sps[0])
			sps = sps[1:]
		}
		if len(sps) > 0 {
			copy(sa.isExtended(), "\x80")
		}
		return sps
	}

	makeInput := func(format Format, size string, sps ...string) (out []byte) {
		// Write the initial GNU header.
		var blk block
		gnu := blk.toGNU()
		sparse := gnu.sparse()
		copy(gnu.realSize(), size)
		sps = populateSparseMap(sparse, sps)
		if format != FormatUnknown {
			blk.setFormat(format)
		}
		out = append(out, blk[:]...)

		// Write extended sparse blocks.
		for len(sps) > 0 {
			var blk block
			sps = populateSparseMap(blk.toSparse(), sps)
			out = append(out, blk[:]...)
		}
		return out
	}

	makeSparseStrings := func(sp []sparseEntry) (out []string) {
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
		input:   makeInput(FormatUnknown, ""),
		wantErr: ErrHeader,
	}, {
		input:    makeInput(FormatGNU, "1234", "fewa"),
		wantSize: 01234,
		wantErr:  ErrHeader,
	}, {
		input:    makeInput(FormatGNU, "0031"),
		wantSize: 031,
	}, {
		input:   makeInput(FormatGNU, "80"),
		wantErr: ErrHeader,
	}, {
		input: makeInput(FormatGNU, "1234",
			makeSparseStrings(sparseDatas{{0, 0}, {1, 1}})...),
		wantMap:  sparseDatas{{0, 0}, {1, 1}},
		wantSize: 01234,
	}, {
		input: makeInput(FormatGNU, "1234",
			append(makeSparseStrings(sparseDatas{{0, 0}, {1, 1}}), []string{"", "blah"}...)...),
		wantMap:  sparseDatas{{0, 0}, {1, 1}},
		wantSize: 01234,
	}, {
		input: makeInput(FormatGNU, "3333",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}})...),
		wantMap:  sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}},
		wantSize: 03333,
	}, {
		input: makeInput(FormatGNU, "",
			append(append(
				makeSparseStrings(sparseDatas{{0, 1}, {2, 1}}),
				[]string{"", ""}...),
				makeSparseStrings(sparseDatas{{4, 1}, {6, 1}})...)...),
		wantMap: sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}},
	}, {
		input: makeInput(FormatGNU, "",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}})...)[:blockSize],
		wantErr: io.ErrUnexpectedEOF,
	}, {
		input: makeInput(FormatGNU, "",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}})...)[:3*blockSize/2],
		wantErr: io.ErrUnexpectedEOF,
	}, {
		input: makeInput(FormatGNU, "",
			makeSparseStrings(sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}})...),
		wantMap: sparseDatas{{0, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {10, 1}},
	}, {
		input: makeInput(FormatGNU, "",
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
				spd = append(spd, sparseEntry{int64(i) << 30, 512})
			}
			return spd
		}(),
	}}

	for i, v := range vectors {
		var hdr Header
		hdr.PAXRecords = v.inputHdrs
		r := strings.NewReader(v.inputData + "#") // Add canary byte
		tr := Reader{curr: &regFileReader{r, int64(r.Len())}}
		got, err := tr.readGNUSparsePAXHeaders(&hdr)
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

// testNonEmptyReader wraps an io.Reader and ensures that
// Read is never called with an empty buffer.
type testNonEmptyReader struct{ io.Reader }

func (r testNonEmptyReader) Read(b []byte) (int, error) {
	if len(b) == 0 {
		return 0, errors.New("unexpected empty Read call")
	}
	return r.Reader.Read(b)
}

func TestFileReader(t *testing.T) {
	type (
		testRead struct { // Read(cnt) == (wantStr, wantErr)
			cnt     int
			wantStr string
			wantErr error
		}
		testWriteTo struct { // WriteTo(testFile{ops}) == (wantCnt, wantErr)
			ops     fileOps
			wantCnt int64
			wantErr error
		}
		testRemaining struct { // logicalRemaining() == wantLCnt, physicalRemaining() == wantPCnt
			wantLCnt int64
			wantPCnt int64
		}
		testFnc any // testRead | testWriteTo | testRemaining
	)

	type (
		makeReg struct {
			str  string
			size int64
		}
		makeSparse struct {
			makeReg makeReg
			spd     sparseDatas
			size    int64
		}
		fileMaker any // makeReg | makeSparse
	)

	vectors := []struct {
		maker fileMaker
		tests []testFnc
	}{{
		maker: makeReg{"", 0},
		tests: []testFnc{
			testRemaining{0, 0},
			testRead{0, "", io.EOF},
			testRead{1, "", io.EOF},
			testWriteTo{nil, 0, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{"", 1},
		tests: []testFnc{
			testRemaining{1, 1},
			testRead{5, "", io.ErrUnexpectedEOF},
			testWriteTo{nil, 0, io.ErrUnexpectedEOF},
			testRemaining{1, 1},
		},
	}, {
		maker: makeReg{"hello", 5},
		tests: []testFnc{
			testRemaining{5, 5},
			testRead{5, "hello", io.EOF},
			testRemaining{0, 0},
		},
	}, {
		maker: makeReg{"hello, world", 50},
		tests: []testFnc{
			testRemaining{50, 50},
			testRead{7, "hello, ", nil},
			testRemaining{43, 43},
			testRead{5, "world", nil},
			testRemaining{38, 38},
			testWriteTo{nil, 0, io.ErrUnexpectedEOF},
			testRead{1, "", io.ErrUnexpectedEOF},
			testRemaining{38, 38},
		},
	}, {
		maker: makeReg{"hello, world", 5},
		tests: []testFnc{
			testRemaining{5, 5},
			testRead{0, "", nil},
			testRead{4, "hell", nil},
			testRemaining{1, 1},
			testWriteTo{fileOps{"o"}, 1, nil},
			testRemaining{0, 0},
			testWriteTo{nil, 0, nil},
			testRead{0, "", io.EOF},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{0, 2}, {5, 3}}, 8},
		tests: []testFnc{
			testRemaining{8, 5},
			testRead{3, "ab\x00", nil},
			testRead{10, "\x00\x00cde", io.EOF},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{0, 2}, {5, 3}}, 8},
		tests: []testFnc{
			testRemaining{8, 5},
			testWriteTo{fileOps{"ab", int64(3), "cde"}, 8, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{0, 2}, {5, 3}}, 10},
		tests: []testFnc{
			testRemaining{10, 5},
			testRead{100, "ab\x00\x00\x00cde\x00\x00", io.EOF},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{"abc", 5}, sparseDatas{{0, 2}, {5, 3}}, 10},
		tests: []testFnc{
			testRemaining{10, 5},
			testRead{100, "ab\x00\x00\x00c", io.ErrUnexpectedEOF},
			testRemaining{4, 2},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 2}}, 8},
		tests: []testFnc{
			testRemaining{8, 5},
			testRead{8, "\x00abc\x00\x00de", io.EOF},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 0}, {6, 0}, {6, 2}}, 8},
		tests: []testFnc{
			testRemaining{8, 5},
			testRead{8, "\x00abc\x00\x00de", io.EOF},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 0}, {6, 0}, {6, 2}}, 8},
		tests: []testFnc{
			testRemaining{8, 5},
			testWriteTo{fileOps{int64(1), "abc", int64(2), "de"}, 8, nil},
			testRemaining{0, 0},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 2}}, 10},
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de\x00\x00", io.EOF},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 2}}, 10},
		tests: []testFnc{
			testWriteTo{fileOps{int64(1), "abc", int64(2), "de", int64(1), "\x00"}, 10, nil},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 2}, {8, 0}, {8, 0}, {8, 0}, {8, 0}}, 10},
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de\x00\x00", io.EOF},
		},
	}, {
		maker: makeSparse{makeReg{"", 0}, sparseDatas{}, 2},
		tests: []testFnc{
			testRead{100, "\x00\x00", io.EOF},
		},
	}, {
		maker: makeSparse{makeReg{"", 8}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00", io.ErrUnexpectedEOF},
		},
	}, {
		maker: makeSparse{makeReg{"ab", 2}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00ab", errMissData},
		},
	}, {
		maker: makeSparse{makeReg{"ab", 8}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00ab", io.ErrUnexpectedEOF},
		},
	}, {
		maker: makeSparse{makeReg{"abc", 3}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00", errMissData},
		},
	}, {
		maker: makeSparse{makeReg{"abc", 8}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00", io.ErrUnexpectedEOF},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de", errMissData},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 5}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testWriteTo{fileOps{int64(1), "abc", int64(2), "de"}, 8, errMissData},
		},
	}, {
		maker: makeSparse{makeReg{"abcde", 8}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRead{100, "\x00abc\x00\x00de", io.ErrUnexpectedEOF},
		},
	}, {
		maker: makeSparse{makeReg{"abcdefghEXTRA", 13}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRemaining{15, 13},
			testRead{100, "\x00abc\x00\x00defgh\x00\x00\x00\x00", errUnrefData},
			testWriteTo{nil, 0, errUnrefData},
			testRemaining{0, 5},
		},
	}, {
		maker: makeSparse{makeReg{"abcdefghEXTRA", 13}, sparseDatas{{1, 3}, {6, 5}}, 15},
		tests: []testFnc{
			testRemaining{15, 13},
			testWriteTo{fileOps{int64(1), "abc", int64(2), "defgh", int64(4)}, 15, errUnrefData},
			testRead{100, "", errUnrefData},
			testRemaining{0, 5},
		},
	}}

	for i, v := range vectors {
		var fr fileReader
		switch maker := v.maker.(type) {
		case makeReg:
			r := testNonEmptyReader{strings.NewReader(maker.str)}
			fr = &regFileReader{r, maker.size}
		case makeSparse:
			if !validateSparseEntries(maker.spd, maker.size) {
				t.Fatalf("invalid sparse map: %v", maker.spd)
			}
			sph := invertSparseEntries(maker.spd, maker.size)
			r := testNonEmptyReader{strings.NewReader(maker.makeReg.str)}
			fr = &regFileReader{r, maker.makeReg.size}
			fr = &sparseFileReader{fr, sph, 0}
		default:
			t.Fatalf("test %d, unknown make operation: %T", i, maker)
		}

		for j, tf := range v.tests {
			switch tf := tf.(type) {
			case testRead:
				b := make([]byte, tf.cnt)
				n, err := fr.Read(b)
				if got := string(b[:n]); got != tf.wantStr || err != tf.wantErr {
					t.Errorf("test %d.%d, Read(%d):\ngot  (%q, %v)\nwant (%q, %v)", i, j, tf.cnt, got, err, tf.wantStr, tf.wantErr)
				}
			case testWriteTo:
				f := &testFile{ops: tf.ops}
				got, err := fr.WriteTo(f)
				if _, ok := err.(testError); ok {
					t.Errorf("test %d.%d, WriteTo(): %v", i, j, err)
				} else if got != tf.wantCnt || err != tf.wantErr {
					t.Errorf("test %d.%d, WriteTo() = (%d, %v), want (%d, %v)", i, j, got, err, tf.wantCnt, tf.wantErr)
				}
				if len(f.ops) > 0 {
					t.Errorf("test %d.%d, expected %d more operations", i, j, len(f.ops))
				}
			case testRemaining:
				if got := fr.logicalRemaining(); got != tf.wantLCnt {
					t.Errorf("test %d.%d, logicalRemaining() = %d, want %d", i, j, got, tf.wantLCnt)
				}
				if got := fr.physicalRemaining(); got != tf.wantPCnt {
					t.Errorf("test %d.%d, physicalRemaining() = %d, want %d", i, j, got, tf.wantPCnt)
				}
			default:
				t.Fatalf("test %d.%d, unknown test operation: %T", i, j, tf)
			}
		}
	}
}
