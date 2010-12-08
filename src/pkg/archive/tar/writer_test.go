// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"testing"
	"testing/iotest"
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
	&writerTest{
		file: "testdata/writer.tar",
		entries: []*writerTestEntry{
			&writerTestEntry{
				header: &Header{
					Name:     "small.txt",
					Mode:     0640,
					Uid:      73025,
					Gid:      5000,
					Size:     5,
					Mtime:    1246508266,
					Typeflag: '0',
					Uname:    "dsymonds",
					Gname:    "eng",
				},
				contents: "Kilts",
			},
			&writerTestEntry{
				header: &Header{
					Name:     "small2.txt",
					Mode:     0640,
					Uid:      73025,
					Gid:      5000,
					Size:     11,
					Mtime:    1245217492,
					Typeflag: '0',
					Uname:    "dsymonds",
					Gname:    "eng",
				},
				contents: "Google.com\n",
			},
		},
	},
	// The truncated test file was produced using these commands:
	//   dd if=/dev/zero bs=1048576 count=16384 > /tmp/16gig.txt
	//   tar -b 1 -c -f- /tmp/16gig.txt | dd bs=512 count=8 > writer-big.tar
	&writerTest{
		file: "testdata/writer-big.tar",
		entries: []*writerTestEntry{
			&writerTestEntry{
				header: &Header{
					Name:     "tmp/16gig.txt",
					Mode:     0640,
					Uid:      73025,
					Gid:      5000,
					Size:     16 << 30,
					Mtime:    1254699560,
					Typeflag: '0',
					Uname:    "dsymonds",
					Gname:    "eng",
				},
				// no contents
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
		for j, entry := range test.entries {
			if err := tw.WriteHeader(entry.header); err != nil {
				t.Errorf("test %d, entry %d: Failed writing header: %v", i, j, err)
				continue testLoop
			}
			if _, err := io.WriteString(tw, entry.contents); err != nil {
				t.Errorf("test %d, entry %d: Failed writing contents: %v", i, j, err)
				continue testLoop
			}
		}
		if err := tw.Close(); err != nil {
			t.Errorf("test %d: Failed closing archive: %v", i, err)
			continue testLoop
		}

		actual := buf.Bytes()
		if !bytes.Equal(expected, actual) {
			t.Errorf("test %d: Incorrect result: (-=expected, +=actual)\n%v",
				i, bytediff(expected, actual))
		}
	}
}
