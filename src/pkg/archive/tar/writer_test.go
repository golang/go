// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tar

import (
	"archive/tar";
	"bytes";
	"fmt";
	"io";
	"os";
	"reflect";
	"strings";
	"testing";
)

type writerTestEntry struct {
	header *Header;
	contents string;
}

type writerTest struct {
	file string;  // filename of expected output
	entries []*writerTestEntry;
}

var writerTests = []*writerTest{
	&writerTest{
		file: "testdata/writer.tar",
		entries: []*writerTestEntry{
			&writerTestEntry{
				header: &Header{
					Name: "small.txt",
					Mode: 0640,
					Uid: 73025,
					Gid: 5000,
					Size: 5,
					Mtime: 1246508266,
					Typeflag: '0',
					Uname: "dsymonds",
					Gname: "eng",
				},
				contents: `Kilts`,
			},
			&writerTestEntry{
				header: &Header{
					Name: "small2.txt",
					Mode: 0640,
					Uid: 73025,
					Gid: 5000,
					Size: 11,
					Mtime: 1245217492,
					Typeflag: '0',
					Uname: "dsymonds",
					Gname: "eng",
				},
				contents: "Google.com\n",
			},
		}
	},
}

// Render byte array in a two-character hexadecimal string, spaced for easy visual inspection.
func bytestr(b []byte) string {
	s := fmt.Sprintf("(%d bytes)\n", len(b));
	const rowLen = 32;
	for i, ch := range b {
		if i % rowLen == 0 {
			// start of line: hex offset
			s += fmt.Sprintf("%04x", i);
		}
		switch {
		case '0' <= ch && ch <= '9', 'A' <= ch && ch <= 'Z', 'a' <= ch && ch <= 'z':
			s += fmt.Sprintf("  %c", ch);
		default:
			s += fmt.Sprintf(" %02x", ch);
		}
		if (i + 1) % rowLen == 0 {
			// end of line
			s += "\n";
		} else if (i + 1) % (rowLen / 2) == 0 {
			// extra space
			s += " ";
		}
	}
	if s[len(s)-1] != '\n' {
		s += "\n"
	}
	return s
}

func TestWriter(t *testing.T) {
testLoop:
	for i, test := range writerTests {
		expected, err := io.ReadFile(test.file);
		if err != nil {
			t.Errorf("test %d: Unexpected error: %v", i, err);
			continue
		}

		buf := new(bytes.Buffer);
		tw := NewWriter(buf);
		for j, entry := range test.entries {
			if err := tw.WriteHeader(entry.header); err != nil {
				t.Errorf("test %d, entry %d: Failed writing header: %v", i, j, err);
				continue testLoop
			}
			if n, err := io.WriteString(tw, entry.contents); err != nil {
				t.Errorf("test %d, entry %d: Failed writing contents: %v", i, j, err);
				continue testLoop
			}
		}
		tw.Close();

		actual := buf.Data();
		if !bytes.Equal(expected, actual) {
			t.Errorf("test %d: Incorrect result:\n%v\nwant:\n%v",
				 i, bytestr(actual), bytestr(expected));
		}
	}
}
