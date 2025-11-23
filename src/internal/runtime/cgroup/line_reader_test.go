// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgroup_test

import (
	"internal/runtime/cgroup"
	"io"
	"strings"
	"testing"
)

type nextLine struct {
	line       string
	incomplete bool // next call before this line should return incomplete
}

func complete(s string) nextLine {
	return nextLine{line: s}
}
func incomplete(s string) nextLine {
	return nextLine{line: s, incomplete: true}
}

const scratchSize = 8

var readerTests = []struct {
	name     string
	contents string
	want     []nextLine
}{
	{
		name:     "empty",
		contents: "",
	},
	{
		name:     "single",
		contents: "1234\n",
		want: []nextLine{
			complete("1234"),
		},
	},
	{
		name:     "single-incomplete",
		contents: "1234",
		want: []nextLine{
			incomplete("1234"),
		},
	},
	{
		name:     "single-exact",
		contents: "1234567\n",
		want: []nextLine{
			complete("1234567"),
		},
	},
	{
		name:     "single-exact-incomplete",
		contents: "12345678",
		want: []nextLine{
			incomplete("12345678"),
		},
	},
	{
		name: "multi",
		contents: `1234
5678
`,
		want: []nextLine{
			complete("1234"),
			complete("5678"),
		},
	},
	{
		name: "multi-short",
		contents: `12
34
56
78
`,
		want: []nextLine{
			complete("12"),
			complete("34"),
			complete("56"),
			complete("78"),
		},
	},
	{
		name: "multi-notrailingnewline",
		contents: `1234
5678`,
		want: []nextLine{
			complete("1234"),
			incomplete("5678"),
		},
	},
	{
		name: "middle-too-long",
		contents: `1234
1234567890
5678
`,
		want: []nextLine{
			complete("1234"),
			incomplete("12345678"),
			complete("5678"),
		},
	},
	{
		// Multiple reads required to find newline.
		name: "middle-way-too-long",
		contents: `1234
12345678900000000000000000000000000000000000000000000000000
5678
`,
		want: []nextLine{
			complete("1234"),
			incomplete("12345678"),
			complete("5678"),
		},
	},
}

func readString(contents string) func(fd int, b []byte) (int, uintptr) {
	r := strings.NewReader(contents)
	return func(fd int, b []byte) (int, uintptr) {
		n, err := r.Read(b)
		if err != nil && err != io.EOF {
			const dummyErrno = 42
			return n, dummyErrno
		}
		return n, 0
	}
}

func TestLineReader(t *testing.T) {
	for _, tc := range readerTests {
		t.Run(tc.name, func(t *testing.T) {
			var scratch [scratchSize]byte
			l := cgroup.NewLineReader(0, scratch[:], readString(tc.contents))

			var got []nextLine
			for {
				err := l.Next()
				if err == cgroup.ErrEOF {
					break
				} else if err == cgroup.ErrIncompleteLine {
					got = append(got, incomplete(string(l.Line())))
				} else if err != nil {
					t.Fatalf("next got err %v", err)
				} else {
					got = append(got, complete(string(l.Line())))
				}
			}

			if len(got) != len(tc.want) {
				t.Logf("got lines %+v", got)
				t.Logf("want lines %+v", tc.want)
				t.Fatalf("lineReader got %d lines, want %d", len(got), len(tc.want))
			}

			for i := range got {
				if got[i].line != tc.want[i].line {
					t.Errorf("line %d got %q want %q", i, got[i].line, tc.want[i].line)
				}
				if got[i].incomplete != tc.want[i].incomplete {
					t.Errorf("line %d got incomplete %v want %v", i, got[i].incomplete, tc.want[i].incomplete)
				}
			}
		})
	}
}

func FuzzLineReader(f *testing.F) {
	for _, tc := range readerTests {
		f.Add(tc.contents)
	}
	f.Fuzz(func(t *testing.T, input string) {
		scratch := make([]byte, scratchSize)
		reader := cgroup.NewLineReader(0, scratch, readString(input))
		for expected := range strings.Lines(input) {
			err := reader.Next()
			line := reader.Line()

			var expectedErr error
			if len(expected) > scratchSize {
				expected = expected[:scratchSize]
				expectedErr = cgroup.ErrIncompleteLine
			} else if expected[len(expected)-1] == '\n' {
				expected = expected[:len(expected)-1]
			} else {
				expectedErr = cgroup.ErrIncompleteLine
			}

			if err != expectedErr {
				t.Fatalf("got err %v, want %v", err, expectedErr)
			}

			if string(line) != expected {
				t.Fatalf("got %q, want %q", string(line), expected)
			}
		}
		err := reader.Next()
		if err != cgroup.ErrEOF {
			t.Fatalf("got %v, want EOF", err)
		}
	})
}
