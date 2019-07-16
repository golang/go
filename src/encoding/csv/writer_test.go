// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csv

import (
	"bytes"
	"errors"
	"testing"
)

var writeTests = []struct {
	Input   [][]string
	Output  string
	Error   error
	UseCRLF bool
	Comma   rune
}{
	{Input: [][]string{{"abc"}}, Output: "abc\n"},
	{Input: [][]string{{"abc"}}, Output: "abc\r\n", UseCRLF: true},
	{Input: [][]string{{`"abc"`}}, Output: `"""abc"""` + "\n"},
	{Input: [][]string{{`a"b`}}, Output: `"a""b"` + "\n"},
	{Input: [][]string{{`"a"b"`}}, Output: `"""a""b"""` + "\n"},
	{Input: [][]string{{" abc"}}, Output: `" abc"` + "\n"},
	{Input: [][]string{{"abc,def"}}, Output: `"abc,def"` + "\n"},
	{Input: [][]string{{"abc", "def"}}, Output: "abc,def\n"},
	{Input: [][]string{{"abc"}, {"def"}}, Output: "abc\ndef\n"},
	{Input: [][]string{{"abc\ndef"}}, Output: "\"abc\ndef\"\n"},
	{Input: [][]string{{"abc\ndef"}}, Output: "\"abc\r\ndef\"\r\n", UseCRLF: true},
	{Input: [][]string{{"abc\rdef"}}, Output: "\"abcdef\"\r\n", UseCRLF: true},
	{Input: [][]string{{"abc\rdef"}}, Output: "\"abc\rdef\"\n", UseCRLF: false},
	{Input: [][]string{{""}}, Output: "\n"},
	{Input: [][]string{{"", ""}}, Output: ",\n"},
	{Input: [][]string{{"", "", ""}}, Output: ",,\n"},
	{Input: [][]string{{"", "", "a"}}, Output: ",,a\n"},
	{Input: [][]string{{"", "a", ""}}, Output: ",a,\n"},
	{Input: [][]string{{"", "a", "a"}}, Output: ",a,a\n"},
	{Input: [][]string{{"a", "", ""}}, Output: "a,,\n"},
	{Input: [][]string{{"a", "", "a"}}, Output: "a,,a\n"},
	{Input: [][]string{{"a", "a", ""}}, Output: "a,a,\n"},
	{Input: [][]string{{"a", "a", "a"}}, Output: "a,a,a\n"},
	{Input: [][]string{{`\.`}}, Output: "\"\\.\"\n"},
	{Input: [][]string{{"x09\x41\xb4\x1c", "aktau"}}, Output: "x09\x41\xb4\x1c,aktau\n"},
	{Input: [][]string{{",x09\x41\xb4\x1c", "aktau"}}, Output: "\",x09\x41\xb4\x1c\",aktau\n"},
	{Input: [][]string{{"a", "a", ""}}, Output: "a|a|\n", Comma: '|'},
	{Input: [][]string{{",", ",", ""}}, Output: ",|,|\n", Comma: '|'},
	{Input: [][]string{{"foo"}}, Comma: '"', Error: errInvalidDelim},
}

func TestWrite(t *testing.T) {
	for n, tt := range writeTests {
		b := &bytes.Buffer{}
		f := NewWriter(b)
		f.UseCRLF = tt.UseCRLF
		if tt.Comma != 0 {
			f.Comma = tt.Comma
		}
		err := f.WriteAll(tt.Input)
		if err != tt.Error {
			t.Errorf("Unexpected error:\ngot  %v\nwant %v", err, tt.Error)
		}
		out := b.String()
		if out != tt.Output {
			t.Errorf("#%d: out=%q want %q", n, out, tt.Output)
		}
	}
}

type errorWriter struct{}

func (e errorWriter) Write(b []byte) (int, error) {
	return 0, errors.New("Test")
}

func TestError(t *testing.T) {
	b := &bytes.Buffer{}
	f := NewWriter(b)
	f.Write([]string{"abc"})
	f.Flush()
	err := f.Error()

	if err != nil {
		t.Errorf("Unexpected error: %s\n", err)
	}

	f = NewWriter(errorWriter{})
	f.Write([]string{"abc"})
	f.Flush()
	err = f.Error()

	if err == nil {
		t.Error("Error should not be nil")
	}
}
