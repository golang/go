// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"strings"
	"testing"
)

func TestHorizontalWhitespace(t *testing.T) {
	if !onlyHorizontalWhitespace([]byte(" \t")) {
		t.Error("expected pass")
	}
	if onlyHorizontalWhitespace([]byte("foo bar")) {
		t.Error("expected failure")
	}
}

func TestBoundaryLine(t *testing.T) {
	mr := NewReader(strings.NewReader(""), "myBoundary")
	if !mr.isBoundaryDelimiterLine([]byte("--myBoundary\r\n")) {
		t.Error("expected")
	}
	if !mr.isBoundaryDelimiterLine([]byte("--myBoundary \r\n")) {
		t.Error("expected")
	}
	if !mr.isBoundaryDelimiterLine([]byte("--myBoundary \n")) {
		t.Error("expected")
	}
	if mr.isBoundaryDelimiterLine([]byte("--myBoundary bogus \n")) {
		t.Error("expected fail")
	}
	if mr.isBoundaryDelimiterLine([]byte("--myBoundary bogus--")) {
		t.Error("expected fail")
	}
}

func escapeString(v string) string {
	bytes, _ := json.Marshal(v)
	return string(bytes)
}

func expectEq(t *testing.T, expected, actual, what string) {
	if expected == actual {
		return
	}
	t.Errorf("Unexpected value for %s; got %s (len %d) but expected: %s (len %d)",
		what, escapeString(actual), len(actual), escapeString(expected), len(expected))
}

func TestNameAccessors(t *testing.T) {
	tests := [...][3]string{
		{`form-data; name="foo"`, "foo", ""},
		{` form-data ; name=foo`, "foo", ""},
		{`FORM-DATA;name="foo"`, "foo", ""},
		{` FORM-DATA ; name="foo"`, "foo", ""},
		{` FORM-DATA ; name="foo"`, "foo", ""},
		{` FORM-DATA ; name=foo`, "foo", ""},
		{` FORM-DATA ; filename="foo.txt"; name=foo; baz=quux`, "foo", "foo.txt"},
		{` not-form-data ; filename="bar.txt"; name=foo; baz=quux`, "", "bar.txt"},
	}
	for i, test := range tests {
		p := &Part{Header: make(map[string][]string)}
		p.Header.Set("Content-Disposition", test[0])
		if g, e := p.FormName(), test[1]; g != e {
			t.Errorf("test %d: FormName() = %q; want %q", i, g, e)
		}
		if g, e := p.FileName(), test[2]; g != e {
			t.Errorf("test %d: FileName() = %q; want %q", i, g, e)
		}
	}
}

var longLine = strings.Repeat("\n\n\r\r\r\n\r\000", (1<<20)/8)

func testMultipartBody(sep string) string {
	testBody := `
This is a multi-part message.  This line is ignored.
--MyBoundary
Header1: value1
HEADER2: value2
foo-bar: baz

My value
The end.
--MyBoundary
name: bigsection

[longline]
--MyBoundary
Header1: value1b
HEADER2: value2b
foo-bar: bazb

Line 1
Line 2
Line 3 ends in a newline, but just one.

--MyBoundary

never read data
--MyBoundary--


useless trailer
`
	testBody = strings.Replace(testBody, "\n", sep, -1)
	return strings.Replace(testBody, "[longline]", longLine, 1)
}

func TestMultipart(t *testing.T) {
	bodyReader := strings.NewReader(testMultipartBody("\r\n"))
	testMultipart(t, bodyReader, false)
}

func TestMultipartOnlyNewlines(t *testing.T) {
	bodyReader := strings.NewReader(testMultipartBody("\n"))
	testMultipart(t, bodyReader, true)
}

func TestMultipartSlowInput(t *testing.T) {
	bodyReader := strings.NewReader(testMultipartBody("\r\n"))
	testMultipart(t, &slowReader{bodyReader}, false)
}

func testMultipart(t *testing.T, r io.Reader, onlyNewlines bool) {
	reader := NewReader(r, "MyBoundary")
	buf := new(bytes.Buffer)

	// Part1
	part, err := reader.NextPart()
	if part == nil || err != nil {
		t.Error("Expected part1")
		return
	}
	if x := part.Header.Get("Header1"); x != "value1" {
		t.Errorf("part.Header.Get(%q) = %q, want %q", "Header1", x, "value1")
	}
	if x := part.Header.Get("foo-bar"); x != "baz" {
		t.Errorf("part.Header.Get(%q) = %q, want %q", "foo-bar", x, "baz")
	}
	if x := part.Header.Get("Foo-Bar"); x != "baz" {
		t.Errorf("part.Header.Get(%q) = %q, want %q", "Foo-Bar", x, "baz")
	}
	buf.Reset()
	if _, err := io.Copy(buf, part); err != nil {
		t.Errorf("part 1 copy: %v", err)
	}

	adjustNewlines := func(s string) string {
		if onlyNewlines {
			return strings.Replace(s, "\r\n", "\n", -1)
		}
		return s
	}

	expectEq(t, adjustNewlines("My value\r\nThe end."), buf.String(), "Value of first part")

	// Part2
	part, err = reader.NextPart()
	if err != nil {
		t.Fatalf("Expected part2; got: %v", err)
		return
	}
	if e, g := "bigsection", part.Header.Get("name"); e != g {
		t.Errorf("part2's name header: expected %q, got %q", e, g)
	}
	buf.Reset()
	if _, err := io.Copy(buf, part); err != nil {
		t.Errorf("part 2 copy: %v", err)
	}
	s := buf.String()
	if len(s) != len(longLine) {
		t.Errorf("part2 body expected long line of length %d; got length %d",
			len(longLine), len(s))
	}
	if s != longLine {
		t.Errorf("part2 long body didn't match")
	}

	// Part3
	part, err = reader.NextPart()
	if part == nil || err != nil {
		t.Error("Expected part3")
		return
	}
	if part.Header.Get("foo-bar") != "bazb" {
		t.Error("Expected foo-bar: bazb")
	}
	buf.Reset()
	if _, err := io.Copy(buf, part); err != nil {
		t.Errorf("part 3 copy: %v", err)
	}
	expectEq(t, adjustNewlines("Line 1\r\nLine 2\r\nLine 3 ends in a newline, but just one.\r\n"),
		buf.String(), "body of part 3")

	// Part4
	part, err = reader.NextPart()
	if part == nil || err != nil {
		t.Error("Expected part 4 without errors")
		return
	}

	// Non-existent part5
	part, err = reader.NextPart()
	if part != nil {
		t.Error("Didn't expect a fifth part.")
	}
	if err != io.EOF {
		t.Errorf("On fifth part expected io.EOF; got %v", err)
	}
}

func TestVariousTextLineEndings(t *testing.T) {
	tests := [...]string{
		"Foo\nBar",
		"Foo\nBar\n",
		"Foo\r\nBar",
		"Foo\r\nBar\r\n",
		"Foo\rBar",
		"Foo\rBar\r",
		"\x00\x01\x02\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10",
	}

	for testNum, expectedBody := range tests {
		body := "--BOUNDARY\r\n" +
			"Content-Disposition: form-data; name=\"value\"\r\n" +
			"\r\n" +
			expectedBody +
			"\r\n--BOUNDARY--\r\n"
		bodyReader := strings.NewReader(body)

		reader := NewReader(bodyReader, "BOUNDARY")
		buf := new(bytes.Buffer)
		part, err := reader.NextPart()
		if part == nil {
			t.Errorf("Expected a body part on text %d", testNum)
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error on text %d: %v", testNum, err)
			continue
		}
		written, err := io.Copy(buf, part)
		expectEq(t, expectedBody, buf.String(), fmt.Sprintf("test %d", testNum))
		if err != nil {
			t.Errorf("Error copying multipart; bytes=%v, error=%v", written, err)
		}

		part, err = reader.NextPart()
		if part != nil {
			t.Errorf("Unexpected part in test %d", testNum)
		}
		if err != io.EOF {
			t.Errorf("On test %d expected io.EOF; got %v", testNum, err)
		}

	}
}

type maliciousReader struct {
	t *testing.T
	n int
}

const maxReadThreshold = 1 << 20

func (mr *maliciousReader) Read(b []byte) (n int, err error) {
	mr.n += len(b)
	if mr.n >= maxReadThreshold {
		mr.t.Fatal("too much was read")
		return 0, io.EOF
	}
	return len(b), nil
}

func TestLineLimit(t *testing.T) {
	mr := &maliciousReader{t: t}
	r := NewReader(mr, "fooBoundary")
	part, err := r.NextPart()
	if part != nil {
		t.Errorf("unexpected part read")
	}
	if err == nil {
		t.Errorf("expected an error")
	}
	if mr.n >= maxReadThreshold {
		t.Errorf("expected to read < %d bytes; read %d", maxReadThreshold, mr.n)
	}
}

func TestMultipartTruncated(t *testing.T) {
	testBody := `
This is a multi-part message.  This line is ignored.
--MyBoundary
foo-bar: baz

Oh no, premature EOF!
`
	body := strings.Replace(testBody, "\n", "\r\n", -1)
	bodyReader := strings.NewReader(body)
	r := NewReader(bodyReader, "MyBoundary")

	part, err := r.NextPart()
	if err != nil {
		t.Fatalf("didn't get a part")
	}
	_, err = io.Copy(ioutil.Discard, part)
	if err != io.ErrUnexpectedEOF {
		t.Fatalf("expected error io.ErrUnexpectedEOF; got %v", err)
	}
}

func TestZeroLengthBody(t *testing.T) {
	testBody := strings.Replace(`
This is a multi-part message.  This line is ignored.
--MyBoundary
foo: bar


--MyBoundary--
`, "\n", "\r\n", -1)
	r := NewReader(strings.NewReader(testBody), "MyBoundary")
	part, err := r.NextPart()
	if err != nil {
		t.Fatalf("didn't get a part")
	}
	n, err := io.Copy(ioutil.Discard, part)
	if err != nil {
		t.Errorf("error reading part: %v", err)
	}
	if n != 0 {
		t.Errorf("read %d bytes; expected 0", n)
	}
}

type slowReader struct {
	r io.Reader
}

func (s *slowReader) Read(p []byte) (int, error) {
	if len(p) == 0 {
		return s.r.Read(p)
	}
	return s.r.Read(p[:1])
}

func TestLineContinuation(t *testing.T) {
	// This body, extracted from an email, contains headers that span multiple
	// lines.

	// TODO: The original mail ended with a double-newline before the
	// final delimiter; this was manually edited to use a CRLF.
	testBody :=
		"\n--Apple-Mail-2-292336769\nContent-Transfer-Encoding: 7bit\nContent-Type: text/plain;\n\tcharset=US-ASCII;\n\tdelsp=yes;\n\tformat=flowed\n\nI'm finding the same thing happening on my system (10.4.1).\n\n\n--Apple-Mail-2-292336769\nContent-Transfer-Encoding: quoted-printable\nContent-Type: text/html;\n\tcharset=ISO-8859-1\n\n<HTML><BODY>I'm finding the same thing =\nhappening on my system (10.4.1).=A0 But I built it with XCode =\n2.0.</BODY></=\nHTML>=\n\r\n--Apple-Mail-2-292336769--\n"

	r := NewReader(strings.NewReader(testBody), "Apple-Mail-2-292336769")

	for i := 0; i < 2; i++ {
		part, err := r.NextPart()
		if err != nil {
			t.Fatalf("didn't get a part")
		}
		n, err := io.Copy(ioutil.Discard, part)
		if err != nil {
			t.Errorf("error reading part: %v", err)
		}
		if n <= 0 {
			t.Errorf("read %d bytes; expected >0", n)
		}
	}
}
