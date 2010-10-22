// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package multipart

import (
	"bytes"
	"fmt"
	"io"
	"json"
	"regexp"
	"strings"
	"testing"
)

func TestHorizontalWhitespace(t *testing.T) {
	if !onlyHorizontalWhitespace(" \t") {
		t.Error("expected pass")
	}
	if onlyHorizontalWhitespace("foo bar") {
		t.Error("expected failure")
	}
}

func TestBoundaryLine(t *testing.T) {
	boundary := "myBoundary"
	prefix := "--" + boundary
	if !isBoundaryDelimiterLine("--myBoundary\r\n", prefix) {
		t.Error("expected")
	}
	if !isBoundaryDelimiterLine("--myBoundary \r\n", prefix) {
		t.Error("expected")
	}
	if !isBoundaryDelimiterLine("--myBoundary \n", prefix) {
		t.Error("expected")
	}
	if isBoundaryDelimiterLine("--myBoundary bogus \n", prefix) {
		t.Error("expected fail")
	}
	if isBoundaryDelimiterLine("--myBoundary bogus--", prefix) {
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

func TestFormName(t *testing.T) {
	p := new(Part)
	p.Header = make(map[string]string)
	tests := [...][2]string{
		{`form-data; name="foo"`, "foo"},
		{` form-data ; name=foo`, "foo"},
		{`FORM-DATA;name="foo"`, "foo"},
		{` FORM-DATA ; name="foo"`, "foo"},
		{` FORM-DATA ; name="foo"`, "foo"},
		{` FORM-DATA ; name=foo`, "foo"},
		{` FORM-DATA ; filename="foo.txt"; name=foo; baz=quux`, "foo"},
	}
	for _, test := range tests {
		p.Header["Content-Disposition"] = test[0]
		expected := test[1]
		actual := p.FormName()
		if actual != expected {
			t.Errorf("expected \"%s\"; got: \"%s\"", expected, actual)
		}
	}
}

func TestMultipart(t *testing.T) {
	testBody := `
This is a multi-part message.  This line is ignored.
--MyBoundary
Header1: value1
HEADER2: value2
foo-bar: baz

My value
The end.
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
`
	testBody = regexp.MustCompile("\n").ReplaceAllString(testBody, "\r\n")
	bodyReader := strings.NewReader(testBody)

	reader := NewReader(bodyReader, "MyBoundary")
	buf := new(bytes.Buffer)

	// Part1
	part, err := reader.NextPart()
	if part == nil || err != nil {
		t.Error("Expected part1")
		return
	}
	if part.Header["Header1"] != "value1" {
		t.Error("Expected Header1: value")
	}
	if part.Header["foo-bar"] != "baz" {
		t.Error("Expected foo-bar: baz")
	}
	buf.Reset()
	io.Copy(buf, part)
	expectEq(t, "My value\r\nThe end.",
		buf.String(), "Value of first part")

	// Part2
	part, err = reader.NextPart()
	if part == nil || err != nil {
		t.Error("Expected part2")
		return
	}
	if part.Header["foo-bar"] != "bazb" {
		t.Error("Expected foo-bar: bazb")
	}
	buf.Reset()
	io.Copy(buf, part)
	expectEq(t, "Line 1\r\nLine 2\r\nLine 3 ends in a newline, but just one.\r\n",
		buf.String(), "Value of second part")

	// Part3
	part, err = reader.NextPart()
	if part == nil || err != nil {
		t.Error("Expected part3 without errors")
		return
	}

	// Non-existent part4
	part, err = reader.NextPart()
	if part != nil {
		t.Error("Didn't expect a third part.")
	}
	if err != nil {
		t.Errorf("Unexpected error getting third part: %v", err)
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
		if err != nil {
			t.Errorf("Unexpected error in test %d: %v", testNum, err)
		}

	}
}
