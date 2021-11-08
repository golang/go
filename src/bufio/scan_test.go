// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bufio_test

import (
	. "bufio"
	"bytes"
	"errors"
	"io"
	"strings"
	"testing"
	"unicode"
	"unicode/utf8"
)

const smallMaxTokenSize = 256 // Much smaller for more efficient testing.

// Test white space table matches the Unicode definition.
func TestSpace(t *testing.T) {
	for r := rune(0); r <= utf8.MaxRune; r++ {
		if IsSpace(r) != unicode.IsSpace(r) {
			t.Fatalf("white space property disagrees: %#U should be %t", r, unicode.IsSpace(r))
		}
	}
}

var scanTests = []string{
	"",
	"a",
	"¼",
	"☹",
	"\x81",   // UTF-8 error
	"\uFFFD", // correctly encoded RuneError
	"abcdefgh",
	"abc def\n\t\tgh    ",
	"abc¼☹\x81\uFFFD日本語\x82abc",
}

func TestScanByte(t *testing.T) {
	for n, test := range scanTests {
		buf := strings.NewReader(test)
		s := NewScanner(buf)
		s.Split(ScanBytes)
		var i int
		for i = 0; s.Scan(); i++ {
			if b := s.Bytes(); len(b) != 1 || b[0] != test[i] {
				t.Errorf("#%d: %d: expected %q got %q", n, i, test, b)
			}
		}
		if i != len(test) {
			t.Errorf("#%d: termination expected at %d; got %d", n, len(test), i)
		}
		err := s.Err()
		if err != nil {
			t.Errorf("#%d: %v", n, err)
		}
	}
}

// Test that the rune splitter returns same sequence of runes (not bytes) as for range string.
func TestScanRune(t *testing.T) {
	for n, test := range scanTests {
		buf := strings.NewReader(test)
		s := NewScanner(buf)
		s.Split(ScanRunes)
		var i, runeCount int
		var expect rune
		// Use a string range loop to validate the sequence of runes.
		for i, expect = range string(test) {
			if !s.Scan() {
				break
			}
			runeCount++
			got, _ := utf8.DecodeRune(s.Bytes())
			if got != expect {
				t.Errorf("#%d: %d: expected %q got %q", n, i, expect, got)
			}
		}
		if s.Scan() {
			t.Errorf("#%d: scan ran too long, got %q", n, s.Text())
		}
		testRuneCount := utf8.RuneCountInString(test)
		if runeCount != testRuneCount {
			t.Errorf("#%d: termination expected at %d; got %d", n, testRuneCount, runeCount)
		}
		err := s.Err()
		if err != nil {
			t.Errorf("#%d: %v", n, err)
		}
	}
}

var wordScanTests = []string{
	"",
	" ",
	"\n",
	"a",
	" a ",
	"abc def",
	" abc def ",
	" abc\tdef\nghi\rjkl\fmno\vpqr\u0085stu\u00a0\n",
}

// Test that the word splitter returns the same data as strings.Fields.
func TestScanWords(t *testing.T) {
	for n, test := range wordScanTests {
		buf := strings.NewReader(test)
		s := NewScanner(buf)
		s.Split(ScanWords)
		words := strings.Fields(test)
		var wordCount int
		for wordCount = 0; wordCount < len(words); wordCount++ {
			if !s.Scan() {
				break
			}
			got := s.Text()
			if got != words[wordCount] {
				t.Errorf("#%d: %d: expected %q got %q", n, wordCount, words[wordCount], got)
			}
		}
		if s.Scan() {
			t.Errorf("#%d: scan ran too long, got %q", n, s.Text())
		}
		if wordCount != len(words) {
			t.Errorf("#%d: termination expected at %d; got %d", n, len(words), wordCount)
		}
		err := s.Err()
		if err != nil {
			t.Errorf("#%d: %v", n, err)
		}
	}
}

// slowReader is a reader that returns only a few bytes at a time, to test the incremental
// reads in Scanner.Scan.
type slowReader struct {
	max int
	buf io.Reader
}

func (sr *slowReader) Read(p []byte) (n int, err error) {
	if len(p) > sr.max {
		p = p[0:sr.max]
	}
	return sr.buf.Read(p)
}

// genLine writes to buf a predictable but non-trivial line of text of length
// n, including the terminal newline and an occasional carriage return.
// If addNewline is false, the \r and \n are not emitted.
func genLine(buf *bytes.Buffer, lineNum, n int, addNewline bool) {
	buf.Reset()
	doCR := lineNum%5 == 0
	if doCR {
		n--
	}
	for i := 0; i < n-1; i++ { // Stop early for \n.
		c := 'a' + byte(lineNum+i)
		if c == '\n' || c == '\r' { // Don't confuse us.
			c = 'N'
		}
		buf.WriteByte(c)
	}
	if addNewline {
		if doCR {
			buf.WriteByte('\r')
		}
		buf.WriteByte('\n')
	}
}

// Test the line splitter, including some carriage returns but no long lines.
func TestScanLongLines(t *testing.T) {
	// Build a buffer of lots of line lengths up to but not exceeding smallMaxTokenSize.
	tmp := new(bytes.Buffer)
	buf := new(bytes.Buffer)
	lineNum := 0
	j := 0
	for i := 0; i < 2*smallMaxTokenSize; i++ {
		genLine(tmp, lineNum, j, true)
		if j < smallMaxTokenSize {
			j++
		} else {
			j--
		}
		buf.Write(tmp.Bytes())
		lineNum++
	}
	s := NewScanner(&slowReader{1, buf})
	s.Split(ScanLines)
	s.MaxTokenSize(smallMaxTokenSize)
	j = 0
	for lineNum := 0; s.Scan(); lineNum++ {
		genLine(tmp, lineNum, j, false)
		if j < smallMaxTokenSize {
			j++
		} else {
			j--
		}
		line := tmp.String() // We use the string-valued token here, for variety.
		if s.Text() != line {
			t.Errorf("%d: bad line: %d %d\n%.100q\n%.100q\n", lineNum, len(s.Bytes()), len(line), s.Text(), line)
		}
	}
	err := s.Err()
	if err != nil {
		t.Fatal(err)
	}
}

// Test that the line splitter errors out on a long line.
func TestScanLineTooLong(t *testing.T) {
	const smallMaxTokenSize = 256 // Much smaller for more efficient testing.
	// Build a buffer of lots of line lengths up to but not exceeding smallMaxTokenSize.
	tmp := new(bytes.Buffer)
	buf := new(bytes.Buffer)
	lineNum := 0
	j := 0
	for i := 0; i < 2*smallMaxTokenSize; i++ {
		genLine(tmp, lineNum, j, true)
		j++
		buf.Write(tmp.Bytes())
		lineNum++
	}
	s := NewScanner(&slowReader{3, buf})
	s.Split(ScanLines)
	s.MaxTokenSize(smallMaxTokenSize)
	j = 0
	for lineNum := 0; s.Scan(); lineNum++ {
		genLine(tmp, lineNum, j, false)
		if j < smallMaxTokenSize {
			j++
		} else {
			j--
		}
		line := tmp.Bytes()
		if !bytes.Equal(s.Bytes(), line) {
			t.Errorf("%d: bad line: %d %d\n%.100q\n%.100q\n", lineNum, len(s.Bytes()), len(line), s.Bytes(), line)
		}
	}
	err := s.Err()
	if err != ErrTooLong {
		t.Fatalf("expected ErrTooLong; got %s", err)
	}
}

// Test that the line splitter handles a final line without a newline.
func testNoNewline(text string, lines []string, t *testing.T) {
	buf := strings.NewReader(text)
	s := NewScanner(&slowReader{7, buf})
	s.Split(ScanLines)
	for lineNum := 0; s.Scan(); lineNum++ {
		line := lines[lineNum]
		if s.Text() != line {
			t.Errorf("%d: bad line: %d %d\n%.100q\n%.100q\n", lineNum, len(s.Bytes()), len(line), s.Bytes(), line)
		}
	}
	err := s.Err()
	if err != nil {
		t.Fatal(err)
	}
}

// Test that the line splitter handles a final line without a newline.
func TestScanLineNoNewline(t *testing.T) {
	const text = "abcdefghijklmn\nopqrstuvwxyz"
	lines := []string{
		"abcdefghijklmn",
		"opqrstuvwxyz",
	}
	testNoNewline(text, lines, t)
}

// Test that the line splitter handles a final line with a carriage return but no newline.
func TestScanLineReturnButNoNewline(t *testing.T) {
	const text = "abcdefghijklmn\nopqrstuvwxyz\r"
	lines := []string{
		"abcdefghijklmn",
		"opqrstuvwxyz",
	}
	testNoNewline(text, lines, t)
}

// Test that the line splitter handles a final empty line.
func TestScanLineEmptyFinalLine(t *testing.T) {
	const text = "abcdefghijklmn\nopqrstuvwxyz\n\n"
	lines := []string{
		"abcdefghijklmn",
		"opqrstuvwxyz",
		"",
	}
	testNoNewline(text, lines, t)
}

// Test that the line splitter handles a final empty line with a carriage return but no newline.
func TestScanLineEmptyFinalLineWithCR(t *testing.T) {
	const text = "abcdefghijklmn\nopqrstuvwxyz\n\r"
	lines := []string{
		"abcdefghijklmn",
		"opqrstuvwxyz",
		"",
	}
	testNoNewline(text, lines, t)
}

var testError = errors.New("testError")

// Test the correct error is returned when the split function errors out.
func TestSplitError(t *testing.T) {
	// Create a split function that delivers a little data, then a predictable error.
	numSplits := 0
	const okCount = 7
	errorSplit := func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		if atEOF {
			panic("didn't get enough data")
		}
		if numSplits >= okCount {
			return 0, nil, testError
		}
		numSplits++
		return 1, data[0:1], nil
	}
	// Read the data.
	const text = "abcdefghijklmnopqrstuvwxyz"
	buf := strings.NewReader(text)
	s := NewScanner(&slowReader{1, buf})
	s.Split(errorSplit)
	var i int
	for i = 0; s.Scan(); i++ {
		if len(s.Bytes()) != 1 || text[i] != s.Bytes()[0] {
			t.Errorf("#%d: expected %q got %q", i, text[i], s.Bytes()[0])
		}
	}
	// Check correct termination location and error.
	if i != okCount {
		t.Errorf("unexpected termination; expected %d tokens got %d", okCount, i)
	}
	err := s.Err()
	if err != testError {
		t.Fatalf("expected %q got %v", testError, err)
	}
}

// Test that an EOF is overridden by a user-generated scan error.
func TestErrAtEOF(t *testing.T) {
	s := NewScanner(strings.NewReader("1 2 33"))
	// This splitter will fail on last entry, after s.err==EOF.
	split := func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		advance, token, err = ScanWords(data, atEOF)
		if len(token) > 1 {
			if s.ErrOrEOF() != io.EOF {
				t.Fatal("not testing EOF")
			}
			err = testError
		}
		return
	}
	s.Split(split)
	for s.Scan() {
	}
	if s.Err() != testError {
		t.Fatal("wrong error:", s.Err())
	}
}

// Test for issue 5268.
type alwaysError struct{}

func (alwaysError) Read(p []byte) (int, error) {
	return 0, io.ErrUnexpectedEOF
}

func TestNonEOFWithEmptyRead(t *testing.T) {
	scanner := NewScanner(alwaysError{})
	for scanner.Scan() {
		t.Fatal("read should fail")
	}
	err := scanner.Err()
	if err != io.ErrUnexpectedEOF {
		t.Errorf("unexpected error: %v", err)
	}
}

// Test that Scan finishes if we have endless empty reads.
type endlessZeros struct{}

func (endlessZeros) Read(p []byte) (int, error) {
	return 0, nil
}

func TestBadReader(t *testing.T) {
	scanner := NewScanner(endlessZeros{})
	for scanner.Scan() {
		t.Fatal("read should fail")
	}
	err := scanner.Err()
	if err != io.ErrNoProgress {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestScanWordsExcessiveWhiteSpace(t *testing.T) {
	const word = "ipsum"
	s := strings.Repeat(" ", 4*smallMaxTokenSize) + word
	scanner := NewScanner(strings.NewReader(s))
	scanner.MaxTokenSize(smallMaxTokenSize)
	scanner.Split(ScanWords)
	if !scanner.Scan() {
		t.Fatalf("scan failed: %v", scanner.Err())
	}
	if token := scanner.Text(); token != word {
		t.Fatalf("unexpected token: %v", token)
	}
}

// Test that empty tokens, including at end of line or end of file, are found by the scanner.
// Issue 8672: Could miss final empty token.

func commaSplit(data []byte, atEOF bool) (advance int, token []byte, err error) {
	for i := 0; i < len(data); i++ {
		if data[i] == ',' {
			return i + 1, data[:i], nil
		}
	}
	return 0, data, ErrFinalToken
}

func testEmptyTokens(t *testing.T, text string, values []string) {
	s := NewScanner(strings.NewReader(text))
	s.Split(commaSplit)
	var i int
	for i = 0; s.Scan(); i++ {
		if i >= len(values) {
			t.Fatalf("got %d fields, expected %d", i+1, len(values))
		}
		if s.Text() != values[i] {
			t.Errorf("%d: expected %q got %q", i, values[i], s.Text())
		}
	}
	if i != len(values) {
		t.Fatalf("got %d fields, expected %d", i, len(values))
	}
	if err := s.Err(); err != nil {
		t.Fatal(err)
	}
}

func TestEmptyTokens(t *testing.T) {
	testEmptyTokens(t, "1,2,3,", []string{"1", "2", "3", ""})
}

func TestWithNoEmptyTokens(t *testing.T) {
	testEmptyTokens(t, "1,2,3", []string{"1", "2", "3"})
}

func loopAtEOFSplit(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if len(data) > 0 {
		return 1, data[:1], nil
	}
	return 0, data, nil
}

func TestDontLoopForever(t *testing.T) {
	s := NewScanner(strings.NewReader("abc"))
	s.Split(loopAtEOFSplit)
	// Expect a panic
	defer func() {
		err := recover()
		if err == nil {
			t.Fatal("should have panicked")
		}
		if msg, ok := err.(string); !ok || !strings.Contains(msg, "empty tokens") {
			panic(err)
		}
	}()
	for count := 0; s.Scan(); count++ {
		if count > 1000 {
			t.Fatal("looping")
		}
	}
	if s.Err() != nil {
		t.Fatal("after scan:", s.Err())
	}
}

func TestBlankLines(t *testing.T) {
	s := NewScanner(strings.NewReader(strings.Repeat("\n", 1000)))
	for count := 0; s.Scan(); count++ {
		if count > 2000 {
			t.Fatal("looping")
		}
	}
	if s.Err() != nil {
		t.Fatal("after scan:", s.Err())
	}
}

type countdown int

func (c *countdown) split(data []byte, atEOF bool) (advance int, token []byte, err error) {
	if *c > 0 {
		*c--
		return 1, data[:1], nil
	}
	return 0, nil, nil
}

// Check that the looping-at-EOF check doesn't trigger for merely empty tokens.
func TestEmptyLinesOK(t *testing.T) {
	c := countdown(10000)
	s := NewScanner(strings.NewReader(strings.Repeat("\n", 10000)))
	s.Split(c.split)
	for s.Scan() {
	}
	if s.Err() != nil {
		t.Fatal("after scan:", s.Err())
	}
	if c != 0 {
		t.Fatalf("stopped with %d left to process", c)
	}
}

// Make sure we can read a huge token if a big enough buffer is provided.
func TestHugeBuffer(t *testing.T) {
	text := strings.Repeat("x", 2*MaxScanTokenSize)
	s := NewScanner(strings.NewReader(text + "\n"))
	s.Buffer(make([]byte, 100), 3*MaxScanTokenSize)
	for s.Scan() {
		token := s.Text()
		if token != text {
			t.Errorf("scan got incorrect token of length %d", len(token))
		}
	}
	if s.Err() != nil {
		t.Fatal("after scan:", s.Err())
	}
}

// negativeEOFReader returns an invalid -1 at the end, as though it
// were wrapping the read system call.
type negativeEOFReader int

func (r *negativeEOFReader) Read(p []byte) (int, error) {
	if *r > 0 {
		c := int(*r)
		if c > len(p) {
			c = len(p)
		}
		for i := 0; i < c; i++ {
			p[i] = 'a'
		}
		p[c-1] = '\n'
		*r -= negativeEOFReader(c)
		return c, nil
	}
	return -1, io.EOF
}

// Test that the scanner doesn't panic and returns ErrBadReadCount
// on a reader that returns a negative count of bytes read (issue 38053).
func TestNegativeEOFReader(t *testing.T) {
	r := negativeEOFReader(10)
	scanner := NewScanner(&r)
	c := 0
	for scanner.Scan() {
		c++
		if c > 1 {
			t.Error("read too many lines")
			break
		}
	}
	if got, want := scanner.Err(), ErrBadReadCount; got != want {
		t.Errorf("scanner.Err: got %v, want %v", got, want)
	}
}

// largeReader returns an invalid count that is larger than the number
// of bytes requested.
type largeReader struct{}

func (largeReader) Read(p []byte) (int, error) {
	return len(p) + 1, nil
}

// Test that the scanner doesn't panic and returns ErrBadReadCount
// on a reader that returns an impossibly large count of bytes read (issue 38053).
func TestLargeReader(t *testing.T) {
	scanner := NewScanner(largeReader{})
	for scanner.Scan() {
	}
	if got, want := scanner.Err(), ErrBadReadCount; got != want {
		t.Errorf("scanner.Err: got %v, want %v", got, want)
	}
}
