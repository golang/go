// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csv

import (
	"io"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestRead(t *testing.T) {
	tests := []struct {
		Name   string
		Input  string
		Output [][]string
		Error  error

		// These fields are copied into the Reader
		Comma              rune
		Comment            rune
		UseFieldsPerRecord bool // false (default) means FieldsPerRecord is -1
		FieldsPerRecord    int
		LazyQuotes         bool
		TrimLeadingSpace   bool
		ReuseRecord        bool
	}{{
		Name:   "Simple",
		Input:  "a,b,c\n",
		Output: [][]string{{"a", "b", "c"}},
	}, {
		Name:   "CRLF",
		Input:  "a,b\r\nc,d\r\n",
		Output: [][]string{{"a", "b"}, {"c", "d"}},
	}, {
		Name:   "BareCR",
		Input:  "a,b\rc,d\r\n",
		Output: [][]string{{"a", "b\rc", "d"}},
	}, {
		Name: "RFC4180test",
		Input: `#field1,field2,field3
"aaa","bb
b","ccc"
"a,a","b""bb","ccc"
zzz,yyy,xxx
`,
		Output: [][]string{
			{"#field1", "field2", "field3"},
			{"aaa", "bb\nb", "ccc"},
			{"a,a", `b"bb`, "ccc"},
			{"zzz", "yyy", "xxx"},
		},
		UseFieldsPerRecord: true,
		FieldsPerRecord:    0,
	}, {
		Name:   "NoEOLTest",
		Input:  "a,b,c",
		Output: [][]string{{"a", "b", "c"}},
	}, {
		Name:   "Semicolon",
		Input:  "a;b;c\n",
		Output: [][]string{{"a", "b", "c"}},
		Comma:  ';',
	}, {
		Name: "MultiLine",
		Input: `"two
line","one line","three
line
field"`,
		Output: [][]string{{"two\nline", "one line", "three\nline\nfield"}},
	}, {
		Name:  "BlankLine",
		Input: "a,b,c\n\nd,e,f\n\n",
		Output: [][]string{
			{"a", "b", "c"},
			{"d", "e", "f"},
		},
	}, {
		Name:  "BlankLineFieldCount",
		Input: "a,b,c\n\nd,e,f\n\n",
		Output: [][]string{
			{"a", "b", "c"},
			{"d", "e", "f"},
		},
		UseFieldsPerRecord: true,
		FieldsPerRecord:    0,
	}, {
		Name:             "TrimSpace",
		Input:            " a,  b,   c\n",
		Output:           [][]string{{"a", "b", "c"}},
		TrimLeadingSpace: true,
	}, {
		Name:   "LeadingSpace",
		Input:  " a,  b,   c\n",
		Output: [][]string{{" a", "  b", "   c"}},
	}, {
		Name:    "Comment",
		Input:   "#1,2,3\na,b,c\n#comment",
		Output:  [][]string{{"a", "b", "c"}},
		Comment: '#',
	}, {
		Name:   "NoComment",
		Input:  "#1,2,3\na,b,c",
		Output: [][]string{{"#1", "2", "3"}, {"a", "b", "c"}},
	}, {
		Name:       "LazyQuotes",
		Input:      `a "word","1"2",a","b`,
		Output:     [][]string{{`a "word"`, `1"2`, `a"`, `b`}},
		LazyQuotes: true,
	}, {
		Name:       "BareQuotes",
		Input:      `a "word","1"2",a"`,
		Output:     [][]string{{`a "word"`, `1"2`, `a"`}},
		LazyQuotes: true,
	}, {
		Name:       "BareDoubleQuotes",
		Input:      `a""b,c`,
		Output:     [][]string{{`a""b`, `c`}},
		LazyQuotes: true,
	}, {
		Name:  "BadDoubleQuotes",
		Input: `a""b,c`,
		Error: &ParseError{StartLine: 1, Line: 1, Column: 1, Err: ErrBareQuote},
	}, {
		Name:             "TrimQuote",
		Input:            ` "a"," b",c`,
		Output:           [][]string{{"a", " b", "c"}},
		TrimLeadingSpace: true,
	}, {
		Name:  "BadBareQuote",
		Input: `a "word","b"`,
		Error: &ParseError{StartLine: 1, Line: 1, Column: 2, Err: ErrBareQuote},
	}, {
		Name:  "BadTrailingQuote",
		Input: `"a word",b"`,
		Error: &ParseError{StartLine: 1, Line: 1, Column: 10, Err: ErrBareQuote},
	}, {
		Name:  "ExtraneousQuote",
		Input: `"a "word","b"`,
		Error: &ParseError{StartLine: 1, Line: 1, Column: 3, Err: ErrQuote},
	}, {
		Name:               "BadFieldCount",
		Input:              "a,b,c\nd,e",
		Error:              &ParseError{StartLine: 2, Line: 2, Err: ErrFieldCount},
		UseFieldsPerRecord: true,
		FieldsPerRecord:    0,
	}, {
		Name:               "BadFieldCount1",
		Input:              `a,b,c`,
		Error:              &ParseError{StartLine: 1, Line: 1, Err: ErrFieldCount},
		UseFieldsPerRecord: true,
		FieldsPerRecord:    2,
	}, {
		Name:   "FieldCount",
		Input:  "a,b,c\nd,e",
		Output: [][]string{{"a", "b", "c"}, {"d", "e"}},
	}, {
		Name:   "TrailingCommaEOF",
		Input:  "a,b,c,",
		Output: [][]string{{"a", "b", "c", ""}},
	}, {
		Name:   "TrailingCommaEOL",
		Input:  "a,b,c,\n",
		Output: [][]string{{"a", "b", "c", ""}},
	}, {
		Name:             "TrailingCommaSpaceEOF",
		Input:            "a,b,c, ",
		Output:           [][]string{{"a", "b", "c", ""}},
		TrimLeadingSpace: true,
	}, {
		Name:             "TrailingCommaSpaceEOL",
		Input:            "a,b,c, \n",
		Output:           [][]string{{"a", "b", "c", ""}},
		TrimLeadingSpace: true,
	}, {
		Name:             "TrailingCommaLine3",
		Input:            "a,b,c\nd,e,f\ng,hi,",
		Output:           [][]string{{"a", "b", "c"}, {"d", "e", "f"}, {"g", "hi", ""}},
		TrimLeadingSpace: true,
	}, {
		Name:   "NotTrailingComma3",
		Input:  "a,b,c, \n",
		Output: [][]string{{"a", "b", "c", " "}},
	}, {
		Name: "CommaFieldTest",
		Input: `x,y,z,w
x,y,z,
x,y,,
x,,,
,,,
"x","y","z","w"
"x","y","z",""
"x","y","",""
"x","","",""
"","","",""
`,
		Output: [][]string{
			{"x", "y", "z", "w"},
			{"x", "y", "z", ""},
			{"x", "y", "", ""},
			{"x", "", "", ""},
			{"", "", "", ""},
			{"x", "y", "z", "w"},
			{"x", "y", "z", ""},
			{"x", "y", "", ""},
			{"x", "", "", ""},
			{"", "", "", ""},
		},
	}, {
		Name:  "TrailingCommaIneffective1",
		Input: "a,b,\nc,d,e",
		Output: [][]string{
			{"a", "b", ""},
			{"c", "d", "e"},
		},
		TrimLeadingSpace: true,
	}, {
		Name:  "ReadAllReuseRecord",
		Input: "a,b\nc,d",
		Output: [][]string{
			{"a", "b"},
			{"c", "d"},
		},
		ReuseRecord: true,
	}, {
		Name:  "StartLine1", // Issue 19019
		Input: "a,\"b\nc\"d,e",
		Error: &ParseError{StartLine: 1, Line: 2, Column: 1, Err: ErrQuote},
	}, {
		Name:  "StartLine2",
		Input: "a,b\n\"d\n\n,e",
		Error: &ParseError{StartLine: 2, Line: 5, Column: 0, Err: ErrQuote},
	}, {
		Name:  "CRLFInQuotedField", // Issue 21201
		Input: "A,\"Hello\r\nHi\",B\r\n",
		Output: [][]string{
			{"A", "Hello\nHi", "B"},
		},
	}, {
		Name:   "BinaryBlobField", // Issue 19410
		Input:  "x09\x41\xb4\x1c,aktau",
		Output: [][]string{{"x09A\xb4\x1c", "aktau"}},
	}, {
		Name:   "TrailingCR",
		Input:  "field1,field2\r",
		Output: [][]string{{"field1", "field2"}},
	}, {
		Name:   "QuotedTrailingCR",
		Input:  "\"field\"\r",
		Output: [][]string{{"field"}},
	}, {
		Name:  "QuotedTrailingCRCR",
		Input: "\"field\"\r\r",
		Error: &ParseError{StartLine: 1, Line: 1, Column: 6, Err: ErrQuote},
	}, {
		Name:   "FieldCR",
		Input:  "field\rfield\r",
		Output: [][]string{{"field\rfield"}},
	}, {
		Name:   "FieldCRCR",
		Input:  "field\r\rfield\r\r",
		Output: [][]string{{"field\r\rfield\r"}},
	}, {
		Name:   "FieldCRCRLF",
		Input:  "field\r\r\nfield\r\r\n",
		Output: [][]string{{"field\r"}, {"field\r"}},
	}, {
		Name:   "FieldCRCRLFCR",
		Input:  "field\r\r\n\rfield\r\r\n\r",
		Output: [][]string{{"field\r"}, {"\rfield\r"}},
	}, {
		Name:   "FieldCRCRLFCRCR",
		Input:  "field\r\r\n\r\rfield\r\r\n\r\r",
		Output: [][]string{{"field\r"}, {"\r\rfield\r"}, {"\r"}},
	}, {
		Name:  "MultiFieldCRCRLFCRCR",
		Input: "field1,field2\r\r\n\r\rfield1,field2\r\r\n\r\r,",
		Output: [][]string{
			{"field1", "field2\r"},
			{"\r\rfield1", "field2\r"},
			{"\r\r", ""},
		},
	}, {
		Name:             "NonASCIICommaAndComment",
		Input:            "a£b,c£ \td,e\n€ comment\n",
		Output:           [][]string{{"a", "b,c", "d,e"}},
		TrimLeadingSpace: true,
		Comma:            '£',
		Comment:          '€',
	}, {
		Name:    "NonASCIICommaAndCommentWithQuotes",
		Input:   "a€\"  b,\"€ c\nλ comment\n",
		Output:  [][]string{{"a", "  b,", " c"}},
		Comma:   '€',
		Comment: 'λ',
	}, {
		// λ and θ start with the same byte.
		// This tests that the parser doesn't confuse such characters.
		Name:    "NonASCIICommaConfusion",
		Input:   "\"abθcd\"λefθgh",
		Output:  [][]string{{"abθcd", "efθgh"}},
		Comma:   'λ',
		Comment: '€',
	}, {
		Name:    "NonASCIICommentConfusion",
		Input:   "λ\nλ\nθ\nλ\n",
		Output:  [][]string{{"λ"}, {"λ"}, {"λ"}},
		Comment: 'θ',
	}, {
		Name:   "QuotedFieldMultipleLF",
		Input:  "\"\n\n\n\n\"",
		Output: [][]string{{"\n\n\n\n"}},
	}, {
		Name:  "MultipleCRLF",
		Input: "\r\n\r\n\r\n\r\n",
	}, {
		// The implementation may read each line in several chunks if it doesn't fit entirely
		// in the read buffer, so we should test the code to handle that condition.
		Name:    "HugeLines",
		Input:   strings.Repeat("#ignore\n", 10000) + strings.Repeat("@", 5000) + "," + strings.Repeat("*", 5000),
		Output:  [][]string{{strings.Repeat("@", 5000), strings.Repeat("*", 5000)}},
		Comment: '#',
	}, {
		Name:  "QuoteWithTrailingCRLF",
		Input: "\"foo\"bar\"\r\n",
		Error: &ParseError{StartLine: 1, Line: 1, Column: 4, Err: ErrQuote},
	}, {
		Name:       "LazyQuoteWithTrailingCRLF",
		Input:      "\"foo\"bar\"\r\n",
		Output:     [][]string{{`foo"bar`}},
		LazyQuotes: true,
	}, {
		Name:   "DoubleQuoteWithTrailingCRLF",
		Input:  "\"foo\"\"bar\"\r\n",
		Output: [][]string{{`foo"bar`}},
	}, {
		Name:   "EvenQuotes",
		Input:  `""""""""`,
		Output: [][]string{{`"""`}},
	}, {
		Name:  "OddQuotes",
		Input: `"""""""`,
		Error: &ParseError{StartLine: 1, Line: 1, Column: 7, Err: ErrQuote},
	}, {
		Name:       "LazyOddQuotes",
		Input:      `"""""""`,
		Output:     [][]string{{`"""`}},
		LazyQuotes: true,
	}, {
		Name:  "BadComma1",
		Comma: '\n',
		Error: errInvalidDelim,
	}, {
		Name:  "BadComma2",
		Comma: '\r',
		Error: errInvalidDelim,
	}, {
		Name:  "BadComma3",
		Comma: '"',
		Error: errInvalidDelim,
	}, {
		Name:  "BadComma4",
		Comma: utf8.RuneError,
		Error: errInvalidDelim,
	}, {
		Name:    "BadComment1",
		Comment: '\n',
		Error:   errInvalidDelim,
	}, {
		Name:    "BadComment2",
		Comment: '\r',
		Error:   errInvalidDelim,
	}, {
		Name:    "BadComment3",
		Comment: utf8.RuneError,
		Error:   errInvalidDelim,
	}, {
		Name:    "BadCommaComment",
		Comma:   'X',
		Comment: 'X',
		Error:   errInvalidDelim,
	}}

	for _, tt := range tests {
		t.Run(tt.Name, func(t *testing.T) {
			r := NewReader(strings.NewReader(tt.Input))

			if tt.Comma != 0 {
				r.Comma = tt.Comma
			}
			r.Comment = tt.Comment
			if tt.UseFieldsPerRecord {
				r.FieldsPerRecord = tt.FieldsPerRecord
			} else {
				r.FieldsPerRecord = -1
			}
			r.LazyQuotes = tt.LazyQuotes
			r.TrimLeadingSpace = tt.TrimLeadingSpace
			r.ReuseRecord = tt.ReuseRecord

			out, err := r.ReadAll()
			if !reflect.DeepEqual(err, tt.Error) {
				t.Errorf("ReadAll() error:\ngot  %v\nwant %v", err, tt.Error)
			} else if !reflect.DeepEqual(out, tt.Output) {
				t.Errorf("ReadAll() output:\ngot  %q\nwant %q", out, tt.Output)
			}
		})
	}
}

// nTimes is an io.Reader which yields the string s n times.
type nTimes struct {
	s   string
	n   int
	off int
}

func (r *nTimes) Read(p []byte) (n int, err error) {
	for {
		if r.n <= 0 || r.s == "" {
			return n, io.EOF
		}
		n0 := copy(p, r.s[r.off:])
		p = p[n0:]
		n += n0
		r.off += n0
		if r.off == len(r.s) {
			r.off = 0
			r.n--
		}
		if len(p) == 0 {
			return
		}
	}
}

// benchmarkRead measures reading the provided CSV rows data.
// initReader, if non-nil, modifies the Reader before it's used.
func benchmarkRead(b *testing.B, initReader func(*Reader), rows string) {
	b.ReportAllocs()
	r := NewReader(&nTimes{s: rows, n: b.N})
	if initReader != nil {
		initReader(r)
	}
	for {
		_, err := r.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			b.Fatal(err)
		}
	}
}

const benchmarkCSVData = `x,y,z,w
x,y,z,
x,y,,
x,,,
,,,
"x","y","z","w"
"x","y","z",""
"x","y","",""
"x","","",""
"","","",""
`

func BenchmarkRead(b *testing.B) {
	benchmarkRead(b, nil, benchmarkCSVData)
}

func BenchmarkReadWithFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func(r *Reader) { r.FieldsPerRecord = 4 }, benchmarkCSVData)
}

func BenchmarkReadWithoutFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func(r *Reader) { r.FieldsPerRecord = -1 }, benchmarkCSVData)
}

func BenchmarkReadLargeFields(b *testing.B) {
	benchmarkRead(b, nil, strings.Repeat(`xxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvv
,,zzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
`, 3))
}

func BenchmarkReadReuseRecord(b *testing.B) {
	benchmarkRead(b, func(r *Reader) { r.ReuseRecord = true }, benchmarkCSVData)
}

func BenchmarkReadReuseRecordWithFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func(r *Reader) { r.ReuseRecord = true; r.FieldsPerRecord = 4 }, benchmarkCSVData)
}

func BenchmarkReadReuseRecordWithoutFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func(r *Reader) { r.ReuseRecord = true; r.FieldsPerRecord = -1 }, benchmarkCSVData)
}

func BenchmarkReadReuseRecordLargeFields(b *testing.B) {
	benchmarkRead(b, func(r *Reader) { r.ReuseRecord = true }, strings.Repeat(`xxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvv
,,zzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
`, 3))
}
