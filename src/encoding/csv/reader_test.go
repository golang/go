// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package csv

import (
	"errors"
	"fmt"
	"io"
	"reflect"
	"strings"
	"testing"
	"unicode/utf8"
)

type readTest struct {
	Name      string
	Input     string
	Output    [][]string
	Positions [][][2]int
	Errors    []error

	// These fields are copied into the Reader
	Comma              rune
	Comment            rune
	UseFieldsPerRecord bool // false (default) means FieldsPerRecord is -1
	FieldsPerRecord    int
	LazyQuotes         bool
	TrimLeadingSpace   bool
	ReuseRecord        bool
}

// In these tests, the §, ¶ and ∑ characters in readTest.Input are used to denote
// the start of a field, a record boundary and the position of an error respectively.
// They are removed before parsing and are used to verify the position
// information reported by FieldPos.

var readTests = []readTest{{
	Name:   "Simple",
	Input:  "§a,§b,§c\n",
	Output: [][]string{{"a", "b", "c"}},
}, {
	Name:   "CRLF",
	Input:  "§a,§b\r\n¶§c,§d\r\n",
	Output: [][]string{{"a", "b"}, {"c", "d"}},
}, {
	Name:   "BareCR",
	Input:  "§a,§b\rc,§d\r\n",
	Output: [][]string{{"a", "b\rc", "d"}},
}, {
	Name: "RFC4180test",
	Input: `§#field1,§field2,§field3
¶§"aaa",§"bb
b",§"ccc"
¶§"a,a",§"b""bb",§"ccc"
¶§zzz,§yyy,§xxx
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
	Input:  "§a,§b,§c",
	Output: [][]string{{"a", "b", "c"}},
}, {
	Name:   "Semicolon",
	Input:  "§a;§b;§c\n",
	Output: [][]string{{"a", "b", "c"}},
	Comma:  ';',
}, {
	Name: "MultiLine",
	Input: `§"two
line",§"one line",§"three
line
field"`,
	Output: [][]string{{"two\nline", "one line", "three\nline\nfield"}},
}, {
	Name:  "BlankLine",
	Input: "§a,§b,§c\n\n¶§d,§e,§f\n\n",
	Output: [][]string{
		{"a", "b", "c"},
		{"d", "e", "f"},
	},
}, {
	Name:  "BlankLineFieldCount",
	Input: "§a,§b,§c\n\n¶§d,§e,§f\n\n",
	Output: [][]string{
		{"a", "b", "c"},
		{"d", "e", "f"},
	},
	UseFieldsPerRecord: true,
	FieldsPerRecord:    0,
}, {
	Name:             "TrimSpace",
	Input:            " §a,  §b,   §c\n",
	Output:           [][]string{{"a", "b", "c"}},
	TrimLeadingSpace: true,
}, {
	Name:   "LeadingSpace",
	Input:  "§ a,§  b,§   c\n",
	Output: [][]string{{" a", "  b", "   c"}},
}, {
	Name:    "Comment",
	Input:   "#1,2,3\n§a,§b,§c\n#comment",
	Output:  [][]string{{"a", "b", "c"}},
	Comment: '#',
}, {
	Name:   "NoComment",
	Input:  "§#1,§2,§3\n¶§a,§b,§c",
	Output: [][]string{{"#1", "2", "3"}, {"a", "b", "c"}},
}, {
	Name:       "LazyQuotes",
	Input:      `§a "word",§"1"2",§a",§"b`,
	Output:     [][]string{{`a "word"`, `1"2`, `a"`, `b`}},
	LazyQuotes: true,
}, {
	Name:       "BareQuotes",
	Input:      `§a "word",§"1"2",§a"`,
	Output:     [][]string{{`a "word"`, `1"2`, `a"`}},
	LazyQuotes: true,
}, {
	Name:       "BareDoubleQuotes",
	Input:      `§a""b,§c`,
	Output:     [][]string{{`a""b`, `c`}},
	LazyQuotes: true,
}, {
	Name:   "BadDoubleQuotes",
	Input:  `§a∑""b,c`,
	Errors: []error{&ParseError{Err: ErrBareQuote}},
}, {
	Name:             "TrimQuote",
	Input:            ` §"a",§" b",§c`,
	Output:           [][]string{{"a", " b", "c"}},
	TrimLeadingSpace: true,
}, {
	Name:   "BadBareQuote",
	Input:  `§a ∑"word","b"`,
	Errors: []error{&ParseError{Err: ErrBareQuote}},
}, {
	Name:   "BadTrailingQuote",
	Input:  `§"a word",b∑"`,
	Errors: []error{&ParseError{Err: ErrBareQuote}},
}, {
	Name:   "ExtraneousQuote",
	Input:  `§"a ∑"word","b"`,
	Errors: []error{&ParseError{Err: ErrQuote}},
}, {
	Name:               "BadFieldCount",
	Input:              "§a,§b,§c\n¶∑§d,§e",
	Errors:             []error{nil, &ParseError{Err: ErrFieldCount}},
	Output:             [][]string{{"a", "b", "c"}, {"d", "e"}},
	UseFieldsPerRecord: true,
	FieldsPerRecord:    0,
}, {
	Name:               "BadFieldCountMultiple",
	Input:              "§a,§b,§c\n¶∑§d,§e\n¶∑§f",
	Errors:             []error{nil, &ParseError{Err: ErrFieldCount}, &ParseError{Err: ErrFieldCount}},
	Output:             [][]string{{"a", "b", "c"}, {"d", "e"}, {"f"}},
	UseFieldsPerRecord: true,
	FieldsPerRecord:    0,
}, {
	Name:               "BadFieldCount1",
	Input:              `§∑a,§b,§c`,
	Errors:             []error{&ParseError{Err: ErrFieldCount}},
	Output:             [][]string{{"a", "b", "c"}},
	UseFieldsPerRecord: true,
	FieldsPerRecord:    2,
}, {
	Name:   "FieldCount",
	Input:  "§a,§b,§c\n¶§d,§e",
	Output: [][]string{{"a", "b", "c"}, {"d", "e"}},
}, {
	Name:   "TrailingCommaEOF",
	Input:  "§a,§b,§c,§",
	Output: [][]string{{"a", "b", "c", ""}},
}, {
	Name:   "TrailingCommaEOL",
	Input:  "§a,§b,§c,§\n",
	Output: [][]string{{"a", "b", "c", ""}},
}, {
	Name:             "TrailingCommaSpaceEOF",
	Input:            "§a,§b,§c, §",
	Output:           [][]string{{"a", "b", "c", ""}},
	TrimLeadingSpace: true,
}, {
	Name:             "TrailingCommaSpaceEOL",
	Input:            "§a,§b,§c, §\n",
	Output:           [][]string{{"a", "b", "c", ""}},
	TrimLeadingSpace: true,
}, {
	Name:             "TrailingCommaLine3",
	Input:            "§a,§b,§c\n¶§d,§e,§f\n¶§g,§hi,§",
	Output:           [][]string{{"a", "b", "c"}, {"d", "e", "f"}, {"g", "hi", ""}},
	TrimLeadingSpace: true,
}, {
	Name:   "NotTrailingComma3",
	Input:  "§a,§b,§c,§ \n",
	Output: [][]string{{"a", "b", "c", " "}},
}, {
	Name: "CommaFieldTest",
	Input: `§x,§y,§z,§w
¶§x,§y,§z,§
¶§x,§y,§,§
¶§x,§,§,§
¶§,§,§,§
¶§"x",§"y",§"z",§"w"
¶§"x",§"y",§"z",§""
¶§"x",§"y",§"",§""
¶§"x",§"",§"",§""
¶§"",§"",§"",§""
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
	Input: "§a,§b,§\n¶§c,§d,§e",
	Output: [][]string{
		{"a", "b", ""},
		{"c", "d", "e"},
	},
	TrimLeadingSpace: true,
}, {
	Name:  "ReadAllReuseRecord",
	Input: "§a,§b\n¶§c,§d",
	Output: [][]string{
		{"a", "b"},
		{"c", "d"},
	},
	ReuseRecord: true,
}, {
	Name:   "StartLine1", // Issue 19019
	Input:  "§a,\"b\nc∑\"d,e",
	Errors: []error{&ParseError{Err: ErrQuote}},
}, {
	Name:   "StartLine2",
	Input:  "§a,§b\n¶§\"d\n\n,e∑",
	Errors: []error{nil, &ParseError{Err: ErrQuote}},
	Output: [][]string{{"a", "b"}},
}, {
	Name:  "CRLFInQuotedField", // Issue 21201
	Input: "§A,§\"Hello\r\nHi\",§B\r\n",
	Output: [][]string{
		{"A", "Hello\nHi", "B"},
	},
}, {
	Name:   "BinaryBlobField", // Issue 19410
	Input:  "§x09\x41\xb4\x1c,§aktau",
	Output: [][]string{{"x09A\xb4\x1c", "aktau"}},
}, {
	Name:   "TrailingCR",
	Input:  "§field1,§field2\r",
	Output: [][]string{{"field1", "field2"}},
}, {
	Name:   "QuotedTrailingCR",
	Input:  "§\"field\"\r",
	Output: [][]string{{"field"}},
}, {
	Name:   "QuotedTrailingCRCR",
	Input:  "§\"field∑\"\r\r",
	Errors: []error{&ParseError{Err: ErrQuote}},
}, {
	Name:   "FieldCR",
	Input:  "§field\rfield\r",
	Output: [][]string{{"field\rfield"}},
}, {
	Name:   "FieldCRCR",
	Input:  "§field\r\rfield\r\r",
	Output: [][]string{{"field\r\rfield\r"}},
}, {
	Name:   "FieldCRCRLF",
	Input:  "§field\r\r\n¶§field\r\r\n",
	Output: [][]string{{"field\r"}, {"field\r"}},
}, {
	Name:   "FieldCRCRLFCR",
	Input:  "§field\r\r\n¶§\rfield\r\r\n\r",
	Output: [][]string{{"field\r"}, {"\rfield\r"}},
}, {
	Name:   "FieldCRCRLFCRCR",
	Input:  "§field\r\r\n¶§\r\rfield\r\r\n¶§\r\r",
	Output: [][]string{{"field\r"}, {"\r\rfield\r"}, {"\r"}},
}, {
	Name:  "MultiFieldCRCRLFCRCR",
	Input: "§field1,§field2\r\r\n¶§\r\rfield1,§field2\r\r\n¶§\r\r,§",
	Output: [][]string{
		{"field1", "field2\r"},
		{"\r\rfield1", "field2\r"},
		{"\r\r", ""},
	},
}, {
	Name:             "NonASCIICommaAndComment",
	Input:            "§a£§b,c£ \t§d,e\n€ comment\n",
	Output:           [][]string{{"a", "b,c", "d,e"}},
	TrimLeadingSpace: true,
	Comma:            '£',
	Comment:          '€',
}, {
	Name:    "NonASCIICommaAndCommentWithQuotes",
	Input:   "§a€§\"  b,\"€§ c\nλ comment\n",
	Output:  [][]string{{"a", "  b,", " c"}},
	Comma:   '€',
	Comment: 'λ',
}, {
	// λ and θ start with the same byte.
	// This tests that the parser doesn't confuse such characters.
	Name:    "NonASCIICommaConfusion",
	Input:   "§\"abθcd\"λ§efθgh",
	Output:  [][]string{{"abθcd", "efθgh"}},
	Comma:   'λ',
	Comment: '€',
}, {
	Name:    "NonASCIICommentConfusion",
	Input:   "§λ\n¶§λ\nθ\n¶§λ\n",
	Output:  [][]string{{"λ"}, {"λ"}, {"λ"}},
	Comment: 'θ',
}, {
	Name:   "QuotedFieldMultipleLF",
	Input:  "§\"\n\n\n\n\"",
	Output: [][]string{{"\n\n\n\n"}},
}, {
	Name:  "MultipleCRLF",
	Input: "\r\n\r\n\r\n\r\n",
}, {
	// The implementation may read each line in several chunks if it doesn't fit entirely
	// in the read buffer, so we should test the code to handle that condition.
	Name:    "HugeLines",
	Input:   strings.Repeat("#ignore\n", 10000) + "§" + strings.Repeat("@", 5000) + ",§" + strings.Repeat("*", 5000),
	Output:  [][]string{{strings.Repeat("@", 5000), strings.Repeat("*", 5000)}},
	Comment: '#',
}, {
	Name:   "QuoteWithTrailingCRLF",
	Input:  "§\"foo∑\"bar\"\r\n",
	Errors: []error{&ParseError{Err: ErrQuote}},
}, {
	Name:       "LazyQuoteWithTrailingCRLF",
	Input:      "§\"foo\"bar\"\r\n",
	Output:     [][]string{{`foo"bar`}},
	LazyQuotes: true,
}, {
	Name:   "DoubleQuoteWithTrailingCRLF",
	Input:  "§\"foo\"\"bar\"\r\n",
	Output: [][]string{{`foo"bar`}},
}, {
	Name:   "EvenQuotes",
	Input:  `§""""""""`,
	Output: [][]string{{`"""`}},
}, {
	Name:   "OddQuotes",
	Input:  `§"""""""∑`,
	Errors: []error{&ParseError{Err: ErrQuote}},
}, {
	Name:       "LazyOddQuotes",
	Input:      `§"""""""`,
	Output:     [][]string{{`"""`}},
	LazyQuotes: true,
}, {
	Name:   "BadComma1",
	Comma:  '\n',
	Errors: []error{errInvalidDelim},
}, {
	Name:   "BadComma2",
	Comma:  '\r',
	Errors: []error{errInvalidDelim},
}, {
	Name:   "BadComma3",
	Comma:  '"',
	Errors: []error{errInvalidDelim},
}, {
	Name:   "BadComma4",
	Comma:  utf8.RuneError,
	Errors: []error{errInvalidDelim},
}, {
	Name:    "BadComment1",
	Comment: '\n',
	Errors:  []error{errInvalidDelim},
}, {
	Name:    "BadComment2",
	Comment: '\r',
	Errors:  []error{errInvalidDelim},
}, {
	Name:    "BadComment3",
	Comment: utf8.RuneError,
	Errors:  []error{errInvalidDelim},
}, {
	Name:    "BadCommaComment",
	Comma:   'X',
	Comment: 'X',
	Errors:  []error{errInvalidDelim},
}}

func TestRead(t *testing.T) {
	newReader := func(tt readTest) (*Reader, [][][2]int, map[int][2]int, string) {
		positions, errPositions, input := makePositions(tt.Input)
		r := NewReader(strings.NewReader(input))

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
		return r, positions, errPositions, input
	}

	for _, tt := range readTests {
		t.Run(tt.Name, func { t ->
			r, positions, errPositions, input := newReader(tt)
			out, err := r.ReadAll()
			if wantErr := firstError(tt.Errors, positions, errPositions); wantErr != nil {
				if !reflect.DeepEqual(err, wantErr) {
					t.Fatalf("ReadAll() error mismatch:\ngot  %v (%#v)\nwant %v (%#v)", err, err, wantErr, wantErr)
				}
				if out != nil {
					t.Fatalf("ReadAll() output:\ngot  %q\nwant nil", out)
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected Readall() error: %v", err)
				}
				if !reflect.DeepEqual(out, tt.Output) {
					t.Fatalf("ReadAll() output:\ngot  %q\nwant %q", out, tt.Output)
				}
			}

			// Check input offset after call ReadAll()
			inputByteSize := len(input)
			inputOffset := r.InputOffset()
			if err == nil && int64(inputByteSize) != inputOffset {
				t.Errorf("wrong input offset after call ReadAll():\ngot:  %d\nwant: %d\ninput: %s", inputOffset, inputByteSize, input)
			}

			// Check field and error positions.
			r, _, _, _ = newReader(tt)
			for recNum := 0; ; recNum++ {
				rec, err := r.Read()
				var wantErr error
				if recNum < len(tt.Errors) && tt.Errors[recNum] != nil {
					wantErr = errorWithPosition(tt.Errors[recNum], recNum, positions, errPositions)
				} else if recNum >= len(tt.Output) {
					wantErr = io.EOF
				}
				if !reflect.DeepEqual(err, wantErr) {
					t.Fatalf("Read() error at record %d:\ngot %v (%#v)\nwant %v (%#v)", recNum, err, err, wantErr, wantErr)
				}
				// ErrFieldCount is explicitly non-fatal.
				if err != nil && !errors.Is(err, ErrFieldCount) {
					if recNum < len(tt.Output) {
						t.Fatalf("need more records; got %d want %d", recNum, len(tt.Output))
					}
					break
				}
				if got, want := rec, tt.Output[recNum]; !reflect.DeepEqual(got, want) {
					t.Errorf("Read vs ReadAll mismatch;\ngot %q\nwant %q", got, want)
				}
				pos := positions[recNum]
				if len(pos) != len(rec) {
					t.Fatalf("mismatched position length at record %d", recNum)
				}
				for i := range rec {
					line, col := r.FieldPos(i)
					if got, want := [2]int{line, col}, pos[i]; got != want {
						t.Errorf("position mismatch at record %d, field %d;\ngot %v\nwant %v", recNum, i, got, want)
					}
				}
			}
		})
	}
}

// firstError returns the first non-nil error in errs,
// with the position adjusted according to the error's
// index inside positions.
func firstError(errs []error, positions [][][2]int, errPositions map[int][2]int) error {
	for i, err := range errs {
		if err != nil {
			return errorWithPosition(err, i, positions, errPositions)
		}
	}
	return nil
}

func errorWithPosition(err error, recNum int, positions [][][2]int, errPositions map[int][2]int) error {
	parseErr, ok := err.(*ParseError)
	if !ok {
		return err
	}
	if recNum >= len(positions) {
		panic(fmt.Errorf("no positions found for error at record %d", recNum))
	}
	errPos, ok := errPositions[recNum]
	if !ok {
		panic(fmt.Errorf("no error position found for error at record %d", recNum))
	}
	parseErr1 := *parseErr
	parseErr1.StartLine = positions[recNum][0][0]
	parseErr1.Line = errPos[0]
	parseErr1.Column = errPos[1]
	return &parseErr1
}

// makePositions returns the expected field positions of all
// the fields in text, the positions of any errors, and the text with the position markers
// removed.
//
// The start of each field is marked with a § symbol;
// CSV lines are separated by ¶ symbols;
// Error positions are marked with ∑ symbols.
func makePositions(text string) ([][][2]int, map[int][2]int, string) {
	buf := make([]byte, 0, len(text))
	var positions [][][2]int
	errPositions := make(map[int][2]int)
	line, col := 1, 1
	recNum := 0

	for len(text) > 0 {
		r, size := utf8.DecodeRuneInString(text)
		switch r {
		case '\n':
			line++
			col = 1
			buf = append(buf, '\n')
		case '§':
			if len(positions) == 0 {
				positions = append(positions, [][2]int{})
			}
			positions[len(positions)-1] = append(positions[len(positions)-1], [2]int{line, col})
		case '¶':
			positions = append(positions, [][2]int{})
			recNum++
		case '∑':
			errPositions[recNum] = [2]int{line, col}
		default:
			buf = append(buf, text[:size]...)
			col += size
		}
		text = text[size:]
	}
	return positions, errPositions, string(buf)
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
	benchmarkRead(b, func { r -> r.FieldsPerRecord = 4 }, benchmarkCSVData)
}

func BenchmarkReadWithoutFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func { r -> r.FieldsPerRecord = -1 }, benchmarkCSVData)
}

func BenchmarkReadLargeFields(b *testing.B) {
	benchmarkRead(b, nil, strings.Repeat(`xxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvv
,,zzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
`, 3))
}

func BenchmarkReadReuseRecord(b *testing.B) {
	benchmarkRead(b, func { r -> r.ReuseRecord = true }, benchmarkCSVData)
}

func BenchmarkReadReuseRecordWithFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func { r ->
		r.ReuseRecord = true
		r.FieldsPerRecord = 4
	}, benchmarkCSVData)
}

func BenchmarkReadReuseRecordWithoutFieldsPerRecord(b *testing.B) {
	benchmarkRead(b, func { r ->
		r.ReuseRecord = true
		r.FieldsPerRecord = -1
	}, benchmarkCSVData)
}

func BenchmarkReadReuseRecordLargeFields(b *testing.B) {
	benchmarkRead(b, func { r -> r.ReuseRecord = true }, strings.Repeat(`xxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvv
,,zzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx,yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy,zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz,wwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwwww,vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
`, 3))
}
