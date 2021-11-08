// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tabwriter_test

import (
	"bytes"
	"fmt"
	"io"
	"testing"
	. "text/tabwriter"
)

type buffer struct {
	a []byte
}

func (b *buffer) init(n int) { b.a = make([]byte, 0, n) }

func (b *buffer) clear() { b.a = b.a[0:0] }

func (b *buffer) Write(buf []byte) (written int, err error) {
	n := len(b.a)
	m := len(buf)
	if n+m <= cap(b.a) {
		b.a = b.a[0 : n+m]
		for i := 0; i < m; i++ {
			b.a[n+i] = buf[i]
		}
	} else {
		panic("buffer.Write: buffer too small")
	}
	return len(buf), nil
}

func (b *buffer) String() string { return string(b.a) }

func write(t *testing.T, testname string, w *Writer, src string) {
	written, err := io.WriteString(w, src)
	if err != nil {
		t.Errorf("--- test: %s\n--- src:\n%q\n--- write error: %v\n", testname, src, err)
	}
	if written != len(src) {
		t.Errorf("--- test: %s\n--- src:\n%q\n--- written = %d, len(src) = %d\n", testname, src, written, len(src))
	}
}

func verify(t *testing.T, testname string, w *Writer, b *buffer, src, expected string) {
	err := w.Flush()
	if err != nil {
		t.Errorf("--- test: %s\n--- src:\n%q\n--- flush error: %v\n", testname, src, err)
	}

	res := b.String()
	if res != expected {
		t.Errorf("--- test: %s\n--- src:\n%q\n--- found:\n%q\n--- expected:\n%q\n", testname, src, res, expected)
	}
}

func check(t *testing.T, testname string, minwidth, tabwidth, padding int, padchar byte, flags uint, src, expected string) {
	var b buffer
	b.init(1000)

	var w Writer
	w.Init(&b, minwidth, tabwidth, padding, padchar, flags)

	// write all at once
	title := testname + " (written all at once)"
	b.clear()
	write(t, title, &w, src)
	verify(t, title, &w, &b, src, expected)

	// write byte-by-byte
	title = testname + " (written byte-by-byte)"
	b.clear()
	for i := 0; i < len(src); i++ {
		write(t, title, &w, src[i:i+1])
	}
	verify(t, title, &w, &b, src, expected)

	// write using Fibonacci slice sizes
	title = testname + " (written in fibonacci slices)"
	b.clear()
	for i, d := 0, 0; i < len(src); {
		write(t, title, &w, src[i:i+d])
		i, d = i+d, d+1
		if i+d > len(src) {
			d = len(src) - i
		}
	}
	verify(t, title, &w, &b, src, expected)
}

var tests = []struct {
	testname                    string
	minwidth, tabwidth, padding int
	padchar                     byte
	flags                       uint
	src, expected               string
}{
	{
		"1a",
		8, 0, 1, '.', 0,
		"",
		"",
	},

	{
		"1a debug",
		8, 0, 1, '.', Debug,
		"",
		"",
	},

	{
		"1b esc stripped",
		8, 0, 1, '.', StripEscape,
		"\xff\xff",
		"",
	},

	{
		"1b esc",
		8, 0, 1, '.', 0,
		"\xff\xff",
		"\xff\xff",
	},

	{
		"1c esc stripped",
		8, 0, 1, '.', StripEscape,
		"\xff\t\xff",
		"\t",
	},

	{
		"1c esc",
		8, 0, 1, '.', 0,
		"\xff\t\xff",
		"\xff\t\xff",
	},

	{
		"1d esc stripped",
		8, 0, 1, '.', StripEscape,
		"\xff\"foo\t\n\tbar\"\xff",
		"\"foo\t\n\tbar\"",
	},

	{
		"1d esc",
		8, 0, 1, '.', 0,
		"\xff\"foo\t\n\tbar\"\xff",
		"\xff\"foo\t\n\tbar\"\xff",
	},

	{
		"1e esc stripped",
		8, 0, 1, '.', StripEscape,
		"abc\xff\tdef", // unterminated escape
		"abc\tdef",
	},

	{
		"1e esc",
		8, 0, 1, '.', 0,
		"abc\xff\tdef", // unterminated escape
		"abc\xff\tdef",
	},

	{
		"2",
		8, 0, 1, '.', 0,
		"\n\n\n",
		"\n\n\n",
	},

	{
		"3",
		8, 0, 1, '.', 0,
		"a\nb\nc",
		"a\nb\nc",
	},

	{
		"4a",
		8, 0, 1, '.', 0,
		"\t", // '\t' terminates an empty cell on last line - nothing to print
		"",
	},

	{
		"4b",
		8, 0, 1, '.', AlignRight,
		"\t", // '\t' terminates an empty cell on last line - nothing to print
		"",
	},

	{
		"5",
		8, 0, 1, '.', 0,
		"*\t*",
		"*.......*",
	},

	{
		"5b",
		8, 0, 1, '.', 0,
		"*\t*\n",
		"*.......*\n",
	},

	{
		"5c",
		8, 0, 1, '.', 0,
		"*\t*\t",
		"*.......*",
	},

	{
		"5c debug",
		8, 0, 1, '.', Debug,
		"*\t*\t",
		"*.......|*",
	},

	{
		"5d",
		8, 0, 1, '.', AlignRight,
		"*\t*\t",
		".......**",
	},

	{
		"6",
		8, 0, 1, '.', 0,
		"\t\n",
		"........\n",
	},

	{
		"7a",
		8, 0, 1, '.', 0,
		"a) foo",
		"a) foo",
	},

	{
		"7b",
		8, 0, 1, ' ', 0,
		"b) foo\tbar",
		"b) foo  bar",
	},

	{
		"7c",
		8, 0, 1, '.', 0,
		"c) foo\tbar\t",
		"c) foo..bar",
	},

	{
		"7d",
		8, 0, 1, '.', 0,
		"d) foo\tbar\n",
		"d) foo..bar\n",
	},

	{
		"7e",
		8, 0, 1, '.', 0,
		"e) foo\tbar\t\n",
		"e) foo..bar.....\n",
	},

	{
		"7f",
		8, 0, 1, '.', FilterHTML,
		"f) f&lt;o\t<b>bar</b>\t\n",
		"f) f&lt;o..<b>bar</b>.....\n",
	},

	{
		"7g",
		8, 0, 1, '.', FilterHTML,
		"g) f&lt;o\t<b>bar</b>\t non-terminated entity &amp",
		"g) f&lt;o..<b>bar</b>..... non-terminated entity &amp",
	},

	{
		"7g debug",
		8, 0, 1, '.', FilterHTML | Debug,
		"g) f&lt;o\t<b>bar</b>\t non-terminated entity &amp",
		"g) f&lt;o..|<b>bar</b>.....| non-terminated entity &amp",
	},

	{
		"8",
		8, 0, 1, '*', 0,
		"Hello, world!\n",
		"Hello, world!\n",
	},

	{
		"9a",
		1, 0, 0, '.', 0,
		"1\t2\t3\t4\n" +
			"11\t222\t3333\t44444\n",

		"1.2..3...4\n" +
			"11222333344444\n",
	},

	{
		"9b",
		1, 0, 0, '.', FilterHTML,
		"1\t2<!---\f--->\t3\t4\n" + // \f inside HTML is ignored
			"11\t222\t3333\t44444\n",

		"1.2<!---\f--->..3...4\n" +
			"11222333344444\n",
	},

	{
		"9c",
		1, 0, 0, '.', 0,
		"1\t2\t3\t4\f" + // \f causes a newline and flush
			"11\t222\t3333\t44444\n",

		"1234\n" +
			"11222333344444\n",
	},

	{
		"9c debug",
		1, 0, 0, '.', Debug,
		"1\t2\t3\t4\f" + // \f causes a newline and flush
			"11\t222\t3333\t44444\n",

		"1|2|3|4\n" +
			"---\n" +
			"11|222|3333|44444\n",
	},

	{
		"10a",
		5, 0, 0, '.', 0,
		"1\t2\t3\t4\n",
		"1....2....3....4\n",
	},

	{
		"10b",
		5, 0, 0, '.', 0,
		"1\t2\t3\t4\t\n",
		"1....2....3....4....\n",
	},

	{
		"11",
		8, 0, 1, '.', 0,
		"本\tb\tc\n" +
			"aa\t\u672c\u672c\u672c\tcccc\tddddd\n" +
			"aaa\tbbbb\n",

		"本.......b.......c\n" +
			"aa......本本本.....cccc....ddddd\n" +
			"aaa.....bbbb\n",
	},

	{
		"12a",
		8, 0, 1, ' ', AlignRight,
		"a\tè\tc\t\n" +
			"aa\tèèè\tcccc\tddddd\t\n" +
			"aaa\tèèèè\t\n",

		"       a       è       c\n" +
			"      aa     èèè    cccc   ddddd\n" +
			"     aaa    èèèè\n",
	},

	{
		"12b",
		2, 0, 0, ' ', 0,
		"a\tb\tc\n" +
			"aa\tbbb\tcccc\n" +
			"aaa\tbbbb\n",

		"a  b  c\n" +
			"aa bbbcccc\n" +
			"aaabbbb\n",
	},

	{
		"12c",
		8, 0, 1, '_', 0,
		"a\tb\tc\n" +
			"aa\tbbb\tcccc\n" +
			"aaa\tbbbb\n",

		"a_______b_______c\n" +
			"aa______bbb_____cccc\n" +
			"aaa_____bbbb\n",
	},

	{
		"13a",
		4, 0, 1, '-', 0,
		"4444\t日本語\t22\t1\t333\n" +
			"999999999\t22\n" +
			"7\t22\n" +
			"\t\t\t88888888\n" +
			"\n" +
			"666666\t666666\t666666\t4444\n" +
			"1\t1\t999999999\t0000000000\n",

		"4444------日本語-22--1---333\n" +
			"999999999-22\n" +
			"7---------22\n" +
			"------------------88888888\n" +
			"\n" +
			"666666-666666-666666----4444\n" +
			"1------1------999999999-0000000000\n",
	},

	{
		"13b",
		4, 0, 3, '.', 0,
		"4444\t333\t22\t1\t333\n" +
			"999999999\t22\n" +
			"7\t22\n" +
			"\t\t\t88888888\n" +
			"\n" +
			"666666\t666666\t666666\t4444\n" +
			"1\t1\t999999999\t0000000000\n",

		"4444........333...22...1...333\n" +
			"999999999...22\n" +
			"7...........22\n" +
			"....................88888888\n" +
			"\n" +
			"666666...666666...666666......4444\n" +
			"1........1........999999999...0000000000\n",
	},

	{
		"13c",
		8, 8, 1, '\t', FilterHTML,
		"4444\t333\t22\t1\t333\n" +
			"999999999\t22\n" +
			"7\t22\n" +
			"\t\t\t88888888\n" +
			"\n" +
			"666666\t666666\t666666\t4444\n" +
			"1\t1\t<font color=red attr=日本語>999999999</font>\t0000000000\n",

		"4444\t\t333\t22\t1\t333\n" +
			"999999999\t22\n" +
			"7\t\t22\n" +
			"\t\t\t\t88888888\n" +
			"\n" +
			"666666\t666666\t666666\t\t4444\n" +
			"1\t1\t<font color=red attr=日本語>999999999</font>\t0000000000\n",
	},

	{
		"14",
		1, 0, 2, ' ', AlignRight,
		".0\t.3\t2.4\t-5.1\t\n" +
			"23.0\t12345678.9\t2.4\t-989.4\t\n" +
			"5.1\t12.0\t2.4\t-7.0\t\n" +
			".0\t0.0\t332.0\t8908.0\t\n" +
			".0\t-.3\t456.4\t22.1\t\n" +
			".0\t1.2\t44.4\t-13.3\t\t",

		"    .0          .3    2.4    -5.1\n" +
			"  23.0  12345678.9    2.4  -989.4\n" +
			"   5.1        12.0    2.4    -7.0\n" +
			"    .0         0.0  332.0  8908.0\n" +
			"    .0         -.3  456.4    22.1\n" +
			"    .0         1.2   44.4   -13.3",
	},

	{
		"14 debug",
		1, 0, 2, ' ', AlignRight | Debug,
		".0\t.3\t2.4\t-5.1\t\n" +
			"23.0\t12345678.9\t2.4\t-989.4\t\n" +
			"5.1\t12.0\t2.4\t-7.0\t\n" +
			".0\t0.0\t332.0\t8908.0\t\n" +
			".0\t-.3\t456.4\t22.1\t\n" +
			".0\t1.2\t44.4\t-13.3\t\t",

		"    .0|          .3|    2.4|    -5.1|\n" +
			"  23.0|  12345678.9|    2.4|  -989.4|\n" +
			"   5.1|        12.0|    2.4|    -7.0|\n" +
			"    .0|         0.0|  332.0|  8908.0|\n" +
			"    .0|         -.3|  456.4|    22.1|\n" +
			"    .0|         1.2|   44.4|   -13.3|",
	},

	{
		"15a",
		4, 0, 0, '.', 0,
		"a\t\tb",
		"a.......b",
	},

	{
		"15b",
		4, 0, 0, '.', DiscardEmptyColumns,
		"a\t\tb", // htabs - do not discard column
		"a.......b",
	},

	{
		"15c",
		4, 0, 0, '.', DiscardEmptyColumns,
		"a\v\vb",
		"a...b",
	},

	{
		"15d",
		4, 0, 0, '.', AlignRight | DiscardEmptyColumns,
		"a\v\vb",
		"...ab",
	},

	{
		"16a",
		100, 100, 0, '\t', 0,
		"a\tb\t\td\n" +
			"a\tb\t\td\te\n" +
			"a\n" +
			"a\tb\tc\td\n" +
			"a\tb\tc\td\te\n",

		"a\tb\t\td\n" +
			"a\tb\t\td\te\n" +
			"a\n" +
			"a\tb\tc\td\n" +
			"a\tb\tc\td\te\n",
	},

	{
		"16b",
		100, 100, 0, '\t', DiscardEmptyColumns,
		"a\vb\v\vd\n" +
			"a\vb\v\vd\ve\n" +
			"a\n" +
			"a\vb\vc\vd\n" +
			"a\vb\vc\vd\ve\n",

		"a\tb\td\n" +
			"a\tb\td\te\n" +
			"a\n" +
			"a\tb\tc\td\n" +
			"a\tb\tc\td\te\n",
	},

	{
		"16b debug",
		100, 100, 0, '\t', DiscardEmptyColumns | Debug,
		"a\vb\v\vd\n" +
			"a\vb\v\vd\ve\n" +
			"a\n" +
			"a\vb\vc\vd\n" +
			"a\vb\vc\vd\ve\n",

		"a\t|b\t||d\n" +
			"a\t|b\t||d\t|e\n" +
			"a\n" +
			"a\t|b\t|c\t|d\n" +
			"a\t|b\t|c\t|d\t|e\n",
	},

	{
		"16c",
		100, 100, 0, '\t', DiscardEmptyColumns,
		"a\tb\t\td\n" + // hard tabs - do not discard column
			"a\tb\t\td\te\n" +
			"a\n" +
			"a\tb\tc\td\n" +
			"a\tb\tc\td\te\n",

		"a\tb\t\td\n" +
			"a\tb\t\td\te\n" +
			"a\n" +
			"a\tb\tc\td\n" +
			"a\tb\tc\td\te\n",
	},

	{
		"16c debug",
		100, 100, 0, '\t', DiscardEmptyColumns | Debug,
		"a\tb\t\td\n" + // hard tabs - do not discard column
			"a\tb\t\td\te\n" +
			"a\n" +
			"a\tb\tc\td\n" +
			"a\tb\tc\td\te\n",

		"a\t|b\t|\t|d\n" +
			"a\t|b\t|\t|d\t|e\n" +
			"a\n" +
			"a\t|b\t|c\t|d\n" +
			"a\t|b\t|c\t|d\t|e\n",
	},
}

func Test(t *testing.T) {
	for _, e := range tests {
		check(t, e.testname, e.minwidth, e.tabwidth, e.padding, e.padchar, e.flags, e.src, e.expected)
	}
}

type panicWriter struct{}

func (panicWriter) Write([]byte) (int, error) {
	panic("cannot write")
}

func wantPanicString(t *testing.T, want string) {
	if e := recover(); e != nil {
		got, ok := e.(string)
		switch {
		case !ok:
			t.Errorf("got %v (%T), want panic string", e, e)
		case got != want:
			t.Errorf("wrong panic message: got %q, want %q", got, want)
		}
	}
}

func TestPanicDuringFlush(t *testing.T) {
	defer wantPanicString(t, "tabwriter: panic during Flush")
	var p panicWriter
	w := new(Writer)
	w.Init(p, 0, 0, 5, ' ', 0)
	io.WriteString(w, "a")
	w.Flush()
	t.Errorf("failed to panic during Flush")
}

func TestPanicDuringWrite(t *testing.T) {
	defer wantPanicString(t, "tabwriter: panic during Write")
	var p panicWriter
	w := new(Writer)
	w.Init(p, 0, 0, 5, ' ', 0)
	io.WriteString(w, "a\n\n") // the second \n triggers a call to w.Write and thus a panic
	t.Errorf("failed to panic during Write")
}

func BenchmarkTable(b *testing.B) {
	for _, w := range [...]int{1, 10, 100} {
		// Build a line with w cells.
		line := bytes.Repeat([]byte("a\t"), w)
		line = append(line, '\n')
		for _, h := range [...]int{10, 1000, 100000} {
			b.Run(fmt.Sprintf("%dx%d", w, h), func(b *testing.B) {
				b.Run("new", func(b *testing.B) {
					b.ReportAllocs()
					for i := 0; i < b.N; i++ {
						w := NewWriter(io.Discard, 4, 4, 1, ' ', 0) // no particular reason for these settings
						// Write the line h times.
						for j := 0; j < h; j++ {
							w.Write(line)
						}
						w.Flush()
					}
				})

				b.Run("reuse", func(b *testing.B) {
					b.ReportAllocs()
					w := NewWriter(io.Discard, 4, 4, 1, ' ', 0) // no particular reason for these settings
					for i := 0; i < b.N; i++ {
						// Write the line h times.
						for j := 0; j < h; j++ {
							w.Write(line)
						}
						w.Flush()
					}
				})
			})
		}
	}
}

func BenchmarkPyramid(b *testing.B) {
	for _, x := range [...]int{10, 100, 1000} {
		// Build a line with x cells.
		line := bytes.Repeat([]byte("a\t"), x)
		b.Run(fmt.Sprintf("%d", x), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				w := NewWriter(io.Discard, 4, 4, 1, ' ', 0) // no particular reason for these settings
				// Write increasing prefixes of that line.
				for j := 0; j < x; j++ {
					w.Write(line[:j*2])
					w.Write([]byte{'\n'})
				}
				w.Flush()
			}
		})
	}
}

func BenchmarkRagged(b *testing.B) {
	var lines [8][]byte
	for i, w := range [8]int{6, 2, 9, 5, 5, 7, 3, 8} {
		// Build a line with w cells.
		lines[i] = bytes.Repeat([]byte("a\t"), w)
	}
	for _, h := range [...]int{10, 100, 1000} {
		b.Run(fmt.Sprintf("%d", h), func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				w := NewWriter(io.Discard, 4, 4, 1, ' ', 0) // no particular reason for these settings
				// Write the lines in turn h times.
				for j := 0; j < h; j++ {
					w.Write(lines[j%len(lines)])
					w.Write([]byte{'\n'})
				}
				w.Flush()
			}
		})
	}
}

const codeSnippet = `
some command

foo	# aligned
barbaz	# comments

but
mostly
single
cell
lines
`

func BenchmarkCode(b *testing.B) {
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		w := NewWriter(io.Discard, 4, 4, 1, ' ', 0) // no particular reason for these settings
		// The code is small, so it's reasonable for the tabwriter user
		// to write it all at once, or buffer the writes.
		w.Write([]byte(codeSnippet))
		w.Flush()
	}
}
