// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tabwriter

import (
	"os";
	"io";
	"tabwriter";
	"testing";
)


type Buffer struct {
	a *[]byte;
}


func (b *Buffer) Init(n int) {
	b.a = new([]byte, n)[0 : 0];
}


func (b *Buffer) Clear() {
	b.a = b.a[0 : 0];
}


func (b *Buffer) Write(buf *[]byte) (written int, err *os.Error) {
	n := len(b.a);
	m := len(buf);
	if n + m <= cap(b.a) {
		b.a = b.a[0 : n + m];
		for i := 0; i < m; i++ {
			b.a[n+i] = buf[i];
		}
	} else {
		panicln("buffer too small", n, m, cap(b.a));
	}
	return len(buf), nil;
}


func (b *Buffer) String() string {
	return string(b.a);
}


func Write(t *testing.T, w *tabwriter.Writer, src string) {
	written, err := io.WriteString(w, src);
	if err != nil {
		t.Errorf("--- src:\n%s\n--- write error: %v\n", src, err);
	}
	if written != len(src) {
		t.Errorf("--- src:\n%s\n--- written = %d, len(src) = %d\n", src, written, len(src));
	}
}


func Verify(t *testing.T, w *tabwriter.Writer, b *Buffer, src, expected string) {
	err := w.Flush();
	if err != nil {
		t.Errorf("--- src:\n%s\n--- flush error: %v\n", src, err);
	}

	res := b.String();
	if res != expected {
		t.Errorf("--- src:\n%s\n--- found:\n%s\n--- expected:\n%s\n", src, res, expected)
	}
}


func Check(t *testing.T, tabwidth, padding int, padchar byte, align_left, filter_html bool, src, expected string) {
	var b Buffer;
	b.Init(1000);

	var w tabwriter.Writer;
	w.Init(&b, tabwidth, padding, padchar, align_left, filter_html);

	// write all at once
	b.Clear();
	Write(t, &w, src);
	Verify(t, &w, &b, src, expected);

	// write byte-by-byte
	b.Clear();
	for i := 0; i < len(src); i++ {
		Write(t, &w, src[i : i+1]);
	}
	Verify(t, &w, &b, src, expected);

	// write using Fibonacci slice sizes
	b.Clear();
	for i, d := 0, 0; i < len(src); {
		Write(t, &w, src[i : i+d]);
		i, d = i+d, d+1;
		if i+d > len(src) {
			d = len(src) - i;
		}
	}
	Verify(t, &w, &b, src, expected);
}


export func Test(t *testing.T) {
	Check(
		t, 8, 1, '.', true, false,
		"",
		""
	);

	Check(
		t, 8, 1, '.', true, false,
		"\n\n\n",
		"\n\n\n"
	);

	Check(
		t, 8, 1, '.', true, false,
		"a\nb\nc",
		"a\nb\nc"
	);

	Check(
		t, 8, 1, '.', true, false,
		"\t",  // '\t' terminates an empty cell on last line - nothing to print
		""
	);

	Check(
		t, 8, 1, '.', false, false,
		"\t",  // '\t' terminates an empty cell on last line - nothing to print
		""
	);

	Check(
		t, 8, 1, '.', true, false,
		"*\t*",
		"**"
	);

	Check(
		t, 8, 1, '.', true, false,
		"*\t*\n",
		"*.......*\n"
	);

	Check(
		t, 8, 1, '.', true, false,
		"*\t*\t",
		"*.......*"
	);

	Check(
		t, 8, 1, '.', false, false,
		"*\t*\t",
		".......**"
	);

	Check(
		t, 8, 1, '.', true, false,
		"\t\n",
		"........\n"
	);

	Check(
		t, 8, 1, '.', true, false,
		"a) foo",
		"a) foo"
	);

	Check(
		t, 8, 1, ' ', true, false,
		"b) foo\tbar",  // "bar" is not in any cell - not formatted, just flushed
		"b) foobar"
	);

	Check(
		t, 8, 1, '.', true, false,
		"c) foo\tbar\t",
		"c) foo..bar"
	);

	Check(
		t, 8, 1, '.', true, false,
		"d) foo\tbar\n",
		"d) foo..bar\n"
	);

	Check(
		t, 8, 1, '.', true, false,
		"e) foo\tbar\t\n",
		"e) foo..bar.....\n"
	);

	Check(
		t, 8, 1, '.', true, true,
		"e) f&lt;o\t<b>bar</b>\t\n",
		"e) f&lt;o..<b>bar</b>.....\n"
	);

	Check(
		t, 8, 1, '*', true, false,
		"Hello, world!\n",
		"Hello, world!\n"
	);

	Check(
		t, 0, 0, '.', true, false,
		"1\t2\t3\t4\n"
		"11\t222\t3333\t44444\n",

		"1.2..3...4\n"
		"11222333344444\n"
	);

	Check(
		t, 5, 0, '.', true, false,
		"1\t2\t3\t4\n",
		"1....2....3....4\n"
	);

	Check(
		t, 5, 0, '.', true, false,
		"1\t2\t3\t4\t\n",
		"1....2....3....4....\n"
	);

	Check(
		t, 8, 1, '.', true, false,
		"本\tb\tc\n"
		"aa\t\u672c\u672c\u672c\tcccc\tddddd\n"
		"aaa\tbbbb\n",

		"本.......b.......c\n"
		"aa......本本本.....cccc....ddddd\n"
		"aaa.....bbbb\n"
	);

	Check(
		t, 8, 1, ' ', false, false,
		"a\tè\tc\t\n"
		"aa\tèèè\tcccc\tddddd\t\n"
		"aaa\tèèèè\t\n",

		"       a       è       c\n"
		"      aa     èèè    cccc   ddddd\n"
		"     aaa    èèèè\n"
	);

	Check(
		t, 2, 0, ' ', true, false,
		"a\tb\tc\n"
		"aa\tbbb\tcccc\n"
		"aaa\tbbbb\n",

		"a  b  c\n"
		"aa bbbcccc\n"
		"aaabbbb\n"
	);

	Check(
		t, 8, 1, '_', true, false,
		"a\tb\tc\n"
		"aa\tbbb\tcccc\n"
		"aaa\tbbbb\n",

		"a_______b_______c\n"
		"aa______bbb_____cccc\n"
		"aaa_____bbbb\n"
	);

	Check(
		t, 4, 1, '-', true, false,
		"4444\t日本語\t22\t1\t333\n"
		"999999999\t22\n"
		"7\t22\n"
		"\t\t\t88888888\n"
		"\n"
		"666666\t666666\t666666\t4444\n"
		"1\t1\t999999999\t0000000000\n",

		"4444------日本語-22--1---333\n"
		"999999999-22\n"
		"7---------22\n"
		"------------------88888888\n"
		"\n"
		"666666-666666-666666----4444\n"
		"1------1------999999999-0000000000\n"
	);

	Check(
		t, 4, 3, '.', true, false,
		"4444\t333\t22\t1\t333\n"
		"999999999\t22\n"
		"7\t22\n"
		"\t\t\t88888888\n"
		"\n"
		"666666\t666666\t666666\t4444\n"
		"1\t1\t999999999\t0000000000\n",

		"4444........333...22...1...333\n"
		"999999999...22\n"
		"7...........22\n"
		"....................88888888\n"
		"\n"
		"666666...666666...666666......4444\n"
		"1........1........999999999...0000000000\n"
	);

	Check(
		t, 8, 1, '\t', true, true,
		"4444\t333\t22\t1\t333\n"
		"999999999\t22\n"
		"7\t22\n"
		"\t\t\t88888888\n"
		"\n"
		"666666\t666666\t666666\t4444\n"
		"1\t1\t<font color=red attr=日本語>999999999</font>\t0000000000\n",

		"4444\t\t333\t22\t1\t333\n"
		"999999999\t22\n"
		"7\t\t22\n"
		"\t\t\t\t88888888\n"
		"\n"
		"666666\t666666\t666666\t\t4444\n"
		"1\t1\t<font color=red attr=日本語>999999999</font>\t0000000000\n"
	);

	Check(
		t, 0, 2, ' ', false, false,
		".0\t.3\t2.4\t-5.1\t\n"
		"23.0\t12345678.9\t2.4\t-989.4\t\n"
		"5.1\t12.0\t2.4\t-7.0\t\n"
		".0\t0.0\t332.0\t8908.0\t\n"
		".0\t-.3\t456.4\t22.1\t\n"
		".0\t1.2\t44.4\t-13.3\t\t",

		"    .0          .3    2.4    -5.1\n"
		"  23.0  12345678.9    2.4  -989.4\n"
		"   5.1        12.0    2.4    -7.0\n"
		"    .0         0.0  332.0  8908.0\n"
		"    .0         -.3  456.4    22.1\n"
		"    .0         1.2   44.4   -13.3"
	);
}
