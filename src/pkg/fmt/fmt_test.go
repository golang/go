// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"fmt";
	"io";
	"math";
	"testing";
)

func TestFmtInterface(t *testing.T) {
	var i1 interface{};
	i1 = "abc";
	s := fmt.Sprintf("%s", i1);
	if s != "abc" {
		t.Errorf(`fmt.Sprintf("%%s", empty("abc")) = %q want %q`, s, "abc");
	}
}

type fmtTest struct {
	fmt string;
	val interface { };
	out string;
}

const b32 uint32 = 1<<32 - 1
const b64 uint64 = 1<<64 - 1
var array = []int{1, 2, 3, 4, 5}
var iarray = []interface{}{1, "hello", 2.5, nil}

var fmttests = []fmtTest{
	// basic string
	fmtTest{ "%s",	"abc",	"abc" },
	fmtTest{ "%x",	"abc",	"616263" },
	fmtTest{ "%x",	"xyz",	"78797a" },
	fmtTest{ "%X",	"xyz",	"78797A" },
	fmtTest{ "%q",	"abc",	`"abc"` },

	// basic bytes
	fmtTest{ "%s",	io.StringBytes("abc"),	"abc" },
	fmtTest{ "%x",	io.StringBytes("abc"),	"616263" },
	fmtTest{ "% x",	io.StringBytes("abc"),	"61 62 63" },
	fmtTest{ "%x",	io.StringBytes("xyz"),	"78797a" },
	fmtTest{ "%X",	io.StringBytes("xyz"),	"78797A" },
	fmtTest{ "%q",	io.StringBytes("abc"),	`"abc"` },

	// escaped strings
	fmtTest{ "%#q",	`abc`,		"`abc`" },
	fmtTest{ "%#q",	`"`,		"`\"`" },
	fmtTest{ "1 %#q", `\n`,		"1 `\\n`" },
	fmtTest{ "2 %#q", "\n",		`2 "\n"` },
	fmtTest{ "%q",	`"`,		`"\""` },
	fmtTest{ "%q",	"\a\b\f\r\n\t\v",	`"\a\b\f\r\n\t\v"` },
	fmtTest{ "%q",	"abc\xffdef",		`"abc\xffdef"` },
	fmtTest{ "%q",	"\u263a",	`"\u263a"` },
	fmtTest{ "%q",	"\U0010ffff",	`"\U0010ffff"` },

	// width
	fmtTest{ "%5s",		"abc",	"  abc" },
	fmtTest{ "%-5s",	"abc",	"abc  " },
	fmtTest{ "%05s",	"abc",	"00abc" },

	// integers
	fmtTest{ "%d",		12345,	"12345" },
	fmtTest{ "%d",		-12345,	"-12345" },
	fmtTest{ "%10d",	12345,	"     12345" },
	fmtTest{ "%10d",	-12345,	"    -12345" },
	fmtTest{ "%+10d",	12345,	"    +12345" },
	fmtTest{ "%010d",	12345,	"0000012345" },
	fmtTest{ "%010d",	-12345,	"-000012345" },
	fmtTest{ "%-10d",	12345,	"12345     " },
	fmtTest{ "%010.3d",	1,	"       001" },
	fmtTest{ "%010.3d",	-1,	"      -001" },
	fmtTest{ "%+d",		12345,	"+12345" },
	fmtTest{ "%+d",		-12345,	"-12345" },
	fmtTest{ "% d",		12345,	" 12345" },

	// arrays
	fmtTest{ "%v",		array,			"[1 2 3 4 5]" },
	fmtTest{ "%v",		iarray,			"[1 hello 2.5 <nil>]" },
	fmtTest{ "%v",		&array,			"&[1 2 3 4 5]" },
	fmtTest{ "%v",		&iarray,			"&[1 hello 2.5 <nil>]" },

	// old test/fmt_test.go
	fmtTest{ "%d",		1234,			"1234" },
	fmtTest{ "%d",		-1234,			"-1234" },
	fmtTest{ "%d",		uint(1234),		"1234" },
	fmtTest{ "%d",		uint32(b32),		"4294967295" },
	fmtTest{ "%d",		uint64(b64),		"18446744073709551615" },
	fmtTest{ "%o",		01234,			"1234" },
	fmtTest{ "%#o",		01234,			"01234" },
	fmtTest{ "%o",		uint32(b32),		"37777777777" },
	fmtTest{ "%o",		uint64(b64),		"1777777777777777777777" },
	fmtTest{ "%x",		0x1234abcd,		"1234abcd" },
	fmtTest{ "%#x",		0x1234abcd,		"0x1234abcd" },
	fmtTest{ "%x",		b32-0x1234567,		"fedcba98" },
	fmtTest{ "%X",		0x1234abcd,		"1234ABCD" },
	fmtTest{ "%X",		b32-0x1234567,		"FEDCBA98" },
	fmtTest{ "%#X",		0,		"0X0" },
	fmtTest{ "%x",		b64,			"ffffffffffffffff" },
	fmtTest{ "%b",		7,			"111" },
	fmtTest{ "%b",		b64,			"1111111111111111111111111111111111111111111111111111111111111111" },
	fmtTest{ "%e",		float64(1),		"1.000000e+00" },
	fmtTest{ "%e",		float64(1234.5678e3),	"1.234568e+06" },
	fmtTest{ "%e",		float64(1234.5678e-8),	"1.234568e-05" },
	fmtTest{ "%e",		float64(-7),		"-7.000000e+00" },
	fmtTest{ "%e",		float64(-1e-9),		"-1.000000e-09" },
	fmtTest{ "%f",		float64(1234.5678e3),	"1234567.800000" },
	fmtTest{ "%f",		float64(1234.5678e-8),	"0.000012" },
	fmtTest{ "%f",		float64(-7),		"-7.000000" },
	fmtTest{ "%f",		float64(-1e-9),		"-0.000000" },
	fmtTest{ "%g",		float64(1234.5678e3),	"1.2345678e+06" },
	fmtTest{ "%g",		float32(1234.5678e3),	"1.2345678e+06" },
	fmtTest{ "%g",		float64(1234.5678e-8),	"1.2345678e-05" },
	fmtTest{ "%g",		float64(-7),		"-7" },
	fmtTest{ "%g",		float64(-1e-9),		"-1e-09",	 },
	fmtTest{ "%g",		float32(-1e-9),		"-1e-09" },
	fmtTest{ "%c",		'x',			"x" },
	fmtTest{ "%c",		0xe4,			"ä" },
	fmtTest{ "%c",		0x672c,			"本" },
	fmtTest{ "%c",		'日',			"日" },
	fmtTest{ "%20.8d",	1234,			"            00001234" },
	fmtTest{ "%20.8d",	-1234,			"           -00001234" },
	fmtTest{ "%20d",	1234,			"                1234" },
	fmtTest{ "%-20.8d",	1234,			"00001234            " },
	fmtTest{ "%-20.8d",	-1234,			"-00001234           " },
	fmtTest{ "%-#20.8x",		0x1234abc,		"0x01234abc          " },
	fmtTest{ "%-#20.8X",		0x1234abc,		"0X01234ABC          " },
	fmtTest{ "%-#20.8o",		01234,		"00001234            " },
	fmtTest{ "%.20b",	7,			"00000000000000000111" },
	fmtTest{ "%20.5s",	"qwertyuiop",		"               qwert" },
	fmtTest{ "%.5s",	"qwertyuiop",		"qwert" },
	fmtTest{ "%-20.5s",	"qwertyuiop",		"qwert               " },
	fmtTest{ "%20c",	'x',			"                   x" },
	fmtTest{ "%-20c",	'x',			"x                   " },
	fmtTest{ "%20.6e",	1.2345e3,		"        1.234500e+03" },
	fmtTest{ "%20.6e",	1.2345e-3,		"        1.234500e-03" },
	fmtTest{ "%20e",	1.2345e3,		"        1.234500e+03" },
	fmtTest{ "%20e",	1.2345e-3,		"        1.234500e-03" },
	fmtTest{ "%20.8e",	1.2345e3,		"      1.23450000e+03" },
	fmtTest{ "%20f",	float64(1.23456789e3),	"         1234.567890" },
	fmtTest{ "%20f",	float64(1.23456789e-3),	"            0.001235" },
	fmtTest{ "%20f",	float64(12345678901.23456789),	"  12345678901.234568" },
	fmtTest{ "%-20f",	float64(1.23456789e3),	"1234.567890         " },
	fmtTest{ "%20.8f",	float64(1.23456789e3),	"       1234.56789000" },
	fmtTest{ "%20.8f",	float64(1.23456789e-3),	"          0.00123457" },
	fmtTest{ "%g",		float64(1.23456789e3),	"1234.56789" },
	fmtTest{ "%g",		float64(1.23456789e-3),	"0.00123456789" },
	fmtTest{ "%g",		float64(1.23456789e20),	"1.23456789e+20" },
	fmtTest{ "%20e",	math.Inf(1),		"                +Inf" },
	fmtTest{ "%-20f",	math.Inf(-1),		"-Inf                " },
	fmtTest{ "%20g",	math.NaN(),		"                 NaN" },
}

func TestSprintf(t *testing.T) {
	for i := 0; i < len(fmttests); i++ {
		tt := fmttests[i];
		s := fmt.Sprintf(tt.fmt, tt.val);
		if s != tt.out {
			if ss, ok := tt.val.(string); ok {
				// Don't requote the already-quoted strings.
				// It's too confusing to read the errors.
				t.Errorf("fmt.Sprintf(%q, %q) = %s want %s", tt.fmt, tt.val, s, tt.out);
			} else {
				t.Errorf("fmt.Sprintf(%q, %v) = %q want %q", tt.fmt, tt.val, s, tt.out);
			}
		}
	}
}

type flagPrinter struct { }
func (*flagPrinter) Format(f fmt.State, c int) {
	s := "%";
	for i := 0; i < 128; i++ {
		if f.Flag(i) {
			s += string(i);
		}
	}
	if w, ok := f.Width(); ok {
		s += fmt.Sprintf("%d", w);
	}
	if p, ok := f.Precision(); ok {
		s += fmt.Sprintf(".%d", p);
	}
	s += string(c);
	io.WriteString(f, "["+s+"]");
}

type flagTest struct {
	in string;
	out string;
}

var flagtests = []flagTest {
	flagTest{ "%a", "[%a]" },
	flagTest{ "%-a", "[%-a]" },
	flagTest{ "%+a", "[%+a]" },
	flagTest{ "%#a", "[%#a]" },
	flagTest{ "% a", "[% a]" },
	flagTest{ "%0a", "[%0a]" },
	flagTest{ "%1.2a", "[%1.2a]" },
	flagTest{ "%-1.2a", "[%-1.2a]" },
	flagTest{ "%+1.2a", "[%+1.2a]" },
	flagTest{ "%-+1.2a", "[%+-1.2a]" },
	flagTest{ "%-+1.2abc", "[%+-1.2a]bc" },
	flagTest{ "%-1.2abc", "[%-1.2a]bc" },
}

func TestFlagParser(t *testing.T) {
	var flagprinter flagPrinter;
	for i := 0; i < len(flagtests); i++ {
		tt := flagtests[i];
		s := fmt.Sprintf(tt.in, &flagprinter);
		if s != tt.out {
			t.Errorf("Sprintf(%q, &flagprinter) => %q, want %q", tt.in, s, tt.out);
		}
	}
}

func TestStructPrinter(t *testing.T) {
	var s struct {
		a string;
		b string;
		c int;
	};
	s.a = "abc";
	s.b = "def";
	s.c = 123;
	type Test struct {
		fmt string;
		out string;
	}
	var tests = []Test {
		Test{ "%v", "{abc def 123}" },
		Test{ "%+v", "{a=abc b=def c=123}" },
	};
	for i := 0; i < len(tests); i++ {
		tt := tests[i];
		out := fmt.Sprintf(tt.fmt, s);
		if out != tt.out {
			t.Errorf("Sprintf(%q, &s) = %q, want %q", tt.fmt, out, tt.out);
		}
	}
}
