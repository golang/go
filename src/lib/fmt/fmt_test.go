// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt

import (
	"fmt";
	"io";
	"syscall";
	"testing";
)

export func TestFmtInterface(t *testing.T) {
	var i1 interface{};
	i1 = "abc";
	s := fmt.sprintf("%s", i1);
	if s != "abc" {
		t.Errorf(`fmt.sprintf("%%s", empty("abc")) = %q want %q`, s, "abc");
	}
}

type FmtTest struct {
	fmt string;
	val interface { };
	out string;
}

func Bytes(s string) *[]byte {
	b := new([]byte, len(s)+1);
	syscall.StringToBytes(b, s);
	return b[0:len(s)];
}

const B32 uint32 = 1<<32 - 1
const B64 uint64 = 1<<64 - 1

var fmttests = []FmtTest{
	// basic string
	FmtTest{ "%s",	"abc",	"abc" },
	FmtTest{ "%x",	"abc",	"616263" },
	FmtTest{ "%x",	"xyz",	"78797a" },
	FmtTest{ "%X",	"xyz",	"78797A" },
	FmtTest{ "%q",	"abc",	`"abc"` },

	// basic bytes
	FmtTest{ "%s",	Bytes("abc"),	"abc" },
	FmtTest{ "%x",	Bytes("abc"),	"616263" },
	FmtTest{ "% x",	Bytes("abc"),	"61 62 63" },
	FmtTest{ "%x",	Bytes("xyz"),	"78797a" },
	FmtTest{ "%X",	Bytes("xyz"),	"78797A" },
	FmtTest{ "%q",	Bytes("abc"),	`"abc"` },

	// escaped strings
	FmtTest{ "%#q",	`abc`,		"`abc`" },
	FmtTest{ "%#q",	`"`,		"`\"`" },
	FmtTest{ "1 %#q", `\n`,		"1 `\\n`" },
	FmtTest{ "2 %#q", "\n",		`2 "\n"` },
	FmtTest{ "%q",	`"`,		`"\""` },
	FmtTest{ "%q",	"\a\b\f\r\n\t\v",	`"\a\b\f\r\n\t\v"` },
	FmtTest{ "%q",	"abc\xffdef",		`"abc\xffdef"` },
	FmtTest{ "%q",	"\u263a",	`"\u263a"` },
	FmtTest{ "%q",	"\U0010ffff",	`"\U0010ffff"` },

	// width
	FmtTest{ "%5s",		"abc",	"  abc" },
	FmtTest{ "%-5s",	"abc",	"abc  " },
	FmtTest{ "%05s",	"abc",	"00abc" },

	// integers
	FmtTest{ "%d",		12345,	"12345" },
	FmtTest{ "%d",		-12345,	"-12345" },
	FmtTest{ "%10d",	12345,	"     12345" },
	FmtTest{ "%10d",	-12345,	"    -12345" },
	FmtTest{ "%+10d",	12345,	"    +12345" },
	FmtTest{ "%010d",	12345,	"0000012345" },
	FmtTest{ "%010d",	-12345,	"-000012345" },
	FmtTest{ "%-10d",	12345,	"12345     " },
	FmtTest{ "%010.3d",	1,	"       001" },
	FmtTest{ "%010.3d",	-1,	"      -001" },
	FmtTest{ "%+d",		12345,	"+12345" },
	FmtTest{ "%+d",		-12345,	"-12345" },
	FmtTest{ "% d",		12345,	" 12345" },
	FmtTest{ "% d",		-12345,	"-12345" },

	// old test/fmt_test.go
	FmtTest{ "%d",		1234,			"1234" },
	FmtTest{ "%d",		-1234,			"-1234" },
	FmtTest{ "%d",		uint(1234),		"1234" },
	FmtTest{ "%d",		uint32(B32),		"4294967295" },
	FmtTest{ "%d",		uint64(B64),		"18446744073709551615" },
	FmtTest{ "%o",		01234,			"1234" },
	FmtTest{ "%o",		uint32(B32),		"37777777777" },
	FmtTest{ "%o",		uint64(B64),		"1777777777777777777777" },
	FmtTest{ "%x",		0x1234abcd,		"1234abcd" },
	FmtTest{ "%x",		B32-0x1234567,		"fedcba98" },
	FmtTest{ "%X",		0x1234abcd,		"1234ABCD" },
	FmtTest{ "%X",		B32-0x1234567,		"FEDCBA98" },
	FmtTest{ "%x",		B64,			"ffffffffffffffff" },
	FmtTest{ "%b",		7,			"111" },
	FmtTest{ "%b",		B64,			"1111111111111111111111111111111111111111111111111111111111111111" },
	FmtTest{ "%e",		float64(1),		"1.000000e+00" },
	FmtTest{ "%e",		float64(1234.5678e3),	"1.234568e+06" },
	FmtTest{ "%e",		float64(1234.5678e-8),	"1.234568e-05" },
	FmtTest{ "%e",		float64(-7),		"-7.000000e+00" },
	FmtTest{ "%e",		float64(-1e-9),		"-1.000000e-09" },
	FmtTest{ "%f",		float64(1234.5678e3),	"1234567.800000" },
	FmtTest{ "%f",		float64(1234.5678e-8),	"0.000012" },
	FmtTest{ "%f",		float64(-7),		"-7.000000" },
	FmtTest{ "%f",		float64(-1e-9),		"-0.000000" },
	FmtTest{ "%g",		float64(1234.5678e3),	"1.2345678e+06" },
	FmtTest{ "%g",		float32(1234.5678e3),	"1.2345678e+06" },
	FmtTest{ "%g",		float64(1234.5678e-8),	"1.2345678e-05" },
	FmtTest{ "%g",		float64(-7),		"-7" },
	FmtTest{ "%g",		float64(-1e-9),		"-1e-09",	 },
	FmtTest{ "%g",		float32(-1e-9),		"-1e-09" },
	FmtTest{ "%c",		'x',			"x" },
	FmtTest{ "%c",		0xe4,			"ä" },
	FmtTest{ "%c",		0x672c,			"本" },
	FmtTest{ "%c",		'日',			"日" },
	FmtTest{ "%20.8d",	1234,			"            00001234" },
	FmtTest{ "%20.8d",	-1234,			"           -00001234" },
	FmtTest{ "%20d",	1234,			"                1234" },
	FmtTest{ "%-20.8d",	1234,			"00001234            " },
	FmtTest{ "%-20.8d",	-1234,			"-00001234           " },
	FmtTest{ "%.20b",	7,			"00000000000000000111" },
	FmtTest{ "%20.5s",	"qwertyuiop",		"               qwert" },
	FmtTest{ "%.5s",	"qwertyuiop",		"qwert" },
	FmtTest{ "%-20.5s",	"qwertyuiop",		"qwert               " },
	FmtTest{ "%20c",	'x',			"                   x" },
	FmtTest{ "%-20c",	'x',			"x                   " },
	FmtTest{ "%20.6e",	1.2345e3,		"        1.234500e+03" },
	FmtTest{ "%20.6e",	1.2345e-3,		"        1.234500e-03" },
	FmtTest{ "%20e",	1.2345e3,		"        1.234500e+03" },
	FmtTest{ "%20e",	1.2345e-3,		"        1.234500e-03" },
	FmtTest{ "%20.8e",	1.2345e3,		"      1.23450000e+03" },
	FmtTest{ "%20f",	float64(1.23456789e3),	"         1234.567890" },
	FmtTest{ "%20f",	float64(1.23456789e-3),	"            0.001235" },
	FmtTest{ "%20f",	float64(12345678901.23456789),	"  12345678901.234568" },
	FmtTest{ "%-20f",	float64(1.23456789e3),	"1234.567890         " },
	FmtTest{ "%20.8f",	float64(1.23456789e3),	"       1234.56789000" },
	FmtTest{ "%20.8f",	float64(1.23456789e-3),	"          0.00123457" },
	FmtTest{ "%g",		float64(1.23456789e3),	"1234.56789" },
	FmtTest{ "%g",		float64(1.23456789e-3),	"0.00123456789" },
	FmtTest{ "%g",		float64(1.23456789e20),	"1.23456789e+20" },
	FmtTest{ "%20e",	sys.Inf(1),		"                +Inf" },
	FmtTest{ "%-20f",	sys.Inf(-1),		"-Inf                " },
	FmtTest{ "%20g",	sys.NaN(),		"                 NaN" },
}

export func TestSprintf(t *testing.T) {
	for i := 0; i < len(fmttests); i++ {
		tt := fmttests[i];
		s := fmt.sprintf(tt.fmt, tt.val);
		if s != tt.out {
			if ss, ok := tt.val.(string); ok {
				// Don't requote the already-quoted strings.
				// It's too confusing to read the errors.
				t.Errorf("fmt.sprintf(%q, %q) = %s want %s", tt.fmt, tt.val, s, tt.out);
			} else {
				t.Errorf("fmt.sprintf(%q, %v) = %q want %q", tt.fmt, tt.val, s, tt.out);
			}
		}
	}
}

type FlagPrinter struct { }
func (*FlagPrinter) Format(f fmt.Formatter, c int) {
	s := "%";
	for i := 0; i < 128; i++ {
		if f.Flag(i) {
			s += string(i);
		}
	}
	if w, ok := f.Width(); ok {
		s += fmt.sprintf("%d", w);
	}
	if p, ok := f.Precision(); ok {
		s += fmt.sprintf(".%d", p);
	}
	s += string(c);
	io.WriteString(f, "["+s+"]");
}

type FlagTest struct {
	in string;
	out string;
}

var flagtests = []FlagTest {
	FlagTest{ "%a", "[%a]" },
	FlagTest{ "%-a", "[%-a]" },
	FlagTest{ "%+a", "[%+a]" },
	FlagTest{ "%#a", "[%#a]" },
	FlagTest{ "% a", "[% a]" },
	FlagTest{ "%0a", "[%0a]" },
	FlagTest{ "%1.2a", "[%1.2a]" },
	FlagTest{ "%-1.2a", "[%-1.2a]" },
	FlagTest{ "%+1.2a", "[%+1.2a]" },
	FlagTest{ "%-+1.2a", "[%+-1.2a]" },
	FlagTest{ "%-+1.2abc", "[%+-1.2a]bc" },
	FlagTest{ "%-1.2abc", "[%-1.2a]bc" },
}

export func TestFlagParser(t *testing.T) {
	var flagprinter FlagPrinter;
	for i := 0; i < len(flagtests); i++ {
		tt := flagtests[i];
		s := fmt.sprintf(tt.in, &flagprinter);
		if s != tt.out {
			t.Errorf("sprintf(%q, &flagprinter) => %q, want %q", tt.in, s, tt.out);
		}
	}
}

export func TestStructPrinter(t *testing.T) {
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
		out := fmt.sprintf(tt.fmt, s);
		if out != tt.out {
			t.Errorf("sprintf(%q, &s) = %q, want %q", tt.fmt, out, tt.out);
		}
	}
}
