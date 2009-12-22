// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	. "fmt"
	"io"
	"malloc" // for the malloc count test only
	"math"
	"strings"
	"testing"
)

func TestFmtInterface(t *testing.T) {
	var i1 interface{}
	i1 = "abc"
	s := Sprintf("%s", i1)
	if s != "abc" {
		t.Errorf(`Sprintf("%%s", empty("abc")) = %q want %q`, s, "abc")
	}
}

type fmtTest struct {
	fmt string
	val interface{}
	out string
}

const b32 uint32 = 1<<32 - 1
const b64 uint64 = 1<<64 - 1

var array = []int{1, 2, 3, 4, 5}
var iarray = []interface{}{1, "hello", 2.5, nil}

type A struct {
	i int
	j uint
	s string
	x []int
}

type I int

func (i I) String() string { return Sprintf("<%d>", i) }

type B struct {
	i I
	j int
}

type C struct {
	i int
	B
}

var b byte

var fmttests = []fmtTest{
	// basic string
	fmtTest{"%s", "abc", "abc"},
	fmtTest{"%x", "abc", "616263"},
	fmtTest{"%x", "xyz", "78797a"},
	fmtTest{"%X", "xyz", "78797A"},
	fmtTest{"%q", "abc", `"abc"`},

	// basic bytes
	fmtTest{"%s", strings.Bytes("abc"), "abc"},
	fmtTest{"%x", strings.Bytes("abc"), "616263"},
	fmtTest{"% x", strings.Bytes("abc"), "61 62 63"},
	fmtTest{"%x", strings.Bytes("xyz"), "78797a"},
	fmtTest{"%X", strings.Bytes("xyz"), "78797A"},
	fmtTest{"%q", strings.Bytes("abc"), `"abc"`},

	// escaped strings
	fmtTest{"%#q", `abc`, "`abc`"},
	fmtTest{"%#q", `"`, "`\"`"},
	fmtTest{"1 %#q", `\n`, "1 `\\n`"},
	fmtTest{"2 %#q", "\n", `2 "\n"`},
	fmtTest{"%q", `"`, `"\""`},
	fmtTest{"%q", "\a\b\f\r\n\t\v", `"\a\b\f\r\n\t\v"`},
	fmtTest{"%q", "abc\xffdef", `"abc\xffdef"`},
	fmtTest{"%q", "\u263a", `"\u263a"`},
	fmtTest{"%q", "\U0010ffff", `"\U0010ffff"`},

	// width
	fmtTest{"%5s", "abc", "  abc"},
	fmtTest{"%-5s", "abc", "abc  "},
	fmtTest{"%05s", "abc", "00abc"},

	// integers
	fmtTest{"%d", 12345, "12345"},
	fmtTest{"%d", -12345, "-12345"},
	fmtTest{"%10d", 12345, "     12345"},
	fmtTest{"%10d", -12345, "    -12345"},
	fmtTest{"%+10d", 12345, "    +12345"},
	fmtTest{"%010d", 12345, "0000012345"},
	fmtTest{"%010d", -12345, "-000012345"},
	fmtTest{"%-10d", 12345, "12345     "},
	fmtTest{"%010.3d", 1, "       001"},
	fmtTest{"%010.3d", -1, "      -001"},
	fmtTest{"%+d", 12345, "+12345"},
	fmtTest{"%+d", -12345, "-12345"},
	fmtTest{"%+d", 0, "+0"},
	fmtTest{"% d", 0, " 0"},
	fmtTest{"% d", 12345, " 12345"},

	// floats
	fmtTest{"%+.3e", 0.0, "+0.000e+00"},
	fmtTest{"%+.3e", 1.0, "+1.000e+00"},
	fmtTest{"%+.3f", -1.0, "-1.000"},
	fmtTest{"% .3E", -1.0, "-1.000E+00"},
	fmtTest{"% .3e", 1.0, " 1.000e+00"},
	fmtTest{"%+.3g", 0.0, "+0"},
	fmtTest{"%+.3g", 1.0, "+1"},
	fmtTest{"%+.3g", -1.0, "-1"},
	fmtTest{"% .3g", -1.0, "-1"},
	fmtTest{"% .3g", 1.0, " 1"},

	// erroneous formats
	fmtTest{"", 2, "?(extra int=2)"},
	fmtTest{"%d", "hello", "%d(string=hello)"},

	// old test/fmt_test.go
	fmtTest{"%d", 1234, "1234"},
	fmtTest{"%d", -1234, "-1234"},
	fmtTest{"%d", uint(1234), "1234"},
	fmtTest{"%d", uint32(b32), "4294967295"},
	fmtTest{"%d", uint64(b64), "18446744073709551615"},
	fmtTest{"%o", 01234, "1234"},
	fmtTest{"%#o", 01234, "01234"},
	fmtTest{"%o", uint32(b32), "37777777777"},
	fmtTest{"%o", uint64(b64), "1777777777777777777777"},
	fmtTest{"%x", 0x1234abcd, "1234abcd"},
	fmtTest{"%#x", 0x1234abcd, "0x1234abcd"},
	fmtTest{"%x", b32 - 0x1234567, "fedcba98"},
	fmtTest{"%X", 0x1234abcd, "1234ABCD"},
	fmtTest{"%X", b32 - 0x1234567, "FEDCBA98"},
	fmtTest{"%#X", 0, "0X0"},
	fmtTest{"%x", b64, "ffffffffffffffff"},
	fmtTest{"%b", 7, "111"},
	fmtTest{"%b", b64, "1111111111111111111111111111111111111111111111111111111111111111"},
	fmtTest{"%e", float64(1), "1.000000e+00"},
	fmtTest{"%e", float64(1234.5678e3), "1.234568e+06"},
	fmtTest{"%e", float64(1234.5678e-8), "1.234568e-05"},
	fmtTest{"%e", float64(-7), "-7.000000e+00"},
	fmtTest{"%e", float64(-1e-9), "-1.000000e-09"},
	fmtTest{"%f", float64(1234.5678e3), "1234567.800000"},
	fmtTest{"%f", float64(1234.5678e-8), "0.000012"},
	fmtTest{"%f", float64(-7), "-7.000000"},
	fmtTest{"%f", float64(-1e-9), "-0.000000"},
	fmtTest{"%g", float64(1234.5678e3), "1.2345678e+06"},
	fmtTest{"%g", float32(1234.5678e3), "1.2345678e+06"},
	fmtTest{"%g", float64(1234.5678e-8), "1.2345678e-05"},
	fmtTest{"%g", float64(-7), "-7"},
	fmtTest{"%g", float64(-1e-9), "-1e-09"},
	fmtTest{"%g", float32(-1e-9), "-1e-09"},
	fmtTest{"%E", float64(1), "1.000000E+00"},
	fmtTest{"%E", float64(1234.5678e3), "1.234568E+06"},
	fmtTest{"%E", float64(1234.5678e-8), "1.234568E-05"},
	fmtTest{"%E", float64(-7), "-7.000000E+00"},
	fmtTest{"%E", float64(-1e-9), "-1.000000E-09"},
	fmtTest{"%G", float64(1234.5678e3), "1.2345678E+06"},
	fmtTest{"%G", float32(1234.5678e3), "1.2345678E+06"},
	fmtTest{"%G", float64(1234.5678e-8), "1.2345678E-05"},
	fmtTest{"%G", float64(-7), "-7"},
	fmtTest{"%G", float64(-1e-9), "-1E-09"},
	fmtTest{"%G", float32(-1e-9), "-1E-09"},
	fmtTest{"%c", 'x', "x"},
	fmtTest{"%c", 0xe4, "ä"},
	fmtTest{"%c", 0x672c, "本"},
	fmtTest{"%c", '日', "日"},
	fmtTest{"%20.8d", 1234, "            00001234"},
	fmtTest{"%20.8d", -1234, "           -00001234"},
	fmtTest{"%20d", 1234, "                1234"},
	fmtTest{"%-20.8d", 1234, "00001234            "},
	fmtTest{"%-20.8d", -1234, "-00001234           "},
	fmtTest{"%-#20.8x", 0x1234abc, "0x01234abc          "},
	fmtTest{"%-#20.8X", 0x1234abc, "0X01234ABC          "},
	fmtTest{"%-#20.8o", 01234, "00001234            "},
	fmtTest{"%.20b", 7, "00000000000000000111"},
	fmtTest{"%20.5s", "qwertyuiop", "               qwert"},
	fmtTest{"%.5s", "qwertyuiop", "qwert"},
	fmtTest{"%-20.5s", "qwertyuiop", "qwert               "},
	fmtTest{"%20c", 'x', "                   x"},
	fmtTest{"%-20c", 'x', "x                   "},
	fmtTest{"%20.6e", 1.2345e3, "        1.234500e+03"},
	fmtTest{"%20.6e", 1.2345e-3, "        1.234500e-03"},
	fmtTest{"%20e", 1.2345e3, "        1.234500e+03"},
	fmtTest{"%20e", 1.2345e-3, "        1.234500e-03"},
	fmtTest{"%20.8e", 1.2345e3, "      1.23450000e+03"},
	fmtTest{"%20f", float64(1.23456789e3), "         1234.567890"},
	fmtTest{"%20f", float64(1.23456789e-3), "            0.001235"},
	fmtTest{"%20f", float64(12345678901.23456789), "  12345678901.234568"},
	fmtTest{"%-20f", float64(1.23456789e3), "1234.567890         "},
	fmtTest{"%20.8f", float64(1.23456789e3), "       1234.56789000"},
	fmtTest{"%20.8f", float64(1.23456789e-3), "          0.00123457"},
	fmtTest{"%g", float64(1.23456789e3), "1234.56789"},
	fmtTest{"%g", float64(1.23456789e-3), "0.00123456789"},
	fmtTest{"%g", float64(1.23456789e20), "1.23456789e+20"},
	fmtTest{"%20e", math.Inf(1), "                +Inf"},
	fmtTest{"%-20f", math.Inf(-1), "-Inf                "},
	fmtTest{"%20g", math.NaN(), "                 NaN"},

	// arrays
	fmtTest{"%v", array, "[1 2 3 4 5]"},
	fmtTest{"%v", iarray, "[1 hello 2.5 <nil>]"},
	fmtTest{"%v", &array, "&[1 2 3 4 5]"},
	fmtTest{"%v", &iarray, "&[1 hello 2.5 <nil>]"},

	// structs
	fmtTest{"%v", A{1, 2, "a", []int{1, 2}}, `{1 2 a [1 2]}`},
	fmtTest{"%+v", A{1, 2, "a", []int{1, 2}}, `{i:1 j:2 s:a x:[1 2]}`},

	// +v on structs with Stringable items
	fmtTest{"%+v", B{1, 2}, `{i:<1> j:2}`},
	fmtTest{"%+v", C{1, B{2, 3}}, `{i:1 B:{i:<2> j:3}}`},

	// %p on non-pointers
	fmtTest{"%p", make(chan int), "PTR"},
	fmtTest{"%p", make(map[int]int), "PTR"},
	fmtTest{"%p", make([]int, 1), "PTR"},

	// go syntax
	fmtTest{"%#v", A{1, 2, "a", []int{1, 2}}, `fmt_test.A{i:1, j:0x2, s:"a", x:[]int{1, 2}}`},
	fmtTest{"%#v", &b, "(*uint8)(PTR)"},
	fmtTest{"%#v", TestFmtInterface, "(func(*testing.T))(PTR)"},
	fmtTest{"%#v", make(chan int), "(chan int)(PTR)"},
	fmtTest{"%#v", uint64(1<<64 - 1), "0xffffffffffffffff"},
	fmtTest{"%#v", 1000000000, "1000000000"},
}

func TestSprintf(t *testing.T) {
	for _, tt := range fmttests {
		s := Sprintf(tt.fmt, tt.val)
		if i := strings.Index(s, "0x"); i >= 0 && strings.Index(tt.out, "PTR") >= 0 {
			j := i + 2
			for ; j < len(s); j++ {
				c := s[j]
				if (c < '0' || c > '9') && (c < 'a' || c > 'f') && (c < 'A' || c > 'F') {
					break
				}
			}
			s = s[0:i] + "PTR" + s[j:]
		}
		if s != tt.out {
			if _, ok := tt.val.(string); ok {
				// Don't requote the already-quoted strings.
				// It's too confusing to read the errors.
				t.Errorf("Sprintf(%q, %q) = <%s> want <%s>", tt.fmt, tt.val, s, tt.out)
			} else {
				t.Errorf("Sprintf(%q, %v) = %q want %q", tt.fmt, tt.val, s, tt.out)
			}
		}
	}
}

func BenchmarkSprintfEmpty(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sprintf("")
	}
}

func BenchmarkSprintfString(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sprintf("%s", "hello")
	}
}

func BenchmarkSprintfInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sprintf("%d", 5)
	}
}

func BenchmarkSprintfIntInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sprintf("%d %d", 5, 6)
	}
}

func TestCountMallocs(t *testing.T) {
	mallocs := 0 - malloc.GetStats().Mallocs
	for i := 0; i < 100; i++ {
		Sprintf("")
	}
	mallocs += malloc.GetStats().Mallocs
	Printf("mallocs per Sprintf(\"\"): %d\n", mallocs/100)
	mallocs = 0 - malloc.GetStats().Mallocs
	for i := 0; i < 100; i++ {
		Sprintf("xxx")
	}
	mallocs += malloc.GetStats().Mallocs
	Printf("mallocs per Sprintf(\"xxx\"): %d\n", mallocs/100)
	mallocs = 0 - malloc.GetStats().Mallocs
	for i := 0; i < 100; i++ {
		Sprintf("%x", i)
	}
	mallocs += malloc.GetStats().Mallocs
	Printf("mallocs per Sprintf(\"%%x\"): %d\n", mallocs/100)
	mallocs = 0 - malloc.GetStats().Mallocs
	for i := 0; i < 100; i++ {
		Sprintf("%x %x", i, i)
	}
	mallocs += malloc.GetStats().Mallocs
	Printf("mallocs per Sprintf(\"%%x %%x\"): %d\n", mallocs/100)
}

type flagPrinter struct{}

func (*flagPrinter) Format(f State, c int) {
	s := "%"
	for i := 0; i < 128; i++ {
		if f.Flag(i) {
			s += string(i)
		}
	}
	if w, ok := f.Width(); ok {
		s += Sprintf("%d", w)
	}
	if p, ok := f.Precision(); ok {
		s += Sprintf(".%d", p)
	}
	s += string(c)
	io.WriteString(f, "["+s+"]")
}

type flagTest struct {
	in  string
	out string
}

var flagtests = []flagTest{
	flagTest{"%a", "[%a]"},
	flagTest{"%-a", "[%-a]"},
	flagTest{"%+a", "[%+a]"},
	flagTest{"%#a", "[%#a]"},
	flagTest{"% a", "[% a]"},
	flagTest{"%0a", "[%0a]"},
	flagTest{"%1.2a", "[%1.2a]"},
	flagTest{"%-1.2a", "[%-1.2a]"},
	flagTest{"%+1.2a", "[%+1.2a]"},
	flagTest{"%-+1.2a", "[%+-1.2a]"},
	flagTest{"%-+1.2abc", "[%+-1.2a]bc"},
	flagTest{"%-1.2abc", "[%-1.2a]bc"},
}

func TestFlagParser(t *testing.T) {
	var flagprinter flagPrinter
	for _, tt := range flagtests {
		s := Sprintf(tt.in, &flagprinter)
		if s != tt.out {
			t.Errorf("Sprintf(%q, &flagprinter) => %q, want %q", tt.in, s, tt.out)
		}
	}
}

func TestStructPrinter(t *testing.T) {
	var s struct {
		a string
		b string
		c int
	}
	s.a = "abc"
	s.b = "def"
	s.c = 123
	type Test struct {
		fmt string
		out string
	}
	var tests = []Test{
		Test{"%v", "{abc def 123}"},
		Test{"%+v", "{a:abc b:def c:123}"},
	}
	for _, tt := range tests {
		out := Sprintf(tt.fmt, s)
		if out != tt.out {
			t.Errorf("Sprintf(%q, &s) = %q, want %q", tt.fmt, out, tt.out)
		}
	}
}

// Check map printing using substrings so we don't depend on the print order.
func presentInMap(s string, a []string, t *testing.T) {
	for i := 0; i < len(a); i++ {
		loc := strings.Index(s, a[i])
		if loc < 0 {
			t.Errorf("map print: expected to find %q in %q", a[i], s)
		}
		// make sure the match ends here
		loc += len(a[i])
		if loc >= len(s) || (s[loc] != ' ' && s[loc] != ']') {
			t.Errorf("map print: %q not properly terminated in %q", a[i], s)
		}
	}
}

func TestMapPrinter(t *testing.T) {
	m0 := make(map[int]string)
	s := Sprint(m0)
	if s != "map[]" {
		t.Errorf("empty map printed as %q not %q", s, "map[]")
	}
	m1 := map[int]string{1: "one", 2: "two", 3: "three"}
	a := []string{"1:one", "2:two", "3:three"}
	presentInMap(Sprintf("%v", m1), a, t)
	presentInMap(Sprint(m1), a, t)
}

func TestEmptyMap(t *testing.T) {
	const emptyMapStr = "map[]"
	var m map[string]int
	s := Sprint(m)
	if s != emptyMapStr {
		t.Errorf("nil map printed as %q not %q", s, emptyMapStr)
	}
	m = make(map[string]int)
	s = Sprint(m)
	if s != emptyMapStr {
		t.Errorf("empty map printed as %q not %q", s, emptyMapStr)
	}
}
