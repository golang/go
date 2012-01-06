// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"bytes"
	. "fmt"
	"io"
	"math"
	"runtime" // for the malloc count test only
	"strings"
	"testing"
	"time"
)

type (
	renamedBool       bool
	renamedInt        int
	renamedInt8       int8
	renamedInt16      int16
	renamedInt32      int32
	renamedInt64      int64
	renamedUint       uint
	renamedUint8      uint8
	renamedUint16     uint16
	renamedUint32     uint32
	renamedUint64     uint64
	renamedUintptr    uintptr
	renamedString     string
	renamedBytes      []byte
	renamedFloat32    float32
	renamedFloat64    float64
	renamedComplex64  complex64
	renamedComplex128 complex128
)

func TestFmtInterface(t *testing.T) {
	var i1 interface{}
	i1 = "abc"
	s := Sprintf("%s", i1)
	if s != "abc" {
		t.Errorf(`Sprintf("%%s", empty("abc")) = %q want %q`, s, "abc")
	}
}

const b32 uint32 = 1<<32 - 1
const b64 uint64 = 1<<64 - 1

var array = [5]int{1, 2, 3, 4, 5}
var iarray = [4]interface{}{1, "hello", 2.5, nil}
var slice = array[:]
var islice = iarray[:]

type A struct {
	i int
	j uint
	s string
	x []int
}

type I int

func (i I) String() string { return Sprintf("<%d>", int(i)) }

type B struct {
	I I
	j int
}

type C struct {
	i int
	B
}

type F int

func (f F) Format(s State, c rune) {
	Fprintf(s, "<%c=F(%d)>", c, int(f))
}

type G int

func (g G) GoString() string {
	return Sprintf("GoString(%d)", int(g))
}

type S struct {
	F F // a struct field that Formats
	G G // a struct field that GoStrings
}

type SI struct {
	I interface{}
}

// A type with a String method with pointer receiver for testing %p
type P int

var pValue P

func (p *P) String() string {
	return "String(p)"
}

var b byte

var fmttests = []struct {
	fmt string
	val interface{}
	out string
}{
	{"%d", 12345, "12345"},
	{"%v", 12345, "12345"},
	{"%t", true, "true"},

	// basic string
	{"%s", "abc", "abc"},
	{"%x", "abc", "616263"},
	{"%x", "xyz", "78797a"},
	{"%X", "xyz", "78797A"},
	{"%q", "abc", `"abc"`},

	// basic bytes
	{"%s", []byte("abc"), "abc"},
	{"%x", []byte("abc"), "616263"},
	{"% x", []byte("abc\xff"), "61 62 63 ff"},
	{"% X", []byte("abc\xff"), "61 62 63 FF"},
	{"%x", []byte("xyz"), "78797a"},
	{"%X", []byte("xyz"), "78797A"},
	{"%q", []byte("abc"), `"abc"`},

	// escaped strings
	{"%#q", `abc`, "`abc`"},
	{"%#q", `"`, "`\"`"},
	{"1 %#q", `\n`, "1 `\\n`"},
	{"2 %#q", "\n", `2 "\n"`},
	{"%q", `"`, `"\""`},
	{"%q", "\a\b\f\r\n\t\v", `"\a\b\f\r\n\t\v"`},
	{"%q", "abc\xffdef", `"abc\xffdef"`},
	{"%q", "\u263a", `"☺"`},
	{"%+q", "\u263a", `"\u263a"`},
	{"%q", "\U0010ffff", `"\U0010ffff"`},

	// escaped characters
	{"%q", 'x', `'x'`},
	{"%q", 0, `'\x00'`},
	{"%q", '\n', `'\n'`},
	{"%q", '\u0e00', `'\u0e00'`},         // not a printable rune.
	{"%q", '\U000c2345', `'\U000c2345'`}, // not a printable rune.
	{"%q", int64(0x7FFFFFFF), `%!q(int64=2147483647)`},
	{"%q", uint64(0xFFFFFFFF), `%!q(uint64=4294967295)`},
	{"%q", '"', `'"'`},
	{"%q", '\'', `'\''`},
	{"%q", "\u263a", `"☺"`},
	{"%+q", "\u263a", `"\u263a"`},

	// width
	{"%5s", "abc", "  abc"},
	{"%2s", "\u263a", " ☺"},
	{"%-5s", "abc", "abc  "},
	{"%-8q", "abc", `"abc"   `},
	{"%05s", "abc", "00abc"},
	{"%08q", "abc", `000"abc"`},
	{"%5s", "abcdefghijklmnopqrstuvwxyz", "abcdefghijklmnopqrstuvwxyz"},
	{"%.5s", "abcdefghijklmnopqrstuvwxyz", "abcde"},
	{"%.5s", "日本語日本語", "日本語日本"},
	{"%.5s", []byte("日本語日本語"), "日本語日本"},
	{"%.5q", "abcdefghijklmnopqrstuvwxyz", `"abcde"`},
	{"%.3q", "日本語日本語", `"日本語"`},
	{"%.3q", []byte("日本語日本語"), `"日本語"`},
	{"%10.1q", "日本語日本語", `       "日"`},

	// integers
	{"%d", 12345, "12345"},
	{"%d", -12345, "-12345"},
	{"%10d", 12345, "     12345"},
	{"%10d", -12345, "    -12345"},
	{"%+10d", 12345, "    +12345"},
	{"%010d", 12345, "0000012345"},
	{"%010d", -12345, "-000012345"},
	{"%-10d", 12345, "12345     "},
	{"%010.3d", 1, "       001"},
	{"%010.3d", -1, "      -001"},
	{"%+d", 12345, "+12345"},
	{"%+d", -12345, "-12345"},
	{"%+d", 0, "+0"},
	{"% d", 0, " 0"},
	{"% d", 12345, " 12345"},
	{"%.0d", 0, ""},
	{"%.d", 0, ""},

	// unicode format
	{"%U", 0x1, "U+0001"},
	{"%U", uint(0x1), "U+0001"},
	{"%.8U", 0x2, "U+00000002"},
	{"%U", 0x1234, "U+1234"},
	{"%U", 0x12345, "U+12345"},
	{"%10.6U", 0xABC, "  U+000ABC"},
	{"%-10.6U", 0xABC, "U+000ABC  "},
	{"%U", '\n', `U+000A`},
	{"%#U", '\n', `U+000A`},
	{"%U", 'x', `U+0078`},
	{"%#U", 'x', `U+0078 'x'`},
	{"%U", '\u263a', `U+263A`},
	{"%#U", '\u263a', `U+263A '☺'`},

	// floats
	{"%+.3e", 0.0, "+0.000e+00"},
	{"%+.3e", 1.0, "+1.000e+00"},
	{"%+.3f", -1.0, "-1.000"},
	{"% .3E", -1.0, "-1.000E+00"},
	{"% .3e", 1.0, " 1.000e+00"},
	{"%+.3g", 0.0, "+0"},
	{"%+.3g", 1.0, "+1"},
	{"%+.3g", -1.0, "-1"},
	{"% .3g", -1.0, "-1"},
	{"% .3g", 1.0, " 1"},

	// complex values
	{"%+.3e", 0i, "(+0.000e+00+0.000e+00i)"},
	{"%+.3f", 0i, "(+0.000+0.000i)"},
	{"%+.3g", 0i, "(+0+0i)"},
	{"%+.3e", 1 + 2i, "(+1.000e+00+2.000e+00i)"},
	{"%+.3f", 1 + 2i, "(+1.000+2.000i)"},
	{"%+.3g", 1 + 2i, "(+1+2i)"},
	{"%.3e", 0i, "(0.000e+00+0.000e+00i)"},
	{"%.3f", 0i, "(0.000+0.000i)"},
	{"%.3g", 0i, "(0+0i)"},
	{"%.3e", 1 + 2i, "(1.000e+00+2.000e+00i)"},
	{"%.3f", 1 + 2i, "(1.000+2.000i)"},
	{"%.3g", 1 + 2i, "(1+2i)"},
	{"%.3e", -1 - 2i, "(-1.000e+00-2.000e+00i)"},
	{"%.3f", -1 - 2i, "(-1.000-2.000i)"},
	{"%.3g", -1 - 2i, "(-1-2i)"},
	{"% .3E", -1 - 2i, "(-1.000E+00-2.000E+00i)"},
	{"%+.3g", complex64(1 + 2i), "(+1+2i)"},
	{"%+.3g", complex128(1 + 2i), "(+1+2i)"},

	// erroneous formats
	{"", 2, "%!(EXTRA int=2)"},
	{"%d", "hello", "%!d(string=hello)"},

	// old test/fmt_test.go
	{"%d", 1234, "1234"},
	{"%d", -1234, "-1234"},
	{"%d", uint(1234), "1234"},
	{"%d", uint32(b32), "4294967295"},
	{"%d", uint64(b64), "18446744073709551615"},
	{"%o", 01234, "1234"},
	{"%#o", 01234, "01234"},
	{"%o", uint32(b32), "37777777777"},
	{"%o", uint64(b64), "1777777777777777777777"},
	{"%x", 0x1234abcd, "1234abcd"},
	{"%#x", 0x1234abcd, "0x1234abcd"},
	{"%x", b32 - 0x1234567, "fedcba98"},
	{"%X", 0x1234abcd, "1234ABCD"},
	{"%X", b32 - 0x1234567, "FEDCBA98"},
	{"%#X", 0, "0X0"},
	{"%x", b64, "ffffffffffffffff"},
	{"%b", 7, "111"},
	{"%b", b64, "1111111111111111111111111111111111111111111111111111111111111111"},
	{"%b", -6, "-110"},
	{"%e", 1.0, "1.000000e+00"},
	{"%e", 1234.5678e3, "1.234568e+06"},
	{"%e", 1234.5678e-8, "1.234568e-05"},
	{"%e", -7.0, "-7.000000e+00"},
	{"%e", -1e-9, "-1.000000e-09"},
	{"%f", 1234.5678e3, "1234567.800000"},
	{"%f", 1234.5678e-8, "0.000012"},
	{"%f", -7.0, "-7.000000"},
	{"%f", -1e-9, "-0.000000"},
	{"%g", 1234.5678e3, "1.2345678e+06"},
	{"%g", float32(1234.5678e3), "1.2345678e+06"},
	{"%g", 1234.5678e-8, "1.2345678e-05"},
	{"%g", -7.0, "-7"},
	{"%g", -1e-9, "-1e-09"},
	{"%g", float32(-1e-9), "-1e-09"},
	{"%E", 1.0, "1.000000E+00"},
	{"%E", 1234.5678e3, "1.234568E+06"},
	{"%E", 1234.5678e-8, "1.234568E-05"},
	{"%E", -7.0, "-7.000000E+00"},
	{"%E", -1e-9, "-1.000000E-09"},
	{"%G", 1234.5678e3, "1.2345678E+06"},
	{"%G", float32(1234.5678e3), "1.2345678E+06"},
	{"%G", 1234.5678e-8, "1.2345678E-05"},
	{"%G", -7.0, "-7"},
	{"%G", -1e-9, "-1E-09"},
	{"%G", float32(-1e-9), "-1E-09"},
	{"%c", 'x', "x"},
	{"%c", 0xe4, "ä"},
	{"%c", 0x672c, "本"},
	{"%c", '日', "日"},
	{"%20.8d", 1234, "            00001234"},
	{"%20.8d", -1234, "           -00001234"},
	{"%20d", 1234, "                1234"},
	{"%-20.8d", 1234, "00001234            "},
	{"%-20.8d", -1234, "-00001234           "},
	{"%-#20.8x", 0x1234abc, "0x01234abc          "},
	{"%-#20.8X", 0x1234abc, "0X01234ABC          "},
	{"%-#20.8o", 01234, "00001234            "},
	{"%.20b", 7, "00000000000000000111"},
	{"%20.5s", "qwertyuiop", "               qwert"},
	{"%.5s", "qwertyuiop", "qwert"},
	{"%-20.5s", "qwertyuiop", "qwert               "},
	{"%20c", 'x', "                   x"},
	{"%-20c", 'x', "x                   "},
	{"%20.6e", 1.2345e3, "        1.234500e+03"},
	{"%20.6e", 1.2345e-3, "        1.234500e-03"},
	{"%20e", 1.2345e3, "        1.234500e+03"},
	{"%20e", 1.2345e-3, "        1.234500e-03"},
	{"%20.8e", 1.2345e3, "      1.23450000e+03"},
	{"%20f", 1.23456789e3, "         1234.567890"},
	{"%20f", 1.23456789e-3, "            0.001235"},
	{"%20f", 12345678901.23456789, "  12345678901.234568"},
	{"%-20f", 1.23456789e3, "1234.567890         "},
	{"%20.8f", 1.23456789e3, "       1234.56789000"},
	{"%20.8f", 1.23456789e-3, "          0.00123457"},
	{"%g", 1.23456789e3, "1234.56789"},
	{"%g", 1.23456789e-3, "0.00123456789"},
	{"%g", 1.23456789e20, "1.23456789e+20"},
	{"%20e", math.Inf(1), "                +Inf"},
	{"%-20f", math.Inf(-1), "-Inf                "},
	{"%20g", math.NaN(), "                 NaN"},

	// arrays
	{"%v", array, "[1 2 3 4 5]"},
	{"%v", iarray, "[1 hello 2.5 <nil>]"},
	{"%v", &array, "&[1 2 3 4 5]"},
	{"%v", &iarray, "&[1 hello 2.5 <nil>]"},

	// slices
	{"%v", slice, "[1 2 3 4 5]"},
	{"%v", islice, "[1 hello 2.5 <nil>]"},
	{"%v", &slice, "&[1 2 3 4 5]"},
	{"%v", &islice, "&[1 hello 2.5 <nil>]"},

	// complexes with %v
	{"%v", 1 + 2i, "(1+2i)"},
	{"%v", complex64(1 + 2i), "(1+2i)"},
	{"%v", complex128(1 + 2i), "(1+2i)"},

	// structs
	{"%v", A{1, 2, "a", []int{1, 2}}, `{1 2 a [1 2]}`},
	{"%+v", A{1, 2, "a", []int{1, 2}}, `{i:1 j:2 s:a x:[1 2]}`},

	// +v on structs with Stringable items
	{"%+v", B{1, 2}, `{I:<1> j:2}`},
	{"%+v", C{1, B{2, 3}}, `{i:1 B:{I:<2> j:3}}`},

	// q on Stringable items
	{"%s", I(23), `<23>`},
	{"%q", I(23), `"<23>"`},
	{"%x", I(23), `3c32333e`},
	{"%d", I(23), `23`}, // Stringer applies only to string formats.

	// go syntax
	{"%#v", A{1, 2, "a", []int{1, 2}}, `fmt_test.A{i:1, j:0x2, s:"a", x:[]int{1, 2}}`},
	{"%#v", &b, "(*uint8)(0xPTR)"},
	{"%#v", TestFmtInterface, "(func(*testing.T))(0xPTR)"},
	{"%#v", make(chan int), "(chan int)(0xPTR)"},
	{"%#v", uint64(1<<64 - 1), "0xffffffffffffffff"},
	{"%#v", 1000000000, "1000000000"},
	{"%#v", map[string]int{"a": 1}, `map[string]int{"a":1}`},
	{"%#v", map[string]B{"a": {1, 2}}, `map[string]fmt_test.B{"a":fmt_test.B{I:1, j:2}}`},
	{"%#v", []string{"a", "b"}, `[]string{"a", "b"}`},
	{"%#v", SI{}, `fmt_test.SI{I:interface {}(nil)}`},
	{"%#v", []int(nil), `[]int(nil)`},
	{"%#v", []int{}, `[]int{}`},
	{"%#v", array, `[5]int{1, 2, 3, 4, 5}`},
	{"%#v", &array, `&[5]int{1, 2, 3, 4, 5}`},
	{"%#v", iarray, `[4]interface {}{1, "hello", 2.5, interface {}(nil)}`},
	{"%#v", &iarray, `&[4]interface {}{1, "hello", 2.5, interface {}(nil)}`},
	{"%#v", map[int]byte(nil), `map[int]uint8(nil)`},
	{"%#v", map[int]byte{}, `map[int]uint8{}`},

	// slices with other formats
	{"%#x", []int{1, 2, 15}, `[0x1 0x2 0xf]`},
	{"%x", []int{1, 2, 15}, `[1 2 f]`},
	{"%d", []int{1, 2, 15}, `[1 2 15]`},
	{"%d", []byte{1, 2, 15}, `[1 2 15]`},
	{"%q", []string{"a", "b"}, `["a" "b"]`},

	// renamings
	{"%v", renamedBool(true), "true"},
	{"%d", renamedBool(true), "%!d(fmt_test.renamedBool=true)"},
	{"%o", renamedInt(8), "10"},
	{"%d", renamedInt8(-9), "-9"},
	{"%v", renamedInt16(10), "10"},
	{"%v", renamedInt32(-11), "-11"},
	{"%X", renamedInt64(255), "FF"},
	{"%v", renamedUint(13), "13"},
	{"%o", renamedUint8(14), "16"},
	{"%X", renamedUint16(15), "F"},
	{"%d", renamedUint32(16), "16"},
	{"%X", renamedUint64(17), "11"},
	{"%o", renamedUintptr(18), "22"},
	{"%x", renamedString("thing"), "7468696e67"},
	{"%d", renamedBytes([]byte{1, 2, 15}), `[1 2 15]`},
	{"%q", renamedBytes([]byte("hello")), `"hello"`},
	{"%v", renamedFloat32(22), "22"},
	{"%v", renamedFloat64(33), "33"},
	{"%v", renamedComplex64(3 + 4i), "(3+4i)"},
	{"%v", renamedComplex128(4 - 3i), "(4-3i)"},

	// Formatter
	{"%x", F(1), "<x=F(1)>"},
	{"%x", G(2), "2"},
	{"%+v", S{F(4), G(5)}, "{F:<v=F(4)> G:5}"},

	// GoStringer
	{"%#v", G(6), "GoString(6)"},
	{"%#v", S{F(7), G(8)}, "fmt_test.S{F:<v=F(7)>, G:GoString(8)}"},

	// %T
	{"%T", (4 - 3i), "complex128"},
	{"%T", renamedComplex128(4 - 3i), "fmt_test.renamedComplex128"},
	{"%T", intVal, "int"},
	{"%6T", &intVal, "  *int"},

	// %p
	{"p0=%p", new(int), "p0=0xPTR"},
	{"p1=%s", &pValue, "p1=String(p)"}, // String method...
	{"p2=%p", &pValue, "p2=0xPTR"},     // ... not called with %p
	{"p4=%#p", new(int), "p4=PTR"},

	// %p on non-pointers
	{"%p", make(chan int), "0xPTR"},
	{"%p", make(map[int]int), "0xPTR"},
	{"%p", make([]int, 1), "0xPTR"},
	{"%p", 27, "%!p(int=27)"}, // not a pointer at all

	// %d on Stringer should give integer if possible
	{"%s", time.Time{}.Month(), "January"},
	{"%d", time.Time{}.Month(), "1"},

	// erroneous things
	{"%s %", "hello", "hello %!(NOVERB)"},
	{"%s %.2", "hello", "hello %!(NOVERB)"},
	{"%d", "hello", "%!d(string=hello)"},
	{"no args", "hello", "no args%!(EXTRA string=hello)"},
	{"%s", nil, "%!s(<nil>)"},
	{"%T", nil, "<nil>"},
	{"%-1", 100, "%!(NOVERB)%!(EXTRA int=100)"},
}

func TestSprintf(t *testing.T) {
	for _, tt := range fmttests {
		s := Sprintf(tt.fmt, tt.val)
		if i := strings.Index(tt.out, "PTR"); i >= 0 {
			j := i
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

func BenchmarkSprintfPrefixedInt(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sprintf("This is some meaningless prefix text that needs to be scanned %d", 6)
	}
}

func BenchmarkSprintfFloat(b *testing.B) {
	for i := 0; i < b.N; i++ {
		Sprintf("%g", 5.23184)
	}
}

var mallocBuf bytes.Buffer

var mallocTest = []struct {
	count int
	desc  string
	fn    func()
}{
	{0, `Sprintf("")`, func() { Sprintf("") }},
	{1, `Sprintf("xxx")`, func() { Sprintf("xxx") }},
	{1, `Sprintf("%x")`, func() { Sprintf("%x", 7) }},
	{2, `Sprintf("%s")`, func() { Sprintf("%s", "hello") }},
	{1, `Sprintf("%x %x")`, func() { Sprintf("%x %x", 7, 112) }},
	{1, `Sprintf("%g")`, func() { Sprintf("%g", 3.14159) }},
	{0, `Fprintf(buf, "%x %x %x")`, func() { mallocBuf.Reset(); Fprintf(&mallocBuf, "%x %x %x", 7, 8, 9) }},
	{1, `Fprintf(buf, "%s")`, func() { mallocBuf.Reset(); Fprintf(&mallocBuf, "%s", "hello") }},
}

var _ bytes.Buffer

func TestCountMallocs(t *testing.T) {
	if testing.Short() {
		return
	}
	for _, mt := range mallocTest {
		const N = 100
		runtime.UpdateMemStats()
		mallocs := 0 - runtime.MemStats.Mallocs
		for i := 0; i < N; i++ {
			mt.fn()
		}
		runtime.UpdateMemStats()
		mallocs += runtime.MemStats.Mallocs
		if mallocs/N != uint64(mt.count) {
			t.Errorf("%s: expected %d mallocs, got %d", mt.desc, mt.count, mallocs/N)
		}
	}
}

type flagPrinter struct{}

func (*flagPrinter) Format(f State, c rune) {
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

var flagtests = []struct {
	in  string
	out string
}{
	{"%a", "[%a]"},
	{"%-a", "[%-a]"},
	{"%+a", "[%+a]"},
	{"%#a", "[%#a]"},
	{"% a", "[% a]"},
	{"%0a", "[%0a]"},
	{"%1.2a", "[%1.2a]"},
	{"%-1.2a", "[%-1.2a]"},
	{"%+1.2a", "[%+1.2a]"},
	{"%-+1.2a", "[%+-1.2a]"},
	{"%-+1.2abc", "[%+-1.2a]bc"},
	{"%-1.2abc", "[%-1.2a]bc"},
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
	var tests = []struct {
		fmt string
		out string
	}{
		{"%v", "{abc def 123}"},
		{"%+v", "{a:abc b:def c:123}"},
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

// Check that Sprint (and hence Print, Fprint) puts spaces in the right places,
// that is, between arg pairs in which neither is a string.
func TestBlank(t *testing.T) {
	got := Sprint("<", 1, ">:", 1, 2, 3, "!")
	expect := "<1>:1 2 3!"
	if got != expect {
		t.Errorf("got %q expected %q", got, expect)
	}
}

// Check that Sprintln (and hence Println, Fprintln) puts spaces in the right places,
// that is, between all arg pairs.
func TestBlankln(t *testing.T) {
	got := Sprintln("<", 1, ">:", 1, 2, 3, "!")
	expect := "< 1 >: 1 2 3 !\n"
	if got != expect {
		t.Errorf("got %q expected %q", got, expect)
	}
}

// Check Formatter with Sprint, Sprintln, Sprintf
func TestFormatterPrintln(t *testing.T) {
	f := F(1)
	expect := "<v=F(1)>\n"
	s := Sprint(f, "\n")
	if s != expect {
		t.Errorf("Sprint wrong with Formatter: expected %q got %q", expect, s)
	}
	s = Sprintln(f)
	if s != expect {
		t.Errorf("Sprintln wrong with Formatter: expected %q got %q", expect, s)
	}
	s = Sprintf("%v\n", f)
	if s != expect {
		t.Errorf("Sprintf wrong with Formatter: expected %q got %q", expect, s)
	}
}

func args(a ...interface{}) []interface{} { return a }

var startests = []struct {
	fmt string
	in  []interface{}
	out string
}{
	{"%*d", args(4, 42), "  42"},
	{"%.*d", args(4, 42), "0042"},
	{"%*.*d", args(8, 4, 42), "    0042"},
	{"%0*d", args(4, 42), "0042"},
	{"%-*d", args(4, 42), "42  "},

	// erroneous
	{"%*d", args(nil, 42), "%!(BADWIDTH)42"},
	{"%.*d", args(nil, 42), "%!(BADPREC)42"},
	{"%*d", args(5, "foo"), "%!d(string=  foo)"},
	{"%*% %d", args(20, 5), "% 5"},
	{"%*", args(4), "%!(NOVERB)"},
	{"%*d", args(int32(4), 42), "%!(BADWIDTH)42"},
}

func TestWidthAndPrecision(t *testing.T) {
	for _, tt := range startests {
		s := Sprintf(tt.fmt, tt.in...)
		if s != tt.out {
			t.Errorf("%q: got %q expected %q", tt.fmt, s, tt.out)
		}
	}
}

// A type that panics in String.
type Panic struct {
	message interface{}
}

// Value receiver.
func (p Panic) GoString() string {
	panic(p.message)
}

// Value receiver.
func (p Panic) String() string {
	panic(p.message)
}

// A type that panics in Format.
type PanicF struct {
	message interface{}
}

// Value receiver.
func (p PanicF) Format(f State, c rune) {
	panic(p.message)
}

var panictests = []struct {
	fmt string
	in  interface{}
	out string
}{
	// String
	{"%s", (*Panic)(nil), "<nil>"}, // nil pointer special case
	{"%s", Panic{io.ErrUnexpectedEOF}, "%s(PANIC=unexpected EOF)"},
	{"%s", Panic{3}, "%s(PANIC=3)"},
	// GoString
	{"%#v", (*Panic)(nil), "<nil>"}, // nil pointer special case
	{"%#v", Panic{io.ErrUnexpectedEOF}, "%v(PANIC=unexpected EOF)"},
	{"%#v", Panic{3}, "%v(PANIC=3)"},
	// Format
	{"%s", (*PanicF)(nil), "<nil>"}, // nil pointer special case
	{"%s", PanicF{io.ErrUnexpectedEOF}, "%s(PANIC=unexpected EOF)"},
	{"%s", PanicF{3}, "%s(PANIC=3)"},
}

func TestPanics(t *testing.T) {
	for _, tt := range panictests {
		s := Sprintf(tt.fmt, tt.in)
		if s != tt.out {
			t.Errorf("%q: got %q expected %q", tt.fmt, s, tt.out)
		}
	}
}

// Test that erroneous String routine doesn't cause fatal recursion.
var recurCount = 0

type Recur struct {
	i      int
	failed *bool
}

func (r Recur) String() string {
	if recurCount++; recurCount > 10 {
		*r.failed = true
		return "FAIL"
	}
	// This will call badVerb. Before the fix, that would cause us to recur into
	// this routine to print %!p(value). Now we don't call the user's method
	// during an error.
	return Sprintf("recur@%p value: %d", r, r.i)
}

func TestBadVerbRecursion(t *testing.T) {
	failed := false
	r := Recur{3, &failed}
	Sprintf("recur@%p value: %d\n", &r, r.i)
	if failed {
		t.Error("fail with pointer")
	}
	failed = false
	r = Recur{4, &failed}
	Sprintf("recur@%p, value: %d\n", r, r.i)
	if failed {
		t.Error("fail with value")
	}
}
