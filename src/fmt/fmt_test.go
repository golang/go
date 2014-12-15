// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	"bytes"
	. "fmt"
	"io"
	"math"
	"runtime"
	"strings"
	"testing"
	"time"
	"unicode"
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

// P is a type with a String method with pointer receiver for testing %p.
type P int

var pValue P

func (p *P) String() string {
	return "String(p)"
}

var barray = [5]renamedUint8{1, 2, 3, 4, 5}
var bslice = barray[:]

type byteStringer byte

func (byteStringer) String() string { return "X" }

var byteStringerSlice = []byteStringer{97, 98, 99, 100}

type byteFormatter byte

func (byteFormatter) Format(f State, _ rune) {
	Fprint(f, "X")
}

var byteFormatterSlice = []byteFormatter{97, 98, 99, 100}

var b byte

var fmtTests = []struct {
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
	{"%#x", []byte("abc\xff"), "0x616263ff"},
	{"%#X", []byte("abc\xff"), "0X616263FF"},
	{"%# x", []byte("abc\xff"), "0x61 0x62 0x63 0xff"},
	{"%# X", []byte("abc\xff"), "0X61 0X62 0X63 0XFF"},

	// basic bytes
	{"%s", []byte("abc"), "abc"},
	{"%x", []byte("abc"), "616263"},
	{"% x", []byte("abc\xff"), "61 62 63 ff"},
	{"%#x", []byte("abc\xff"), "0x616263ff"},
	{"%#X", []byte("abc\xff"), "0X616263FF"},
	{"%# x", []byte("abc\xff"), "0x61 0x62 0x63 0xff"},
	{"%# X", []byte("abc\xff"), "0X61 0X62 0X63 0XFF"},
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
	{"%.5x", "abcdefghijklmnopqrstuvwxyz", `6162636465`},
	{"%.5q", []byte("abcdefghijklmnopqrstuvwxyz"), `"abcde"`},
	{"%.5x", []byte("abcdefghijklmnopqrstuvwxyz"), `6162636465`},
	{"%.3q", "日本語日本語", `"日本語"`},
	{"%.3q", []byte("日本語日本語"), `"日本語"`},
	{"%.1q", "日本語", `"日"`},
	{"%.1q", []byte("日本語"), `"日"`},
	{"%.1x", "日本語", `e6`},
	{"%.1X", []byte("日本語"), `E6`},
	{"%10.1q", "日本語日本語", `       "日"`},
	{"%3c", '⌘', "  ⌘"},
	{"%5q", '\u2026', `  '…'`},
	{"%10v", nil, "     <nil>"},
	{"%-10v", nil, "<nil>     "},

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
	{"%+.3F", -1.0, "-1.000"},
	{"%+.3F", float32(-1.0), "-1.000"},
	{"%+07.2f", 1.0, "+001.00"},
	{"%+07.2f", -1.0, "-001.00"},
	{"%+10.2f", +1.0, "     +1.00"},
	{"%+10.2f", -1.0, "     -1.00"},
	{"% .3E", -1.0, "-1.000E+00"},
	{"% .3e", 1.0, " 1.000e+00"},
	{"%+.3g", 0.0, "+0"},
	{"%+.3g", 1.0, "+1"},
	{"%+.3g", -1.0, "-1"},
	{"% .3g", -1.0, "-1"},
	{"% .3g", 1.0, " 1"},
	{"%b", float32(1.0), "8388608p-23"},
	{"%b", 1.0, "4503599627370496p-52"},

	// complex values
	{"%+.3e", 0i, "(+0.000e+00+0.000e+00i)"},
	{"%+.3f", 0i, "(+0.000+0.000i)"},
	{"%+.3g", 0i, "(+0+0i)"},
	{"%+.3e", 1 + 2i, "(+1.000e+00+2.000e+00i)"},
	{"%+.3f", 1 + 2i, "(+1.000+2.000i)"},
	{"%+.3g", 1 + 2i, "(+1+2i)"},
	{"%.3e", 0i, "(0.000e+00+0.000e+00i)"},
	{"%.3f", 0i, "(0.000+0.000i)"},
	{"%.3F", 0i, "(0.000+0.000i)"},
	{"%.3F", complex64(0i), "(0.000+0.000i)"},
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
	{"%b", complex64(1 + 2i), "(8388608p-23+8388608p-22i)"},
	{"%b", 1 + 2i, "(4503599627370496p-52+4503599627370496p-51i)"},

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
	{"%v", barray, "[1 2 3 4 5]"},
	{"%v", &array, "&[1 2 3 4 5]"},
	{"%v", &iarray, "&[1 hello 2.5 <nil>]"},
	{"%v", &barray, "&[1 2 3 4 5]"},

	// slices
	{"%v", slice, "[1 2 3 4 5]"},
	{"%v", islice, "[1 hello 2.5 <nil>]"},
	{"%v", bslice, "[1 2 3 4 5]"},
	{"%v", &slice, "&[1 2 3 4 5]"},
	{"%v", &islice, "&[1 hello 2.5 <nil>]"},
	{"%v", &bslice, "&[1 2 3 4 5]"},

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

	// other formats on Stringable items
	{"%s", I(23), `<23>`},
	{"%q", I(23), `"<23>"`},
	{"%x", I(23), `3c32333e`},
	{"%#x", I(23), `0x3c32333e`},
	{"%# x", I(23), `0x3c 0x32 0x33 0x3e`},
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
	{"%#v", "foo", `"foo"`},
	{"%#v", barray, `[5]fmt_test.renamedUint8{0x1, 0x2, 0x3, 0x4, 0x5}`},
	{"%#v", bslice, `[]fmt_test.renamedUint8{0x1, 0x2, 0x3, 0x4, 0x5}`},
	{"%#v", []byte(nil), "[]byte(nil)"},
	{"%#v", []int32(nil), "[]int32(nil)"},

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
	{"%x", []renamedUint8{'a', 'b', 'c'}, "616263"},
	{"%s", []renamedUint8{'h', 'e', 'l', 'l', 'o'}, "hello"},
	{"%q", []renamedUint8{'h', 'e', 'l', 'l', 'o'}, `"hello"`},
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
	{"%10T", nil, "     <nil>"},
	{"%-10T", nil, "<nil>     "},

	// %p
	{"p0=%p", new(int), "p0=0xPTR"},
	{"p1=%s", &pValue, "p1=String(p)"}, // String method...
	{"p2=%p", &pValue, "p2=0xPTR"},     // ... not called with %p
	{"p3=%p", (*int)(nil), "p3=0x0"},
	{"p4=%#p", new(int), "p4=PTR"},

	// %p on non-pointers
	{"%p", make(chan int), "0xPTR"},
	{"%p", make(map[int]int), "0xPTR"},
	{"%p", make([]int, 1), "0xPTR"},
	{"%p", 27, "%!p(int=27)"}, // not a pointer at all

	// %q on pointers
	{"%q", (*int)(nil), "%!q(*int=<nil>)"},
	{"%q", new(int), "%!q(*int=0xPTR)"},

	// %v on pointers formats 0 as <nil>
	{"%v", (*int)(nil), "<nil>"},
	{"%v", new(int), "0xPTR"},

	// %d etc. pointers use specified base.
	{"%d", new(int), "PTR_d"},
	{"%o", new(int), "PTR_o"},
	{"%x", new(int), "PTR_x"},

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

	// The "<nil>" show up because maps are printed by
	// first obtaining a list of keys and then looking up
	// each key.  Since NaNs can be map keys but cannot
	// be fetched directly, the lookup fails and returns a
	// zero reflect.Value, which formats as <nil>.
	// This test is just to check that it shows the two NaNs at all.
	{"%v", map[float64]int{math.NaN(): 1, math.NaN(): 2}, "map[NaN:<nil> NaN:<nil>]"},

	// Used to crash because nByte didn't allow for a sign.
	{"%b", int64(-1 << 63), zeroFill("-1", 63, "")},

	// Used to panic.
	{"%0100d", 1, zeroFill("", 100, "1")},
	{"%0100d", -1, zeroFill("-", 99, "1")},
	{"%0.100f", 1.0, zeroFill("1.", 100, "")},
	{"%0.100f", -1.0, zeroFill("-1.", 100, "")},

	// Comparison of padding rules with C printf.
	/*
		C program:
		#include <stdio.h>

		char *format[] = {
			"[%.2f]",
			"[% .2f]",
			"[%+.2f]",
			"[%7.2f]",
			"[% 7.2f]",
			"[%+7.2f]",
			"[%07.2f]",
			"[% 07.2f]",
			"[%+07.2f]",
		};

		int main(void) {
			int i;
			for(i = 0; i < 9; i++) {
				printf("%s: ", format[i]);
				printf(format[i], 1.0);
				printf(" ");
				printf(format[i], -1.0);
				printf("\n");
			}
		}

		Output:
			[%.2f]: [1.00] [-1.00]
			[% .2f]: [ 1.00] [-1.00]
			[%+.2f]: [+1.00] [-1.00]
			[%7.2f]: [   1.00] [  -1.00]
			[% 7.2f]: [   1.00] [  -1.00]
			[%+7.2f]: [  +1.00] [  -1.00]
			[%07.2f]: [0001.00] [-001.00]
			[% 07.2f]: [ 001.00] [-001.00]
			[%+07.2f]: [+001.00] [-001.00]
	*/
	{"%.2f", 1.0, "1.00"},
	{"%.2f", -1.0, "-1.00"},
	{"% .2f", 1.0, " 1.00"},
	{"% .2f", -1.0, "-1.00"},
	{"%+.2f", 1.0, "+1.00"},
	{"%+.2f", -1.0, "-1.00"},
	{"%7.2f", 1.0, "   1.00"},
	{"%7.2f", -1.0, "  -1.00"},
	{"% 7.2f", 1.0, "   1.00"},
	{"% 7.2f", -1.0, "  -1.00"},
	{"%+7.2f", 1.0, "  +1.00"},
	{"%+7.2f", -1.0, "  -1.00"},
	{"%07.2f", 1.0, "0001.00"},
	{"%07.2f", -1.0, "-001.00"},
	{"% 07.2f", 1.0, " 001.00"},
	{"% 07.2f", -1.0, "-001.00"},
	{"%+07.2f", 1.0, "+001.00"},
	{"%+07.2f", -1.0, "-001.00"},

	// Complex numbers: exhaustively tested in TestComplexFormatting.
	{"%7.2f", 1 + 2i, "(   1.00  +2.00i)"},
	{"%+07.2f", -1 - 2i, "(-001.00-002.00i)"},
	// Zero padding does not apply to infinities.
	{"%020f", math.Inf(-1), "                -Inf"},
	{"%020f", math.Inf(+1), "                +Inf"},
	{"% 020f", math.Inf(-1), "                -Inf"},
	{"% 020f", math.Inf(+1), "                 Inf"},
	{"%+020f", math.Inf(-1), "                -Inf"},
	{"%+020f", math.Inf(+1), "                +Inf"},
	{"%20f", -1.0, "           -1.000000"},
	// Make sure we can handle very large widths.
	{"%0100f", -1.0, zeroFill("-", 99, "1.000000")},

	// Complex fmt used to leave the plus flag set for future entries in the array
	// causing +2+0i and +3+0i instead of 2+0i and 3+0i.
	{"%v", []complex64{1, 2, 3}, "[(1+0i) (2+0i) (3+0i)]"},
	{"%v", []complex128{1, 2, 3}, "[(1+0i) (2+0i) (3+0i)]"},

	// Incomplete format specification caused crash.
	{"%.", 3, "%!.(int=3)"},

	// Used to panic with out-of-bounds for very large numeric representations.
	// nByte is set to handle one bit per uint64 in %b format, with a negative number.
	// See issue 6777.
	{"%#064x", 1, zeroFill("0x", 64, "1")},
	{"%#064x", -1, zeroFill("-0x", 63, "1")},
	{"%#064b", 1, zeroFill("", 64, "1")},
	{"%#064b", -1, zeroFill("-", 63, "1")},
	{"%#064o", 1, zeroFill("", 64, "1")},
	{"%#064o", -1, zeroFill("-", 63, "1")},
	{"%#064d", 1, zeroFill("", 64, "1")},
	{"%#064d", -1, zeroFill("-", 63, "1")},
	// Test that we handle the crossover above the size of uint64
	{"%#072x", 1, zeroFill("0x", 72, "1")},
	{"%#072x", -1, zeroFill("-0x", 71, "1")},
	{"%#072b", 1, zeroFill("", 72, "1")},
	{"%#072b", -1, zeroFill("-", 71, "1")},
	{"%#072o", 1, zeroFill("", 72, "1")},
	{"%#072o", -1, zeroFill("-", 71, "1")},
	{"%#072d", 1, zeroFill("", 72, "1")},
	{"%#072d", -1, zeroFill("-", 71, "1")},

	// Padding for complex numbers. Has been bad, then fixed, then bad again.
	{"%+10.2f", +104.66 + 440.51i, "(   +104.66   +440.51i)"},
	{"%+10.2f", -104.66 + 440.51i, "(   -104.66   +440.51i)"},
	{"%+10.2f", +104.66 - 440.51i, "(   +104.66   -440.51i)"},
	{"%+10.2f", -104.66 - 440.51i, "(   -104.66   -440.51i)"},
	{"%+010.2f", +104.66 + 440.51i, "(+000104.66+000440.51i)"},
	{"%+010.2f", -104.66 + 440.51i, "(-000104.66+000440.51i)"},
	{"%+010.2f", +104.66 - 440.51i, "(+000104.66-000440.51i)"},
	{"%+010.2f", -104.66 - 440.51i, "(-000104.66-000440.51i)"},

	// []T where type T is a byte with a Stringer method.
	{"%v", byteStringerSlice, "[X X X X]"},
	{"%s", byteStringerSlice, "abcd"},
	{"%q", byteStringerSlice, "\"abcd\""},
	{"%x", byteStringerSlice, "61626364"},
	{"%#v", byteStringerSlice, "[]fmt_test.byteStringer{0x61, 0x62, 0x63, 0x64}"},

	// And the same for Formatter.
	{"%v", byteFormatterSlice, "[X X X X]"},
	{"%s", byteFormatterSlice, "abcd"},
	{"%q", byteFormatterSlice, "\"abcd\""},
	{"%x", byteFormatterSlice, "61626364"},
	// This next case seems wrong, but the docs say the Formatter wins here.
	{"%#v", byteFormatterSlice, "[]fmt_test.byteFormatter{X, X, X, X}"},
}

// zeroFill generates zero-filled strings of the specified width. The length
// of the suffix (but not the prefix) is compensated for in the width calculation.
func zeroFill(prefix string, width int, suffix string) string {
	return prefix + strings.Repeat("0", width-len(suffix)) + suffix
}

func TestSprintf(t *testing.T) {
	for _, tt := range fmtTests {
		s := Sprintf(tt.fmt, tt.val)
		if i := strings.Index(tt.out, "PTR"); i >= 0 {
			pattern := "PTR"
			chars := "0123456789abcdefABCDEF"
			switch {
			case strings.HasPrefix(tt.out[i:], "PTR_d"):
				pattern = "PTR_d"
				chars = chars[:10]
			case strings.HasPrefix(tt.out[i:], "PTR_o"):
				pattern = "PTR_o"
				chars = chars[:8]
			case strings.HasPrefix(tt.out[i:], "PTR_x"):
				pattern = "PTR_x"
			}
			j := i
			for ; j < len(s); j++ {
				c := s[j]
				if !strings.ContainsRune(chars, rune(c)) {
					break
				}
			}
			s = s[0:i] + pattern + s[j:]
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

// TestComplexFormatting checks that a complex always formats to the same
// thing as if done by hand with two singleton prints.
func TestComplexFormatting(t *testing.T) {
	var yesNo = []bool{true, false}
	var values = []float64{1, 0, -1, math.Inf(1), math.Inf(-1), math.NaN()}
	for _, plus := range yesNo {
		for _, zero := range yesNo {
			for _, space := range yesNo {
				for _, char := range "fFeEgG" {
					realFmt := "%"
					if zero {
						realFmt += "0"
					}
					if space {
						realFmt += " "
					}
					if plus {
						realFmt += "+"
					}
					realFmt += "10.2"
					realFmt += string(char)
					// Imaginary part always has a sign, so force + and ignore space.
					imagFmt := "%"
					if zero {
						imagFmt += "0"
					}
					imagFmt += "+"
					imagFmt += "10.2"
					imagFmt += string(char)
					for _, realValue := range values {
						for _, imagValue := range values {
							one := Sprintf(realFmt, complex(realValue, imagValue))
							two := Sprintf("("+realFmt+imagFmt+"i)", realValue, imagValue)
							if one != two {
								t.Error(f, one, two)
							}
						}
					}
				}
			}
		}
	}
}

type SE []interface{} // slice of empty; notational compactness.

var reorderTests = []struct {
	fmt string
	val SE
	out string
}{
	{"%[1]d", SE{1}, "1"},
	{"%[2]d", SE{2, 1}, "1"},
	{"%[2]d %[1]d", SE{1, 2}, "2 1"},
	{"%[2]*[1]d", SE{2, 5}, "    2"},
	{"%6.2f", SE{12.0}, " 12.00"}, // Explicit version of next line.
	{"%[3]*.[2]*[1]f", SE{12.0, 2, 6}, " 12.00"},
	{"%[1]*.[2]*[3]f", SE{6, 2, 12.0}, " 12.00"},
	{"%10f", SE{12.0}, " 12.000000"},
	{"%[1]*[3]f", SE{10, 99, 12.0}, " 12.000000"},
	{"%.6f", SE{12.0}, "12.000000"}, // Explicit version of next line.
	{"%.[1]*[3]f", SE{6, 99, 12.0}, "12.000000"},
	{"%6.f", SE{12.0}, "    12"}, //  // Explicit version of next line; empty precision means zero.
	{"%[1]*.[3]f", SE{6, 3, 12.0}, "    12"},
	// An actual use! Print the same arguments twice.
	{"%d %d %d %#[1]o %#o %#o", SE{11, 12, 13}, "11 12 13 013 014 015"},

	// Erroneous cases.
	{"%[d", SE{2, 1}, "%!d(BADINDEX)"},
	{"%]d", SE{2, 1}, "%!](int=2)d%!(EXTRA int=1)"},
	{"%[]d", SE{2, 1}, "%!d(BADINDEX)"},
	{"%[-3]d", SE{2, 1}, "%!d(BADINDEX)"},
	{"%[99]d", SE{2, 1}, "%!d(BADINDEX)"},
	{"%[3]", SE{2, 1}, "%!(NOVERB)"},
	{"%[1].2d", SE{5, 6}, "%!d(BADINDEX)"},
	{"%[1]2d", SE{2, 1}, "%!d(BADINDEX)"},
	{"%3.[2]d", SE{7}, "%!d(BADINDEX)"},
	{"%.[2]d", SE{7}, "%!d(BADINDEX)"},
	{"%d %d %d %#[1]o %#o %#o %#o", SE{11, 12, 13}, "11 12 13 013 014 015 %!o(MISSING)"},
	{"%[5]d %[2]d %d", SE{1, 2, 3}, "%!d(BADINDEX) 2 3"},
	{"%d %[3]d %d", SE{1, 2}, "1 %!d(BADINDEX) 2"}, // Erroneous index does not affect sequence.
}

func TestReorder(t *testing.T) {
	for _, tt := range reorderTests {
		s := Sprintf(tt.fmt, tt.val...)
		if s != tt.out {
			t.Errorf("Sprintf(%q, %v) = <%s> want <%s>", tt.fmt, tt.val, s, tt.out)
		} else {
		}
	}
}

func BenchmarkSprintfEmpty(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Sprintf("")
		}
	})
}

func BenchmarkSprintfString(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Sprintf("%s", "hello")
		}
	})
}

func BenchmarkSprintfInt(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Sprintf("%d", 5)
		}
	})
}

func BenchmarkSprintfIntInt(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Sprintf("%d %d", 5, 6)
		}
	})
}

func BenchmarkSprintfPrefixedInt(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Sprintf("This is some meaningless prefix text that needs to be scanned %d", 6)
		}
	})
}

func BenchmarkSprintfFloat(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			Sprintf("%g", 5.23184)
		}
	})
}

func BenchmarkManyArgs(b *testing.B) {
	b.RunParallel(func(pb *testing.PB) {
		var buf bytes.Buffer
		for pb.Next() {
			buf.Reset()
			Fprintf(&buf, "%2d/%2d/%2d %d:%d:%d %s %s\n", 3, 4, 5, 11, 12, 13, "hello", "world")
		}
	})
}

func BenchmarkFprintInt(b *testing.B) {
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Fprint(&buf, 123456)
	}
}

func BenchmarkFprintIntNoAlloc(b *testing.B) {
	var x interface{} = 123456
	var buf bytes.Buffer
	for i := 0; i < b.N; i++ {
		buf.Reset()
		Fprint(&buf, x)
	}
}

var mallocBuf bytes.Buffer
var mallocPointer *int // A pointer so we know the interface value won't allocate.

var mallocTest = []struct {
	count int
	desc  string
	fn    func()
}{
	{0, `Sprintf("")`, func() { Sprintf("") }},
	{1, `Sprintf("xxx")`, func() { Sprintf("xxx") }},
	{2, `Sprintf("%x")`, func() { Sprintf("%x", 7) }},
	{2, `Sprintf("%s")`, func() { Sprintf("%s", "hello") }},
	{3, `Sprintf("%x %x")`, func() { Sprintf("%x %x", 7, 112) }},
	{2, `Sprintf("%g")`, func() { Sprintf("%g", float32(3.14159)) }}, // TODO: Can this be 1?
	{1, `Fprintf(buf, "%s")`, func() { mallocBuf.Reset(); Fprintf(&mallocBuf, "%s", "hello") }},
	// If the interface value doesn't need to allocate, amortized allocation overhead should be zero.
	{0, `Fprintf(buf, "%x %x %x")`, func() {
		mallocBuf.Reset()
		Fprintf(&mallocBuf, "%x %x %x", mallocPointer, mallocPointer, mallocPointer)
	}},
}

var _ bytes.Buffer

func TestCountMallocs(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping malloc count in short mode")
	}
	if runtime.GOMAXPROCS(0) > 1 {
		t.Skip("skipping; GOMAXPROCS>1")
	}
	for _, mt := range mallocTest {
		mallocs := testing.AllocsPerRun(100, mt.fn)
		if got, max := mallocs, float64(mt.count); got > max {
			t.Errorf("%s: got %v allocs, want <=%v", mt.desc, got, max)
		}
	}
}

type flagPrinter struct{}

func (flagPrinter) Format(f State, c rune) {
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
	type T struct {
		a string
		b string
		c int
	}
	var s T
	s.a = "abc"
	s.b = "def"
	s.c = 123
	var tests = []struct {
		fmt string
		out string
	}{
		{"%v", "{abc def 123}"},
		{"%+v", "{a:abc b:def c:123}"},
		{"%#v", `fmt_test.T{a:"abc", b:"def", c:123}`},
	}
	for _, tt := range tests {
		out := Sprintf(tt.fmt, s)
		if out != tt.out {
			t.Errorf("Sprintf(%q, s) = %#q, want %#q", tt.fmt, out, tt.out)
		}
		// The same but with a pointer.
		out = Sprintf(tt.fmt, &s)
		if out != "&"+tt.out {
			t.Errorf("Sprintf(%q, &s) = %#q, want %#q", tt.fmt, out, "&"+tt.out)
		}
	}
}

func TestSlicePrinter(t *testing.T) {
	slice := []int{}
	s := Sprint(slice)
	if s != "[]" {
		t.Errorf("empty slice printed as %q not %q", s, "[]")
	}
	slice = []int{1, 2, 3}
	s = Sprint(slice)
	if s != "[1 2 3]" {
		t.Errorf("slice: got %q expected %q", s, "[1 2 3]")
	}
	s = Sprint(&slice)
	if s != "&[1 2 3]" {
		t.Errorf("&slice: got %q expected %q", s, "&[1 2 3]")
	}
}

// presentInMap checks map printing using substrings so we don't depend on the
// print order.
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
	// Pointer to map prints the same but with initial &.
	if !strings.HasPrefix(Sprint(&m1), "&") {
		t.Errorf("no initial & for address of map")
	}
	presentInMap(Sprintf("%v", &m1), a, t)
	presentInMap(Sprint(&m1), a, t)
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

// TestBlank checks that Sprint (and hence Print, Fprint) puts spaces in the
// right places, that is, between arg pairs in which neither is a string.
func TestBlank(t *testing.T) {
	got := Sprint("<", 1, ">:", 1, 2, 3, "!")
	expect := "<1>:1 2 3!"
	if got != expect {
		t.Errorf("got %q expected %q", got, expect)
	}
}

// TestBlankln checks that Sprintln (and hence Println, Fprintln) puts spaces in
// the right places, that is, between all arg pairs.
func TestBlankln(t *testing.T) {
	got := Sprintln("<", 1, ">:", 1, 2, 3, "!")
	expect := "< 1 >: 1 2 3 !\n"
	if got != expect {
		t.Errorf("got %q expected %q", got, expect)
	}
}

// TestFormatterPrintln checks Formatter with Sprint, Sprintln, Sprintf.
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

// Panic is a type that panics in String.
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

// PanicF is a type that panics in Format.
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
	{"%s", Panic{io.ErrUnexpectedEOF}, "%!s(PANIC=unexpected EOF)"},
	{"%s", Panic{3}, "%!s(PANIC=3)"},
	// GoString
	{"%#v", (*Panic)(nil), "<nil>"}, // nil pointer special case
	{"%#v", Panic{io.ErrUnexpectedEOF}, "%!v(PANIC=unexpected EOF)"},
	{"%#v", Panic{3}, "%!v(PANIC=3)"},
	// Format
	{"%s", (*PanicF)(nil), "<nil>"}, // nil pointer special case
	{"%s", PanicF{io.ErrUnexpectedEOF}, "%!s(PANIC=unexpected EOF)"},
	{"%s", PanicF{3}, "%!s(PANIC=3)"},
}

func TestPanics(t *testing.T) {
	for i, tt := range panictests {
		s := Sprintf(tt.fmt, tt.in)
		if s != tt.out {
			t.Errorf("%d: %q: got %q expected %q", i, tt.fmt, s, tt.out)
		}
	}
}

// recurCount tests that erroneous String routine doesn't cause fatal recursion.
var recurCount = 0

type Recur struct {
	i      int
	failed *bool
}

func (r *Recur) String() string {
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
	r := &Recur{3, &failed}
	Sprintf("recur@%p value: %d\n", &r, r.i)
	if failed {
		t.Error("fail with pointer")
	}
	failed = false
	r = &Recur{4, &failed}
	Sprintf("recur@%p, value: %d\n", r, r.i)
	if failed {
		t.Error("fail with value")
	}
}

func TestIsSpace(t *testing.T) {
	// This tests the internal isSpace function.
	// IsSpace = isSpace is defined in export_test.go.
	for i := rune(0); i <= unicode.MaxRune; i++ {
		if IsSpace(i) != unicode.IsSpace(i) {
			t.Errorf("isSpace(%U) = %v, want %v", i, IsSpace(i), unicode.IsSpace(i))
		}
	}
}

func TestNilDoesNotBecomeTyped(t *testing.T) {
	type A struct{}
	type B struct{}
	var a *A = nil
	var b B = B{}
	got := Sprintf("%s %s %s %s %s", nil, a, nil, b, nil) // go vet should complain about this line.
	const expect = "%!s(<nil>) %!s(*fmt_test.A=<nil>) %!s(<nil>) {} %!s(<nil>)"
	if got != expect {
		t.Errorf("expected:\n\t%q\ngot:\n\t%q", expect, got)
	}
}

var formatterFlagTests = []struct {
	in  string
	val interface{}
	out string
}{
	// scalar values with the (unused by fmt) 'a' verb.
	{"%a", flagPrinter{}, "[%a]"},
	{"%-a", flagPrinter{}, "[%-a]"},
	{"%+a", flagPrinter{}, "[%+a]"},
	{"%#a", flagPrinter{}, "[%#a]"},
	{"% a", flagPrinter{}, "[% a]"},
	{"%0a", flagPrinter{}, "[%0a]"},
	{"%1.2a", flagPrinter{}, "[%1.2a]"},
	{"%-1.2a", flagPrinter{}, "[%-1.2a]"},
	{"%+1.2a", flagPrinter{}, "[%+1.2a]"},
	{"%-+1.2a", flagPrinter{}, "[%+-1.2a]"},
	{"%-+1.2abc", flagPrinter{}, "[%+-1.2a]bc"},
	{"%-1.2abc", flagPrinter{}, "[%-1.2a]bc"},

	// composite values with the 'a' verb
	{"%a", [1]flagPrinter{}, "[[%a]]"},
	{"%-a", [1]flagPrinter{}, "[[%-a]]"},
	{"%+a", [1]flagPrinter{}, "[[%+a]]"},
	{"%#a", [1]flagPrinter{}, "[[%#a]]"},
	{"% a", [1]flagPrinter{}, "[[% a]]"},
	{"%0a", [1]flagPrinter{}, "[[%0a]]"},
	{"%1.2a", [1]flagPrinter{}, "[[%1.2a]]"},
	{"%-1.2a", [1]flagPrinter{}, "[[%-1.2a]]"},
	{"%+1.2a", [1]flagPrinter{}, "[[%+1.2a]]"},
	{"%-+1.2a", [1]flagPrinter{}, "[[%+-1.2a]]"},
	{"%-+1.2abc", [1]flagPrinter{}, "[[%+-1.2a]]bc"},
	{"%-1.2abc", [1]flagPrinter{}, "[[%-1.2a]]bc"},

	// simple values with the 'v' verb
	{"%v", flagPrinter{}, "[%v]"},
	{"%-v", flagPrinter{}, "[%-v]"},
	{"%+v", flagPrinter{}, "[%+v]"},
	{"%#v", flagPrinter{}, "[%#v]"},
	{"% v", flagPrinter{}, "[% v]"},
	{"%0v", flagPrinter{}, "[%0v]"},
	{"%1.2v", flagPrinter{}, "[%1.2v]"},
	{"%-1.2v", flagPrinter{}, "[%-1.2v]"},
	{"%+1.2v", flagPrinter{}, "[%+1.2v]"},
	{"%-+1.2v", flagPrinter{}, "[%+-1.2v]"},
	{"%-+1.2vbc", flagPrinter{}, "[%+-1.2v]bc"},
	{"%-1.2vbc", flagPrinter{}, "[%-1.2v]bc"},

	// composite values with the 'v' verb.
	{"%v", [1]flagPrinter{}, "[[%v]]"},
	{"%-v", [1]flagPrinter{}, "[[%-v]]"},
	{"%+v", [1]flagPrinter{}, "[[%+v]]"},
	{"%#v", [1]flagPrinter{}, "[1]fmt_test.flagPrinter{[%#v]}"},
	{"% v", [1]flagPrinter{}, "[[% v]]"},
	{"%0v", [1]flagPrinter{}, "[[%0v]]"},
	{"%1.2v", [1]flagPrinter{}, "[[%1.2v]]"},
	{"%-1.2v", [1]flagPrinter{}, "[[%-1.2v]]"},
	{"%+1.2v", [1]flagPrinter{}, "[[%+1.2v]]"},
	{"%-+1.2v", [1]flagPrinter{}, "[[%+-1.2v]]"},
	{"%-+1.2vbc", [1]flagPrinter{}, "[[%+-1.2v]]bc"},
	{"%-1.2vbc", [1]flagPrinter{}, "[[%-1.2v]]bc"},
}

func TestFormatterFlags(t *testing.T) {
	for _, tt := range formatterFlagTests {
		s := Sprintf(tt.in, tt.val)
		if s != tt.out {
			t.Errorf("Sprintf(%q, %T) = %q, want %q", tt.in, tt.val, s, tt.out)
		}
	}
}
