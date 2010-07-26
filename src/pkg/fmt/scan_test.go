// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fmt_test

import (
	. "fmt"
	"io"
	"os"
	"reflect"
	"strings"
	"testing"
	"utf8"
)

type ScanTest struct {
	text string
	in   interface{}
	out  interface{}
}

type ScanfTest struct {
	format string
	text   string
	in     interface{}
	out    interface{}
}

type ScanfMultiTest struct {
	format string
	text   string
	in     []interface{}
	out    []interface{}
	err    string
}

var (
	boolVal              bool
	intVal               int
	int8Val              int8
	int16Val             int16
	int32Val             int32
	int64Val             int64
	uintVal              uint
	uint8Val             uint8
	uint16Val            uint16
	uint32Val            uint32
	uint64Val            uint64
	floatVal             float
	float32Val           float32
	float64Val           float64
	stringVal            string
	stringVal1           string
	bytesVal             []byte
	complexVal           complex
	complex64Val         complex64
	complex128Val        complex128
	renamedBoolVal       renamedBool
	renamedIntVal        renamedInt
	renamedInt8Val       renamedInt8
	renamedInt16Val      renamedInt16
	renamedInt32Val      renamedInt32
	renamedInt64Val      renamedInt64
	renamedUintVal       renamedUint
	renamedUint8Val      renamedUint8
	renamedUint16Val     renamedUint16
	renamedUint32Val     renamedUint32
	renamedUint64Val     renamedUint64
	renamedUintptrVal    renamedUintptr
	renamedStringVal     renamedString
	renamedBytesVal      renamedBytes
	renamedFloatVal      renamedFloat
	renamedFloat32Val    renamedFloat32
	renamedFloat64Val    renamedFloat64
	renamedComplexVal    renamedComplex
	renamedComplex64Val  renamedComplex64
	renamedComplex128Val renamedComplex128
)

// Xs accepts any non-empty run of the verb character
type Xs string

func (x *Xs) Scan(state ScanState, verb int) os.Error {
	var tok string
	var c int
	var err os.Error
	wid, present := state.Width()
	if !present {
		tok, err = state.Token()
	} else {
		for i := 0; i < wid; i++ {
			c, err = state.GetRune()
			if err != nil {
				break
			}
			tok += string(c)
		}
	}
	if err != nil {
		return err
	}
	if !testing.MustCompile("^" + string(verb) + "+$").MatchString(tok) {
		return os.ErrorString("syntax error for xs")
	}
	*x = Xs(tok)
	return nil
}

var xVal Xs

// myStringReader implements Read but not ReadRune, allowing us to test our readRune wrapper
// type that creates something that can read runes given only Read().
type myStringReader struct {
	r *strings.Reader
}

func (s *myStringReader) Read(p []byte) (n int, err os.Error) {
	return s.r.Read(p)
}

func newReader(s string) *myStringReader {
	return &myStringReader{strings.NewReader(s)}
}

var scanTests = []ScanTest{
	// Numbers
	ScanTest{"T\n", &boolVal, true},  // boolean test vals toggle to be sure they are written
	ScanTest{"F\n", &boolVal, false}, // restored to zero value
	ScanTest{"21\n", &intVal, 21},
	ScanTest{"22\n", &int8Val, int8(22)},
	ScanTest{"23\n", &int16Val, int16(23)},
	ScanTest{"24\n", &int32Val, int32(24)},
	ScanTest{"25\n", &int64Val, int64(25)},
	ScanTest{"127\n", &int8Val, int8(127)},
	ScanTest{"-21\n", &intVal, -21},
	ScanTest{"-22\n", &int8Val, int8(-22)},
	ScanTest{"-23\n", &int16Val, int16(-23)},
	ScanTest{"-24\n", &int32Val, int32(-24)},
	ScanTest{"-25\n", &int64Val, int64(-25)},
	ScanTest{"-128\n", &int8Val, int8(-128)},
	ScanTest{"+21\n", &intVal, +21},
	ScanTest{"+22\n", &int8Val, int8(+22)},
	ScanTest{"+23\n", &int16Val, int16(+23)},
	ScanTest{"+24\n", &int32Val, int32(+24)},
	ScanTest{"+25\n", &int64Val, int64(+25)},
	ScanTest{"+127\n", &int8Val, int8(+127)},
	ScanTest{"26\n", &uintVal, uint(26)},
	ScanTest{"27\n", &uint8Val, uint8(27)},
	ScanTest{"28\n", &uint16Val, uint16(28)},
	ScanTest{"29\n", &uint32Val, uint32(29)},
	ScanTest{"30\n", &uint64Val, uint64(30)},
	ScanTest{"255\n", &uint8Val, uint8(255)},
	ScanTest{"32767\n", &int16Val, int16(32767)},
	ScanTest{"2.3\n", &floatVal, 2.3},
	ScanTest{"2.3e1\n", &float32Val, float32(2.3e1)},
	ScanTest{"2.3e2\n", &float64Val, float64(2.3e2)},
	ScanTest{"2.35\n", &stringVal, "2.35"},
	ScanTest{"2345678\n", &bytesVal, []byte("2345678")},
	ScanTest{"(3.4e1-2i)\n", &complexVal, 3.4e1 - 2i},
	ScanTest{"-3.45e1-3i\n", &complex64Val, complex64(-3.45e1 - 3i)},
	ScanTest{"-.45e1-1e2i\n", &complex128Val, complex128(-.45e1 - 100i)},
	ScanTest{"hello\n", &stringVal, "hello"},

	// Renamed types
	ScanTest{"true\n", &renamedBoolVal, renamedBool(true)},
	ScanTest{"F\n", &renamedBoolVal, renamedBool(false)},
	ScanTest{"101\n", &renamedIntVal, renamedInt(101)},
	ScanTest{"102\n", &renamedIntVal, renamedInt(102)},
	ScanTest{"103\n", &renamedUintVal, renamedUint(103)},
	ScanTest{"104\n", &renamedUintVal, renamedUint(104)},
	ScanTest{"105\n", &renamedInt8Val, renamedInt8(105)},
	ScanTest{"106\n", &renamedInt16Val, renamedInt16(106)},
	ScanTest{"107\n", &renamedInt32Val, renamedInt32(107)},
	ScanTest{"108\n", &renamedInt64Val, renamedInt64(108)},
	ScanTest{"109\n", &renamedUint8Val, renamedUint8(109)},
	ScanTest{"110\n", &renamedUint16Val, renamedUint16(110)},
	ScanTest{"111\n", &renamedUint32Val, renamedUint32(111)},
	ScanTest{"112\n", &renamedUint64Val, renamedUint64(112)},
	ScanTest{"113\n", &renamedUintptrVal, renamedUintptr(113)},
	ScanTest{"114\n", &renamedStringVal, renamedString("114")},
	ScanTest{"115\n", &renamedBytesVal, renamedBytes([]byte("115"))},

	// Custom scanner.
	ScanTest{"  vvv ", &xVal, Xs("vvv")},
}

var scanfTests = []ScanfTest{
	ScanfTest{"%v", "TRUE\n", &boolVal, true},
	ScanfTest{"%t", "false\n", &boolVal, false},
	ScanfTest{"%v", "-71\n", &intVal, -71},
	ScanfTest{"%d", "72\n", &intVal, 72},
	ScanfTest{"%c", "a\n", &intVal, 'a'},
	ScanfTest{"%c", "\u5072\n", &intVal, 0x5072},
	ScanfTest{"%c", "\u1234\n", &intVal, '\u1234'},
	ScanfTest{"%d", "73\n", &int8Val, int8(73)},
	ScanfTest{"%d", "+74\n", &int16Val, int16(74)},
	ScanfTest{"%d", "75\n", &int32Val, int32(75)},
	ScanfTest{"%d", "76\n", &int64Val, int64(76)},
	ScanfTest{"%b", "1001001\n", &intVal, 73},
	ScanfTest{"%o", "075\n", &intVal, 075},
	ScanfTest{"%x", "a75\n", &intVal, 0xa75},
	ScanfTest{"%v", "71\n", &uintVal, uint(71)},
	ScanfTest{"%d", "72\n", &uintVal, uint(72)},
	ScanfTest{"%d", "73\n", &uint8Val, uint8(73)},
	ScanfTest{"%d", "74\n", &uint16Val, uint16(74)},
	ScanfTest{"%d", "75\n", &uint32Val, uint32(75)},
	ScanfTest{"%d", "76\n", &uint64Val, uint64(76)},
	ScanfTest{"%b", "1001001\n", &uintVal, uint(73)},
	ScanfTest{"%o", "075\n", &uintVal, uint(075)},
	ScanfTest{"%x", "a75\n", &uintVal, uint(0xa75)},
	ScanfTest{"%x", "A75\n", &uintVal, uint(0xa75)},

	// Strings
	ScanfTest{"%s", "using-%s\n", &stringVal, "using-%s"},
	ScanfTest{"%x", "7573696e672d2578\n", &stringVal, "using-%x"},
	ScanfTest{"%q", `"quoted\twith\\do\u0075bl\x65s"` + "\n", &stringVal, "quoted\twith\\doubles"},
	ScanfTest{"%q", "`quoted with backs`\n", &stringVal, "quoted with backs"},

	// Byte slices
	ScanfTest{"%s", "bytes-%s\n", &bytesVal, []byte("bytes-%s")},
	ScanfTest{"%x", "62797465732d2578\n", &bytesVal, []byte("bytes-%x")},
	ScanfTest{"%q", `"bytes\rwith\vdo\u0075bl\x65s"` + "\n", &bytesVal, []byte("bytes\rwith\vdoubles")},
	ScanfTest{"%q", "`bytes with backs`\n", &bytesVal, []byte("bytes with backs")},

	// Renamed types
	ScanfTest{"%v\n", "true\n", &renamedBoolVal, renamedBool(true)},
	ScanfTest{"%t\n", "F\n", &renamedBoolVal, renamedBool(false)},
	ScanfTest{"%v", "101\n", &renamedIntVal, renamedInt(101)},
	ScanfTest{"%c", "\u0101\n", &renamedIntVal, renamedInt('\u0101')},
	ScanfTest{"%o", "0146\n", &renamedIntVal, renamedInt(102)},
	ScanfTest{"%v", "103\n", &renamedUintVal, renamedUint(103)},
	ScanfTest{"%d", "104\n", &renamedUintVal, renamedUint(104)},
	ScanfTest{"%d", "105\n", &renamedInt8Val, renamedInt8(105)},
	ScanfTest{"%d", "106\n", &renamedInt16Val, renamedInt16(106)},
	ScanfTest{"%d", "107\n", &renamedInt32Val, renamedInt32(107)},
	ScanfTest{"%d", "108\n", &renamedInt64Val, renamedInt64(108)},
	ScanfTest{"%x", "6D\n", &renamedUint8Val, renamedUint8(109)},
	ScanfTest{"%o", "0156\n", &renamedUint16Val, renamedUint16(110)},
	ScanfTest{"%d", "111\n", &renamedUint32Val, renamedUint32(111)},
	ScanfTest{"%d", "112\n", &renamedUint64Val, renamedUint64(112)},
	ScanfTest{"%d", "113\n", &renamedUintptrVal, renamedUintptr(113)},
	ScanfTest{"%s", "114\n", &renamedStringVal, renamedString("114")},
	ScanfTest{"%q", "\"1155\"\n", &renamedBytesVal, renamedBytes([]byte("1155"))},
	ScanfTest{"%g", "115.1\n", &renamedFloatVal, renamedFloat(115.1)},
	ScanfTest{"%g", "116e1\n", &renamedFloat32Val, renamedFloat32(116e1)},
	ScanfTest{"%g", "-11.7e+1", &renamedFloat64Val, renamedFloat64(-11.7e+1)},
	ScanfTest{"%g", "11+5.1i\n", &renamedComplexVal, renamedComplex(11 + 5.1i)},
	ScanfTest{"%g", "11+6e1i\n", &renamedComplex64Val, renamedComplex64(11 + 6e1i)},
	ScanfTest{"%g", "-11.+7e+1i", &renamedComplex128Val, renamedComplex128(-11. + 7e+1i)},

	// Interesting formats
	ScanfTest{"here is\tthe value:%d", "here is   the\tvalue:118\n", &intVal, 118},
	ScanfTest{"%% %%:%d", "% %:119\n", &intVal, 119},

	// Corner cases
	ScanfTest{"%x", "FFFFFFFF\n", &uint32Val, uint32(0xFFFFFFFF)},

	// Custom scanner.
	ScanfTest{"%s", "  sss ", &xVal, Xs("sss")},
	ScanfTest{"%2s", "sssss", &xVal, Xs("ss")},

	// Fixed bugs
	ScanfTest{"%d\n", "27\n", &intVal, 27},  // ok
	ScanfTest{"%d\n", "28 \n", &intVal, 28}, // was: "unexpected newline"
}

var overflowTests = []ScanTest{
	ScanTest{"128", &int8Val, 0},
	ScanTest{"32768", &int16Val, 0},
	ScanTest{"-129", &int8Val, 0},
	ScanTest{"-32769", &int16Val, 0},
	ScanTest{"256", &uint8Val, 0},
	ScanTest{"65536", &uint16Val, 0},
	ScanTest{"1e100", &float32Val, 0},
	ScanTest{"1e500", &float64Val, 0},
	ScanTest{"(1e100+0i)", &complexVal, 0},
	ScanTest{"(1+1e100i)", &complex64Val, 0},
	ScanTest{"(1-1e500i)", &complex128Val, 0},
}

var i, j, k int
var f float
var s, t string
var c complex
var x, y Xs

func args(a ...interface{}) []interface{} { return a }

var multiTests = []ScanfMultiTest{
	ScanfMultiTest{"", "", nil, nil, ""},
	ScanfMultiTest{"%d", "23", args(&i), args(23), ""},
	ScanfMultiTest{"%2s%3s", "22333", args(&s, &t), args("22", "333"), ""},
	ScanfMultiTest{"%2d%3d", "44555", args(&i, &j), args(44, 555), ""},
	ScanfMultiTest{"%2d.%3d", "66.777", args(&i, &j), args(66, 777), ""},
	ScanfMultiTest{"%d, %d", "23, 18", args(&i, &j), args(23, 18), ""},
	ScanfMultiTest{"%3d22%3d", "33322333", args(&i, &j), args(333, 333), ""},
	ScanfMultiTest{"%6vX=%3fY", "3+2iX=2.5Y", args(&c, &f), args((3 + 2i), float(2.5)), ""},
	ScanfMultiTest{"%d%s", "123abc", args(&i, &s), args(123, "abc"), ""},
	ScanfMultiTest{"%c%c%c", "2\u50c2X", args(&i, &j, &k), args('2', '\u50c2', 'X'), ""},

	// Custom scanner.
	ScanfMultiTest{"%2e%f", "eefffff", args(&x, &y), args(Xs("ee"), Xs("fffff")), ""},

	// Errors
	ScanfMultiTest{"%t", "23 18", args(&i), nil, "bad verb"},
	ScanfMultiTest{"%d %d %d", "23 18", args(&i, &j), args(23, 18), "too few operands"},
	ScanfMultiTest{"%d %d", "23 18 27", args(&i, &j, &k), args(23, 18), "too many operands"},
	ScanfMultiTest{"%c", "\u0100", args(&int8Val), nil, "overflow"},
	ScanfMultiTest{"X%d", "10X", args(&intVal), nil, "input does not match format"},

	// Bad UTF-8: should see every byte.
	ScanfMultiTest{"%c%c%c", "\xc2X\xc2", args(&i, &j, &k), args(utf8.RuneError, 'X', utf8.RuneError), ""},
}

func testScan(name string, t *testing.T, scan func(r io.Reader, a ...interface{}) (int, os.Error)) {
	for _, test := range scanTests {
		var r io.Reader
		if name == "StringReader" {
			r = strings.NewReader(test.text)
		} else {
			r = newReader(test.text)
		}
		n, err := scan(r, test.in)
		if err != nil {
			t.Errorf("%s got error scanning %q: %s", name, test.text, err)
			continue
		}
		if n != 1 {
			t.Errorf("%s count error on entry %q: got %d", name, test.text, n)
			continue
		}
		// The incoming value may be a pointer
		v := reflect.NewValue(test.in)
		if p, ok := v.(*reflect.PtrValue); ok {
			v = p.Elem()
		}
		val := v.Interface()
		if !reflect.DeepEqual(val, test.out) {
			t.Errorf("%s scanning %q: expected %v got %v, type %T", name, test.text, test.out, val, val)
		}
	}
}

func TestScan(t *testing.T) {
	testScan("StringReader", t, Fscan)
}

func TestMyReaderScan(t *testing.T) {
	testScan("myStringReader", t, Fscan)
}

func TestScanln(t *testing.T) {
	testScan("StringReader", t, Fscanln)
}

func TestMyReaderScanln(t *testing.T) {
	testScan("myStringReader", t, Fscanln)
}

func TestScanf(t *testing.T) {
	for _, test := range scanfTests {
		n, err := Sscanf(test.text, test.format, test.in)
		if err != nil {
			t.Errorf("got error scanning (%q, %q): %s", test.format, test.text, err)
			continue
		}
		if n != 1 {
			t.Errorf("count error on entry (%q, %q): got %d", test.format, test.text, n)
			continue
		}
		// The incoming value may be a pointer
		v := reflect.NewValue(test.in)
		if p, ok := v.(*reflect.PtrValue); ok {
			v = p.Elem()
		}
		val := v.Interface()
		if !reflect.DeepEqual(val, test.out) {
			t.Errorf("scanning (%q, %q): expected %v got %v, type %T", test.format, test.text, test.out, val, val)
		}
	}
}

func TestScanOverflow(t *testing.T) {
	// different machines and different types report errors with different strings.
	re := testing.MustCompile("overflow|too large|out of range|not representable")
	for _, test := range overflowTests {
		_, err := Sscan(test.text, test.in)
		if err == nil {
			t.Errorf("expected overflow scanning %q", test.text)
			continue
		}
		if !re.MatchString(err.String()) {
			t.Errorf("expected overflow error scanning %q: %s", test.text, err)
		}
	}
}

// TODO: there's no conversion from []T to ...T, but we can fake it.  These
// functions do the faking.  We index the table by the length of the param list.
var fscanf = []func(io.Reader, string, []interface{}) (int, os.Error){
	0: func(r io.Reader, f string, i []interface{}) (int, os.Error) { return Fscanf(r, f) },
	1: func(r io.Reader, f string, i []interface{}) (int, os.Error) { return Fscanf(r, f, i[0]) },
	2: func(r io.Reader, f string, i []interface{}) (int, os.Error) { return Fscanf(r, f, i[0], i[1]) },
	3: func(r io.Reader, f string, i []interface{}) (int, os.Error) { return Fscanf(r, f, i[0], i[1], i[2]) },
}

func testScanfMulti(name string, t *testing.T) {
	sliceType := reflect.Typeof(make([]interface{}, 1)).(*reflect.SliceType)
	for _, test := range multiTests {
		var r io.Reader
		if name == "StringReader" {
			r = strings.NewReader(test.text)
		} else {
			r = newReader(test.text)
		}
		n, err := fscanf[len(test.in)](r, test.format, test.in)
		if err != nil {
			if test.err == "" {
				t.Errorf("got error scanning (%q, %q): %q", test.format, test.text, err)
			} else if strings.Index(err.String(), test.err) < 0 {
				t.Errorf("got wrong error scanning (%q, %q): %q; expected %q", test.format, test.text, err, test.err)
			}
			continue
		}
		if test.err != "" {
			t.Errorf("expected error %q error scanning (%q, %q)", test.err, test.format, test.text)
		}
		if n != len(test.out) {
			t.Errorf("count error on entry (%q, %q): expected %d got %d", test.format, test.text, len(test.out), n)
			continue
		}
		// Convert the slice of pointers into a slice of values
		resultVal := reflect.MakeSlice(sliceType, n, n)
		for i := 0; i < n; i++ {
			v := reflect.NewValue(test.in[i]).(*reflect.PtrValue).Elem()
			resultVal.Elem(i).(*reflect.InterfaceValue).Set(v)
		}
		result := resultVal.Interface()
		if !reflect.DeepEqual(result, test.out) {
			t.Errorf("scanning (%q, %q): expected %v got %v", test.format, test.text, test.out, result)
		}
	}
}

func TestScanfMulti(t *testing.T) {
	testScanfMulti("StringReader", t)
}

func TestMyReaderScanfMulti(t *testing.T) {
	testScanfMulti("myStringReader", t)
}

func TestScanMultiple(t *testing.T) {
	var a int
	var s string
	n, err := Sscan("123abc", &a, &s)
	if n != 2 {
		t.Errorf("Sscan count error: expected 2: got %d", n)
	}
	if err != nil {
		t.Errorf("Sscan expected no error; got %s", err)
	}
	if a != 123 || s != "abc" {
		t.Errorf("Sscan wrong values: got (%d %q) expected (123 \"abc\")", a, s)
	}
}

func TestScanNotPointer(t *testing.T) {
	r := strings.NewReader("1")
	var a int
	_, err := Fscan(r, a)
	if err == nil {
		t.Error("expected error scanning non-pointer")
	} else if strings.Index(err.String(), "pointer") < 0 {
		t.Errorf("expected pointer error scanning non-pointer, got: %s", err)
	}
}

func TestScanlnNoNewline(t *testing.T) {
	var a int
	_, err := Sscanln("1 x\n", &a)
	if err == nil {
		t.Error("expected error scanning string missing newline")
	} else if strings.Index(err.String(), "newline") < 0 {
		t.Errorf("expected newline error scanning string missing newline, got: %s", err)
	}
}

func TestScanlnWithMiddleNewline(t *testing.T) {
	r := strings.NewReader("123\n456\n")
	var a, b int
	_, err := Fscanln(r, &a, &b)
	if err == nil {
		t.Error("expected error scanning string with extra newline")
	} else if strings.Index(err.String(), "newline") < 0 {
		t.Errorf("expected newline error scanning string with extra newline, got: %s", err)
	}
}

// Special Reader that counts reads at end of file.
type eofCounter struct {
	reader   *strings.Reader
	eofCount int
}

func (ec *eofCounter) Read(b []byte) (n int, err os.Error) {
	n, err = ec.reader.Read(b)
	if n == 0 {
		ec.eofCount++
	}
	return
}

// Verify that when we scan, we see at most EOF once per call to a Scan function,
// and then only when it's really an EOF
func TestEOF(t *testing.T) {
	ec := &eofCounter{strings.NewReader("123\n"), 0}
	var a int
	n, err := Fscanln(ec, &a)
	if err != nil {
		t.Error("unexpected error", err)
	}
	if n != 1 {
		t.Error("expected to scan one item, got", n)
	}
	if ec.eofCount != 0 {
		t.Error("expected zero EOFs", ec.eofCount)
		ec.eofCount = 0 // reset for next test
	}
	n, err = Fscanln(ec, &a)
	if err == nil {
		t.Error("expected error scanning empty string")
	}
	if n != 0 {
		t.Error("expected to scan zero items, got", n)
	}
	if ec.eofCount != 1 {
		t.Error("expected one EOF, got", ec.eofCount)
	}
}
