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
	renamedFloat      float
	renamedFloat32    float32
	renamedFloat64    float64
	renamedComplex    complex
	renamedComplex64  complex64
	renamedComplex128 complex128
)

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
	renamedFloatVal      renamedFloat
	renamedFloat32Val    renamedFloat32
	renamedFloat64Val    renamedFloat64
	renamedComplexVal    renamedComplex
	renamedComplex64Val  renamedComplex64
	renamedComplex128Val renamedComplex128
)

// Xs accepts any non-empty run of x's.
var xPat = testing.MustCompile("x+")

type Xs string

func (x *Xs) Scan(state ScanState) os.Error {
	tok, err := state.Token()
	if err != nil {
		return err
	}
	if !xPat.MatchString(tok) {
		return os.ErrorString("syntax error for xs")
	}
	*x = Xs(tok)
	return nil
}

var xVal Xs

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

	// Custom scanner.
	ScanTest{"  xxx ", &xVal, Xs("xxx")},
}

var scanfTests = []ScanfTest{
	ScanfTest{"%v", "TRUE\n", &boolVal, true},
	ScanfTest{"%t", "false\n", &boolVal, false},
	ScanfTest{"%v", "-71\n", &intVal, -71},
	ScanfTest{"%d", "72\n", &intVal, 72},
	ScanfTest{"%d", "73\n", &int8Val, int8(73)},
	ScanfTest{"%d", "-74\n", &int16Val, int16(-74)},
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

	// Renamed types
	ScanfTest{"%v\n", "true\n", &renamedBoolVal, renamedBool(true)},
	ScanfTest{"%t\n", "F\n", &renamedBoolVal, renamedBool(false)},
	ScanfTest{"%v", "101\n", &renamedIntVal, renamedInt(101)},
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
	ScanfTest{"%g", "115.1\n", &renamedFloatVal, renamedFloat(115.1)},
	ScanfTest{"%g", "116e1\n", &renamedFloat32Val, renamedFloat32(116e1)},
	ScanfTest{"%g", "-11.7e+1", &renamedFloat64Val, renamedFloat64(-11.7e+1)},
	ScanfTest{"%g", "11+5.1i\n", &renamedComplexVal, renamedComplex(11 + 5.1i)},
	ScanfTest{"%g", "11+6e1i\n", &renamedComplex64Val, renamedComplex64(11 + 6e1i)},
	ScanfTest{"%g", "-11.+7e+1i", &renamedComplex128Val, renamedComplex128(-11. + 7e+1i)},

	ScanfTest{"%x", "FFFFFFFF\n", &uint32Val, uint32(0xFFFFFFFF)},
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

func testScan(t *testing.T, scan func(r io.Reader, a ...interface{}) (int, os.Error)) {
	for _, test := range scanTests {
		r := strings.NewReader(test.text)
		n, err := scan(r, test.in)
		if err != nil {
			t.Errorf("got error scanning %q: %s", test.text, err)
			continue
		}
		if n != 1 {
			t.Errorf("count error on entry %q: got %d", test.text, n)
			continue
		}
		// The incoming value may be a pointer
		v := reflect.NewValue(test.in)
		if p, ok := v.(*reflect.PtrValue); ok {
			v = p.Elem()
		}
		val := v.Interface()
		if !reflect.DeepEqual(val, test.out) {
			t.Errorf("scanning %q: expected %v got %v, type %T", test.text, test.out, val, val)
		}
	}
}

func TestScan(t *testing.T) {
	testScan(t, Fscan)
}

func TestScanln(t *testing.T) {
	testScan(t, Fscanln)
}

func TestScanf(t *testing.T) {
	for _, test := range scanfTests {
		r := strings.NewReader(test.text)
		n, err := XXXFscanf(r, test.format, test.in)
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
		r := strings.NewReader(test.text)
		_, err := Fscan(r, test.in)
		if err == nil {
			t.Errorf("expected overflow scanning %q", test.text)
			continue
		}
		if !re.MatchString(err.String()) {
			t.Errorf("expected overflow error scanning %q: %s", test.text, err)
		}
	}
}

func TestScanMultiple(t *testing.T) {
	text := "1 2 3 x"
	r := strings.NewReader(text)
	var a, b, c, d int
	n, err := Fscan(r, &a, &b, &c, &d)
	if n != 3 {
		t.Errorf("count error: expected 3: got %d", n)
	}
	if err == nil {
		t.Errorf("expected error scanning ", text)
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
	r := strings.NewReader("1 x\n")
	var a int
	_, err := Fscanln(r, &a)
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
