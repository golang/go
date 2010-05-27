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

var boolVal bool
var intVal int
var int8Val int8
var int16Val int16
var int32Val int32
var int64Val int64
var uintVal uint
var uint8Val uint8
var uint16Val uint16
var uint32Val uint32
var uint64Val uint64
var floatVal float
var float32Val float32
var float64Val float64
var stringVal string
var complexVal complex
var complex64Val complex64
var complex128Val complex128

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
	ScanTest{"T\n", &boolVal, true},
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

	// Custom scanner.
	ScanTest{"  xxx ", &xVal, Xs("xxx")},
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
	testScan(t, Scan)
}

func TestScanln(t *testing.T) {
	testScan(t, Scanln)
}

func TestScanOverflow(t *testing.T) {
	// different machines and different types report errors with different strings.
	re := testing.MustCompile("overflow|too large|out of range|not representable")
	for _, test := range overflowTests {
		r := strings.NewReader(test.text)
		_, err := Scan(r, test.in)
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
	n, err := Scan(r, &a, &b, &c, &d)
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
	_, err := Scan(r, a)
	if err == nil {
		t.Error("expected error scanning non-pointer")
	} else if strings.Index(err.String(), "pointer") < 0 {
		t.Errorf("expected pointer error scanning non-pointer, got: %s", err)
	}
}

func TestScanlnNoNewline(t *testing.T) {
	r := strings.NewReader("1 x\n")
	var a int
	_, err := Scanln(r, &a)
	if err == nil {
		t.Error("expected error scanning string missing newline")
	} else if strings.Index(err.String(), "newline") < 0 {
		t.Errorf("expected newline error scanning string missing newline, got: %s", err)
	}
}

func TestScanlnWithMiddleNewline(t *testing.T) {
	r := strings.NewReader("123\n456\n")
	var a, b int
	_, err := Scanln(r, &a, &b)
	if err == nil {
		t.Error("expected error scanning string with extra newline")
	} else if strings.Index(err.String(), "newline") < 0 {
		t.Errorf("expected newline error scanning string with extra newline, got: %s", err)
	}
}
