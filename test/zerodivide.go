// run

// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that zero division causes a panic.

package main

import (
	"fmt"
	"math"
	"runtime"
	"strings"
)

type ErrorTest struct {
	name string
	fn   func()
	err  string
}

var (
	i, j, k       int   = 0, 0, 1
	i8, j8, k8    int8  = 0, 0, 1
	i16, j16, k16 int16 = 0, 0, 1
	i32, j32, k32 int32 = 0, 0, 1
	i64, j64, k64 int64 = 0, 0, 1

	u, v, w       uint    = 0, 0, 1
	u8, v8, w8    uint8   = 0, 0, 1
	u16, v16, w16 uint16  = 0, 0, 1
	u32, v32, w32 uint32  = 0, 0, 1
	u64, v64, w64 uint64  = 0, 0, 1
	up, vp, wp    uintptr = 0, 0, 1

	f, g, h                         float64 = 0, 0, 1
	f32, g32, h32                   float32 = 0, 0, 1
	f64, g64, h64, inf, negInf, nan float64 = 0, 0, 1, math.Inf(1), math.Inf(-1), math.NaN()

	c, d, e          complex128 = 0 + 0i, 0 + 0i, 1 + 1i
	c64, d64, e64    complex64  = 0 + 0i, 0 + 0i, 1 + 1i
	c128, d128, e128 complex128 = 0 + 0i, 0 + 0i, 1 + 1i
)

// Fool gccgo into thinking that these variables can change.
func NotCalled() {
	i++
	j++
	k++
	i8++
	j8++
	k8++
	i16++
	j16++
	k16++
	i32++
	j32++
	k32++
	i64++
	j64++
	k64++

	u++
	v++
	w++
	u8++
	v8++
	w8++
	u16++
	v16++
	w16++
	u32++
	v32++
	w32++
	u64++
	v64++
	w64++
	up++
	vp++
	wp++

	f += 1
	g += 1
	h += 1
	f32 += 1
	g32 += 1
	h32 += 1
	f64 += 1
	g64 += 1
	h64 += 1

	c += 1 + 1i
	d += 1 + 1i
	e += 1 + 1i
	c64 += 1 + 1i
	d64 += 1 + 1i
	e64 += 1 + 1i
	c128 += 1 + 1i
	d128 += 1 + 1i
	e128 += 1 + 1i
}

var tmp interface{}

// We could assign to _ but the compiler optimizes it too easily.
func use(v interface{}) {
	tmp = v
}

// Verify error/no error for all types.
var errorTests = []ErrorTest{
	// All integer divide by zero should error.
	ErrorTest{"int 0/0", func() { use(i / j) }, "divide"},
	ErrorTest{"int8 0/0", func() { use(i8 / j8) }, "divide"},
	ErrorTest{"int16 0/0", func() { use(i16 / j16) }, "divide"},
	ErrorTest{"int32 0/0", func() { use(i32 / j32) }, "divide"},
	ErrorTest{"int64 0/0", func() { use(i64 / j64) }, "divide"},

	ErrorTest{"int 1/0", func() { use(k / j) }, "divide"},
	ErrorTest{"int8 1/0", func() { use(k8 / j8) }, "divide"},
	ErrorTest{"int16 1/0", func() { use(k16 / j16) }, "divide"},
	ErrorTest{"int32 1/0", func() { use(k32 / j32) }, "divide"},
	ErrorTest{"int64 1/0", func() { use(k64 / j64) }, "divide"},

	ErrorTest{"uint 0/0", func() { use(u / v) }, "divide"},
	ErrorTest{"uint8 0/0", func() { use(u8 / v8) }, "divide"},
	ErrorTest{"uint16 0/0", func() { use(u16 / v16) }, "divide"},
	ErrorTest{"uint32 0/0", func() { use(u32 / v32) }, "divide"},
	ErrorTest{"uint64 0/0", func() { use(u64 / v64) }, "divide"},
	ErrorTest{"uintptr 0/0", func() { use(up / vp) }, "divide"},

	ErrorTest{"uint 1/0", func() { use(w / v) }, "divide"},
	ErrorTest{"uint8 1/0", func() { use(w8 / v8) }, "divide"},
	ErrorTest{"uint16 1/0", func() { use(w16 / v16) }, "divide"},
	ErrorTest{"uint32 1/0", func() { use(w32 / v32) }, "divide"},
	ErrorTest{"uint64 1/0", func() { use(w64 / v64) }, "divide"},
	ErrorTest{"uintptr 1/0", func() { use(wp / vp) }, "divide"},

	// All float64ing divide by zero should not error.
	ErrorTest{"float64 0/0", func() { use(f / g) }, ""},
	ErrorTest{"float32 0/0", func() { use(f32 / g32) }, ""},
	ErrorTest{"float64 0/0", func() { use(f64 / g64) }, ""},

	ErrorTest{"float64 1/0", func() { use(h / g) }, ""},
	ErrorTest{"float32 1/0", func() { use(h32 / g32) }, ""},
	ErrorTest{"float64 1/0", func() { use(h64 / g64) }, ""},
	ErrorTest{"float64 inf/0", func() { use(inf / g64) }, ""},
	ErrorTest{"float64 -inf/0", func() { use(negInf / g64) }, ""},
	ErrorTest{"float64 nan/0", func() { use(nan / g64) }, ""},

	// All complex divide by zero should not error.
	ErrorTest{"complex 0/0", func() { use(c / d) }, ""},
	ErrorTest{"complex64 0/0", func() { use(c64 / d64) }, ""},
	ErrorTest{"complex128 0/0", func() { use(c128 / d128) }, ""},

	ErrorTest{"complex 1/0", func() { use(e / d) }, ""},
	ErrorTest{"complex64 1/0", func() { use(e64 / d64) }, ""},
	ErrorTest{"complex128 1/0", func() { use(e128 / d128) }, ""},
}

func error_(fn func()) (error string) {
	defer func() {
		if e := recover(); e != nil {
			error = e.(runtime.Error).Error()
		}
	}()
	fn()
	return ""
}

type FloatTest struct {
	f, g float64
	out  float64
}

var float64Tests = []FloatTest{
	FloatTest{0, 0, nan},
	FloatTest{nan, 0, nan},
	FloatTest{inf, 0, inf},
	FloatTest{negInf, 0, negInf},
}

func alike(a, b float64) bool {
	switch {
	case math.IsNaN(a) && math.IsNaN(b):
		return true
	case a == b:
		return math.Signbit(a) == math.Signbit(b)
	}
	return false
}

func main() {
	bad := false
	for _, t := range errorTests {
		if t.err != "" {
			continue
		}
		err := error_(t.fn)
		switch {
		case t.err == "" && err == "":
			// fine
		case t.err != "" && err == "":
			if !bad {
				bad = true
				fmt.Printf("BUG\n")
			}
			fmt.Printf("%s: expected %q; got no error\n", t.name, t.err)
		case t.err == "" && err != "":
			if !bad {
				bad = true
				fmt.Printf("BUG\n")
			}
			fmt.Printf("%s: expected no error; got %q\n", t.name, err)
		case t.err != "" && err != "":
			if strings.Index(err, t.err) < 0 {
				if !bad {
					bad = true
					fmt.Printf("BUG\n")
				}
				fmt.Printf("%s: expected %q; got %q\n", t.name, t.err, err)
				continue
			}
		}
	}

	// At this point we know we don't error on the values we're testing
	for _, t := range float64Tests {
		x := t.f / t.g
		if !alike(x, t.out) {
			if !bad {
				bad = true
				fmt.Printf("BUG\n")
			}
			fmt.Printf("%v/%v: expected %g error; got %g\n", t.f, t.g, t.out, x)
		}
	}
}
