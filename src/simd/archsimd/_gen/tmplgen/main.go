// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// this generates type-instantiated boilerplate code for
// slice operations and tests

import (
	"_gen/sgutil"
	"bufio"
	"bytes"
	"flag"
	"fmt"
	"go/format"
	"io"
	"os"
	"strings"
	"text/template"
)

type resultTypeFunc func(t string, w, c int) (ot string, ow int, oc int)

// shapes describes a combination of vector widths and various element types
type shapes struct {
	vecs   []int // Vector bit width for this shape.
	ints   []int // Int element bit width(s) for this shape
	uints  []int // Unsigned int element bit width(s) for this shape
	floats []int // Float element bit width(s) for this shape
	output resultTypeFunc
}

// shapeAndTemplate is a template and the set of shapes on which it will be expanded
type shapeAndTemplate struct {
	s *shapes
	t *template.Template
}

func Map[T, U any](f func(T) U, in []T) []U {
	x := make([]U, len(in))
	for i, v := range in {
		x[i] = f(v)
	}
	return x
}

// smallSAT filters a shapeAndTemplate to keep only those
// for vector lengths <= 128 (effectively, for those == 128).
// 128 is the available-everywhere SIMD size, so many SIMD
// things across architectures can be lumped together as "128"
func smallSAT(sat shapeAndTemplate) shapeAndTemplate {
	r := sat
	s := *r.s
	s.vecs = []int{}
	for _, v := range r.s.vecs {
		if v <= 128 {
			s.vecs = append(s.vecs, v)
		}
	}
	r.s = &s
	return r
}

// largeSAT filters a shapeAndTemplate to keep only those
// for vector lengths > 128.  These tend to be less share-able
// than length 128
func largeSAT(sat shapeAndTemplate) shapeAndTemplate {
	r := sat
	s := *r.s
	s.vecs = []int{}
	for _, v := range r.s.vecs {
		if v > 128 {
			s.vecs = append(s.vecs, v)
		}
	}
	r.s = &s
	return r
}

func (sat shapeAndTemplate) target(outType string, width int) shapeAndTemplate {
	newSat := sat
	newShape := *sat.s
	newShape.output = func(t string, w, c int) (ot string, ow int, oc int) {
		oc = c
		if width*c > 512 {
			oc = 512 / width
		} else if width*c < 128 {
			oc = 128 / width
		}
		return outType, width, oc
	}
	newSat.s = &newShape
	return newSat
}

// arm64Target is like target but caps output at 128 bits (ARM64 NEON has no wider vectors).
func (sat shapeAndTemplate) arm64Target(outType string, width int) shapeAndTemplate {
	newSat := sat
	newShape := *sat.s
	newShape.output = func(t string, w, c int) (ot string, ow int, oc int) {
		oc = c
		if width*c > 128 {
			oc = 128 / width
		} else if width*c < 128 {
			oc = 128 / width
		}
		return outType, width, oc
	}
	newSat.s = &newShape
	return newSat
}

func (sat shapeAndTemplate) targetFixed(outType string, width, count int) shapeAndTemplate {
	newSat := sat
	newShape := *sat.s
	newShape.output = func(t string, w, c int) (ot string, ow int, oc int) {
		return outType, width, count
	}
	newSat.s = &newShape
	return newSat
}

func (s *shapes) forAllShapes(f func(seq int, t, upperT string, w, c int, out io.Writer), out io.Writer) {
	vecs := s.vecs
	ints := s.ints
	uints := s.uints
	floats := s.floats
	seq := 0
	for _, v := range vecs {
		for _, w := range ints {
			c := v / w
			f(seq, "int", "Int", w, c, out)
			seq++
		}
		for _, w := range uints {
			c := v / w
			f(seq, "uint", "Uint", w, c, out)
			seq++
		}
		for _, w := range floats {
			c := v / w
			f(seq, "float", "Float", w, c, out)
			seq++
		}
	}
}

var allShapes = &shapes{
	vecs:   []int{128, 256, 512},
	ints:   []int{8, 16, 32, 64},
	uints:  []int{8, 16, 32, 64},
	floats: []int{32, 64},
}

var intShapes = &shapes{
	vecs: []int{128, 256, 512},
	ints: []int{8, 16, 32, 64},
}

var uintShapes = &shapes{
	vecs:  []int{128, 256, 512},
	uints: []int{8, 16, 32, 64},
}

var floatShapes = &shapes{
	vecs:   []int{128, 256, 512},
	floats: []int{32, 64},
}

var integerShapes = &shapes{
	vecs:  []int{128, 256, 512},
	ints:  []int{8, 16, 32, 64},
	uints: []int{8, 16, 32, 64},
}

var avx512Shapes = &shapes{
	vecs:   []int{512},
	ints:   []int{8, 16, 32, 64},
	uints:  []int{8, 16, 32, 64},
	floats: []int{32, 64},
}

var avx2Shapes = &shapes{
	vecs:   []int{128, 256},
	ints:   []int{8, 16, 32, 64},
	uints:  []int{8, 16, 32, 64},
	floats: []int{32, 64},
}

var avx2MaskedLoadShapes = &shapes{
	vecs:   []int{128, 256},
	ints:   []int{32, 64},
	uints:  []int{32, 64},
	floats: []int{32, 64},
}

// arm64Shapes defines the SIMD shapes for ARM64 NEON (128-bit only)
var arm64Shapes = &shapes{
	vecs:   []int{128},
	ints:   []int{8, 16, 32, 64},
	uints:  []int{8, 16, 32, 64},
	floats: []int{32, 64},
}

// arm64IntegerShapes defines ARM64 NEON integer shapes (128-bit only, no float)
var arm64IntegerShapes = &shapes{
	vecs:  []int{128},
	ints:  []int{8, 16, 32, 64},
	uints: []int{8, 16, 32, 64},
}

// arm64UintToIntShapes maps unsigned shapes to signed output type for mixed-type shift helpers.
// The output function maps uint→int so templates can use OVType/OEtype for the second operand type.
var arm64UintToIntShapes = &shapes{
	vecs:  []int{128},
	uints: []int{8, 16, 32, 64},
	output: func(t string, w, c int) (string, int, int) {
		return "int", w, c // uint → int, same width and count
	},
}

var avx2SmallLoadPunShapes = &shapes{
	// ints are done by hand, these are type-punned to int.
	// 128-bit puns are now cross-platform and in a hand-written file.
	vecs:  []int{256},
	uints: []int{8, 16},
}

var unaryFlaky = &shapes{ // for tests that support flaky equality
	vecs:   []int{128, 256, 512},
	floats: []int{32, 64},
}

var ternaryFlaky = &shapes{ // for tests that support flaky equality
	vecs:   []int{128, 256, 512},
	floats: []int{32},
}

var avx2SignedComparisons = &shapes{
	vecs: []int{128, 256},
	ints: []int{8, 16, 32, 64},
}

var avx2UnsignedComparisons = &shapes{
	vecs:  []int{128, 256},
	uints: []int{8, 16, 32, 64},
}

// The shift-all shapes are for rotate emulation
var amdIntShiftAllShapes = &shapes{
	vecs: []int{128, 256, 512},
	ints: []int{16, 32, 64}, // has 32 and 64 rotate on AVX512 that is too hard to use, and no 8-bit shiftall
}

var amdUintShiftAllShapes = &shapes{
	vecs:  []int{128, 256, 512},
	uints: []int{16, 32, 64}, // has 32 and 64 rotate on AVX512 that is too hard to use, and no 8-bit shiftall
}

var neonIntShiftAllShapes = &shapes{
	vecs: []int{128},
	ints: []int{8, 16, 32, 64},
}

var neonUintShiftAllShapes = &shapes{
	vecs:  []int{128},
	uints: []int{8, 16, 32, 64},
}

type templateData struct {
	VType  string // the type of the vector, e.g. Float32x4
	AOrAn  string // for documentation, the article "a" or "an"
	EWidth int    // the bit width of the element type, e.g. 32
	Vwidth int    // the width of the vector type, e.g. 128
	Count  int    // the number of elements, e.g. 4
	WxC    string // the width-by-type string, e.g., "32x4"
	BxC    string // as if bytes, in the proper count, e.g., "8x16" (W==8)
	Base   string // the title-case Base Type of the vector, e.g., "Float"
	Etype  string // the element type, e.g. "float32"
	OxFF   string // a mask for the lowest 'count' bits

	OVType string // type of output vector
	OEtype string // output element type
	OEType string // output element type, title-case
	OCount int    // output element count
}

func (t templateData) As128BitVec() string {
	return fmt.Sprintf("%s%dx%d", t.Base, t.EWidth, 128/t.EWidth)
}

func oneTemplate(t *template.Template, baseType string, width, count int, out io.Writer, rtf resultTypeFunc) {
	b := width * count
	if b < 128 || b > 512 {
		return
	}

	ot, ow, oc := baseType, width, count
	if rtf != nil {
		ot, ow, oc = rtf(ot, ow, oc)
		if ow*oc > 512 || ow*oc < 128 || ow < 8 || ow > 64 {
			return
		}
		// TODO someday we will support conversions to 16-bit floats
		if ot == "float" && ow < 32 {
			return
		}
	}
	ovType := fmt.Sprintf("%s%dx%d", strings.ToUpper(ot[:1])+ot[1:], ow, oc)
	oeType := fmt.Sprintf("%s%d", ot, ow)
	oEType := fmt.Sprintf("%s%d", strings.ToUpper(ot[:1])+ot[1:], ow)

	wxc := fmt.Sprintf("%dx%d", width, count)
	BaseType := strings.ToUpper(baseType[:1]) + baseType[1:]
	vType := fmt.Sprintf("%s%s", BaseType, wxc)
	eType := fmt.Sprintf("%s%d", baseType, width)

	bxc := fmt.Sprintf("%dx%d", 8, count*(width/8))
	aOrAn := "a"
	if strings.Contains("aeiou", baseType[:1]) {
		aOrAn = "an"
	}
	oxFF := fmt.Sprintf("0x%x", uint64((1<<count)-1))
	t.Execute(out, templateData{
		VType:  vType,
		AOrAn:  aOrAn,
		EWidth: width,
		Vwidth: b,
		Count:  count,
		WxC:    wxc,
		BxC:    bxc,
		Base:   BaseType,
		Etype:  eType,
		OxFF:   oxFF,
		OVType: ovType,
		OEtype: oeType,
		OCount: oc,
		OEType: oEType,
	})
}

// forTemplates expands the template sat.t for each shape
// in sat.s, writing to out.
func (sat shapeAndTemplate) forTemplates(out io.Writer) {
	t, s := sat.t, sat.s
	vecs := s.vecs
	ints := s.ints
	uints := s.uints
	floats := s.floats
	for _, v := range vecs {
		for _, w := range ints {
			c := v / w
			oneTemplate(t, "int", w, c, out, sat.s.output)
		}
		for _, w := range uints {
			c := v / w
			oneTemplate(t, "uint", w, c, out, sat.s.output)
		}
		for _, w := range floats {
			c := v / w
			oneTemplate(t, "float", w, c, out, sat.s.output)
		}
	}
}

func prologue(s, ba string, out io.Writer) {
	fmt.Fprintf(out,
		`// Code generated by '%s'; DO NOT EDIT.

//go:build goexperiment.simd

package archsimd

`, s)
}

func ssaPrologue(s string, out io.Writer) {
	fmt.Fprintf(out,
		`// Code generated by '%s'; DO NOT EDIT.

package ssa

`, s)
}

func unsafePrologue(s, ba string, out io.Writer) {
	fmt.Fprintf(out,
		`// Code generated by '%s'; DO NOT EDIT.

//go:build goexperiment.simd

package archsimd

import "unsafe"

`, s)
}

func testPrologue(t, s, ba string, out io.Writer) {
	fmt.Fprintf(out,
		`// Code generated by '%s'; DO NOT EDIT.

//go:build goexperiment.simd && %s

// This file contains functions testing %s.
// Each function in this file is specialized for a
// particular simd type <BaseType><Width>x<Count>.

package simd_test

import (
	"simd/archsimd"
	"testing"
)

`, s, ba, t)
}

func curryTestPrologue(t string) func(s, ba string, out io.Writer) {
	return func(s, ba string, out io.Writer) {
		testPrologue(t, s, ba, out)
	}
}

func templateOf(name, temp string) shapeAndTemplate {
	return shapeAndTemplate{s: allShapes,
		t: template.Must(template.New(name).Parse(temp))}
}

func shapedTemplateOf(s *shapes, name, temp string) shapeAndTemplate {
	return shapeAndTemplate{s: s,
		t: template.Must(template.New(name).Parse(temp))}
}

const sliceTemplateText = `
// Load{{.VType}} loads {{.AOrAn}} {{.VType}} from a slice of elements.
// If s does not have at least {{.Count}} elements, it panics.
func Load{{.VType}}(s []{{.Etype}}) {{.VType}} {
	return Load{{.VType}}Array((*[{{.Count}}]{{.Etype}})(s))
}

// Store stores the elements of x into a slice.
// If s does not have at least {{.Count}} elements, it panics.
func (x {{.VType}}) Store(s []{{.Etype}}) {
	x.StoreArray((*[{{.Count}}]{{.Etype}})(s))
}
`

var sliceTemplate = templateOf("slice", sliceTemplateText)
var sliceTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_slice", sliceTemplateText)

const unaryTestTemplate = `
// test{{.VType}}Unary tests the simd unary method f against the expected behavior generated by want
func test{{.VType}}Unary(t *testing.T, f func(_ archsimd.{{.VType}}) archsimd.{{.VType}}, want func(_ []{{.Etype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		g := make([]{{.Etype}}, n)
		f(a).Store(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`

var unaryTemplate = templateOf("unary_helpers", unaryTestTemplate)
var unaryTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_unary_helpers", unaryTestTemplate)

var unaryFlakyTemplate = shapedTemplateOf(unaryFlaky, "unary_flaky_helpers", `
// test{{.VType}}UnaryFlaky tests the simd unary method f against the expected behavior generated by want,
// but using a flakiness parameter because we haven't exactly figured out how simd floating point works
func test{{.VType}}UnaryFlaky(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.{{.VType}}, want func(x []{{.Etype}}) []{{.Etype}}, flakiness float64) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		g := make([]{{.Etype}}, n)
		f(a).Store(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, flakiness, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var convertTemplate = templateOf("convert_helpers", `
// test{{.VType}}ConvertTo{{.OEType}} tests the simd conversion method f against the expected behavior generated by want.
// This is for count-preserving conversions, so if there is a change in size, then there is a change in vector width,
// (extended to at least 128 bits, or truncated to at most 512 bits).
func test{{.VType}}ConvertTo{{.OEType}}(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.{{.OVType}}, want func(x []{{.Etype}}) []{{.OEtype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		g := make([]{{.OEtype}}, {{.OCount}})
		f(a).Store(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var (
	// templates and shapes for conversion.
	// TODO: this includes shapes where in and out have the same element type,
	// which are not needed.
	unaryToInt8    = convertTemplate.target("int", 8)
	unaryToUint8   = convertTemplate.target("uint", 8)
	unaryToInt16   = convertTemplate.target("int", 16)
	unaryToUint16  = convertTemplate.target("uint", 16)
	unaryToInt32   = convertTemplate.target("int", 32)
	unaryToUint32  = convertTemplate.target("uint", 32)
	unaryToInt64   = convertTemplate.target("int", 64)
	unaryToUint64  = convertTemplate.target("uint", 64)
	unaryToFloat32 = convertTemplate.target("float", 32)
	unaryToFloat64 = convertTemplate.target("float", 64)
)

var convertLoTemplate = shapedTemplateOf(integerShapes, "convert_lo_helpers", `
// test{{.VType}}ConvertLoTo{{.OVType}} tests the simd conversion method f against the expected behavior generated by want.
// This converts only the low {{.OCount}} elements.
func test{{.VType}}ConvertLoTo{{.OVType}}(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.{{.OVType}}, want func(x []{{.Etype}}) []{{.OEtype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		g := make([]{{.OEtype}}, {{.OCount}})
		f(a).Store(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var (
	// templates and shapes for conversion of low elements.
	// The output is fixed to 128- or 256-bits (no 512-bit, as the
	// regular convertTemplate covers that).
	// TODO: this includes shapes where in and out have the same element
	// type or length, which are not needed.
	unaryToInt64x2  = convertLoTemplate.targetFixed("int", 64, 2)
	unaryToInt64x4  = convertLoTemplate.targetFixed("int", 64, 4)
	unaryToUint64x2 = convertLoTemplate.targetFixed("uint", 64, 2)
	unaryToUint64x4 = convertLoTemplate.targetFixed("uint", 64, 4)
	unaryToInt32x4  = convertLoTemplate.targetFixed("int", 32, 4)
	unaryToInt32x8  = convertLoTemplate.targetFixed("int", 32, 8)
	unaryToUint32x4 = convertLoTemplate.targetFixed("uint", 32, 4)
	unaryToUint32x8 = convertLoTemplate.targetFixed("uint", 32, 8)
	unaryToInt16x8  = convertLoTemplate.targetFixed("int", 16, 8)
	unaryToUint16x8 = convertLoTemplate.targetFixed("uint", 16, 8)
)

const binaryTestTemplate = `
// test{{.VType}}Binary tests the simd binary method f against the expected behavior generated by want
func test{{.VType}}Binary(t *testing.T, f func(_, _ archsimd.{{.VType}}) archsimd.{{.VType}}, want func(_, _ []{{.Etype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSlicePair(t, {{.Etype}}s, n, func(x, y []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.VType}}(y)
		g := make([]{{.Etype}}, n)
		f(a, b).Store(g)
		w := want(x, y)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); })
	})
}
`

var binaryTemplate = templateOf("binary_helpers", binaryTestTemplate)
var binaryTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_binary_helpers", binaryTestTemplate)

// ARM64 shift test helper templates

var shiftConstTestTemplateArm64 = shapedTemplateOf(arm64IntegerShapes, "arm64_shift_const_helpers", `
// test{{.VType}}ShiftConst tests a const-shift method (unary + immediate).
func test{{.VType}}ShiftConst(t *testing.T, f func(_ archsimd.{{.VType}}, _ uint64) archsimd.{{.VType}}, want func(_ []{{.Etype}}, _ uint64) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
		t.Helper()
		for _, amt := range []uint64{0, 1, 3, {{.EWidth}}-1} {
			a := archsimd.Load{{.VType}}(x)
			g := make([]{{.Etype}}, n)
			f(a, amt).Store(g)
			w := want(x, amt)
			if !checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, amt) }) {
				return false
			}
		}
		return true
	})
}
`)

var shiftAllTestTemplateArm64 = shapedTemplateOf(arm64IntegerShapes, "arm64_shift_all_helpers", `
// test{{.VType}}ShiftAll tests a shift-all method (unary + scalar uint64).
func test{{.VType}}ShiftAll(t *testing.T, f func(_ archsimd.{{.VType}}, _ uint64) archsimd.{{.VType}}, want func(_ []{{.Etype}}, _ uint64) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
		t.Helper()
		for _, amt := range testShiftAllAmts {
			a := archsimd.Load{{.VType}}(x)
			g := make([]{{.Etype}}, n)
			f(a, amt).Store(g)
			w := want(x, amt)
			if !checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v, amt=%d", x, amt) }) {
				return false
			}
		}
		return true
	})
}
`)

var shiftMixedTestTemplateArm64 = shapedTemplateOf(arm64UintToIntShapes, "arm64_shift_mixed_helpers", `
// test{{.VType}}Shift tests a shift-like method where the first operand is {{.VType}}
// and the second operand is {{.OVType}} (mixed-type shift).
func test{{.VType}}Shift(t *testing.T, f func(_ archsimd.{{.VType}}, _ archsimd.{{.OVType}}) archsimd.{{.VType}}, want func(_ []{{.Etype}}, _ []{{.OEtype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSliceMixed(t, {{.Etype}}s, {{.OEtype}}s, n, func(x []{{.Etype}}, y []{{.OEtype}}) bool {
		t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.OVType}}(y)
		g := make([]{{.Etype}}, n)
		f(a, b).Store(g)
		w := want(x, y)
		return checkSlicesLogInput(t, g, w, 0.0, func() { t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y) })
	})
}
`)

var convertTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_convert_helpers", `
// test{{.VType}}ConvertTo{{.OVType}} tests the simd conversion method f against the expected behavior generated by want.
func test{{.VType}}ConvertTo{{.OVType}}(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.{{.OVType}}, want func(x []{{.Etype}}) []{{.OEtype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		g := make([]{{.OEtype}}, {{.OCount}})
		f(a).Store(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var (
	arm64ToInt8    = convertTemplateArm64.arm64Target("int", 8)
	arm64ToUint8   = convertTemplateArm64.arm64Target("uint", 8)
	arm64ToInt16   = convertTemplateArm64.arm64Target("int", 16)
	arm64ToUint16  = convertTemplateArm64.arm64Target("uint", 16)
	arm64ToInt32   = convertTemplateArm64.arm64Target("int", 32)
	arm64ToUint32  = convertTemplateArm64.arm64Target("uint", 32)
	arm64ToInt64   = convertTemplateArm64.arm64Target("int", 64)
	arm64ToUint64  = convertTemplateArm64.arm64Target("uint", 64)
	arm64ToFloat32 = convertTemplateArm64.arm64Target("float", 32)
	arm64ToFloat64 = convertTemplateArm64.arm64Target("float", 64)
)

const ternaryTestTemplateText = `
// test{{.VType}}Ternary tests the simd ternary method f against the expected behavior generated by want
func test{{.VType}}Ternary(t *testing.T, f func(_, _, _ archsimd.{{.VType}}) archsimd.{{.VType}}, want func(_, _, _ []{{.Etype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSliceTriple(t, {{.Etype}}s, n, func(x, y, z []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.VType}}(y)
		c := archsimd.Load{{.VType}}(z)
		g := make([]{{.Etype}}, n)
		f(a, b, c).Store(g)
		w := want(x, y, z)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); t.Logf("z=%v", z); })
	})
}
`

const ternaryFlakyTestTemplateText = `
// test{{.VType}}TernaryFlaky tests the simd ternary method f against the expected behavior generated by want,
// but using a flakiness parameter because we haven't exactly figured out how simd floating point works
func test{{.VType}}TernaryFlaky(t *testing.T, f func(x, y, z archsimd.{{.VType}}) archsimd.{{.VType}}, want func(x, y, z []{{.Etype}}) []{{.Etype}}, flakiness float64) {
	n := {{.Count}}
	t.Helper()
	forSliceTriple(t, {{.Etype}}s, n, func(x, y, z []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.VType}}(y)
		c := archsimd.Load{{.VType}}(z)
		g := make([]{{.Etype}}, n)
		f(a, b, c).Store(g)
		w := want(x, y, z)
		return checkSlicesLogInput(t, g, w, flakiness, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); t.Logf("z=%v", z); })
	})
}
`

var ternaryTemplate = templateOf("ternary_helpers", ternaryTestTemplateText)
var ternaryFlakyTemplate = shapedTemplateOf(ternaryFlaky, "ternary_helpers", ternaryFlakyTestTemplateText)
var ternaryTemplateArm64 = shapedTemplateOf(arm64Shapes, "ternary_arm64_helpers", ternaryTestTemplateText)
var ternaryFlakyTemplateArm64 = shapedTemplateOf(arm64Shapes, "ternary_arm64_helpers", ternaryFlakyTestTemplateText)

var compareTemplate = templateOf("compare_helpers", `
// test{{.VType}}Compare tests the simd comparison method f against the expected behavior generated by want
func test{{.VType}}Compare(t *testing.T, f func(_, _ archsimd.{{.VType}}) archsimd.Mask{{.WxC}}, want func(_, _ []{{.Etype}}) []int64) {
	n := {{.Count}}
	t.Helper()
	forSlicePair(t, {{.Etype}}s, n, func(x, y []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.VType}}(y)
		g := make([]int{{.EWidth}}, n)
		f(a, b).ToInt{{.WxC}}().Store(g)
		w := want(x, y)
		return checkSlicesLogInput(t, s64(g), w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); })
	})
}
`)

var compareUnaryTemplate = shapedTemplateOf(floatShapes, "compare_unary_helpers", `
// test{{.VType}}UnaryCompare tests the simd unary comparison method f against the expected behavior generated by want
func test{{.VType}}UnaryCompare(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.Mask{{.WxC}}, want func(x []{{.Etype}}) []int64) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		g := make([]int{{.EWidth}}, n)
		f(a).ToInt{{.WxC}}().Store(g)
		w := want(x)
		return checkSlicesLogInput(t, s64(g), w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

// TODO this has not been tested yet.
var compareMaskedTemplate = templateOf("comparemasked_helpers", `
// test{{.VType}}CompareMasked tests the simd masked comparison method f against the expected behavior generated by want
// The mask is applied to the output of want; anything not in the mask, is zeroed.
func test{{.VType}}CompareMasked(t *testing.T,
	f func(_, _ archsimd.{{.VType}}, m archsimd.Mask{{.WxC}}) archsimd.Mask{{.WxC}},
	want func(_, _ []{{.Etype}}) []int64) {
	n := {{.Count}}
	t.Helper()
	forSlicePairMasked(t, {{.Etype}}s, n, func(x, y []{{.Etype}}, m []bool) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.VType}}(y)
		k := archsimd.LoadInt{{.WxC}}(toVect[int{{.EWidth}}](m)).ToMask()
		g := make([]int{{.EWidth}}, n)
		f(a, b, k).ToInt{{.WxC}}().Store(g)
		w := want(x, y)
		for i := range m {
			if !m[i] {
				w[i] = 0
			}
		}
		return checkSlicesLogInput(t, s64(g), w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); t.Logf("m=%v", m); })
	})
}
`)

var avx512MaskedLoadSliceTemplate = shapedTemplateOf(avx512Shapes, "avx 512 load slice part", `
// Load{{.VType}}Part loads a {{.VType}} from the slice s, it returns the loaded vector and the
// number of elements loaded.
// If s has fewer than {{.Count}} elements, the remaining elements of the vector are filled with zeroes.
// If s has {{.Count}} or more elements, the function is equivalent to Load{{.VType}}.
func Load{{.VType}}Part(s []{{.Etype}}) ({{.VType}}, int) {
	l := len(s)
	if l >= {{.Count}} {
		return Load{{.VType}}(s), {{.Count}}
	}
	if l == 0 {
		var x {{.VType}}
		return x, 0
	}
	mask := Mask{{.WxC}}FromBits({{.OxFF}} >> ({{.Count}} - l))
	return Load{{.VType}}Array(pa{{.VType}}(s)).Masked(mask), l
}

// StorePart stores the {{.Count}} elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has {{.Count}} or more elements, the method is equivalent to x.Store.
func (x {{.VType}}) StorePart(s []{{.Etype}}) {
	l := len(s)
	if l >= {{.Count}} {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	mask := Mask{{.WxC}}FromBits({{.OxFF}} >> ({{.Count}} - l))
	x.StoreArrayMasked(pa{{.VType}}(s), mask)
}
`)

var avx2MaskedLoadSliceTemplate = shapedTemplateOf(avx2MaskedLoadShapes, "avx 2 load slice part", `
// Load{{.VType}}Part loads a {{.VType}} from the slice s, it returns the loaded vector and the
// number of elements loaded.
// If s has fewer than {{.Count}} elements, the remaining elements of the vector are filled with zeroes.
// If s has {{.Count}} or more elements, the function is equivalent to Load{{.VType}}.
func Load{{.VType}}Part(s []{{.Etype}}) ({{.VType}}, int) {
	l := len(s)
	if l >= {{.Count}} {
		return Load{{.VType}}(s), {{.Count}}
	}
	if l == 0 {
		var x {{.VType}}
		return x, 0
	}
	mask := vecMask{{.EWidth}}[len(vecMask{{.EWidth}})/2-l:]
	return Load{{.VType}}Array(pa{{.VType}}(s)).Masked(LoadInt{{.WxC}}(mask).asMask()), l
}

// StorePart stores the {{.Count}} elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has {{.Count}} or more elements, the method is equivalent to x.Store.
func (x {{.VType}}) StorePart(s []{{.Etype}}) {
	l := len(s)
	if l >= {{.Count}} {
		x.Store(s)
		return
	}
	if l == 0 {
		return
	}
	mask := vecMask{{.EWidth}}[len(vecMask{{.EWidth}})/2-l:]
	x.StoreArrayMasked(pa{{.VType}}(s), LoadInt{{.WxC}}(mask).asMask())
}
`)

var avx2SmallLoadSliceTemplate = shapedTemplateOf(avx2SmallLoadPunShapes, "avx 2 small load slice part", `
// Load{{.VType}}Part loads a {{.VType}} from the slice s, it returns the loaded vector and the
// number of elements loaded.
// If s has fewer than {{.Count}} elements, the remaining elements of the vector are filled with zeroes.
// If s has {{.Count}} or more elements, the function is equivalent to Load{{.VType}}.
func Load{{.VType}}Part(s []{{.Etype}}) ({{.VType}}, int) {
	if len(s) == 0 {
		var zero {{.VType}}
		return zero, 0
	}
	t := unsafe.Slice((*int{{.EWidth}})(unsafe.Pointer(&s[0])), len(s))
	v, l := LoadInt{{.WxC}}Part(t)
	return v.As{{.VType}}(), l
}

// StorePart stores the {{.Count}} elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has {{.Count}} or more elements, the method is equivalent to x.Store.
func (x {{.VType}}) StorePart(s []{{.Etype}}) {
	if len(s) == 0 {
		return
	}
	t := unsafe.Slice((*int{{.EWidth}})(unsafe.Pointer(&s[0])), len(s))
	x.AsInt{{.WxC}}().StorePart(t)
}
`)

func (t templateData) CPUfeature() string {
	switch t.Vwidth {
	case 128:
		return "AVX"
	case 256:
		return "AVX2"
	case 512:
		return "AVX512"
	}
	panic(fmt.Errorf("unexpected vector width %d", t.Vwidth))
}

var avx2SignedComparisonsTemplate = shapedTemplateOf(avx2SignedComparisons, "avx2 signed comparisons", `
// Less returns a mask whose elements indicate whether x < y.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) Less(y {{.VType}}) Mask{{.WxC}} {
	return y.Greater(x)
}

// GreaterEqual returns a mask whose elements indicate whether x >= y.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) GreaterEqual(y {{.VType}}) Mask{{.WxC}} {
	ones := x.Equal(x).ToInt{{.WxC}}()
	return y.Greater(x).ToInt{{.WxC}}().Xor(ones).asMask()
}

// LessEqual returns a mask whose elements indicate whether x <= y.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) LessEqual(y {{.VType}}) Mask{{.WxC}} {
	ones := x.Equal(x).ToInt{{.WxC}}()
	return x.Greater(y).ToInt{{.WxC}}().Xor(ones).asMask()
}

// NotEqual returns a mask whose elements indicate whether x != y.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) NotEqual(y {{.VType}}) Mask{{.WxC}} {
	ones := x.Equal(x).ToInt{{.WxC}}()
	return x.Equal(y).ToInt{{.WxC}}().Xor(ones).asMask()	
}
`)

var intRotateAllTemplate = sgutil.TemplateNamed("intRotateAll", `
// RotateAllLeft rotates all elements left by the specified amount
//
// Emulated
func (x {{.VType}}) RotateAllLeft(dist uint64) {{.VType}} {
	dist = dist & ({{.EWidth}}-1)
	ndist := {{.EWidth}} - dist
	return x.ToBits().ShiftAllLeft(dist).Or(x.ToBits().ShiftAllRight(ndist)).BitsToInt{{.EWidth}}()
}

// RotateAllRight rotates all elements right by the specified amount
//
// Emulated
func (x {{.VType}}) RotateAllRight(dist uint64) {{.VType}} {
	dist = dist & ({{.EWidth}}-1)
	ndist := {{.EWidth}} - dist
	return x.ToBits().ShiftAllLeft(ndist).Or(x.ToBits().ShiftAllRight(dist)).BitsToInt{{.EWidth}}()
}
`)

var uintRotateAllTemplate = sgutil.TemplateNamed("intRotateAll", `
// RotateAllLeft rotates all elements left by the specified amount
//
// Emulated
func (x {{.VType}}) RotateAllLeft(dist uint64) {{.VType}} {
	dist = dist & ({{.EWidth}}-1)
	ndist := {{.EWidth}} - dist
	return x.ShiftAllLeft(dist).Or(x.ShiftAllRight(ndist))
}

// RotateAllRight rotates all elements right by the specified amount
//
// Emulated
func (x {{.VType}}) RotateAllRight(dist uint64) {{.VType}} {
	dist = dist & ({{.EWidth}}-1)
	ndist := {{.EWidth}} - dist
	return x.ShiftAllLeft(ndist).Or(x.ShiftAllRight(dist))
}
`)

var bitWiseIntTemplate = shapedTemplateOf(intShapes, "bitwise int complement", `
// Not returns the bitwise complement of x.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) Not() {{.VType}} {
	return x.Xor(x.Equal(x).ToInt{{.WxC}}())
}

// Neg returns the element-wise negation of x.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) Neg() {{.VType}} {
	var zero {{.VType}}
	return zero.Sub(x)
}

`)

var bitWiseUintTemplate = shapedTemplateOf(uintShapes, "bitwise uint complement", `
// Not returns the bitwise complement of x.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) Not() {{.VType}} {
	return x.Xor(x.Equal(x).ToInt{{.WxC}}().As{{.VType}}())
}
`)

// CPUfeatureAVX2if8 return AVX2 if the element width is 8,
// otherwise, it returns CPUfeature.  This is for the cpufeature
// of unsigned comparison emulation, which uses shifts for all
// the sizes > 8 (shifts are AVX) but must use broadcast (AVX2)
// for bytes.
func (t templateData) CPUfeatureAVX2if8() string {
	if t.EWidth == 8 {
		return "AVX2"
	}
	return t.CPUfeature()
}

var avx2UnsignedComparisonsTemplate = shapedTemplateOf(avx2UnsignedComparisons, "avx2 unsigned comparisons", `
// Greater returns a mask whose elements indicate whether x > y.
//
// Emulated, CPU Feature: {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) Greater(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	ones := x.Equal(x).ToInt{{.WxC}}()
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return a.Xor(signs).Greater(b.Xor(signs))
}

// Less returns a mask whose elements indicate whether x < y.
//
// Emulated, CPU Feature: {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) Less(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	ones := x.Equal(x).ToInt{{.WxC}}()
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return b.Xor(signs).Greater(a.Xor(signs))
}

// GreaterEqual returns a mask whose elements indicate whether x >= y.
//
// Emulated, CPU Feature: {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) GreaterEqual(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
	ones := x.Equal(x).ToInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return b.Xor(signs).Greater(a.Xor(signs)).ToInt{{.WxC}}().Xor(ones).asMask()
}

// LessEqual returns a mask whose elements indicate whether x <= y.
//
// Emulated, CPU Feature: {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) LessEqual(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
	ones := x.Equal(x).ToInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return a.Xor(signs).Greater(b.Xor(signs)).ToInt{{.WxC}}().Xor(ones).asMask()
}

// NotEqual returns a mask whose elements indicate whether x != y.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) NotEqual(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
	ones := x.Equal(x).ToInt{{.WxC}}()
	return a.Equal(b).ToInt{{.WxC}}().Xor(ones).asMask()
}
`)

var unsafePATemplate = templateOf("unsafe PA helper", `
// pa{{.VType}} returns a type-unsafe pointer to array that can
// only be used with partial load/store operations that only
// access the known-safe portions of the array.
//
//go:nocheckptr
func pa{{.VType}}(s []{{.Etype}}) *[{{.Count}}]{{.Etype}} {
	return (*[{{.Count}}]{{.Etype}})(unsafe.Pointer(&s[0]))
}
`)

var avx2MaskedTemplate = shapedTemplateOf(avx2Shapes, "avx2 .Masked methods", `
// Masked returns x but with elements zeroed where mask is false.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) Masked(mask Mask{{.WxC}}) {{.VType}} {
	im := mask.ToInt{{.WxC}}()
{{- if eq .Base "Int" }}
	return im.And(x)
{{- else}}
    return x.AsInt{{.WxC}}().And(im).As{{.VType}}()
{{- end -}}
}

// Merge returns x but with elements set to y where mask is false.
//
// Emulated, CPU Feature: {{.CPUfeature}}
//
// Deprecated: use x.IfElse(mask, y)
//
//go:fix inline
func (x {{.VType}}) Merge(y {{.VType}}, mask Mask{{.WxC}}) {{.VType}} {
   return x.IfElse(mask, y)
}

// IfElse returns x but with elements set to y where mask is false.
//
// Emulated, CPU Feature: {{.CPUfeature}}
func (x {{.VType}}) IfElse(mask Mask{{.WxC}}, y {{.VType}}) {{.VType}} {
{{- if eq .BxC .WxC -}}
	im := mask.ToInt{{.BxC}}()
{{- else}}
    im := mask.ToInt{{.WxC}}().AsInt{{.BxC}}()
{{- end -}}
{{- if and (eq .Base "Int") (eq .BxC .WxC) }}
	return y.blend(x, im)
{{- else}}
	ix := x.AsInt{{.BxC}}()
	iy := y.AsInt{{.BxC}}()
	return iy.blend(ix, im).As{{.VType}}()
{{- end -}}
}
`)

// TODO perhaps write these in ways that work better on AVX512
var avx512MaskedTemplate = shapedTemplateOf(avx512Shapes, "avx512 .Masked methods", `
// Masked returns x but with elements zeroed where mask is false.
//
// Emulated, CPU Feature: AVX512
func (x {{.VType}}) Masked(mask Mask{{.WxC}}) {{.VType}} {
	im := mask.ToInt{{.WxC}}()
{{- if eq .Base "Int" }}
	return im.And(x)
{{- else}}
    return x.AsInt{{.WxC}}().And(im).As{{.VType}}()
{{- end -}}
}

// Merge returns x but with elements set to y where mask is false.
//
// Emulated, CPU Feature: AVX512
//
// Deprecated: use x.IfElse(mask, y)
//
//go:fix inline
func (x {{.VType}}) Merge(y {{.VType}}, mask Mask{{.WxC}}) {{.VType}} {
   return x.IfElse(mask, y)
}

func (x {{.VType}}) IfElse(mask Mask{{.WxC}}, y {{.VType}}) {{.VType}} {
{{- if eq .Base "Int" }}
	return y.blendMasked(x, mask)
{{- else}}
	ix := x.AsInt{{.WxC}}()
	iy := y.AsInt{{.WxC}}()
	return iy.blendMasked(ix, mask).As{{.VType}}()
{{- end -}}
}
`)

func (t templateData) CPUfeatureBC() string {
	switch t.Vwidth {
	case 128:
		return "AVX2"
	case 256:
		return "AVX2"
	case 512:
		if t.EWidth <= 16 {
			return "AVX512BW"
		}
		return "AVX512F"
	}
	panic(fmt.Errorf("unexpected vector width %d", t.Vwidth))
}

var broadcastTemplate = templateOf("Broadcast functions", `
// Broadcast{{.VType}} returns a vector with the input
// x assigned to all elements of the output.
//
// Emulated, CPU Feature: {{.CPUfeatureBC}}
func Broadcast{{.VType}}(x {{.Etype}}) {{.VType}} {
	var z {{.As128BitVec }}
	return z.SetElem(0, x).broadcast1To{{.Count}}()
}
`)

var broadcastTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_broadcast", `
// Broadcast{{.VType}} returns a vector with the input
// x assigned to all elements of the output.
func Broadcast{{.VType}}(x {{.Etype}}) {{.VType}} {
	var z {{.VType}}
	return z.SetElem(0, x).broadcast1To{{.Count}}()
}
`)

var stringTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_String methods", `
// String returns a string representation of SIMD vector x.
func (x {{.VType}}) String() string {
	var s [{{.Count}}]{{.Etype}}
	x.StoreArray(&s)
	return sliceToString(s[:])
}
`)

var setHiTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_SetHi methods", `
// SetHi returns a vector with the lower 64 bits of x preserved and the upper
// 64 bits replaced with the lower 64 bits of the parameter lo.
func (x {{.VType}}) SetHi(lo {{.VType}}) {{.VType}} {
{{- if and (eq .Base "Float") (eq .EWidth 64)}}
	return x.SetElem(1, lo.GetElem(0))
{{- else}}
	return x.AsFloat64x2().SetElem(1, lo.AsFloat64x2().GetElem(0)).As{{.VType}}()
{{- end}}
}
`)

var getHiTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_GetHi methods", `
// GetHi returns a vector with the upper 64 bits zeroed and the lower
// 64 bits replaced with the upper 64 bits of x.
func (x {{.VType}}) GetHi() {{.VType}} {
	var z {{.VType}}
{{- if and (eq .Base "Float") (eq .EWidth 64)}}
	return z.SetElem(0, x.GetElem(1))
{{- else}}
	return z.AsFloat64x2().SetElem(0, x.AsFloat64x2().GetElem(1)).As{{.VType}}()
{{- end}}
}
`)

var maskCvtTemplate = shapedTemplateOf(intShapes, "Mask conversions", `
// ToMask returns a mask whose i'th element is set if x[i] is non-zero.
func (from {{.Base}}{{.WxC}}) ToMask() (to Mask{{.WxC}}) {
	return from.NotEqual({{.Base}}{{.WxC}}{})
}
`)

var arm64MaskCvtTemplate = shapedTemplateOf(arm64IntegerShapes, "Mask conversions", `
// ToMask returns a mask whose i'th element is set if x[i] is non-zero.
func (from {{.Base}}{{.WxC}}) ToMask() (to Mask{{.WxC}}) {
	return from.NotEqual({{.Base}}{{.WxC}}{})
}
`)

// ARM64 derived comparison templates.
// On ARM64 NEON, Equal, Greater, and GreaterEqual are hardware-backed.
// Less, LessEqual, and NotEqual are derived.

var arm64LessTemplate = shapedTemplateOf(arm64Shapes, "arm64_less", `
// Less returns a mask whose elements indicate whether x < y.
func (x {{.VType}}) Less(y {{.VType}}) Mask{{.WxC}} {
	return y.Greater(x)
}
`)

var arm64LessEqualTemplate = shapedTemplateOf(arm64Shapes, "arm64_less_equal", `
// LessEqual returns a mask whose elements indicate whether x <= y.
func (x {{.VType}}) LessEqual(y {{.VType}}) Mask{{.WxC}} {
	return y.GreaterEqual(x)
}
`)

var arm64NotEqualTemplate = shapedTemplateOf(arm64Shapes, "arm64_not_equal", `
// NotEqual returns a mask whose elements indicate whether x != y.
func (x {{.VType}}) NotEqual(y {{.VType}}) Mask{{.WxC}} {
	return x.Equal(y).Not()
}
`)

// ARM64 Masked/Merge templates using bitSelect (picks y's bits where mask=1, keeps x's bits where mask=0).

var arm64MaskedMergeTemplate = shapedTemplateOf(arm64Shapes, "arm64_masked_merge", `
// Masked returns x but with elements zeroed where mask is false.
func (x {{.VType}}) Masked(mask Mask{{.WxC}}) {{.VType}} {
	im := mask.ToInt{{.WxC}}()
{{- if eq .Base "Int" }}
	return im.And(x)
{{- else }}
	return im.And(x.AsInt{{.WxC}}()).As{{.VType}}()
{{- end }}
}

// Merge returns x but with elements set to y where mask is true.
//
// Deprecated: use x.IfElse(mask, y)
func (x {{.VType}}) Merge(y {{.VType}}, mask Mask{{.WxC}}) {{.VType}} {
	return x.IfElse(mask, y)
}

// IfElse returns x but with elements set to y where mask is true.
func (x {{.VType}}) IfElse(mask Mask{{.WxC}}, y {{.VType}}) {{.VType}} {
{{- if eq .WxC "8x16" }}
{{-   if eq .Base "Int" }}
	return x.bitSelect(y, mask.ToInt8x16())
{{-   else }}
	return x.AsInt8x16().bitSelect(y.AsInt8x16(), mask.ToInt8x16()).As{{.VType}}()
{{-   end }}
{{- else if eq .Base "Int" }}
	im := mask.ToInt{{.WxC}}().AsInt8x16()
	ix := x.AsInt8x16()
	iy := y.AsInt8x16()
	return ix.bitSelect(iy, im).As{{.VType}}()
{{- else }}
	im := mask.ToInt{{.WxC}}().AsInt8x16()
	ix := x.AsInt{{.WxC}}().AsInt8x16()
	iy := y.AsInt{{.WxC}}().AsInt8x16()
	return ix.bitSelect(iy, im).As{{.VType}}()
{{- end }}
}
`)

var compareTemplateArm64 = shapedTemplateOf(arm64Shapes, "arm64_compare_helpers", `
// test{{.VType}}Compare tests the simd comparison method f against the expected behavior generated by want
func test{{.VType}}Compare(t *testing.T, f func(_, _ archsimd.{{.VType}}) archsimd.Mask{{.WxC}}, want func(_, _ []{{.Etype}}) []int64) {
	n := {{.Count}}
	t.Helper()
	forSlicePair(t, {{.Etype}}s, n, func(x, y []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}(x)
		b := archsimd.Load{{.VType}}(y)
		g := make([]int{{.EWidth}}, n)
		f(a, b).ToInt{{.WxC}}().Store(g)
		w := want(x, y)
		return checkSlicesLogInput(t, s64(g), w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); })
	})
}
`)

var arm64IntShapes = &shapes{
	vecs: []int{128},
	ints: []int{8, 16, 32, 64},
}

var arm64MaskToString = shapedTemplateOf(arm64IntShapes, "arm64_maskToString", `
// String returns a string representation of SIMD mask x.
func (x Mask{{.WxC}}) String() string {
	var s [{{.Count}}]{{.Etype}}
	x.ToInt{{.WxC}}().Neg().StoreArray(&s)
	return sliceToString(s[:])
}
`)

var stringTemplate = shapedTemplateOf(allShapes, "String methods", `
// String returns a string representation of SIMD vector x.
func (x {{.VType}}) String() string {
	var s [{{.Count}}]{{.Etype}}
	x.StoreArray(&s)
	return sliceToString(s[:])
}
`)

var maskToString = shapedTemplateOf(intShapes, "maskToString", `
// String returns a string representation of SIMD mask x.
func (x Mask{{.WxC}}) String() string {
	var s [{{.Count}}]{{.Etype}}
	x.ToInt{{.WxC}}().Neg().StoreArray(&s)
	return sliceToString(s[:])
}
`)

const SIMD = "../../"
const TD = "../../internal/simd_test/"
const SSA = "../../../../cmd/compile/internal/ssa/"

func main() {
	sl := flag.String("sl", SIMD+"slice_gen_amd64.go", "file name for slice operations")
	cm := flag.String("cm", SIMD+"compare_gen_amd64.go", "file name for comparison operations")
	mm := flag.String("mm", SIMD+"maskmerge_gen_amd64.go", "file name for mask/merge operations")
	op := flag.String("op", SIMD+"other_gen_amd64.go", "file name for other operations")
	ush := flag.String("ush", SIMD+"unsafe_helpers.go", "file name for unsafe helpers")
	bh := flag.String("bh", TD+"binary_helpers_%W_test.go", "file name for binary test helpers")
	uh := flag.String("uh", TD+"unary_helpers_%W_test.go", "file name for unary test helpers")
	cvh := flag.String("cvh", TD+"convert_helpers_test.go", "file name for conversion test helpers")
	th := flag.String("th", TD+"ternary_helpers_test.go", "file name for ternary test helpers")
	ch := flag.String("ch", TD+"compare_helpers_%W_test.go", "file name for compare test helpers")
	cmh := flag.String("cmh", TD+"comparemasked_helpers_test.go", "file name for compare-masked test helpers")
	// ARM64-specific
	bhArm64 := flag.String("bhArm64", TD+"arm64_binary_helpers_test.go", "file name for ARM64 binary test helpers")
	slArm64 := flag.String("slArm64", SIMD+"slice_gen_arm64.go", "file name for ARM64 slice operations")
	opArm64 := flag.String("opArm64", SIMD+"other_gen_arm64.go", "file name for ARM64 other operations")
	shArm64 := flag.String("shArm64", TD+"arm64_shift_helpers_test.go", "file name for ARM64 shift test helpers")
	uhArm64 := flag.String("uhArm64", TD+"arm64_unary_helpers_test.go", "file name for ARM64 unary test helpers")
	cmArm64 := flag.String("cmArm64", SIMD+"compare_gen_arm64.go", "file name for ARM64 comparison operations")
	mmArm64 := flag.String("mmArm64", SIMD+"maskmerge_gen_arm64.go", "file name for ARM64 mask/merge operations")
	chArm64 := flag.String("chArm64", TD+"arm64_compare_helpers_test.go", "file name for ARM64 compare test helpers")
	thArm64 := flag.String("thArm64", TD+"ternary_arm64_helpers_test.go", "file name for ARM64 ternary test helpers")
	flag.Parse()

	if *sl != "" {
		one(*sl, unsafePrologue,
			sliceTemplate,
			avx512MaskedLoadSliceTemplate,
			avx2MaskedLoadSliceTemplate,
			avx2SmallLoadSliceTemplate,
		)
	}
	if *cm != "" {
		one(*cm, prologue,
			avx2SignedComparisonsTemplate,
			avx2UnsignedComparisonsTemplate,
		)
	}
	if *mm != "" {
		one(*mm, prologue,
			avx2MaskedTemplate,
			avx512MaskedTemplate,
		)
	}
	if *op != "" {
		one(*op, prologue,
			broadcastTemplate,
			maskCvtTemplate,
			bitWiseIntTemplate,
			bitWiseUintTemplate,
			stringTemplate,
			maskToString,
			shapeAndTemplate{amdIntShiftAllShapes, intRotateAllTemplate},
			shapeAndTemplate{amdUintShiftAllShapes, uintRotateAllTemplate},
		)
	}
	if *ush != "" {
		one(*ush, unsafePrologue, unsafePATemplate)
	}
	if *uh != "" {
		one(*uh, curryTestPrologue("unary simd methods"), unaryTemplate)
	}
	if *cvh != "" {
		one(*cvh, curryTestPrologue("conversion simd methods"),
			unaryToInt8, unaryToUint8, unaryToInt16, unaryToUint16,
			unaryToInt32, unaryToUint32, unaryToInt64, unaryToUint64,
			unaryToFloat32, unaryToFloat64,
			unaryToInt64x2, unaryToInt64x4,
			unaryToUint64x2, unaryToUint64x4,
			unaryToInt32x4, unaryToInt32x8,
			unaryToUint32x4, unaryToUint32x8,
			unaryToInt16x8, unaryToUint16x8,
			unaryFlakyTemplate,
		)
	}
	if *bh != "" {
		one(*bh, curryTestPrologue("binary simd methods"), binaryTemplate)
	}
	if *th != "" {
		one(*th, curryTestPrologue("ternary simd methods"), ternaryTemplate, ternaryFlakyTemplate)
	}
	if *ch != "" {
		one(*ch, curryTestPrologue("simd methods that compare two operands"), compareTemplate, compareUnaryTemplate)
	}
	if *cmh != "" {
		one(*cmh, curryTestPrologue("simd methods that compare two operands under a mask"), compareMaskedTemplate)
	}

	// ARM64-specific generation
	if *slArm64 != "" {
		one(*slArm64, prologue, sliceTemplateArm64)
	}
	if *bhArm64 != "" {
		oneArch(*bhArm64, "arm64", curryTestPrologue("binary simd methods"), binaryTemplateArm64)
	}
	if *opArm64 != "" {
		one(*opArm64, prologue,
			broadcastTemplateArm64,
			stringTemplateArm64,
			setHiTemplateArm64,
			getHiTemplateArm64,
			arm64MaskCvtTemplate,
			shapeAndTemplate{neonIntShiftAllShapes, intRotateAllTemplate},
			shapeAndTemplate{neonUintShiftAllShapes, uintRotateAllTemplate},
		)
	}
	if *shArm64 != "" {
		oneArch(*shArm64, "arm64", curryTestPrologue("shift simd methods"),
			shiftConstTestTemplateArm64,
			shiftAllTestTemplateArm64,
			shiftMixedTestTemplateArm64,
		)
	}
	if *uhArm64 != "" {
		oneArch(*uhArm64, "arm64", curryTestPrologue("unary simd methods"),
			arm64ToInt8, arm64ToUint8,
			arm64ToInt16, arm64ToUint16,
			arm64ToInt32, arm64ToUint32,
			arm64ToInt64, arm64ToUint64,
			arm64ToFloat32, arm64ToFloat64,
			unaryTemplateArm64,
		)
	}
	if *cmArm64 != "" {
		one(*cmArm64, prologue,
			arm64LessTemplate,
			arm64LessEqualTemplate,
			arm64NotEqualTemplate,
		)
	}
	if *mmArm64 != "" {
		one(*mmArm64, prologue,
			arm64MaskedMergeTemplate,
			arm64MaskToString,
		)
	}
	if *chArm64 != "" {
		oneArch(*chArm64, "arm64", curryTestPrologue("simd methods that compare two operands"), compareTemplateArm64)
	}
	if *thArm64 != "" {
		oneArch(*thArm64, "arm64", curryTestPrologue("ternary simd methods"), ternaryTemplateArm64, ternaryFlakyTemplateArm64)
	}

	nonTemplateRewrites(SSA+"tern_helpers.go", ssaPrologue, classifyBooleanSIMD, ternOpForLogical)

}

func ternOpForLogical(out io.Writer) {
	fmt.Fprintf(out, `
func ternOpForLogical(op Op) Op {
	switch op {
`)

	intShapes.forAllShapes(func(seq int, t, upperT string, w, c int, out io.Writer) {
		wt, ct := w, c
		if wt < 32 {
			wt = 32
			ct = (w * c) / wt
		}
		fmt.Fprintf(out, "case OpAndInt%[1]dx%[2]d, OpOrInt%[1]dx%[2]d, OpXorInt%[1]dx%[2]d,OpAndNotInt%[1]dx%[2]d: return OpternInt%dx%d\n", w, c, wt, ct)
		fmt.Fprintf(out, "case OpAndUint%[1]dx%[2]d, OpOrUint%[1]dx%[2]d, OpXorUint%[1]dx%[2]d,OpAndNotUint%[1]dx%[2]d: return OpternUint%dx%d\n", w, c, wt, ct)
	}, out)

	fmt.Fprintf(out, `
	}
	return op
}
`)

}

func classifyBooleanSIMD(out io.Writer) {
	fmt.Fprintf(out, `
type SIMDLogicalOP uint8
const (
	// boolean simd operations, for reducing expression to VPTERNLOG* instructions
	// sloInterior is set for non-root nodes in logical-op expression trees.
	// the operations are even-numbered.
	sloInterior SIMDLogicalOP = 1
	sloNone SIMDLogicalOP = 2 * iota
	sloAnd
	sloOr
	sloAndNot
	sloXor
	sloNot
)
func classifyBooleanSIMD(v *Value) SIMDLogicalOP {
	switch v.Op {
		case `)
	intShapes.forAllShapes(func(seq int, t, upperT string, w, c int, out io.Writer) {
		op := "And"
		if seq > 0 {
			fmt.Fprintf(out, ",Op%s%s%dx%d", op, upperT, w, c)
		} else {
			fmt.Fprintf(out, "Op%s%s%dx%d", op, upperT, w, c)
		}
		seq++
	}, out)

	fmt.Fprintf(out, `:
		return sloAnd

		case `)
	intShapes.forAllShapes(func(seq int, t, upperT string, w, c int, out io.Writer) {
		op := "Or"
		if seq > 0 {
			fmt.Fprintf(out, ",Op%s%s%dx%d", op, upperT, w, c)
		} else {
			fmt.Fprintf(out, "Op%s%s%dx%d", op, upperT, w, c)
		}
		seq++
	}, out)

	fmt.Fprintf(out, `:
		return sloOr

		case `)
	intShapes.forAllShapes(func(seq int, t, upperT string, w, c int, out io.Writer) {
		op := "AndNot"
		if seq > 0 {
			fmt.Fprintf(out, ",Op%s%s%dx%d", op, upperT, w, c)
		} else {
			fmt.Fprintf(out, "Op%s%s%dx%d", op, upperT, w, c)
		}
		seq++
	}, out)

	fmt.Fprintf(out, `:
		return sloAndNot
`)

	// "Not" is encoded as x.Xor(x.Equal(x).AsInt8x16())
	// i.e. xor.Args[0] == x, xor.Args[1].Op == As...
	// but AsInt8x16 is a pun/passthrough.

	intShapes.forAllShapes(
		func(seq int, t, upperT string, w, c int, out io.Writer) {
			fmt.Fprintf(out, "case OpXor%s%dx%d: ", upperT, w, c)
			fmt.Fprintf(out, `
				if y := v.Args[1]; y.Op == OpEqual%s%dx%d &&
				   y.Args[0] == y.Args[1] {
				   		return sloNot
				}
				`, upperT, w, c)
			fmt.Fprintf(out, "return sloXor\n")
		}, out)

	fmt.Fprintf(out, `
	}
	return sloNone
}
`)
}

// numberLines takes a slice of bytes, and returns a string where each line
// is numbered, starting from 1.
func numberLines(data []byte) string {
	var buf bytes.Buffer
	r := bytes.NewReader(data)
	s := bufio.NewScanner(r)
	for i := 1; s.Scan(); i++ {
		fmt.Fprintf(&buf, "%d: %s\n", i, s.Text())
	}
	return buf.String()
}

func nonTemplateRewrites(filename string, prologue func(s string, out io.Writer), rewrites ...func(out io.Writer)) {
	if filename == "" {
		return
	}

	ofile := os.Stdout

	if filename != "-" {
		var err error
		ofile, err = os.Create(filename)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Could not create the output file %s for the generated code, %v", filename, err)
			os.Exit(1)
		}
	}

	out := new(bytes.Buffer)

	prologue("tmplgen", out)
	for _, rewrite := range rewrites {
		rewrite(out)
	}

	b, err := format.Source(out.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "There was a problem formatting the generated code for %s, %v\n", filename, err)
		fmt.Fprintf(os.Stderr, "%s\n", numberLines(out.Bytes()))
		fmt.Fprintf(os.Stderr, "There was a problem formatting the generated code for %s, %v\n", filename, err)
		os.Exit(1)
	} else {
		ofile.Write(b)
		ofile.Close()
	}

}

func one(filename string, prologue func(s, buildArch string, out io.Writer), sats ...shapeAndTemplate) {
	if filename == "" {
		return
	}

	if strings.Contains(filename, "%W") {
		smallFile := strings.ReplaceAll(filename, "%W", "128")
		largeFile := strings.ReplaceAll(filename, "%W", "wider")
		oneArch(smallFile, "(amd64 || wasm)", prologue, Map(smallSAT, sats)...)
		oneArch(largeFile, "amd64", prologue, Map(largeSAT, sats)...)
		return
	}
	oneArch(filename, "amd64", prologue, sats...)
}

func oneArch(filename, buildArch string, prologue func(s, buildArch string, out io.Writer), sats ...shapeAndTemplate) {

	ofile := os.Stdout

	if filename != "-" {
		var err error
		ofile, err = os.Create(filename)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Could not create the output file %s for the generated code, %v", filename, err)
			os.Exit(1)
		}
	}

	out := new(bytes.Buffer)

	prologue("tmplgen", buildArch, out)
	for _, sat := range sats {
		sat.forTemplates(out)
	}

	b, err := format.Source(out.Bytes())
	if err != nil {
		fmt.Fprintf(os.Stderr, "There was a problem formatting the generated code for %s, %v\n", filename, err)
		fmt.Fprintf(os.Stderr, "%s\n", numberLines(out.Bytes()))
		fmt.Fprintf(os.Stderr, "There was a problem formatting the generated code for %s, %v\n", filename, err)
		os.Exit(1)
	} else {
		ofile.Write(b)
		ofile.Close()
	}

}
