// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// this generates type-instantiated boilerplate code for
// slice operations and tests

import (
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

func (sat shapeAndTemplate) target(outType string, width int) shapeAndTemplate {
	newSat := sat
	newShape := *sat.s
	newShape.output = func(t string, w, c int) (ot string, ow int, oc int) {
		return outType, width, c
	}
	newSat.s = &newShape
	return newSat
}

func (sat shapeAndTemplate) shrinkTo(outType string, by int) shapeAndTemplate {
	newSat := sat
	newShape := *sat.s
	newShape.output = func(t string, w, c int) (ot string, ow int, oc int) {
		return outType, w / by, c * by
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

var avx2SmallLoadPunShapes = &shapes{
	// ints are done by hand, these are type-punned to int.
	vecs:  []int{128, 256},
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

func prologue(s string, out io.Writer) {
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

func unsafePrologue(s string, out io.Writer) {
	fmt.Fprintf(out,
		`// Code generated by '%s'; DO NOT EDIT.

//go:build goexperiment.simd

package archsimd

import "unsafe"

`, s)
}

func testPrologue(t, s string, out io.Writer) {
	fmt.Fprintf(out,
		`// Code generated by '%s'; DO NOT EDIT.

//go:build goexperiment.simd

// This file contains functions testing %s.
// Each function in this file is specialized for a
// particular simd type <BaseType><Width>x<Count>.

package simd_test

import (
	"simd/archsimd"
	"testing"
)

`, s, t)
}

func curryTestPrologue(t string) func(s string, out io.Writer) {
	return func(s string, out io.Writer) {
		testPrologue(t, s, out)
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

var sliceTemplate = templateOf("slice", `
// Load{{.VType}}Slice loads {{.AOrAn}} {{.VType}} from a slice of at least {{.Count}} {{.Etype}}s
func Load{{.VType}}Slice(s []{{.Etype}}) {{.VType}} {
	return Load{{.VType}}((*[{{.Count}}]{{.Etype}})(s))
}

// StoreSlice stores x into a slice of at least {{.Count}} {{.Etype}}s
func (x {{.VType}}) StoreSlice(s []{{.Etype}}) {
	x.Store((*[{{.Count}}]{{.Etype}})(s))
}
`)

var unaryTemplate = templateOf("unary_helpers", `
// test{{.VType}}Unary tests the simd unary method f against the expected behavior generated by want
func test{{.VType}}Unary(t *testing.T, f func(_ archsimd.{{.VType}}) archsimd.{{.VType}}, want func(_ []{{.Etype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		g := make([]{{.Etype}}, n)
		f(a).StoreSlice(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var unaryFlakyTemplate = shapedTemplateOf(unaryFlaky, "unary_flaky_helpers", `
// test{{.VType}}UnaryFlaky tests the simd unary method f against the expected behavior generated by want,
// but using a flakiness parameter because we haven't exactly figured out how simd floating point works
func test{{.VType}}UnaryFlaky(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.{{.VType}}, want func(x []{{.Etype}}) []{{.Etype}}, flakiness float64) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		g := make([]{{.Etype}}, n)
		f(a).StoreSlice(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, flakiness, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var convertTemplate = templateOf("convert_helpers", `
// test{{.VType}}ConvertTo{{.OEType}} tests the simd conversion method f against the expected behavior generated by want
// This is for count-preserving conversions, so if there is a change in size, then there is a change in vector width.
func test{{.VType}}ConvertTo{{.OEType}}(t *testing.T, f func(x archsimd.{{.VType}}) archsimd.{{.OVType}}, want func(x []{{.Etype}}) []{{.OEtype}}) {
	n := {{.Count}}
	t.Helper()
	forSlice(t, {{.Etype}}s, n, func(x []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		g := make([]{{.OEtype}}, n)
		f(a).StoreSlice(g)
		w := want(x)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x)})
	})
}
`)

var unaryToInt32 = convertTemplate.target("int", 32)
var unaryToUint32 = convertTemplate.target("uint", 32)
var unaryToUint16 = convertTemplate.target("uint", 16)

var binaryTemplate = templateOf("binary_helpers", `
// test{{.VType}}Binary tests the simd binary method f against the expected behavior generated by want
func test{{.VType}}Binary(t *testing.T, f func(_, _ archsimd.{{.VType}}) archsimd.{{.VType}}, want func(_, _ []{{.Etype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSlicePair(t, {{.Etype}}s, n, func(x, y []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		b := archsimd.Load{{.VType}}Slice(y)
		g := make([]{{.Etype}}, n)
		f(a, b).StoreSlice(g)
		w := want(x, y)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); })
	})
}
`)

var ternaryTemplate = templateOf("ternary_helpers", `
// test{{.VType}}Ternary tests the simd ternary method f against the expected behavior generated by want
func test{{.VType}}Ternary(t *testing.T, f func(_, _, _ archsimd.{{.VType}}) archsimd.{{.VType}}, want func(_, _, _ []{{.Etype}}) []{{.Etype}}) {
	n := {{.Count}}
	t.Helper()
	forSliceTriple(t, {{.Etype}}s, n, func(x, y, z []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		b := archsimd.Load{{.VType}}Slice(y)
		c := archsimd.Load{{.VType}}Slice(z)
		g := make([]{{.Etype}}, n)
		f(a, b, c).StoreSlice(g)
		w := want(x, y, z)
		return checkSlicesLogInput(t, g, w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); t.Logf("z=%v", z); })
	})
}
`)

var ternaryFlakyTemplate = shapedTemplateOf(ternaryFlaky, "ternary_helpers", `
// test{{.VType}}TernaryFlaky tests the simd ternary method f against the expected behavior generated by want,
// but using a flakiness parameter because we haven't exactly figured out how simd floating point works
func test{{.VType}}TernaryFlaky(t *testing.T, f func(x, y, z archsimd.{{.VType}}) archsimd.{{.VType}}, want func(x, y, z []{{.Etype}}) []{{.Etype}}, flakiness float64) {
	n := {{.Count}}
	t.Helper()
	forSliceTriple(t, {{.Etype}}s, n, func(x, y, z []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		b := archsimd.Load{{.VType}}Slice(y)
		c := archsimd.Load{{.VType}}Slice(z)
		g := make([]{{.Etype}}, n)
		f(a, b, c).StoreSlice(g)
		w := want(x, y, z)
		return checkSlicesLogInput(t, g, w, flakiness, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); t.Logf("z=%v", z); })
	})
}
`)

var compareTemplate = templateOf("compare_helpers", `
// test{{.VType}}Compare tests the simd comparison method f against the expected behavior generated by want
func test{{.VType}}Compare(t *testing.T, f func(_, _ archsimd.{{.VType}}) archsimd.Mask{{.WxC}}, want func(_, _ []{{.Etype}}) []int64) {
	n := {{.Count}}
	t.Helper()
	forSlicePair(t, {{.Etype}}s, n, func(x, y []{{.Etype}}) bool {
	 	t.Helper()
		a := archsimd.Load{{.VType}}Slice(x)
		b := archsimd.Load{{.VType}}Slice(y)
		g := make([]int{{.EWidth}}, n)
		f(a, b).AsInt{{.WxC}}().StoreSlice(g)
		w := want(x, y)
		return checkSlicesLogInput(t, s64(g), w, 0.0, func() {t.Helper(); t.Logf("x=%v", x); t.Logf("y=%v", y); })
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
		a := archsimd.Load{{.VType}}Slice(x)
		b := archsimd.Load{{.VType}}Slice(y)
		k := archsimd.LoadInt{{.WxC}}Slice(toVect[int{{.EWidth}}](m)).ToMask()
		g := make([]int{{.EWidth}}, n)
		f(a, b, k).AsInt{{.WxC}}().StoreSlice(g)
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

var avx512MaskedLoadSlicePartTemplate = shapedTemplateOf(avx512Shapes, "avx 512 load slice part", `
// Load{{.VType}}SlicePart loads a {{.VType}} from the slice s.
// If s has fewer than {{.Count}} elements, the remaining elements of the vector are filled with zeroes.
// If s has {{.Count}} or more elements, the function is equivalent to Load{{.VType}}Slice.
func Load{{.VType}}SlicePart(s []{{.Etype}}) {{.VType}} {
	l := len(s)
	if l >= {{.Count}} {
		return Load{{.VType}}Slice(s)
	}
	if l == 0 {
		var x {{.VType}}
		return x
	}
	mask := Mask{{.WxC}}FromBits({{.OxFF}} >> ({{.Count}} - l))
	return LoadMasked{{.VType}}(pa{{.VType}}(s), mask)
}

// StoreSlicePart stores the {{.Count}} elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has {{.Count}} or more elements, the method is equivalent to x.StoreSlice.
func (x {{.VType}}) StoreSlicePart(s []{{.Etype}}) {
	l := len(s)
	if l >= {{.Count}} {
		x.StoreSlice(s)
		return
	}
	if l == 0 {
		return
	}
	mask := Mask{{.WxC}}FromBits({{.OxFF}} >> ({{.Count}} - l))
	x.StoreMasked(pa{{.VType}}(s), mask)
}
`)

var avx2MaskedLoadSlicePartTemplate = shapedTemplateOf(avx2MaskedLoadShapes, "avx 2 load slice part", `
// Load{{.VType}}SlicePart loads a {{.VType}} from the slice s.
// If s has fewer than {{.Count}} elements, the remaining elements of the vector are filled with zeroes.
// If s has {{.Count}} or more elements, the function is equivalent to Load{{.VType}}Slice.
func Load{{.VType}}SlicePart(s []{{.Etype}}) {{.VType}} {
	l := len(s)
	if l >= {{.Count}} {
		return Load{{.VType}}Slice(s)
	}
	if l == 0 {
		var x {{.VType}}
		return x
	}
	mask := vecMask{{.EWidth}}[len(vecMask{{.EWidth}})/2-l:]
	return LoadMasked{{.VType}}(pa{{.VType}}(s), LoadInt{{.WxC}}Slice(mask).asMask())
}

// StoreSlicePart stores the {{.Count}} elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has {{.Count}} or more elements, the method is equivalent to x.StoreSlice.
func (x {{.VType}}) StoreSlicePart(s []{{.Etype}}) {
	l := len(s)
	if l >= {{.Count}} {
		x.StoreSlice(s)
		return
	}
	if l == 0 {
		return
	}
	mask := vecMask{{.EWidth}}[len(vecMask{{.EWidth}})/2-l:]
	x.StoreMasked(pa{{.VType}}(s), LoadInt{{.WxC}}Slice(mask).asMask())
}
`)

var avx2SmallLoadSlicePartTemplate = shapedTemplateOf(avx2SmallLoadPunShapes, "avx 2 small load slice part", `
// Load{{.VType}}SlicePart loads a {{.VType}} from the slice s.
// If s has fewer than {{.Count}} elements, the remaining elements of the vector are filled with zeroes.
// If s has {{.Count}} or more elements, the function is equivalent to Load{{.VType}}Slice.
func Load{{.VType}}SlicePart(s []{{.Etype}}) {{.VType}} {
	if len(s) == 0 {
		var zero {{.VType}}
		return zero
	}
	t := unsafe.Slice((*int{{.EWidth}})(unsafe.Pointer(&s[0])), len(s))
	return LoadInt{{.WxC}}SlicePart(t).As{{.VType}}()
}

// StoreSlicePart stores the {{.Count}} elements of x into the slice s.
// It stores as many elements as will fit in s.
// If s has {{.Count}} or more elements, the method is equivalent to x.StoreSlice.
func (x {{.VType}}) StoreSlicePart(s []{{.Etype}}) {
	if len(s) == 0 {
		return
	}
	t := unsafe.Slice((*int{{.EWidth}})(unsafe.Pointer(&s[0])), len(s))
	x.AsInt{{.WxC}}().StoreSlicePart(t)
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
// Less returns a mask whose elements indicate whether x < y
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) Less(y {{.VType}}) Mask{{.WxC}} {
	return y.Greater(x)
}

// GreaterEqual returns a mask whose elements indicate whether x >= y
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) GreaterEqual(y {{.VType}}) Mask{{.WxC}} {
	ones := x.Equal(x).AsInt{{.WxC}}()
	return y.Greater(x).AsInt{{.WxC}}().Xor(ones).asMask()
}

// LessEqual returns a mask whose elements indicate whether x <= y
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) LessEqual(y {{.VType}}) Mask{{.WxC}} {
	ones := x.Equal(x).AsInt{{.WxC}}()
	return x.Greater(y).AsInt{{.WxC}}().Xor(ones).asMask()
}

// NotEqual returns a mask whose elements indicate whether x != y
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) NotEqual(y {{.VType}}) Mask{{.WxC}} {
	ones := x.Equal(x).AsInt{{.WxC}}()
	return x.Equal(y).AsInt{{.WxC}}().Xor(ones).asMask()	
}
`)

var bitWiseIntTemplate = shapedTemplateOf(intShapes, "bitwise int complement", `
// Not returns the bitwise complement of x
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) Not() {{.VType}} {
	return x.Xor(x.Equal(x).As{{.VType}}())
}
`)

var bitWiseUintTemplate = shapedTemplateOf(uintShapes, "bitwise uint complement", `
// Not returns the bitwise complement of x
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) Not() {{.VType}} {
	return x.Xor(x.Equal(x).AsInt{{.WxC}}().As{{.VType}}())
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
// Greater returns a mask whose elements indicate whether x > y
//
// Emulated, CPU Feature {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) Greater(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	ones := x.Equal(x).AsInt{{.WxC}}()
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return a.Xor(signs).Greater(b.Xor(signs))
}

// Less returns a mask whose elements indicate whether x < y
//
// Emulated, CPU Feature {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) Less(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	ones := x.Equal(x).AsInt{{.WxC}}()
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return b.Xor(signs).Greater(a.Xor(signs))
}

// GreaterEqual returns a mask whose elements indicate whether x >= y
//
// Emulated, CPU Feature {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) GreaterEqual(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
	ones := x.Equal(x).AsInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return b.Xor(signs).Greater(a.Xor(signs)).AsInt{{.WxC}}().Xor(ones).asMask()
}

// LessEqual returns a mask whose elements indicate whether x <= y
//
// Emulated, CPU Feature {{.CPUfeatureAVX2if8}}
func (x {{.VType}}) LessEqual(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
	ones := x.Equal(x).AsInt{{.WxC}}()
{{- if eq .EWidth 8}}
	signs := BroadcastInt{{.WxC}}(-1 << ({{.EWidth}}-1))
{{- else}}
	signs := ones.ShiftAllLeft({{.EWidth}}-1)
{{- end }}
	return a.Xor(signs).Greater(b.Xor(signs)).AsInt{{.WxC}}().Xor(ones).asMask()
}

// NotEqual returns a mask whose elements indicate whether x != y
//
// Emulated, CPU Feature {{.CPUfeature}}
func (x {{.VType}}) NotEqual(y {{.VType}}) Mask{{.WxC}} {
	a, b := x.AsInt{{.WxC}}(), y.AsInt{{.WxC}}()
	ones := x.Equal(x).AsInt{{.WxC}}()
	return a.Equal(b).AsInt{{.WxC}}().Xor(ones).asMask()
}
`)

var unsafePATemplate = templateOf("unsafe PA helper", `
// pa{{.VType}} returns a type-unsafe pointer to array that can
// only be used with partial load/store operations that only
// access the known-safe portions of the array.
func pa{{.VType}}(s []{{.Etype}}) *[{{.Count}}]{{.Etype}} {
	return (*[{{.Count}}]{{.Etype}})(unsafe.Pointer(&s[0]))
}
`)

var avx2MaskedTemplate = shapedTemplateOf(avx2Shapes, "avx2 .Masked methods", `
// Masked returns x but with elements zeroed where mask is false.
func (x {{.VType}}) Masked(mask Mask{{.WxC}}) {{.VType}} {
	im := mask.AsInt{{.WxC}}()
{{- if eq .Base "Int" }}
	return im.And(x)
{{- else}}
    return x.AsInt{{.WxC}}().And(im).As{{.VType}}()
{{- end -}}
}

// Merge returns x but with elements set to y where mask is false.
func (x {{.VType}}) Merge(y {{.VType}}, mask Mask{{.WxC}}) {{.VType}} {
{{- if eq .BxC .WxC -}}
	im := mask.AsInt{{.BxC}}()
{{- else}}
    im := mask.AsInt{{.WxC}}().AsInt{{.BxC}}()
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
func (x {{.VType}}) Masked(mask Mask{{.WxC}}) {{.VType}} {
	im := mask.AsInt{{.WxC}}()
{{- if eq .Base "Int" }}
	return im.And(x)
{{- else}}
    return x.AsInt{{.WxC}}().And(im).As{{.VType}}()
{{- end -}}
}

// Merge returns x but with elements set to y where m is false.
func (x {{.VType}}) Merge(y {{.VType}}, mask Mask{{.WxC}}) {{.VType}} {
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
// Emulated, CPU Feature {{.CPUfeatureBC}}
func Broadcast{{.VType}}(x {{.Etype}}) {{.VType}} {
	var z {{.As128BitVec }}
	return z.SetElem(0, x).Broadcast{{.Vwidth}}()
}
`)

var maskCvtTemplate = shapedTemplateOf(intShapes, "Mask conversions", `
// ToMask converts from {{.Base}}{{.WxC}} to Mask{{.WxC}}, mask element is set to true when the corresponding vector element is non-zero.
func (from {{.Base}}{{.WxC}}) ToMask() (to Mask{{.WxC}}) {
	return from.NotEqual({{.Base}}{{.WxC}}{})
}
`)

var stringTemplate = shapedTemplateOf(allShapes, "String methods", `
// String returns a string representation of SIMD vector x
func (x {{.VType}}) String() string {
	var s [{{.Count}}]{{.Etype}}
	x.Store(&s)
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
	bh := flag.String("bh", TD+"binary_helpers_test.go", "file name for binary test helpers")
	uh := flag.String("uh", TD+"unary_helpers_test.go", "file name for unary test helpers")
	th := flag.String("th", TD+"ternary_helpers_test.go", "file name for ternary test helpers")
	ch := flag.String("ch", TD+"compare_helpers_test.go", "file name for compare test helpers")
	cmh := flag.String("cmh", TD+"comparemasked_helpers_test.go", "file name for compare-masked test helpers")
	flag.Parse()

	if *sl != "" {
		one(*sl, unsafePrologue,
			sliceTemplate,
			avx512MaskedLoadSlicePartTemplate,
			avx2MaskedLoadSlicePartTemplate,
			avx2SmallLoadSlicePartTemplate,
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
		)
	}
	if *ush != "" {
		one(*ush, unsafePrologue, unsafePATemplate)
	}
	if *uh != "" {
		one(*uh, curryTestPrologue("unary simd methods"), unaryTemplate, unaryToInt32, unaryToUint32, unaryToUint16, unaryFlakyTemplate)
	}
	if *bh != "" {
		one(*bh, curryTestPrologue("binary simd methods"), binaryTemplate)
	}
	if *th != "" {
		one(*th, curryTestPrologue("ternary simd methods"), ternaryTemplate, ternaryFlakyTemplate)
	}
	if *ch != "" {
		one(*ch, curryTestPrologue("simd methods that compare two operands"), compareTemplate)
	}
	if *cmh != "" {
		one(*cmh, curryTestPrologue("simd methods that compare two operands under a mask"), compareMaskedTemplate)
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

	prologue("go run genfiles.go", out)
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

func one(filename string, prologue func(s string, out io.Writer), sats ...shapeAndTemplate) {
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

	prologue("go run genfiles.go", out)
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
