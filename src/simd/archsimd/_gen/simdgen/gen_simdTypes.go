// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"cmp"
	"fmt"
	"maps"
	"slices"
	"strings"
	"text/template"
	"unicode"

	"simd/archsimd/_gen/sgutil"
	"simd/archsimd/_gen/simdgen/types"
	"simd/archsimd/_gen/specgen/specexpr"
)

type simdType struct {
	Shape  specexpr.Vector // Shape represents the element type, element bits, and vector width.
	Fields string          // The struct fields, it should be right formatted.
	HasNot bool            // True when this mask type supports Not()
}

func (x simdType) IsMask() bool {
	return strings.EqualFold(x.Shape.Elem.Base, "mask")
}

func (x simdType) Type() string {
	if x.IsMask() {
		return "mask"
	}
	return "vreg"
}

func (x simdType) Name() string {
	return x.Shape.String()
}

func (x simdType) Base() string {
	if x.IsMask() {
		return fmt.Sprintf("int%d", x.ElemBits())
	}
	return fmt.Sprintf("%s%d", strings.ToLower(x.Shape.Elem.Base), x.ElemBits())
}

func (x simdType) ElemBits() int {
	return int(x.Shape.Elem.Bits)
}

// ElemBytes is the element width in bytes.
func (x simdType) ElemBytes() int {
	return x.ElemBits() / 8
}

// Lanes returns the number of elements in a fixed-width vector.
// Panics if the vector is scalable.
func (x simdType) Lanes() int {
	if x.IsScalable() {
		panic(fmt.Sprintf("cannot get fixed lane count of scalable type %s", x.Name()))
	}
	l, err := x.Shape.Width.Div(x.Shape.Elem.Bits)
	if err != nil {
		panic(err)
	}
	return int(l.(specexpr.Int))
}

// MaxLanes returns the maximum number of elements for scalable vectors (based on
// MaxVectorBits) or the fixed lane count for fixed-width vectors.
func (x simdType) MaxLanes() int {
	if x.IsScalable() {
		return types.MaxVectorBits / x.ElemBits()
	}
	return x.Lanes()
}

// Size returns the total bit width of a fixed-width vector.
// Panics if the vector is scalable.
func (x simdType) Size() int {
	if x.IsScalable() {
		panic(fmt.Sprintf("cannot get fixed bit width of scalable type %s", x.Name()))
	}
	return int(x.Shape.Width.(specexpr.Int))
}

// VectorCounterpart returns the counterpart vector type name for a mask type.
func (x simdType) VectorCounterpart() string {
	v := specexpr.Vector{
		Elem:  specexpr.Basic{Base: "int", Bits: x.Shape.Elem.Bits},
		Width: x.Shape.Width,
	}
	return v.String()
}

// ReshapedVectorWithAndOr returns the 32-bit-element vector type name with matching width.
func (x simdType) ReshapedVectorWithAndOr() string {
	v := specexpr.Vector{
		Elem:  specexpr.Basic{Base: "int", Bits: 32},
		Width: x.Shape.Width,
	}
	return v.String()
}

// PredUint16s is the number of uint16s that hold a whole SVE predicate: one bit
// per vector byte, at the maximum vector length. It bounds the scratch buffer a
// mask's String needs to read its own bits.
func (x simdType) PredUint16s() int {
	return (types.MaxVectorBits/8 + 15) / 16
}

// IsScalable reports whether this vector type's length is only known at run time.
func (x simdType) IsScalable() bool {
	return x.Shape.Scalable()
}

// LenExpr is the body expression of the type's Len() method. A fixed-width type
// has a constant lane count; a scalable vector's active lane count is the
// runtime vector length (bytes) divided by the element size.
func (x simdType) LenExpr() string {
	if !x.IsScalable() {
		return fmt.Sprint(x.Lanes())
	}
	if elemBytes := x.ElemBits() / 8; elemBytes > 1 {
		return fmt.Sprintf("vl() / %d", elemBytes)
	}
	return "vl()"
}

// Name_ implements sgutil.TforAsBits.
func (x simdType) Name_() string {
	return x.Name()
}

func (x simdType) Article() string {
	if strings.HasPrefix(x.Name(), "Int") {
		return "an"
	}
	return "a" // Float, Uint
}

// LanesContainer returns the smallest int/uint bit size that is
// large enough to hold one bit for each lane.  E.g., Mask32x4
// is 4 lanes, and a uint8 is the smallest uint that has 4 bits.
func (x simdType) LanesContainer() int {
	lanes := x.Lanes()
	if lanes > 64 {
		panic("too many lanes")
	}
	if lanes > 32 {
		return 64
	}
	if lanes > 16 {
		return 32
	}
	if lanes > 8 {
		return 16
	}
	return 8
}

// MaskedLoadStoreFilter encodes which simd type type currently
// get masked loads/stores generated, it is used in two places,
// this forces coordination.
func (x simdType) MaskedLoadStoreFilter() bool {
	if x.IsScalable() {
		return false
	}
	return x.Size() == 512 || (x.ElemBits() >= 32 && !x.IsMask())
}

func (x simdType) IntelSizeSuffix() string {
	switch x.ElemBits() {
	case 8:
		return "B"
	case 16:
		return "W"
	case 32:
		return "D"
	case 64:
		return "Q"
	}
	panic("oops")
}

func (x simdType) MaskedLoadDoc() string {
	if x.Size() == 512 || x.ElemBits() < 32 {
		return fmt.Sprintf("// Asm: VMOVDQU%d.Z, CPU Feature: AVX512", x.ElemBits())
	} else {
		return fmt.Sprintf("// Asm: VMASKMOV%s, CPU Feature: AVX2", x.IntelSizeSuffix())
	}
}

func (x simdType) MaskedStoreDoc() string {
	if x.Size() == 512 || x.ElemBits() < 32 {
		return fmt.Sprintf("// Asm: VMOVDQU%d, CPU Feature: AVX512", x.ElemBits())
	} else {
		return fmt.Sprintf("// Asm: VMASKMOV%s, CPU Feature: AVX2", x.IntelSizeSuffix())
	}
}

func (x simdType) ToBitsDoc() string {
	if x.IsScalable() {
		panic("ToBitsDoc is not supported for scalable types")
	}
	if x.Size() == 512 || x.ElemBits() == 16 {
		return fmt.Sprintf("// Asm: KMOV%s, CPU Features: AVX512", x.IntelSizeSuffix())
	}
	// 128/256 bit vectors with 8, 32, 64 bit elements
	var asm string
	var feat string
	switch x.ElemBits() {
	case 8:
		asm = "VPMOVMSKB"
		if x.Size() == 256 {
			feat = "AVX2"
		} else {
			feat = "AVX"
		}
	case 32:
		asm = "VMOVMSKPS"
		feat = "AVX"
	case 64:
		asm = "VMOVMSKPD"
		feat = "AVX"
	default:
		panic("unexpected ElemBits")
	}
	return fmt.Sprintf("// Asm: %s, CPU Features: %s", asm, feat)
}

func compareWidths(a, b specexpr.Num) int {
	if c, ok := a.Compare(b); ok {
		return c
	}
	_, aScale := a.(specexpr.ScalableWidth)
	_, bScale := b.(specexpr.ScalableWidth)
	if !aScale && bScale {
		return -1
	}
	return 1
}

func compareSimdTypes(x, y simdType) int {
	// "vreg" then "mask"
	if c := -compareNatural(x.Type(), y.Type()); c != 0 {
		return c
	}
	// want "flo" < "int" < "uin" (and then 8 < 16 < 32 < 64),
	// not "int16" < "int32" < "int64" < "int8")
	// so limit comparison to first 3 bytes in string.
	if c := compareNatural(x.Base()[:3], y.Base()[:3]); c != 0 {
		return c
	}
	// base type size, 8 < 16 < 32 < 64
	if c := x.ElemBits() - y.ElemBits(); c != 0 {
		return c
	}
	// vector size last
	return compareWidths(x.Shape.Width, y.Shape.Width)
}

type simdTypeMap map[specexpr.Num][]simdType

type simdTypePair struct {
	Tsrc simdType
	Tdst simdType
}

func compareSimdTypePairs(x, y simdTypePair) int {
	c := compareSimdTypes(x.Tsrc, y.Tsrc)
	if c != 0 {
		return c
	}
	return compareSimdTypes(x.Tdst, y.Tdst)
}

func simdPackageHeader() string {
	// A shared-package target's Go API files (SVE: types_sve.go, ops_sve.go) carry a
	// "sve" name suffix, which is not a GOARCH, so unlike types_arm64.go they get
	// no implicit build constraint. Add the GOARCH explicitly so they only build
	// on their host arch and don't clash with other arches' tag types (e.g. v256).
	constraint := "goexperiment.simd"
	if a := CurrentArch(); a.sharesBackendPackage() {
		constraint += " && " + a.Arch
	}
	return generatedHeader() + `
//go:build ` + constraint + `

package archsimd
`
}

const simdTypesTemplates = `
{{define "sizeTmpl"}}
// v{{.}} is a tag type that tells the compiler that this is really {{.}}-bit SIMD
type v{{.}} struct {
	_{{.}} [0]func() // uncomparable
}
{{end}}

{{define "typeTmpl"}}
{{- if eq .Type "mask"}}
// {{.Name}} is a mask for a SIMD vector of {{.Lanes}} {{.ElemBits}}-bit elements.
{{- else}}
// {{.Name}} is a {{.Size}}-bit SIMD vector of {{.Lanes}} {{.Base}}s.
{{- end}}
type {{.Name}} struct {
{{.Fields}}
}

{{end}}

{{define "scalableTypeTmpl"}}
{{- if eq .Type "mask"}}
// {{.Name}} is a scalable mask for a SIMD vector of {{.ElemBits}}-bit elements.
//
// An SVE predicate holds one bit per byte of the vector it governs, so a
// {{.Name}} carries one bit for each byte of the runtime vector length, and
// lane i is governed by bit {{if gt .ElemBytes 1}}{{.ElemBytes}}*i. The bits in between are ignored{{else}}i{{end}}.
{{- else}}
// {{.Name}} is a scalable SIMD vector of {{.Base}}s.
{{- end}}
type {{.Name}} struct {
{{.Fields}}
}

{{end}}

{{define "sveMaskLoadStore"}}
// Load{{.Name}} loads a {{.Name}} from the predicate bits packed into bits.
// The bits are concatenated in little-endian order: bit i of bits[j] governs
// vector byte 16*j+i, and so lane k is governed by bit {{if gt .ElemBytes 1}}{{.ElemBytes}}*k{{else}}k{{end}}.
//
// One uint16 covers 16 bytes of vector, the length of the smallest vector SVE
// defines, so bits must hold one uint16 per 16 bytes of the runtime vector
// length. Load{{.Name}} panics if bits is shorter than that.
//
// Asm: Emulated (a length check that can panic, then PLDR (predicate)).
func Load{{.Name}}(bits []uint16) {{.Name}} {
	if len(bits) < (vl()+15)/16 {
		panic("simd: Load{{.Name}}: bits is too short to hold the predicate")
	}
	return load{{.Name}}(bits)
}

//go:noescape
func load{{.Name}}(bits []uint16) {{.Name}}

// Store stores m's predicate bits into bits, concatenated in little-endian
// order: bit i of bits[j] governs vector byte 16*j+i, and so lane k is
// governed by bit {{if gt .ElemBytes 1}}{{.ElemBytes}}*k{{else}}k{{end}}.
//
// bits must hold one uint16 per 16 bytes of the runtime vector length; Store
// panics if it is shorter.
//
// Asm: Emulated (a length check that can panic, then PSTR (predicate)).
func (m {{.Name}}) Store(bits []uint16) {
	if len(bits) < (vl()+15)/16 {
		panic("simd: {{.Name}}.Store: bits is too short to hold the predicate")
	}
	m.store(bits)
}

//go:noescape
func (m {{.Name}}) store(bits []uint16)
{{end}}

{{define "sveIfElseTmpl"}}
// IfElse returns the elements of x where the corresponding element of mask is
// true, and the elements of y where it is false.
//
// Asm: ZSEL
func (x {{.Name}}) IfElse(mask Mask{{.ElemBits}}s, y {{.Name}}) {{.Name}}

// Masked returns the elements of x where the corresponding element of mask is
// true, and zero where it is false.
//
// Asm: Emulated
func (x {{.Name}}) Masked(mask Mask{{.ElemBits}}s) {{.Name}} {
	var zero {{.Name}}
	return x.IfElse(mask, zero)
}
{{end}}

{{define "sveStringTmpl"}}
{{- if eq .Type "mask"}}
// String returns a string representation of SIMD mask m: 1 for an active lane,
// 0 for an inactive one. Only the {{.LenExpr}} lanes that exist at the runtime
// vector length are shown.
func (m {{.Name}}) String() string {
	var bits [{{.PredUint16s}}]uint16
	m.Store(bits[:])
	var s [{{.MaxLanes}}]{{.Base}}
	n := {{.LenExpr}}
	for i := range n {
		if b := i{{if gt .ElemBytes 1}} * {{.ElemBytes}}{{end}}; bits[b/16]>>(b%16)&1 != 0 {
			s[i] = 1
		}
	}
	return sliceToString(s[:n])
}
{{- else}}
// String returns a string representation of SIMD vector x. Only the x.Len()
// elements that exist at the runtime vector length are shown.
func (x {{.Name}}) String() string {
	var s [{{.MaxLanes}}]{{.Base}}
	n := x.Len()
	x.Store(s[:])
	return sliceToString(s[:n])
}
{{- end}}
{{end}}
`

const simdFeaturesTemplate = `
import "internal/cpu"

type X86Features struct {}

var X86 X86Features

{{range .}}
{{$f := .}}
{{- if eq .Feature "AVX512"}}
// {{.Feature}} returns whether the CPU supports the AVX512F+CD+BW+DQ+VL features.
//
// These five CPU features are bundled together, and no use of AVX-512
// is allowed unless all of these features are supported together.
// Nearly every CPU that has shipped with any support for AVX-512 has
// supported all five of these features.
{{- else -}}
// {{.Feature}} returns whether the CPU supports the {{.Feature}} feature.
{{- end}}
{{- if ne .ImpliesAll ""}}
//
// If it returns true, then the CPU also supports {{.ImpliesAll}}.
{{- end}}
//
// {{.Feature}} is defined on all GOARCHes, but will only return true on
// GOARCH {{.GoArch}}.
func ({{.FeatureVar}}Features) {{.Feature}}() bool {
{{- if .Virtual}}
	return {{range $i, $dep := .Implies}}{{if $i}} && {{end}}cpu.{{$f.FeatureVar}}.Has{{$dep}}{{end}}
{{- else}}
	return cpu.{{.FeatureVar}}.Has{{.Feature}}
{{- end}}
}
{{end}}
`

const simdLoadStoreTemplate = `
// Len returns the number of elements in {{.Article}} {{.Name}}.
func (x {{.Name}}) Len() int { return {{.LenExpr}} }

// Load{{.Name}}Array loads {{.Article}} {{.Name}} from an array.
//
//go:noescape
func Load{{.Name}}Array(y *[{{.Lanes}}]{{.Base}}) {{.Name}}

// StoreArray stores {{.Article}} {{.Name}} to an array.
//
//go:noescape
func (x {{.Name}}) StoreArray(y *[{{.Lanes}}]{{.Base}})
`

// simdScalableLoadStoreTemplate is the load/store surface for scalable (SVE)
// types: partial and slice-based rather than fixed-array. The exported Load/Store
// functions are ordinary Go (the "emulation"): they compute how many elements to
// move — min(len(s), Len()), and nothing for an empty or nil slice — and call the
// unexported raw predicated load/store intrinsic. Keeping the length logic in Go
// (rather than in the intrinsic) makes the bounds behavior explicit and safe.
const simdScalableLoadStoreTemplate = `
// Len returns the number of elements in {{.Article}} {{.Name}}.
func (x {{.Name}}) Len() int { return {{.LenExpr}} }

// Load{{.Name}} loads {{.Article}} {{.Name}} from the first Len() elements of s.
// It panics if len(s) < Len().
//
// Asm: Emulated (a length check that can panic, then ZLDR).
func Load{{.Name}}(s []{{.Base}}) {{.Name}} {
	var z {{.Name}}
	if len(s) < z.Len() {
		panic("simd: Load{{.Name}}: slice shorter than the vector")
	}
	return load{{.Name}}(s)
}

//go:noescape
func load{{.Name}}(s []{{.Base}}) {{.Name}}

// Store stores x's Len() elements into the first Len() elements of s. It panics
// if len(s) < Len().
//
// Asm: Emulated (a length check that can panic, then ZSTR).
func (x {{.Name}}) Store(s []{{.Base}}) {
	if len(s) < x.Len() {
		panic("simd: {{.Name}}.Store: slice shorter than the vector")
	}
	x.store(s)
}

//go:noescape
func (x {{.Name}}) store(s []{{.Base}})

// Load{{.Name}}Part loads {{.Article}} {{.Name}} from s, reading n = min(len(s),
// Len()) elements and returning the vector and n; the remaining elements are
// zero.
//
// Asm: Emulated (predicate construction + LD1B).
func Load{{.Name}}Part(s []{{.Base}}) ({{.Name}}, int) {
	if len(s) == 0 {
		return {{.Name}}{}, 0
	}
	return load{{.Name}}Part(s), min(len(s), {{.Name}}{}.Len())
}

//go:noescape
func load{{.Name}}Part(s []{{.Base}}) {{.Name}}

// StorePart stores the low n = min(len(s), Len()) elements of x into s and
// returns n.
//
// Asm: Emulated (predicate construction + ST1B).
func (x {{.Name}}) StorePart(s []{{.Base}}) int {
	if len(s) == 0 {
		return 0
	}
	x.storePart(s)
	return min(len(s), x.Len())
}

//go:noescape
func (x {{.Name}}) storePart(s []{{.Base}})
`

const simdMaskFromValTemplate = `
// {{.Name}}FromBits constructs a {{.Name}} from a bitmap value, where 1 means set for the indexed element, 0 means unset.
{{- if ne .Lanes .LanesContainer}}
// Only the lower {{.Lanes}} bits of y are used.
{{- end}}
//
// Asm: KMOV{{.IntelSizeSuffix}}, CPU Feature: AVX512
func {{.Name}}FromBits(y uint{{.LanesContainer}}) {{.Name}}

// ToBits constructs a bitmap from a {{.Name}}, where 1 means set for the indexed element, 0 means unset.
{{- if ne .Lanes .LanesContainer}}
// Only the lower {{.Lanes}} bits of y are used.
{{- end}}
//
{{.ToBitsDoc}}
func (x {{.Name}}) ToBits() uint{{.LanesContainer}}
`

const simdMaskedLoadStoreTemplate = `
// StoreArrayMasked stores {{.Article}} {{.Name}} to an array,
// at those elements enabled by mask.
//
{{.MaskedStoreDoc}}
//
//go:noescape
func (x {{.Name}}) StoreArrayMasked(y *[{{.Lanes}}]{{.Base}}, mask Mask{{.ElemBits}}x{{.Lanes}})
`

// Helper type to make template map initialization less repetitive
// (and also remove a chance for errors.)
type stubTemplateMap struct {
	sgutil.InsertMap[string, *template.Template]
}

// Add creates a template named "name" after appending "\n" to the
// template, and returns the input so that additions may be chained.
// This helps make template initialization easy to order and easy to read.
func (rtm *stubTemplateMap) Add(name string, templ string) *stubTemplateMap {
	// Wrap in newlines.
	templ = "\n" + templ + "\n"
	ct := sgutil.TemplateNamed(name, templ)
	rtm.InsertMap.Put(name, ct)
	return rtm
}

var stubTemplates = new(stubTemplateMap)

func init() {
	st := stubTemplates

	st.Add("op1", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}() {{.GoType}}`)

	st.Add("op2", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}) {{.GoType}}`)

	st.Add("op2_21", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}) {{.GoType}}`)

	st.Add("op2_21Type1", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}) {{.GoType}}`)

	st.Add("op3", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op2NameAndType "z"}}) {{.GoType}}`)

	st.Add("op3_31Zero3", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op2NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}) {{.GoType}}`)

	st.Add("op3_21", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}, {{.Op2NameAndType "z"}}) {{.GoType}}`)

	st.Add("op3_21Type1", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}, {{.Op2NameAndType "z"}}) {{.GoType}}`)

	st.Add("op3_231Type1", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.Op0NameAndType "z"}}) {{.GoType}}`)

	st.Add("op2VecAsScalar", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1Name "y"}} uint{{(index .In 1).TreatLikeAScalarOfSize}}) {{(index .Out 0).Go}}`)

	st.Add("op2ImmVecAsScalar", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op2NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, v float{{(index .In 3).ElemBits}}) {{(index .Out 0).Go}}`)

	st.Add("op3VecAsScalar", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1Name "y"}} uint{{(index .In 1).TreatLikeAScalarOfSize}}, {{.Op2NameAndType "z"}}) {{(index .Out 0).Go}}`)

	st.Add("op4", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op2NameAndType "z"}}, {{.Op3NameAndType "u"}}) {{.GoType}}`)

	st.Add("op4_231Type1", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.Op0NameAndType "z"}}, {{.Op3NameAndType "u"}}) {{.GoType}}`)

	st.Add("op4_31", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op2NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op0NameAndType "z"}}, {{.Op3NameAndType "u"}}) {{.GoType}}`)

	st.Add("op1Imm", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}) {{.GoType}}`)

	st.Add("op1Imm8", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}) {{.GoType}}`)

	st.Add("op2Imm", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, {{.Op2NameAndType "y"}}) {{.GoType}}`)

	st.Add("op2Imm8", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, {{.Op2NameAndType "y"}}) {{.GoType}}`)

	// Special case for the instruction (in some versions at least) takes an immediate but treat it as a regular operand
	st.Add("op1Imm8_rotate", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// Emulated
func ({{.Op1NameAndType "x"}}) {{.Go}}(dist uint64) {{.GoType}}`)

	st.Add("op2Imm8_2I", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.ImmName}} {{.ImmType}}) {{.GoType}}`)

	st.Add("op2Imm_2I", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.ImmName}} {{.ImmType}}) {{.GoType}}`)

	st.Add("op2Imm8_II", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} should be between 0 and 3, inclusive; other values may result in a runtime panic.
//
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, {{.Op2NameAndType "y"}}) {{.GoType}}`)

	st.Add("op2Imm8_SHA1RNDS4", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, {{.Op2NameAndType "y"}}) {{.GoType}}`)

	st.Add("op3Imm8", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, {{.Op2NameAndType "y"}}, {{.Op3NameAndType "z"}}) {{.GoType}}`)

	st.Add("op3Imm8_2I", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.ImmName}} {{.ImmType}}, {{.Op3NameAndType "z"}}) {{.GoType}}`)

	st.Add("op4Imm8", `{{if .Documentation}}{{.Documentation}}
//{{end}}
// A non-constant value of {{.ImmName}} may result in significantly worse performance for this operation.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} {{.ImmType}}, {{.Op2NameAndType "y"}}, {{.Op3NameAndType "z"}}, {{.Op4NameAndType "u"}}) {{.GoType}}`)

	st.Add("mask", `// To{{.VectorCounterpart}} converts from {{.Name}} to {{.VectorCounterpart}}.
// If element i in the mask is "true", all bits in element i of the resulting
// vector will be set.
func (from {{.Name}}) To{{.VectorCounterpart}}() (to {{.VectorCounterpart}})

// asMask converts from {{.VectorCounterpart}} to {{.Name}}.
func (from {{.VectorCounterpart}}) asMask() (to {{.Name}})

func (x {{.Name}}) And(y {{.Name}}) {{.Name}}

func (x {{.Name}}) Or(y {{.Name}}) {{.Name}}
{{if .HasNot}}
func (x {{.Name}}) Not() {{.Name}}
{{end}}`)

}

func structFields(shape specexpr.Vector) string {
	if shape.Elem.Base == "mask" && CurrentArch().isSVE() {
		return fmt.Sprintf("\t%s psve\n\tvals uint%d", strings.ToLower(shape.String()), types.MaxVectorBits/8)
	}
	elemBits := int(shape.Elem.Bits)
	base := strings.ToLower(shape.Elem.Base)
	if shape.Elem.Base == "mask" {
		base = "int"
	}
	elemBase := fmt.Sprintf("%s%d", base, elemBits)
	lanes, width := types.MaxVectorBits/elemBits, types.MaxVectorBits
	if !shape.Scalable() {
		width = int(shape.Width.(specexpr.Int))
		lanes = width / elemBits
	}
	tagFieldNameS := fmt.Sprintf("%sx%d", elemBase, lanes)
	tagFieldS := fmt.Sprintf("%s v%d", tagFieldNameS, width)
	valFieldS := fmt.Sprintf("vals%s[%d]%s", strings.Repeat(" ", len(tagFieldNameS)-3), lanes, elemBase)
	return fmt.Sprintf("\t%s\n\t%s", tagFieldS, valFieldS)
}

// parseSIMDTypes groups go simd types by their vector widths, and
// returns a map whose key is the vector width (specexpr.Num), value is the simd type.
func parseSIMDTypes(ops []Operation) simdTypeMap {
	// TODO: maybe instead of going over ops, let's try go over types.yaml.
	ret := map[specexpr.Num][]simdType{}
	seen := map[string]struct{}{}
	processArg := func(arg types.Operand) {
		if arg.Class == "immediate" || arg.Class == "greg" {
			// Immediates and general-purpose registers are not encoded as vector types.
			return
		}
		if arg.Lanes != nil && *arg.Lanes <= 1 {
			// Scalar vreg operands (e.g. float32/float64) are not SIMD vector types.
			return
		}
		if _, ok := seen[*arg.Go]; ok {
			return
		}
		seen[*arg.Go] = struct{}{}

		base := *arg.Base
		if arg.Class == "mask" {
			base = "mask"
		}
		elem := specexpr.Basic{
			Base: base,
			Bits: specexpr.Int(*arg.ElemBits),
		}
		width := arg.Bits.Num()
		shape := specexpr.Vector{Elem: elem, Width: width}

		fields := structFields(shape)
		hasNot := CurrentArch().Arch == "arm64"
		ret[width] = append(ret[width], simdType{Shape: shape, Fields: fields, HasNot: hasNot})
		if !shape.Scalable() && shape.Elem.Base == "mask" {
			// In case the vector counterpart of a fixed-width mask is not present, put its vector counterpart typedef into the map as well.
			vCounterpartShape := specexpr.Vector{
				Elem:  specexpr.Basic{Base: "int", Bits: elem.Bits},
				Width: width,
			}
			vcName := vCounterpartShape.String()
			if _, ok := seen[vcName]; !ok {
				seen[vcName] = struct{}{}
				ret[width] = append(ret[width], simdType{
					Shape:  vCounterpartShape,
					Fields: structFields(vCounterpartShape),
					HasNot: hasNot,
				})
			}
		}
	}
	for _, op := range ops {
		for _, arg := range op.In {
			processArg(arg)
		}
		for _, arg := range op.Out {
			processArg(arg)
		}
	}
	for _, v := range ret {
		slices.SortFunc(v, compareSimdTypes)
	}

	return ret
}

func vConvertFromTypeMap(typeMap simdTypeMap) []simdTypePair {
	v := []simdTypePair{}
	for _, ts := range typeMap {
		for i, tsrc := range ts {
			for j, tdst := range ts {
				if i != j && !tsrc.IsMask() && !tdst.IsMask() {
					v = append(v, simdTypePair{tsrc, tdst})
				}
			}
		}
	}
	slices.SortFunc(v, compareSimdTypePairs)
	return v
}

func masksFromTypeMap(typeMap simdTypeMap) []simdType {
	m := []simdType{}
	for _, ts := range typeMap {
		for _, tsrc := range ts {
			if tsrc.IsMask() {
				m = append(m, tsrc)
			}
		}
	}
	slices.SortFunc(m, compareSimdTypes)
	return m
}

func typesFromTypeMap(typeMap simdTypeMap) []simdType {
	m := []simdType{}
	for _, ts := range typeMap {
		m = append(m, ts...)
	}
	slices.SortFunc(m, compareSimdTypes)
	return m
}

// writeSIMDTypes generates the simd vector types into a bytes.Buffer
func writeSIMDTypes(buffer *bytes.Buffer, typeMap simdTypeMap) {
	t := templateOf(simdTypesTemplates, "types_amd64")
	loadStore := templateOf(simdLoadStoreTemplate, "loadstore_amd64")
	scalableLoadStore := templateOf(simdScalableLoadStoreTemplate, "loadstore_scalable")
	maskedLoadStore := templateOf(simdMaskedLoadStoreTemplate, "maskedloadstore_amd64")
	maskFromVal := templateOf(simdMaskFromValTemplate, "maskFromVal_amd64")

	buffer.WriteString(simdPackageHeader())

	if CurrentArch().isSVE() {
		// SVE predicates are represented as-is (a P register), not as data vectors,
		// so their Go types are tagged with psve rather than a v<N> vector tag.
		buffer.WriteString(`
// psve is a tag type that tells the compiler that this is an SVE predicate.
type psve struct {
	_sve [0]func() // uncomparable
}
`)
	}

	widths := slices.SortedFunc(maps.Keys(typeMap), compareWidths)

	for _, width := range widths {
		if wInt, ok := width.(specexpr.Int); ok {
			if err := t.ExecuteTemplate(buffer, "sizeTmpl", int(wInt)); err != nil {
				panic(fmt.Errorf("failed to execute size template for size %d: %w", wInt, err))
			}
		} else if CurrentArch().isSVE() {
			if err := t.ExecuteTemplate(buffer, "sizeTmpl", types.MaxVectorBits); err != nil {
				panic(fmt.Errorf("failed to execute size template for size %d: %w", types.MaxVectorBits, err))
			}
		}
		for _, typeDef := range typeMap[width] {
			typeTmplName := "typeTmpl"
			if typeDef.IsScalable() {
				typeTmplName = "scalableTypeTmpl"
			}
			if err := t.ExecuteTemplate(buffer, typeTmplName, typeDef); err != nil {
				panic(fmt.Errorf("failed to execute type template for type %s: %w", typeDef.Name(), err))
			}
			if !typeDef.IsMask() {
				// Scalable (SVE) types get a partial, slice-based load/store; the
				// fixed-width types get the array load/store.
				ls, lsName := loadStore, "loadstore_amd64"
				if typeDef.IsScalable() {
					ls, lsName = scalableLoadStore, "loadstore_scalable"
				}
				if err := ls.ExecuteTemplate(buffer, lsName, typeDef); err != nil {
					panic(fmt.Errorf("failed to execute loadstore template for type %s: %w", typeDef.Name(), err))
				}
				// restrict to AVX2 masked loads/stores first.
				if CurrentArch().Arch == "amd64" && typeDef.MaskedLoadStoreFilter() {
					if err := maskedLoadStore.ExecuteTemplate(buffer, "maskedloadstore_amd64", typeDef); err != nil {
						panic(fmt.Errorf("failed to execute maskedloadstore template for type %s: %w", typeDef.Name(), err))
					}
				}
			} else if CurrentArch().isSVE() {
				// SVE predicates expose raw-bit memory APIs: exported wrappers that
				// bounds-check (and may panic) around unexported PLDR/PSTR intrinsics.
				if err := t.ExecuteTemplate(buffer, "sveMaskLoadStore", typeDef); err != nil {
					panic(fmt.Errorf("failed to execute sveMaskLoadStore template for type %s: %w", typeDef.Name(), err))
				}
			} else {
				// ARM64 NEON comparisons produce all-0/all-1 per lane, so
				// FromBits/ToBits (x86 mask register conversions) are not needed.
				if CurrentArch().Arch != "arm64" {
					if err := maskFromVal.ExecuteTemplate(buffer, "maskFromVal_amd64", typeDef); err != nil {
						panic(fmt.Errorf("failed to execute maskFromVal template for type %s: %w", typeDef.Name(), err))
					}
				}
			}
			// TODO: these type utility methods can also be generated by tmplgen, or we can move other arches from
			// tmplgen to here.
			if typeDef.IsScalable() && typeDef.Type() != "mask" {
				if err := t.ExecuteTemplate(buffer, "sveIfElseTmpl", typeDef); err != nil {
					panic(fmt.Errorf("failed to execute sveIfElseTmpl template for type %s: %w", typeDef.Name(), err))
				}
			}
			if typeDef.IsScalable() {
				if err := t.ExecuteTemplate(buffer, "sveStringTmpl", typeDef); err != nil {
					panic(fmt.Errorf("failed to execute sveStringTmpl template for type %s: %w", typeDef.Name(), err))
				}
			}
		}
	}
}

type goarchFeatures struct {
	// featureVar is the name of the exported feature-check variable for this
	// architecture.
	featureVar string

	// features records per-feature information.
	features map[string]featureInfo
}

type featureInfo struct {
	// Implies is a list of other CPU features that are required for this
	// feature. These are allowed to chain.
	//
	// For example, if the Frob feature lists "Baz", then if X.Frob() returns
	// true, it must also be true that the CPU has feature Baz.
	Implies []string

	// Virtual means this feature is not represented directly in internal/cpu,
	// but is instead the logical AND of the features in Implies.
	Virtual bool
}

// goarchFeatureInfo maps from GOARCH to CPU feature to additional information
// about that feature. Not all features need to be in this map.
var goarchFeatureInfo = make(map[string]goarchFeatures)

func registerFeatureInfo(goArch string, features goarchFeatures) {
	goarchFeatureInfo[goArch] = features
}

func featureImplies(goarch string, base string) string {
	// Compute the transitive closure of base.
	var list []string
	var visit func(f string)
	visit = func(f string) {
		list = append(list, f)
		for _, dep := range goarchFeatureInfo[goarch].features[f].Implies {
			visit(dep)
		}
	}
	visit(base)
	// Drop base
	list = list[1:]
	// Put in "nice" order
	slices.Reverse(list)
	// Combine into a comment-ready form
	switch len(list) {
	case 0:
		return ""
	case 1:
		return list[0]
	case 2:
		return list[0] + " and " + list[1]
	default:
		list[len(list)-1] = "and " + list[len(list)-1]
		return strings.Join(list, ", ")
	}
}

func writeSIMDFeatures(buffer *bytes.Buffer, ops []Operation) {
	// Gather all features
	type featureKey struct {
		GoArch  string
		Feature string
	}
	featureSet := make(map[featureKey]struct{})
	for _, op := range ops {
		// Generate a feature check for each independant feature in a
		// composite feature.
		for feature := range strings.SplitSeq(op.CPUFeature, ",") {
			feature = strings.TrimSpace(feature)
			featureSet[featureKey{op.GOARCH, feature}] = struct{}{}
		}
	}
	featureKeys := slices.SortedFunc(maps.Keys(featureSet), func(a, b featureKey) int {
		if c := cmp.Compare(a.GoArch, b.GoArch); c != 0 {
			return c
		}
		return compareNatural(a.Feature, b.Feature)
	})

	// TODO: internal/cpu doesn't enforce these at all. You can even do
	// GODEBUG=cpu.avx=off and it will happily turn off AVX without turning off
	// AVX2. We need to push these dependencies into it somehow.
	type feature struct {
		featureKey
		FeatureVar string
		Virtual    bool
		Implies    []string
		ImpliesAll string
	}
	var features []feature
	for _, k := range featureKeys {
		featureVar := goarchFeatureInfo[k.GoArch].featureVar
		fi := goarchFeatureInfo[k.GoArch].features[k.Feature]
		features = append(features, feature{
			featureKey: k,
			FeatureVar: featureVar,
			Virtual:    fi.Virtual,
			Implies:    fi.Implies,
			ImpliesAll: featureImplies(k.GoArch, k.Feature),
		})
	}

	// If we ever have the same feature name on more than one GOARCH, we'll have
	// to be more careful about this.
	t := templateOf(simdFeaturesTemplate, "features")

	buffer.WriteString(simdPackageHeader())

	if err := t.Execute(buffer, features); err != nil {
		panic(fmt.Errorf("failed to execute features template: %w", err))
	}
}

// writeSIMDStubs returns two bytes.Buffers containing the declarations for the public
// and internal-use vector intrinsics.
func writeSIMDStubs(f, fI *bytes.Buffer, ops []Operation, typeMap simdTypeMap, doDeprecatedPuns bool) {
	f.WriteString(simdPackageHeader())
	fI.WriteString(simdPackageHeader())

	slices.SortFunc(ops, compareOperations)

	for i, op := range ops {
		if op.NoTypes != nil && *op.NoTypes == "true" {
			continue
		}
		if op.SkipMaskedMethod() {
			continue
		}
		idxVecAsScalar, err := checkVecAsScalar(op)
		if err != nil {
			panic(err)
		}
		if s, op, err := classifyOp(op); err == nil {
			if op.NoGenericOps != nil && *op.NoGenericOps == "true" {
				continue
			}
			if idxVecAsScalar != -1 {
				if s == "op2" || s == "op3" || s == "op2Imm" {
					s += "VecAsScalar"
				} else {
					panic(fmt.Errorf("simdgen only supports op2, op2Imm or op3, not %s with TreatLikeAScalarOfSize", s))
				}
			}
			if i == 0 || op.Go != ops[i-1].Go {
				if unicode.IsUpper([]rune(op.Go)[0]) {
					fmt.Fprintf(f, "\n/* %s */\n", op.Go)
				} else {
					fmt.Fprintf(fI, "\n/* %s */\n", op.Go)
				}
			}
			tpl := stubTemplates.Get(s)
			if tpl == nil {
				panic(fmt.Errorf("template %s not found", s))
			}
			if unicode.IsUpper([]rune(op.Go)[0]) {
				if err := tpl.Execute(f, op); err != nil {
					panic(fmt.Errorf("failed to execute template %s for op %v: %w", s, op, err))
				}
			} else {
				if err := tpl.Execute(fI, op); err != nil {
					panic(fmt.Errorf("failed to execute template %s for op %v: %w", s, op, err))
				}
			}
		} else {
			panic(fmt.Errorf("failed to classify op %v: %w", op.Go, err))
		}
	}

	vectorConversions := vConvertFromTypeMap(typeMap)
	for _, conv := range vectorConversions {
		from, to := &conv.Tsrc, &conv.Tdst

		if doDeprecatedPuns {
			if err := sgutil.AsOp.Execute(f, sgutil.Conversion(from, to)); err != nil {
				panic(fmt.Errorf("failed to execute vectorConversion template: %w", err))
			}
		}

		// New style factored conversion intrinsics
		if from.Name()[0] != 'U' && to.Name()[0] != 'U' {
			continue
		}
		// Only emit the intrinsic if element sizes are equal OR both are unsigned
		if from.ElemBits() != to.ElemBits() && (from.Name()[0] != 'U' || to.Name()[0] != 'U') {
			continue
		}
		switch to.Name()[0] {
		case 'F': // U -> F
			sgutil.ToFloatsDcl.Execute(f, sgutil.Conversion(from, to))
			sgutil.ToBitsDcl.Execute(f, sgutil.Conversion(to, from))
		case 'I': // U -> I
			sgutil.ToIntsDcl.Execute(f, sgutil.Conversion(from, to))
			sgutil.ToBitsDcl.Execute(f, sgutil.Conversion(to, from))
		case 'U': // U -> U
			if from.Name()[0] != 'U' {
				continue
			}
			sgutil.ReshapeDcl.Execute(f, sgutil.Conversion(from, to))
		default:
			panic("unexpected type in reinterpret-declaration")
		}
	}

	// The AVX mask stub declares mask methods (To/asMask/And/Or/Not) that treat a
	// mask as its data-vector counterpart. An SVE predicate is a P-register with no
	// such counterpart; its raw-bit memory APIs (Load/Store) are generated into
	// types_sve.go instead, so nothing is emitted here for SVE masks.
	if !CurrentArch().isSVE() {
		for _, mask := range masksFromTypeMap(typeMap) {
			tpl := stubTemplates.Get("mask")
			if tpl == nil {
				panic(fmt.Errorf("template mask not found"))
			}
			if err := tpl.Execute(f, mask); err != nil {
				panic(fmt.Errorf("failed to execute mask template for mask %s: %w", mask.Name(), err))
			}
		}
	}
}
