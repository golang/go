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
	"sort"
	"strings"
	"unicode"
)

type simdType struct {
	Name                    string // The go type name of this simd type, for example Int32x4.
	Lanes                   int    // The number of elements in this vector/mask.
	Base                    string // The element's type, like for Int32x4 it will be int32.
	Fields                  string // The struct fields, it should be right formatted.
	Type                    string // Either "mask" or "vreg"
	VectorCounterpart       string // For mask use only: just replacing the "Mask" in [simdType.Name] with "Int"
	ReshapedVectorWithAndOr string // For mask use only: vector AND and OR are only available in some shape with element width 32.
	Size                    int    // The size of the vector type
}

func (x simdType) ElemBits() int {
	return x.Size / x.Lanes
}

// LanesContainer returns the smallest int/uint bit size that is
// large enough to hold one bit for each lane.  E.g., Mask32x4
// is 4 lanes, and a uint8 is the smallest uint that has 4 bits.
func (x simdType) LanesContainer() int {
	if x.Lanes > 64 {
		panic("too many lanes")
	}
	if x.Lanes > 32 {
		return 64
	}
	if x.Lanes > 16 {
		return 32
	}
	if x.Lanes > 8 {
		return 16
	}
	return 8
}

// MaskedLoadStoreFilter encodes which simd type type currently
// get masked loads/stores generated, it is used in two places,
// this forces coordination.
func (x simdType) MaskedLoadStoreFilter() bool {
	return x.Size == 512 || x.ElemBits() >= 32 && x.Type != "mask"
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
	if x.Size == 512 || x.ElemBits() < 32 {
		return fmt.Sprintf("// Asm: VMOVDQU%d.Z, CPU Feature: AVX512", x.ElemBits())
	} else {
		return fmt.Sprintf("// Asm: VMASKMOV%s, CPU Feature: AVX2", x.IntelSizeSuffix())
	}
}

func (x simdType) MaskedStoreDoc() string {
	if x.Size == 512 || x.ElemBits() < 32 {
		return fmt.Sprintf("// Asm: VMOVDQU%d, CPU Feature: AVX512", x.ElemBits())
	} else {
		return fmt.Sprintf("// Asm: VMASKMOV%s, CPU Feature: AVX2", x.IntelSizeSuffix())
	}
}

func compareSimdTypes(x, y simdType) int {
	// "vreg" then "mask"
	if c := -compareNatural(x.Type, y.Type); c != 0 {
		return c
	}
	// want "flo" < "int" < "uin" (and then 8 < 16 < 32 < 64),
	// not "int16" < "int32" < "int64" < "int8")
	// so limit comparison to first 3 bytes in string.
	if c := compareNatural(x.Base[:3], y.Base[:3]); c != 0 {
		return c
	}
	// base type size, 8 < 16 < 32 < 64
	if c := x.ElemBits() - y.ElemBits(); c != 0 {
		return c
	}
	// vector size last
	return x.Size - y.Size
}

type simdTypeMap map[int][]simdType

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

const simdPackageHeader = generatedHeader + `
//go:build goexperiment.simd

package simd
`

const simdTypesTemplates = `
{{define "sizeTmpl"}}
// v{{.}} is a tag type that tells the compiler that this is really {{.}}-bit SIMD
type v{{.}} struct {
	_{{.}} [0]func() // uncomparable
}
{{end}}

{{define "typeTmpl"}}
// {{.Name}} is a {{.Size}}-bit SIMD vector of {{.Lanes}} {{.Base}}
type {{.Name}} struct {
{{.Fields}}
}

{{end}}
`

const simdFeaturesTemplate = `
import "internal/cpu"

type X86Features struct {}

var X86 X86Features

{{range .}}
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
//
// {{.Feature}} is defined on all GOARCHes, but will only return true on
// GOARCH {{.GoArch}}.
func (X86Features) {{.Feature}}() bool {
	return cpu.X86.Has{{.Feature}}
}
{{end}}
`

const simdLoadStoreTemplate = `
// Len returns the number of elements in a {{.Name}}
func (x {{.Name}}) Len() int { return {{.Lanes}} }

// Load{{.Name}} loads a {{.Name}} from an array
//
//go:noescape
func Load{{.Name}}(y *[{{.Lanes}}]{{.Base}}) {{.Name}}

// Store stores a {{.Name}} to an array
//
//go:noescape
func (x {{.Name}}) Store(y *[{{.Lanes}}]{{.Base}})
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
// Asm: KMOV{{.IntelSizeSuffix}}, CPU Features: AVX512
func (x {{.Name}}) ToBits() uint{{.LanesContainer}}
`

const simdMaskedLoadStoreTemplate = `
// LoadMasked{{.Name}} loads a {{.Name}} from an array,
// at those elements enabled by mask
//
{{.MaskedLoadDoc}}
//
//go:noescape
func LoadMasked{{.Name}}(y *[{{.Lanes}}]{{.Base}}, mask Mask{{.ElemBits}}x{{.Lanes}}) {{.Name}}

// StoreMasked stores a {{.Name}} to an array,
// at those elements enabled by mask
//
{{.MaskedStoreDoc}}
//
//go:noescape
func (x {{.Name}}) StoreMasked(y *[{{.Lanes}}]{{.Base}}, mask Mask{{.ElemBits}}x{{.Lanes}})
`

const simdStubsTmpl = `
{{define "op1"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}() {{.GoType}}
{{end}}

{{define "op2"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}) {{.GoType}}
{{end}}

{{define "op2_21"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}) {{.GoType}}
{{end}}

{{define "op2_21Type1"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}) {{.GoType}}
{{end}}

{{define "op3"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op2NameAndType "z"}}) {{.GoType}}
{{end}}

{{define "op3_31"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op2NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op0NameAndType "z"}}) {{.GoType}}
{{end}}

{{define "op3_21"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}, {{.Op2NameAndType "z"}}) {{.GoType}}
{{end}}

{{define "op3_21Type1"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op0NameAndType "y"}}, {{.Op2NameAndType "z"}}) {{.GoType}}
{{end}}

{{define "op3_231Type1"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.Op0NameAndType "z"}}) {{.GoType}}
{{end}}

{{define "op2VecAsScalar"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}(y uint{{(index .In 1).TreatLikeAScalarOfSize}}) {{(index .Out 0).Go}}
{{end}}

{{define "op3VecAsScalar"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}(y uint{{(index .In 1).TreatLikeAScalarOfSize}}, {{.Op2NameAndType "z"}}) {{(index .Out 0).Go}}
{{end}}

{{define "op4"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op0NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op2NameAndType "z"}}, {{.Op3NameAndType "u"}}) {{.GoType}}
{{end}}

{{define "op4_231Type1"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.Op0NameAndType "z"}}, {{.Op3NameAndType "u"}}) {{.GoType}}
{{end}}

{{define "op4_31"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op2NameAndType "x"}}) {{.Go}}({{.Op1NameAndType "y"}}, {{.Op0NameAndType "z"}}, {{.Op3NameAndType "u"}}) {{.GoType}}
{{end}}

{{define "op1Imm8"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} uint8) {{.GoType}}
{{end}}

{{define "op2Imm8"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} uint8, {{.Op2NameAndType "y"}}) {{.GoType}}
{{end}}

{{define "op2Imm8_2I"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.ImmName}} uint8) {{.GoType}}
{{end}}

{{define "op2Imm8_II"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} result in better performance when they are constants, non-constant values will be translated into a jump table.
// {{.ImmName}} should be between 0 and 3, inclusive; other values will result in a runtime panic.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} uint8, {{.Op2NameAndType "y"}}) {{.GoType}}
{{end}}

{{define "op2Imm8_SHA1RNDS4"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} uint8, {{.Op2NameAndType "y"}}) {{.GoType}}
{{end}}

{{define "op3Imm8"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} uint8, {{.Op2NameAndType "y"}}, {{.Op3NameAndType "z"}}) {{.GoType}}
{{end}}

{{define "op3Imm8_2I"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.Op2NameAndType "y"}}, {{.ImmName}} uint8, {{.Op3NameAndType "z"}}) {{.GoType}}
{{end}}


{{define "op4Imm8"}}
{{if .Documentation}}{{.Documentation}}
//{{end}}
// {{.ImmName}} results in better performance when it's a constant, a non-constant value will be translated into a jump table.
//
// Asm: {{.Asm}}, CPU Feature: {{.CPUFeature}}
func ({{.Op1NameAndType "x"}}) {{.Go}}({{.ImmName}} uint8, {{.Op2NameAndType "y"}}, {{.Op3NameAndType "z"}}, {{.Op4NameAndType "u"}}) {{.GoType}}
{{end}}

{{define "vectorConversion"}}
// {{.Tdst.Name}} converts from {{.Tsrc.Name}} to {{.Tdst.Name}}
func (from {{.Tsrc.Name}}) As{{.Tdst.Name}}() (to {{.Tdst.Name}})
{{end}}

{{define "mask"}}
// As{{.VectorCounterpart}} converts from {{.Name}} to {{.VectorCounterpart}}
func (from {{.Name}}) As{{.VectorCounterpart}}() (to {{.VectorCounterpart}})

// asMask converts from {{.VectorCounterpart}} to {{.Name}}
func (from {{.VectorCounterpart}}) asMask() (to {{.Name}})

func (x {{.Name}}) And(y {{.Name}}) {{.Name}}

func (x {{.Name}}) Or(y {{.Name}}) {{.Name}}
{{end}}
`

// parseSIMDTypes groups go simd types by their vector sizes, and
// returns a map whose key is the vector size, value is the simd type.
func parseSIMDTypes(ops []Operation) simdTypeMap {
	// TODO: maybe instead of going over ops, let's try go over types.yaml.
	ret := map[int][]simdType{}
	seen := map[string]struct{}{}
	processArg := func(arg Operand) {
		if arg.Class == "immediate" || arg.Class == "greg" {
			// Immediates are not encoded as vector types.
			return
		}
		if _, ok := seen[*arg.Go]; ok {
			return
		}
		seen[*arg.Go] = struct{}{}

		lanes := *arg.Lanes
		base := fmt.Sprintf("%s%d", *arg.Base, *arg.ElemBits)
		tagFieldNameS := fmt.Sprintf("%sx%d", base, lanes)
		tagFieldS := fmt.Sprintf("%s v%d", tagFieldNameS, *arg.Bits)
		valFieldS := fmt.Sprintf("vals%s[%d]%s", strings.Repeat(" ", len(tagFieldNameS)-3), lanes, base)
		fields := fmt.Sprintf("\t%s\n\t%s", tagFieldS, valFieldS)
		if arg.Class == "mask" {
			vectorCounterpart := strings.ReplaceAll(*arg.Go, "Mask", "Int")
			reshapedVectorWithAndOr := fmt.Sprintf("Int32x%d", *arg.Bits/32)
			ret[*arg.Bits] = append(ret[*arg.Bits], simdType{*arg.Go, lanes, base, fields, arg.Class, vectorCounterpart, reshapedVectorWithAndOr, *arg.Bits})
			// In case the vector counterpart of a mask is not present, put its vector counterpart typedef into the map as well.
			if _, ok := seen[vectorCounterpart]; !ok {
				seen[vectorCounterpart] = struct{}{}
				ret[*arg.Bits] = append(ret[*arg.Bits], simdType{vectorCounterpart, lanes, base, fields, "vreg", "", "", *arg.Bits})
			}
		} else {
			ret[*arg.Bits] = append(ret[*arg.Bits], simdType{*arg.Go, lanes, base, fields, arg.Class, "", "", *arg.Bits})
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
	return ret
}

func vConvertFromTypeMap(typeMap simdTypeMap) []simdTypePair {
	v := []simdTypePair{}
	for _, ts := range typeMap {
		for i, tsrc := range ts {
			for j, tdst := range ts {
				if i != j && tsrc.Type == tdst.Type && tsrc.Type == "vreg" &&
					tsrc.Lanes > 1 && tdst.Lanes > 1 {
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
			if tsrc.Type == "mask" {
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
		for _, tsrc := range ts {
			if tsrc.Lanes > 1 {
				m = append(m, tsrc)
			}
		}
	}
	slices.SortFunc(m, compareSimdTypes)
	return m
}

// writeSIMDTypes generates the simd vector types into a bytes.Buffer
func writeSIMDTypes(typeMap simdTypeMap) *bytes.Buffer {
	t := templateOf(simdTypesTemplates, "types_amd64")
	loadStore := templateOf(simdLoadStoreTemplate, "loadstore_amd64")
	maskedLoadStore := templateOf(simdMaskedLoadStoreTemplate, "maskedloadstore_amd64")
	maskFromVal := templateOf(simdMaskFromValTemplate, "maskFromVal_amd64")

	buffer := new(bytes.Buffer)
	buffer.WriteString(simdPackageHeader)

	sizes := make([]int, 0, len(typeMap))
	for size, types := range typeMap {
		slices.SortFunc(types, compareSimdTypes)
		sizes = append(sizes, size)
	}
	sort.Ints(sizes)

	for _, size := range sizes {
		if size <= 64 {
			// these are scalar
			continue
		}
		if err := t.ExecuteTemplate(buffer, "sizeTmpl", size); err != nil {
			panic(fmt.Errorf("failed to execute size template for size %d: %w", size, err))
		}
		for _, typeDef := range typeMap[size] {
			if typeDef.Lanes == 1 {
				continue
			}
			if err := t.ExecuteTemplate(buffer, "typeTmpl", typeDef); err != nil {
				panic(fmt.Errorf("failed to execute type template for type %s: %w", typeDef.Name, err))
			}
			if typeDef.Type != "mask" {
				if err := loadStore.ExecuteTemplate(buffer, "loadstore_amd64", typeDef); err != nil {
					panic(fmt.Errorf("failed to execute loadstore template for type %s: %w", typeDef.Name, err))
				}
				// restrict to AVX2 masked loads/stores first.
				if typeDef.MaskedLoadStoreFilter() {
					if err := maskedLoadStore.ExecuteTemplate(buffer, "maskedloadstore_amd64", typeDef); err != nil {
						panic(fmt.Errorf("failed to execute maskedloadstore template for type %s: %w", typeDef.Name, err))
					}
				}
			} else {
				if err := maskFromVal.ExecuteTemplate(buffer, "maskFromVal_amd64", typeDef); err != nil {
					panic(fmt.Errorf("failed to execute maskFromVal template for type %s: %w", typeDef.Name, err))
				}
			}
		}
	}

	return buffer
}

func writeSIMDFeatures(ops []Operation) *bytes.Buffer {
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
			featureSet[featureKey{op.GoArch, feature}] = struct{}{}
		}
	}
	features := slices.SortedFunc(maps.Keys(featureSet), func(a, b featureKey) int {
		if c := cmp.Compare(a.GoArch, b.GoArch); c != 0 {
			return c
		}
		return compareNatural(a.Feature, b.Feature)
	})

	// If we ever have the same feature name on more than one GOARCH, we'll have
	// to be more careful about this.
	t := templateOf(simdFeaturesTemplate, "features")

	buffer := new(bytes.Buffer)
	buffer.WriteString(simdPackageHeader)

	if err := t.Execute(buffer, features); err != nil {
		panic(fmt.Errorf("failed to execute features template: %w", err))
	}

	return buffer
}

// writeSIMDStubs returns two bytes.Buffers containing the declarations for the public
// and internal-use vector intrinsics.
func writeSIMDStubs(ops []Operation, typeMap simdTypeMap) (f, fI *bytes.Buffer) {
	t := templateOf(simdStubsTmpl, "simdStubs")
	f = new(bytes.Buffer)
	fI = new(bytes.Buffer)
	f.WriteString(simdPackageHeader)
	fI.WriteString(simdPackageHeader)

	slices.SortFunc(ops, compareOperations)

	for i, op := range ops {
		if op.NoTypes != nil && *op.NoTypes == "true" {
			continue
		}
		idxVecAsScalar, err := checkVecAsScalar(op)
		if err != nil {
			panic(err)
		}
		if s, op, err := classifyOp(op); err == nil {
			if idxVecAsScalar != -1 {
				if s == "op2" || s == "op3" {
					s += "VecAsScalar"
				} else {
					panic(fmt.Errorf("simdgen only supports op2 or op3 with TreatLikeAScalarOfSize"))
				}
			}
			if i == 0 || op.Go != ops[i-1].Go {
				if unicode.IsUpper([]rune(op.Go)[0]) {
					fmt.Fprintf(f, "\n/* %s */\n", op.Go)
				} else {
					fmt.Fprintf(fI, "\n/* %s */\n", op.Go)
				}
			}
			if unicode.IsUpper([]rune(op.Go)[0]) {
				if err := t.ExecuteTemplate(f, s, op); err != nil {
					panic(fmt.Errorf("failed to execute template %s for op %v: %w", s, op, err))
				}
			} else {
				if err := t.ExecuteTemplate(fI, s, op); err != nil {
					panic(fmt.Errorf("failed to execute template %s for op %v: %w", s, op, err))
				}
			}
		} else {
			panic(fmt.Errorf("failed to classify op %v: %w", op.Go, err))
		}
	}

	vectorConversions := vConvertFromTypeMap(typeMap)
	for _, conv := range vectorConversions {
		if err := t.ExecuteTemplate(f, "vectorConversion", conv); err != nil {
			panic(fmt.Errorf("failed to execute vectorConversion template: %w", err))
		}
	}

	masks := masksFromTypeMap(typeMap)
	for _, mask := range masks {
		if err := t.ExecuteTemplate(f, "mask", mask); err != nil {
			panic(fmt.Errorf("failed to execute mask template for mask %s: %w", mask.Name, err))
		}
	}

	return
}
