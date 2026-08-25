// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"simd/archsimd/_gen/sgutil"
	"slices"
	"text/template"
)

// Helper type to make template map initialization less repetitive
// (and also remove a chance for errors.)
type intrinsicTemplateMap struct {
	sgutil.InsertMap[string, *template.Template]
}

func templateNamed(name string, templ string) *template.Template {
	// Append  end of line
	templ += "\n"

	t := template.New(name)

	archInfo := CurrentArch()
	sysArch := "sys." + archInfo.ArchUpper

	t.Funcs(template.FuncMap{
		"GetSysArch": func() string {
			return sysArch
		},
		"GetArchUpper": func() string {
			return archInfo.ArchUpper
		},
		"GetSIMDTag": func() string {
			return archInfo.SIMDTag
		},
		"Hasmask": func() bool {
			return archInfo.Arch == "amd64"
		},
	})

	return template.Must(t.Parse(templ))
}

// Add creates a template named "name" after appending "\n" to the
// template, and returns the input so that additions may be chained.
// This helps make template initialization easy to order and easy to read.
func (rtm *intrinsicTemplateMap) Add(name string, templ string) *intrinsicTemplateMap {

	rtm.InsertMap.Put(name, templateNamed(name, templ))
	return rtm
}

// writeSIMDIntrinsics generates the intrinsic mappings and writes it to simdintrinsics.go
// within the specified directory.
func writeSIMDIntrinsics(buffer *bytes.Buffer, ops []Operation, typeMap simdTypeMap) {

	// These are defined here to avoid init-order problems with GetSysArch GetArchUpper etc which depend on flag values

	var header = templateNamed("header", `package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

func simd{{GetSIMDTag}}Intrinsics(addF func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily)) {
`)

	var intrinsicTemplates = new(intrinsicTemplateMap).
		Add("op1", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen1(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op2", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op2_21", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2_21(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op2_21Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2_21(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_21", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3_21(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_21Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_21(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_231Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_231(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_31Zero3", `addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen3_31Zero3(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op4", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen4(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op4_231Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4_231(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op4_31", `		addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen4_31(ssaop.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op1Imm", `		addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op1Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm8(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm", `		addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op2Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm8_2I", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_2I(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm_2I", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm_2I(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op2Imm8_II", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_II(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm8_SHA1RNDS4", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_SHA1RNDS4(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2ImmVecAsScalar", `addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen2Imm(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op3Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op3Imm8_2I", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8_2I(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op4Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4Imm8(ssaop.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`)

	var loadStore = templateNamed("loadStore", `	addF(simdPackage, "Load{{.Name}}Array", simdLoad(), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.StoreArray", simdStore(), {{GetSysArch}})`)

	var mask = templateNamed("mask", `	addF(simdPackage, "{{.Name}}.To{{.VectorCounterpart}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
	addF(simdPackage, "{{.VectorCounterpart}}.asMask", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.And", opLen2(ssaop.OpAnd{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.Or", opLen2(ssaop.OpOr{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
{{- if eq GetSysArch "sys.ARM64"}}
	addF(simdPackage, "{{.Name}}.Not", opLen1(ssaop.OpNot{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
{{- else}}
	addF(simdPackage, "{{.Name}}FromBits", simdCvtVToMask({{.ElemBits}}, {{.Lanes}}), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.ToBits", simdCvtMaskToV({{.ElemBits}}, {{.Lanes}}), {{GetSysArch}})
{{- end}}`)

	// SVE predicates are P-registers, moved to/from memory by the hand-written
	// sveLoadWhole/sveStoreWhole builders (a generic Load/Store of a mask value,
	// lowered to PLDR/PSTR); only this registration of the raw intrinsics is
	// generated (the exported Load/Store wrappers are generated Go in types_sve.go).
	var sveMask = templateNamed("sveMask", `	addF(simdPackage, "{{.Name}}.store", sveStoreWhole(), {{GetSysArch}})
	addF(simdPackage, "load{{.Name}}", sveLoadWhole(), {{GetSysArch}})`)

	var maskedLoadStore = templateNamed("maskedLoadStore", `	addF(simdPackage, "{{.Name}}.StoreArrayMasked", simdMaskedStore(ssaop.OpStoreMasked{{.ElemBits}}), sys.AMD64)`)

	var vectorConversion = templateNamed("vectorConversion", `	addF(simdPackage, "{{.Tsrc.Name}}.As{{.Tdst.Name}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})`)

	var footer = `}`

	slices.SortFunc(ops, compareOperations)

	buffer.WriteString(generatedHeader())

	doTemplate := func(tpl *template.Template, data any) {
		if err := tpl.Execute(buffer, data); err != nil {
			panic(fmt.Errorf("failed to execute template %s: %w", tpl.Name(), err))
		}
	}

	doTemplate(header, nil)

	doIntrinsic := func(name string, data any) {
		tpl := intrinsicTemplates.Get(name)
		if tpl == nil {
			panic(fmt.Errorf("template %s not found", name))
		}
		doTemplate(tpl, data)
	}

	for _, op := range ops {
		if op.NoTypes != nil && *op.NoTypes == "true" {
			continue
		}
		if op.SkipMaskedMethod() {
			continue
		}
		// Cannot have an intrinsic w/o generics, at least for now.
		if op.NoGenericOps != nil && *op.NoGenericOps == "true" {
			continue
		}
		if s, op, err := classifyOp(op); err == nil {
			if s == "op2Imm" {
				idxVecAsScalar, err := checkVecAsScalar(op)
				if err != nil {
					panic(err)
				}
				if idxVecAsScalar >= 0 {
					s += "VecAsScalar"
				}
			}
			doIntrinsic(s, op)
		} else {
			panic(fmt.Errorf("failed to classify op %v: %w", op.Go, err))
		}
	}

	var TypeDotMethodIntrinsicAMD64 = templateOf(`addF(simdPackage, "{{.TypeDotMethod}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, sys.AMD64)
	`, "amd64 type dot method intrinsics")

	var TypeDotMethodIntrinsicARM64 = templateOf(`addF(simdPackage, "{{.TypeDotMethod}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, sys.ARM64)
	`, "arm64 type dot method intrinsics")

	for _, conv := range vConvertFromTypeMap(typeMap) {
		// Old As intrinsic
		from, to := &conv.Tsrc, &conv.Tdst
		doTemplate(vectorConversion, conv)

		// New style factored conversion intrinsics always involve at least one unsigned type
		if from.Name()[0] != 'U' && to.Name()[0] != 'U' {
			continue
		}
		// Only emit the intrinsic if element sizes are equal OR both are unsigned
		if from.ElemBits() != to.ElemBits() && (from.Name()[0] != 'U' || to.Name()[0] != 'U') {
			continue
		}
		var typeDotMethodIntrinsic *template.Template
		switch CurrentArch().Arch {
		case "amd64":
			typeDotMethodIntrinsic = TypeDotMethodIntrinsicAMD64
		case "arm64":
			typeDotMethodIntrinsic = TypeDotMethodIntrinsicARM64
		default:
			panic(fmt.Errorf("unsupported arch %q for type dot method intrinsics", CurrentArch().Arch))
		}
		sgutil.Conversion(from, to).ExecuteIntrinsicTemplateOfTypeDotMethod(buffer, typeDotMethodIntrinsic)
	}

	for _, typ := range typesFromTypeMap(typeMap) {
		// Scalable (SVE) types have no fixed-array load/store; their slice-based
		// LoadPart/StorePart are hand-registered in ssagen for now.
		// TODO: generate them here once simdgen supports predicates (mask CL).
		if !typ.IsMask() && !typ.IsScalable() {
			loadStore.Execute(buffer, typ)
		}
	}

	// Masked loads/stores are AVX2/AVX512 only (not available on ARM64 NEON).
	// TODO: Reconsider for ARM64 SVE which supports predicated loads/stores.
	if CurrentArch().Arch == "amd64" {
		for _, typ := range typesFromTypeMap(typeMap) {
			if typ.MaskedLoadStoreFilter() {
				doTemplate(maskedLoadStore, typ)
			}
		}
	}

	// The AVX mask template treats a mask as a data vector (no-op To/asMask
	// conversions, And/Or/Not via reshaped vector ops, FromBits/ToBits); an SVE
	// predicate is a P-register with just the memory APIs (Store/LoadMask). The
	// predicate-consuming ops (Masked, IfElse, ...) are peephole optimizations of
	// the data-vector ops, not mask methods, so they are not generated here.
	maskTpl := mask
	if CurrentArch().isSVE() {
		maskTpl = sveMask
	}
	for _, m := range masksFromTypeMap(typeMap) {
		doTemplate(maskTpl, m)
	}

	buffer.WriteString(footer)
}
