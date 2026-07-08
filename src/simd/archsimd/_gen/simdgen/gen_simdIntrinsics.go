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
func writeSIMDIntrinsics(ops []Operation, typeMap simdTypeMap) *bytes.Buffer {

	// These are defined here to avoid init-order problems with GetSysArch GetArchUpper etc which depend on flag values

	var header = templateNamed("header", `package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

func simd{{GetArchUpper}}Intrinsics(addF func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily)) {
`)

	var intrinsicTemplates = new(intrinsicTemplateMap).
		Add("op1", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen1(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op2", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op2_21", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op2_21Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_21", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_21Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_231Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_231(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op3_31Zero3", `addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen3_31Zero3(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op4", `		addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen4(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op4_231Type1", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4_231(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op4_31", `		addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen4_31(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})`).
		Add("op1Imm", `		addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op1Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm", `		addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op2Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm8_2I", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm_2I", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op2Imm8_II", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_II(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2Imm8_SHA1RNDS4", `addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_SHA1RNDS4(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op2ImmVecAsScalar", `addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen2Imm(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})`).
		Add("op3Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op3Imm8_2I", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`).
		Add("op4Imm8", `	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})`)

	var loadStore = templateNamed("loadStore", `	addF(simdPackage, "Load{{.Name}}Array", simdLoad(), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.StoreArray", simdStore(), {{GetSysArch}})`)

	var mask = templateNamed("mask", `	addF(simdPackage, "{{.Name}}.To{{.VectorCounterpart}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
	addF(simdPackage, "{{.VectorCounterpart}}.asMask", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.And", opLen2(ssa.OpAnd{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.Or", opLen2(ssa.OpOr{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
{{- if eq GetSysArch "sys.ARM64"}}
	addF(simdPackage, "{{.Name}}.Not", opLen1(ssa.OpNot{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
{{- else}}
	addF(simdPackage, "{{.Name}}FromBits", simdCvtVToMask({{.ElemBits}}, {{.Lanes}}), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.ToBits", simdCvtMaskToV({{.ElemBits}}, {{.Lanes}}), {{GetSysArch}})
{{- end}}`)

	var maskedLoadStore = templateNamed("maskedLoadStore", `	addF(simdPackage, "{{.Name}}.StoreArrayMasked", simdMaskedStore(ssa.OpStoreMasked{{.ElemBits}}), sys.AMD64)`)

	var vectorConversion = templateNamed("vectorConversion", `	addF(simdPackage, "{{.Tsrc.Name}}.As{{.Tdst.Name}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})`)

	var footer = `}`

	slices.SortFunc(ops, compareOperations)

	buffer := new(bytes.Buffer)
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
		if from.Name[0] != 'U' && to.Name[0] != 'U' {
			continue
		}
		// Only emit the intrinsic if lanes are equal OR both are unsigned
		if from.Lanes != to.Lanes && (from.Name[0] != 'U' || to.Name[0] != 'U') {
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
		if typ.Type != "mask" {
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

	for _, m := range masksFromTypeMap(typeMap) {
		doTemplate(mask, m)
	}

	buffer.WriteString(footer)

	return buffer
}
