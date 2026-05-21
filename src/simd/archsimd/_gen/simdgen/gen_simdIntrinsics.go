// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"_gen/sgutil"
	"bytes"
	"fmt"
	"slices"
	"text/template"
)

const simdIntrinsicsTmpl = `
{{define "header"}}
package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

func simd{{GetArchUpper}}Intrinsics(addF func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily)) {
{{end}}

{{define "op1"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen1(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op2"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op2_21"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op2_21Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op3"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op3_21"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op3_21Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_21(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op3_231Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_231(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op3_31Zero3"}}	addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen3_31Zero3(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op4"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen4(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op4_231Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4_231(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op4_31"}}	addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen4_31(ssa.Op{{.GenericName}}, {{.SSAType}}), {{GetSysArch}})
{{end}}
{{define "op1Imm"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})
{{end}}
{{define "op1Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op2Imm"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})
{{end}}
{{define "op2Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op2Imm8_2I"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op2Imm_2I"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})
{{end}}
{{define "op2Imm8_II"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_II(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op2Imm8_SHA1RNDS4"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_SHA1RNDS4(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op2ImmVecAsScalar"}} addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen2Imm(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}, {{(index .In 0).ImmMax}}), {{GetSysArch}})
{{end}}
{{define "op3Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op3Imm8_2I"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}
{{define "op4Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), {{GetSysArch}})
{{end}}

{{define "vectorConversion"}}	addF(simdPackage, "{{.Tsrc.Name}}.As{{.Tdst.Name}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
{{end}}

{{define "loadStore"}}	addF(simdPackage, "Load{{.Name}}Array", simdLoad(), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.StoreArray", simdStore(), {{GetSysArch}})
{{end}}

{{define "maskedLoadStore"}}
	addF(simdPackage, "{{.Name}}.StoreArrayMasked", simdMaskedStore(ssa.OpStoreMasked{{.ElemBits}}), {{GetSysArch}})
{{end}}

{{define "mask"}}	addF(simdPackage, "{{.Name}}.To{{.VectorCounterpart}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
	addF(simdPackage, "{{.VectorCounterpart}}.asMask", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.And", opLen2(ssa.OpAnd{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.Or", opLen2(ssa.OpOr{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
{{- if eq GetSysArch "sys.ARM64"}}
	addF(simdPackage, "{{.Name}}.Not", opLen1(ssa.OpNot{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), {{GetSysArch}})
{{else}}
	addF(simdPackage, "{{.Name}}FromBits", simdCvtVToMask({{.ElemBits}}, {{.Lanes}}), {{GetSysArch}})
	addF(simdPackage, "{{.Name}}.ToBits", simdCvtMaskToV({{.ElemBits}}, {{.Lanes}}), {{GetSysArch}})
{{- end}}
{{end}}

{{define "footer"}}}
{{end}}
`

// writeSIMDIntrinsics generates the intrinsic mappings and writes it to simdintrinsics.go
// within the specified directory.
func writeSIMDIntrinsics(ops []Operation, typeMap simdTypeMap) *bytes.Buffer {
	archInfo := CurrentArch()
	sysArch := "sys." + archInfo.ArchUpper

	tmpl := template.New("simdintrinsics")
	tmpl.Funcs(template.FuncMap{
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
	t := template.Must(tmpl.Parse(simdIntrinsicsTmpl))
	buffer := new(bytes.Buffer)
	buffer.WriteString(generatedHeader())

	if err := t.ExecuteTemplate(buffer, "header", nil); err != nil {
		panic(fmt.Errorf("failed to execute header template: %w", err))
	}

	slices.SortFunc(ops, compareOperations)

	for _, op := range ops {
		if op.NoTypes != nil && *op.NoTypes == "true" {
			continue
		}
		if op.SkipMaskedMethod() {
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

			if err := t.ExecuteTemplate(buffer, s, op); err != nil {
				panic(fmt.Errorf("failed to execute template %s for op %s: %w", s, op.Go, err))
			}

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
		if err := t.ExecuteTemplate(buffer, "vectorConversion", conv); err != nil {
			panic(fmt.Errorf("failed to execute vectorConversion template: %w", err))
		}
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
			if err := t.ExecuteTemplate(buffer, "loadStore", typ); err != nil {
				panic(fmt.Errorf("failed to execute loadStore template: %w", err))
			}
		}
	}

	// Masked loads/stores are AVX2/AVX512 only (not available on ARM64 NEON).
	// TODO: Reconsider for ARM64 SVE which supports predicated loads/stores.
	if CurrentArch().Arch == "amd64" {
		for _, typ := range typesFromTypeMap(typeMap) {
			if typ.MaskedLoadStoreFilter() {
				if err := t.ExecuteTemplate(buffer, "maskedLoadStore", typ); err != nil {
					panic(fmt.Errorf("failed to execute maskedLoadStore template: %w", err))
				}
			}
		}
	}

	for _, mask := range masksFromTypeMap(typeMap) {
		if err := t.ExecuteTemplate(buffer, "mask", mask); err != nil {
			panic(fmt.Errorf("failed to execute mask template: %w", err))
		}
	}

	if err := t.ExecuteTemplate(buffer, "footer", nil); err != nil {
		panic(fmt.Errorf("failed to execute footer template: %w", err))
	}

	return buffer
}
