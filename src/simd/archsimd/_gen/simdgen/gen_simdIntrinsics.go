// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"_gen/sgutil"
	"bytes"
	"fmt"
	"slices"
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

func simdAMD64Intrinsics(addF func(pkg, fn string, b intrinsicBuilder, archFamilies ...sys.ArchFamily)) {
{{end}}

{{define "op1"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen1(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op2"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op2_21"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen2_21(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op2_21Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2_21(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op3"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op3_21"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen3_21(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op3_21Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_21(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op3_231Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3_231(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op3_31Zero3"}}	addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen3_31Zero3(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op4"}}	addF(simdPackage, "{{(index .In 0).Go}}.{{.Go}}", opLen4(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op4_231Type1"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4_231(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op4_31"}}	addF(simdPackage, "{{(index .In 2).Go}}.{{.Go}}", opLen4_31(ssa.Op{{.GenericName}}, {{.SSAType}}), sys.AMD64)
{{end}}
{{define "op1Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen1Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op2Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op2Imm8_2I"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op2Imm8_II"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_II(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op2Imm8_SHA1RNDS4"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen2Imm8_SHA1RNDS4(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op3Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op3Imm8_2I"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen3Imm8_2I(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}
{{define "op4Imm8"}}	addF(simdPackage, "{{(index .In 1).Go}}.{{.Go}}", opLen4Imm8(ssa.Op{{.GenericName}}, {{.SSAType}}, {{(index .In 0).ImmOffset}}), sys.AMD64)
{{end}}

{{define "vectorConversion"}}	addF(simdPackage, "{{.Tsrc.Name}}.As{{.Tdst.Name}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, sys.AMD64)
{{end}}

{{define "loadStore"}}	addF(simdPackage, "Load{{.Name}}Array", simdLoad(), sys.AMD64)
	addF(simdPackage, "{{.Name}}.StoreArray", simdStore(), sys.AMD64)
{{end}}

{{define "maskedLoadStore"}}
	addF(simdPackage, "{{.Name}}.StoreArrayMasked", simdMaskedStore(ssa.OpStoreMasked{{.ElemBits}}), sys.AMD64)
{{end}}

{{define "mask"}}	addF(simdPackage, "{{.Name}}.To{{.VectorCounterpart}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, sys.AMD64)
	addF(simdPackage, "{{.VectorCounterpart}}.asMask", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, sys.AMD64)
	addF(simdPackage, "{{.Name}}.And", opLen2(ssa.OpAnd{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), sys.AMD64)
	addF(simdPackage, "{{.Name}}.Or", opLen2(ssa.OpOr{{.ReshapedVectorWithAndOr}}, types.TypeVec{{.Size}}), sys.AMD64)
	addF(simdPackage, "{{.Name}}FromBits", simdCvtVToMask({{.ElemBits}}, {{.Lanes}}), sys.AMD64)
	addF(simdPackage, "{{.Name}}.ToBits", simdCvtMaskToV({{.ElemBits}}, {{.Lanes}}), sys.AMD64)
{{end}}

{{define "footer"}}}
{{end}}
`

// writeSIMDIntrinsics generates the intrinsic mappings and writes it to simdintrinsics.go
// within the specified directory.
func writeSIMDIntrinsics(ops []Operation, typeMap simdTypeMap) *bytes.Buffer {
	t := templateOf(simdIntrinsicsTmpl, "simdintrinsics")
	buffer := new(bytes.Buffer)
	buffer.WriteString(generatedHeader)

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
			if err := t.ExecuteTemplate(buffer, s, op); err != nil {
				panic(fmt.Errorf("failed to execute template %s for op %s: %w", s, op.Go, err))
			}

		} else {
			panic(fmt.Errorf("failed to classify op %v: %w", op.Go, err))
		}
	}

	var FooIntrinsic = templateOf(`addF(simdPackage, "{{.Foo}}", func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value { return args[0] }, sys.AMD64)
	`, "amd64 foo intrinsics")

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
		sgutil.Conversion(from, to).ExecuteIntrinsicTemplateOfFoo(buffer, FooIntrinsic)
	}

	for _, typ := range typesFromTypeMap(typeMap) {
		if typ.Type != "mask" {
			if err := t.ExecuteTemplate(buffer, "loadStore", typ); err != nil {
				panic(fmt.Errorf("failed to execute loadStore template: %w", err))
			}
		}
	}

	for _, typ := range typesFromTypeMap(typeMap) {
		if typ.MaskedLoadStoreFilter() {
			if err := t.ExecuteTemplate(buffer, "maskedLoadStore", typ); err != nil {
				panic(fmt.Errorf("failed to execute maskedLoadStore template: %w", err))
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
