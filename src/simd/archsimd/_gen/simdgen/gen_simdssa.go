// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"log"
	"sort"
	"strings"
	"text/template"
)

var (
	ssaTemplates = template.Must(template.New("simdSSA").Parse(`{{define "header"}}{{.GeneratedHeader}}
package {{.Arch}}

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssagen"
	"cmd/internal/obj"
	"cmd/internal/obj/{{.ObjArch}}"
)

func ssaGenSIMDValue(s *ssagen.State, v *ssa.Value) bool {
	var p *obj.Prog
	switch v.Op {{"{"}}{{end}}
{{define "case"}}
	case {{.Cases}}:
		p = {{.Helper}}(s, v{{if .Arrangement}}, {{.Arrangement}}{{end}})
{{end}}
{{define "footer"}}
	default:
		// Unknown reg shape
		return false
	}
{{end}}
{{define "zeroing"}}
	// Masked operation are always compiled with zeroing.
	switch v.Op {
	case {{.}}:
		x86.ParseSuffix(p, "Z")
	}
{{end}}
{{define "ending"}}
	// Ensure p is marked as used (may not be used in all generated code paths)
	_ = p
	return true
}
{{end}}`))
)

type tplSSAData struct {
	Cases       string
	Helper      string
	Arrangement string // Optional arrangement constant for ARM64, e.g. arm64.ARNG_4S
}

type tplSSAHeader struct {
	Arch            string
	ObjArch         string
	GeneratedHeader string
}

// getArrangementFromOp extracts the arrangement constant from an SSA op name for ARM64.
// For example, "ssa.OpARM64VFADD4S" returns "arm64.ARNG_4S".
func getArrangementFromOp(archInfo ArchInfo, caseStr string) string {
	for _, a := range archInfo.Arrangements {
		if strings.Contains(caseStr, a) {
			return archInfo.Arch + ".ARNG_" + a
		}
	}
	return ""
}

// writeSIMDSSA generates the ssa to prog lowering codes and writes it to simdssa.go
// within the specified directory.
func writeSIMDSSA(ops []Operation) *bytes.Buffer {
	archInfo := CurrentArch()
	var ZeroingMask []string
	regInfoKeys := archInfo.RegInfoKeys
	regInfoSet := map[string][]string{}
	for _, key := range regInfoKeys {
		regInfoSet[key] = []string{}
	}

	seen := map[string]struct{}{}
	allUnseen := make(map[string][]Operation)
	allUnseenCaseStr := make(map[string][]string)
	// computeBaseRegShape computes the base regShape for an op.
	computeBaseRegShape := func(op Operation, mem memShape, shapeIn inShape, shapeOut outShape, immOpArg string, immType immShape) (string, error) {
		regShape, err := op.regShape(mem)
		if err != nil {
			return "", err
		}
		if regShape == "v01load" {
			regShape = "vload"
		}
		if shapeOut == OneVregOutAtIn {
			regShape += "ResultInArg0"
		} else if shapeOut == OneVregOutScalar {
			regShape += "Scalar"
		}
		if shapeIn == OneImmIn || shapeIn == OneKmaskImmIn {
			if immOpArg != "" {
				regShape += "Imm"
				regShape += immOpArg
			} else if immType == VarImmLim {
				regShape += "Imm" // limited range immediate (ImmMax set)
			} else {
				regShape += "Imm8" // full 8-bit range (0-255)
			}
		}
		if shapeIn == VlistIn {
			regShape += "List"
		}
		regShape, err = rewriteVecAsScalarRegInfo(op, regShape)
		if err != nil {
			return "", err
		}
		return regShape, nil
	}
	registerRegShape := func(regShape string, caseStr string, op Operation) {
		if _, ok := regInfoSet[regShape]; !ok {
			allUnseen[regShape] = append(allUnseen[regShape], op)
			allUnseenCaseStr[regShape] = append(allUnseenCaseStr[regShape], caseStr)
		}
		regInfoSet[regShape] = append(regInfoSet[regShape], caseStr)
	}
	classifyOp := func(op Operation, maskType maskShape, shapeIn inShape, shapeOut outShape, caseStr string, mem memShape, immOpArg string, immType immShape) error {
		regShape, err := computeBaseRegShape(op, mem, shapeIn, shapeOut, immOpArg, immType)
		if err != nil {
			return err
		}
		// For hi-half base ops, append the kind suffix for lowering dispatch.
		if op.HiHalfAsm != nil {
			kind := op.hiHalfKind()
			if kind != "" {
				regShape += capitalizeFirst(kind) // e.g., "v11Imm" + "Narrow" = "v11ImmNarrow"
			}
		}
		registerRegShape(regShape, caseStr, op)
		if mem == NoMem && op.hasMaskedMerging(maskType, shapeOut) {
			regShapeMerging := regShape
			if shapeOut != OneVregOutAtIn {
				// We have to copy the slice here because the sort will be visible from other
				// aliases when no reslicing is happening.
				newIn := make([]Operand, len(op.In), len(op.In)+1)
				copy(newIn, op.In)
				op.In = newIn
				op.In = append(op.In, op.Out[0])
				op.sortOperand()
				regShapeMerging, err = op.regShape(mem)
				regShapeMerging += "ResultInArg0"
			}
			if err != nil {
				return err
			}
			registerRegShape(regShapeMerging, caseStr+"Merging", op)
		}
		return nil
	}
	// classifyHiHalfOp computes the lowering dispatch regShape for a hi-half "2" variant.
	// It derives the regShape from the base op's shape and applies hi-half transformation.
	classifyHiHalfOp := func(op Operation, kind string, caseStr string, immOpArg string, immType immShape) error {
		shapeIn, shapeOut, _, _, _, _ := op.shape()
		regShape, err := computeBaseRegShape(op, NoMem, shapeIn, shapeOut, immOpArg, immType)
		if err != nil {
			return err
		}
		regShape = hiHalfLoweringRegShape(regShape, kind, true)
		registerRegShape(regShape, caseStr, op)
		return nil
	}
	for _, op := range ops {
		shapeIn, shapeOut, maskType, immType, gOp, immOpArg := op.shape()
		asm := machineOpName(maskType, gOp)
		if _, ok := seen[asm]; ok {
			continue
		}
		seen[asm] = struct{}{}
		caseStr := fmt.Sprintf("ssa.Op%s%s", archInfo.ArchUpper, asm)
		isZeroMasking := false
		if shapeIn == OneKmaskIn || shapeIn == OneKmaskImmIn {
			if gOp.Zeroing == nil || *gOp.Zeroing {
				ZeroingMask = append(ZeroingMask, caseStr)
				isZeroMasking = true
			}
		}
		if err := classifyOp(op, maskType, shapeIn, shapeOut, caseStr, NoMem, immOpArg, immType); err != nil {
			panic(err)
		}

		// Generate hi-half "2" variant SSA lowering case.
		// The base op gets a suffixed regShape (e.g., "v11ImmNarrow"), and
		// the "2" variant gets a derived regShape (e.g., "v21ImmNarrow2").
		if gOp.HiHalfAsm != nil {
			kind := op.hiHalfKind()
			if kind != "" {
				asm2 := hiHalfOpName(*gOp.HiHalfAsm, gOp)
				caseStr2 := fmt.Sprintf("ssa.Op%s%s", archInfo.ArchUpper, asm2)
				if _, ok2 := seen[asm2]; !ok2 {
					seen[asm2] = struct{}{}
					if err := classifyHiHalfOp(op, kind, caseStr2, immOpArg, immType); err != nil {
						panic(err)
					}
				}
			}
		}

		if op.MemFeatures != nil && *op.MemFeatures == "vbcst" {
			// Make a full vec memory variant
			op = rewriteLastVregToMem(op)
			// Ignore the error
			// an error could be triggered by [checkVecAsScalar].
			// TODO: make [checkVecAsScalar] aware of mem ops.
			if err := classifyOp(op, maskType, shapeIn, shapeOut, caseStr+"load", VregMemIn, immOpArg, immType); err != nil {
				if *Verbose {
					log.Printf("Seen error: %e", err)
				}
			} else if isZeroMasking {
				ZeroingMask = append(ZeroingMask, caseStr+"load")
			}
		}
	}
	if len(allUnseen) != 0 {
		allKeys := make([]string, 0)
		for k := range allUnseen {
			allKeys = append(allKeys, k)
		}
		panic(fmt.Errorf("unsupported register constraint for prog, please update gen_simdssa.go and amd64/ssa.go: %+v\nAll keys: %v\n, cases: %v\n", allUnseen, allKeys, allUnseenCaseStr))
	}

	buffer := new(bytes.Buffer)

	headerData := tplSSAHeader{
		Arch:            archInfo.Arch,
		ObjArch:         archInfo.ObjArch,
		GeneratedHeader: archInfo.GeneratedHeader,
	}
	if err := ssaTemplates.ExecuteTemplate(buffer, "header", headerData); err != nil {
		panic(fmt.Errorf("failed to execute header template: %w", err))
	}

	for _, regShape := range regInfoKeys {
		// Stable traversal of regInfoSet
		cases := regInfoSet[regShape]
		if len(cases) == 0 {
			continue
		}

		// Group cases by arrangement (for ARM64)
		arrangementGroups := make(map[string][]string)
		for _, caseStr := range cases {
			arrangement := getArrangementFromOp(archInfo, caseStr)
			arrangementGroups[arrangement] = append(arrangementGroups[arrangement], caseStr)
		}

		// Sort arrangement keys for deterministic output
		var arrangements []string
		for arrangement := range arrangementGroups {
			arrangements = append(arrangements, arrangement)
		}
		sort.Strings(arrangements)

		// Generate cases for each arrangement group in sorted order
		for _, arrangement := range arrangements {
			groupCases := arrangementGroups[arrangement]
			data := tplSSAData{
				Cases:  strings.Join(groupCases, ",\n\t\t"),
				Helper: "simd" + capitalizeFirst(regShape),
			}
			if arrangement != "" {
				data.Arrangement = arrangement
			}
			if err := ssaTemplates.ExecuteTemplate(buffer, "case", data); err != nil {
				panic(fmt.Errorf("failed to execute case template for %s: %w", regShape, err))
			}
		}
	}

	if err := ssaTemplates.ExecuteTemplate(buffer, "footer", nil); err != nil {
		panic(fmt.Errorf("failed to execute footer template: %w", err))
	}

	if len(ZeroingMask) != 0 {
		if err := ssaTemplates.ExecuteTemplate(buffer, "zeroing", strings.Join(ZeroingMask, ",\n\t\t")); err != nil {
			panic(fmt.Errorf("failed to execute footer template: %w", err))
		}
	}

	if err := ssaTemplates.ExecuteTemplate(buffer, "ending", headerData); err != nil {
		panic(fmt.Errorf("failed to execute ending template: %w", err))
	}

	return buffer
}
