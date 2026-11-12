// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"log"
	"maps"
	"simd/archsimd/_gen/simdgen/types"
	"slices"
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
	"cmd/compile/internal/ssa/ssaop"
)

func ssaGenSIMD{{.FuncInfix}}Value(s *ssagen.State, v *ssa.Value) bool {
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
{{define "narrowRegs"}}
	// Some operands are in registers narrower than their Go types.
	switch v.Op {
{{- range .}}
	case {{.Cases}}:
		{{.Fix}}
{{- end}}
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

// tplSSANarrow is a group of machine ops whose operands that are narrower than
// their Go types are fixed up by the same statements, Fix. These rename those
// operands' registers to the narrower register (see [narrowRegFixes] and
// simdNarrowArg in cmd/compile/internal/amd64).
type tplSSANarrow struct {
	Cases string
	Fix   string
}

// narrowRegFixes returns the statements that fix up the registers of the
// operands of op that are narrower than their Go types (see
// [types.Operand.NarrowReg]). Memory operands (for example, from folding a load
// into op) and mask operands don't need fixing, but count as arguments.
func narrowRegFixes(op Operation) []string {
	if !op.hasNarrowReg() {
		return nil
	}
	var fixes []string
	arg := 0
	for _, in := range op.In {
		if in.Class == "immediate" {
			continue
		}
		if in.Class == "vreg" && in.NarrowReg() {
			fixes = append(fixes, fmt.Sprintf("simdNarrowArg(p, v.Args[%d], %d)", arg, in.Bits.N()/8))
		}
		arg++
	}
	for _, out := range op.Out {
		if out.NarrowReg() {
			fixes = append(fixes, fmt.Sprintf("p.To.Reg = simdRegBySize(v.Reg(), %d)", out.Bits.N()/8))
		}
	}
	return fixes
}

// mergingOp returns the merging masked form of op, whose result is also its
// first input (the merge source). This is the operand order of op's merging
// machine op.
func (op Operation) mergingOp() Operation {
	// Copy In, since sorting it would otherwise be visible through other
	// aliases of the slice.
	newIn := make([]types.Operand, len(op.In), len(op.In)+1)
	copy(newIn, op.In)
	op.In = append(newIn, op.Out[0])
	op.sortOperand()
	return op
}

type tplSSAHeader struct {
	Arch            string
	ObjArch         string
	GeneratedHeader string
	// FuncInfix disambiguates the generated ssaGenSIMD<FuncInfix>Value function
	// name when a target shares another's backend package.
	// Empty for the primary target.
	FuncInfix string
}

// getArrangementFromOp extracts the arrangement constant from an SSA op name for ARM64.
// For example, "ssa.OpARM64VFADD4S" returns "arm64.ARNG_4S".
func getArrangementFromOp(archInfo ArchInfo, caseStr string) string {
	if archInfo.isSVE() {
		// SVE machine op names end in a single element-size letter (ZADD -> ZADDB,
		// ZSQADD -> ZSQADDD), then the ssaVariant if any (ZLSLBconst). Match the
		// suffix, not any occurrence: "ZSQADDD" contains "S" (from SQADD) but its
		// arrangement is the trailing "D".
		name := strings.TrimSuffix(caseStr, "const")
		for _, a := range archInfo.Arrangements {
			if strings.HasSuffix(name, a) {
				return archInfo.Arch + ".ARNG_" + a
			}
		}
		return ""
	}
	for _, a := range archInfo.Arrangements {
		if strings.Contains(caseStr, a) {
			return archInfo.Arch + ".ARNG_" + a
		}
	}
	return ""
}

// writeSIMDSSA generates the ssa to prog lowering codes and writes it to simdssa.go
// within the specified directory.
func writeSIMDSSA(buffer *bytes.Buffer, ops []Operation) {
	archInfo := CurrentArch()
	var ZeroingMask []string
	// narrowFix maps the case expression of a machine op, such as
	// "ssaop.OpAMD64VPMOVSXBD256" (or its Merging or load form), to the Go
	// statements that fix up its narrow registers (see narrowRegFixes).
	// Several operations can share a machine op (for example,
	// Int32x8.TruncToInt8 and Uint32x8.TruncToUint8 both use
	// VPMOVDB256), so this collects fixes from all of them, not just the
	// first.
	narrowFix := map[string]string{}
	addNarrow := func(op Operation, caseStr string) {
		fixes := narrowRegFixes(op)
		if len(fixes) == 0 {
			return
		}
		fix := strings.Join(fixes, "\n\t\t")
		if old, ok := narrowFix[caseStr]; ok && old != fix {
			panic(fmt.Errorf("conflicting narrow register fixes for %s:\n%s\n%s", caseStr, old, fix))
		}
		narrowFix[caseStr] = fix
	}
	generated := map[string]bool{} // cases that get a codegen case
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
		if shapeOut == OneVregOutAtIn && !strings.HasSuffix(regShape, "Acc") {
			// An Acc shape's first input is by definition the destination
			// (see sveAccumulatorInput), so the suffix would be redundant.
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
		if sveWideElements(op) {
			regShape += "Wide"
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
		generated[caseStr] = true
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
				op = op.mergingOp()
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
	// An SVE inVariant implies machine ops that are not operations of their own,
	// so expand them here too; each needs its own ssa-to-prog case.
	expanded := make([]Operation, 0, len(ops))
	for _, op := range ops {
		expanded = append(expanded, op)
		for _, pred := range op.svePredicatedOps() {
			expanded = append(expanded, pred)
			if prefixed := pred.sveMergingPrefixedOp(); prefixed != nil {
				expanded = append(expanded, *prefixed)
			}
		}
	}
	ops = expanded
	for _, op := range ops {
		shapeIn, shapeOut, maskType, immType, gOp, immOpArg := op.shape()
		asm := machineOpName(maskType, gOp)
		// Collect the narrow register fixes for op's machine op and
		// its merging and load-folded forms. This comes before the
		// seen check below because op may share its machine op with
		// an earlier operation that has no narrow operands. The fix
		// is harmless for the other operation, since it renames a
		// register to the name it already has. The same holds when
		// the compiler's "simd enlarge" pass (see
		// writeSIMDEnlargeTables) gives a machine op an input
		// narrower than its Go type: VPMOVSXBD256 can take an
		// Int8x16 input from x.Enlarge().ExtendToInt32(), whose X
		// register already has the name the fix gives it.
		addNarrow(op, fmt.Sprintf("ssaop.Op%s%s", archInfo.ArchUpper, asm))
		if op.hasMaskedMerging(maskType, shapeOut) {
			mop := op
			if shapeOut != OneVregOutAtIn {
				mop = op.mergingOp()
			}
			addNarrow(mop, fmt.Sprintf("ssaop.Op%s%sMerging", archInfo.ArchUpper, asm))
		}
		if op.MemFeatures != nil && *op.MemFeatures == "vbcst" {
			addNarrow(rewriteLastVregToMem(op), fmt.Sprintf("ssaop.Op%s%sload", archInfo.ArchUpper, asm))
		}
		if _, ok := seen[asm]; ok {
			continue
		}
		seen[asm] = struct{}{}
		caseStr := fmt.Sprintf("ssaop.Op%s%s", archInfo.ArchUpper, asm)
		isZeroMasking := false
		if shapeIn == OneKmaskIn || shapeIn == OneKmaskImmIn {
			if (gOp.Zeroing == nil || *gOp.Zeroing) && !CurrentArch().isSVE() {
				// x86 spells the zeroing/merging choice as an assembler suffix on
				// the masked instruction. SVE encodes it in the governing predicate
				// operand itself (Pg/Z or Pg/M), which the ssa-to-prog helper
				// already emits, so there is no suffix to parse.
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
				caseStr2 := fmt.Sprintf("ssaop.Op%s%s", archInfo.ArchUpper, asm2)
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

	headerData := tplSSAHeader{
		Arch:            archInfo.Arch,
		ObjArch:         archInfo.ObjArch,
		GeneratedHeader: archInfo.GeneratedHeader,
		FuncInfix:       archInfo.ssaGenFuncInfix(),
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

	// Emit the narrow register fixes as a switch after the main one, with one
	// case per distinct fix listing all the machine ops that need it. Skip
	// machine ops that didn't get a codegen case above.
	narrowCases := map[string][]string{} // fix -> cases
	for _, caseStr := range slices.Sorted(maps.Keys(narrowFix)) {
		if generated[caseStr] {
			fix := narrowFix[caseStr]
			narrowCases[fix] = append(narrowCases[fix], caseStr)
		}
	}
	if len(narrowCases) != 0 {
		var groups []tplSSANarrow
		for _, fix := range slices.Sorted(maps.Keys(narrowCases)) {
			groups = append(groups, tplSSANarrow{
				Cases: strings.Join(narrowCases[fix], ",\n\t\t"),
				Fix:   fix,
			})
		}
		if err := ssaTemplates.ExecuteTemplate(buffer, "narrowRegs", groups); err != nil {
			panic(fmt.Errorf("failed to execute narrowRegs template: %w", err))
		}
	}

	if err := ssaTemplates.ExecuteTemplate(buffer, "ending", headerData); err != nil {
		panic(fmt.Errorf("failed to execute ending template: %w", err))
	}
}
