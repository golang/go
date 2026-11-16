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
)

const simdMachineOpsTmpl = `
package main

func simd{{.SIMDTag}}Ops({{.RegInfoParams}}) []opData {
{{- if .ZeroVariants}}
	// The *z forms of regInfos permit all vector inputs to be in zeroReg,
	// the zero register. They're for ops that don't overwrite their first
	// input (neither resultInArg0 nor clobbersArg0). Whether an op
	// overwrites its first input is a property of the op, not the regInfo,
	// so the caller checks this for the ops this returns.
	zeroForm := func(r regInfo) regInfo {
		r = r.clone()
		r.inputs[0] = r.inputs[0].union(zeroReg)
		return r
	}
{{- range .ZeroVariants}}
	{{.}}z := zeroForm({{.}})
{{- end}}
{{end}}
	return []opData{
{{- range .OpsData }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}",{{if .Comm}} commutative: true,{{end}} typ: "{{.Type}}"{{if .ResultInArg0}}, resultInArg0: true{{end}}},
{{- end }}
{{- range .OpsDataImm }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}", aux: "UInt8",{{if .Comm}} commutative: true,{{end}} typ: "{{.Type}}"{{if .ResultInArg0}}, resultInArg0: true{{end}}},
{{- end }}
{{- range .OpsDataLoad}}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}",{{if .Comm}} commutative: true,{{end}} typ: "{{.Type}}", aux: "SymOff", symEffect: "Read"{{if .ResultInArg0}}, resultInArg0: true{{end}}},
{{- end}}
{{- range .OpsDataImmLoad}}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}",{{if .Comm}} commutative: true,{{end}} typ: "{{.Type}}", aux: "SymValAndOff", symEffect: "Read"{{if .ResultInArg0}}, resultInArg0: true{{end}}},
{{- end}}
{{- range .OpsDataMerging }}
		{name: "{{.OpName}}Merging", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}", typ: "{{.Type}}", resultInArg0: true},
{{- end }}
{{- range .OpsDataImmMerging }}
		{name: "{{.OpName}}Merging", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}", aux: "UInt8", typ: "{{.Type}}", resultInArg0: true},
{{- end }}
	}
}
`

// writeSIMDMachineOps generates the machine ops and writes it to simdAMD64ops.go
// within the specified directory.
// isWidthAgnostic reports whether the operation is an SVE width-agnostic
// bitwise op (see types.RawOperation.WidthAgnostic).
func isWidthAgnostic(gOp Operation) bool {
	return gOp.WidthAgnostic != nil && *gOp.WidthAgnostic
}

func writeSIMDMachineOps(buffer *bytes.Buffer, ops []Operation) {
	t := templateOf(simdMachineOpsTmpl, "simdAMD64Ops")
	buffer.WriteString(generatedHeader())

	type opData struct {
		OpName       string
		Asm          string
		OpInLen      int
		RegInfo      string
		Comm         bool
		Type         string
		ResultInArg0 bool
	}
	type machineOpsData struct {
		SIMDTag           string
		RegInfoParams     string
		ZeroVariants      []string
		OpsData           []opData
		OpsDataImm        []opData
		OpsDataLoad       []opData
		OpsDataImmLoad    []opData
		OpsDataMerging    []opData
		OpsDataImmMerging []opData
	}

	archInfo := CurrentArch()

	regInfoSet := archInfo.RegInfoSet
	opsData := make([]opData, 0)
	opsDataImm := make([]opData, 0)
	opsDataLoad := make([]opData, 0)
	opsDataImmLoad := make([]opData, 0)
	opsDataMerging := make([]opData, 0)
	opsDataImmMerging := make([]opData, 0)

	// Determine the "best" version of an instruction to use
	best := make(map[string]Operation)
	var mOpOrder []string
	countOverrides := func(s []types.Operand) int {
		a := 0
		for _, o := range s {
			if o.OverwriteBase != nil {
				a++
			}
		}
		return a
	}
	for _, op := range ops {
		_, _, maskType, _, gOp, _ := op.shape()
		asm := machineOpName(maskType, gOp)
		if isWidthAgnostic(gOp) {
			// The unpredicated machine op of a width-agnostic bitwise operation
			// collapses to one .D instruction, but its predicated forms merge at
			// a real element granularity, so every width's def must survive this
			// dedup to generate them; the shared unpredicated opData is deduped
			// at the append instead.
			asm = fmt.Sprintf("%s#%d", asm, *gOp.Out[0].ElemBits)
		}
		other, ok := best[asm]
		if !ok {
			best[asm] = op
			mOpOrder = append(mOpOrder, asm)
			continue
		}
		if !op.Commutative && other.Commutative { // if there's a non-commutative version of the op, it wins.
			best[asm] = op
			continue
		}
		// see if "op" is better than "other"
		if countOverrides(op.In)+countOverrides(op.Out) < countOverrides(other.In)+countOverrides(other.Out) {
			best[asm] = op
		}
	}

	regInfoErrs := make([]error, 0)
	regInfoMissing := make(map[string]bool, 0)
	// zeroUsed is the set of regInfos whose *z form some op uses.
	zeroUsed := make(map[string]bool)
	// hasVregInput reports whether op has a vreg input, which is the first
	// input of its machine op.
	hasVregInput := func(op Operation) bool {
		_, _, _, _, gOp, _ := op.shape()
		for _, in := range gOp.In {
			if in.Class == "vreg" {
				return true
			}
		}
		return false
	}
	// sveSeenUnpred dedups the base machine op across the defs that share it:
	// only SVE produces such sharing (a width-agnostic bitwise operation is one
	// .D instruction serving every element width); on other architectures every
	// def has its own machine-op name.
	sveSeenUnpred := make(map[string]bool)
	for _, asm := range mOpOrder {
		op := best[asm]
		shapeIn, shapeOut, maskType, _, gOp, _ := op.shape()
		asm = machineOpName(maskType, gOp)

		// TODO: all our masked operations are now zeroing, we need to generate machine ops with merging masks, maybe copy
		// one here with a name suffix "Merging". The rewrite rules will need them.
		makeRegInfo := func(op Operation, mem memShape) (string, error) {
			regInfo, err := op.regShape(mem)
			if err != nil {
				panic(err)
			}
			regInfo, err = rewriteVecAsScalarRegInfo(op, regInfo)
			if err != nil {
				if mem == NoMem || mem == InvalidMem {
					panic(err)
				}
				return "", err
			}
			if regInfo == "v01load" {
				regInfo = "vload"
			}
			// Makes AVX512 instructions (which are EVEX-encoded) use upper registers.
			// Check rawOperation.CPUFeature rather than op.CPUFeature: a VEX
			// instruction like Enlarge's VMOVDQA may require AVX512 at the Go API
			// level because its result is a 512-bit vector, while still needing
			// X0-X15 for VEX encoding.
			if strings.Contains(op.rawOperation.CPUFeature, "AVX512") {
				regInfo = strings.ReplaceAll(regInfo, "v", "w")
			}
			if _, ok := regInfoSet[regInfo]; !ok {
				regInfoErrs = append(regInfoErrs, fmt.Errorf("unsupported register constraint, please update the template and AMD64Ops.go: %s.  Op is %s", regInfo, op))
				regInfoMissing[regInfo] = true
			}
			return regInfo, nil
		}
		regInfo, err := makeRegInfo(op, NoMem)
		if err != nil {
			panic(err)
		}
		var outType string
		if shapeOut == OneVregOut || shapeOut == OneVregOutAtIn || shapeOut == OneVregOutScalar || gOp.Out[0].OverwriteClass != nil {
			// If class overwrite is happening, that's not really a mask but a vreg.
			if gOp.Out[0].Bits.Scalable {
				outType = fmt.Sprintf("Vec%d", types.MaxVectorBits)
			} else {
				outType = fmt.Sprintf("Vec%d", gOp.Out[0].GoWidth().N())
			}
		} else if shapeOut == OneFpregOut {
			outType = fmt.Sprintf("Float%d", gOp.Out[0].Bits.N())
			if *gOp.Out[0].Base != "float" {
				// An integer in a SIMD&FP register, which may be 8 or 16 bits
				// (SMAXV <V><d>). Writing <V><d> zeroes the rest of the register,
				// so its low 64 bits are the integer zero-extended, which the
				// lowering moves to a general register with FMOVDfpgp.
				outType = "Float64"
			}
		} else if shapeOut == OneGregOut {
			outType = gOp.GoType() // this is a straight Go type, not a VecNNN type
		} else if shapeOut == OneKmaskOut {
			outType = "Mask"
		} else {
			panic(fmt.Errorf("simdgen does not recognize this output shape: %d", shapeOut))
		}
		resultInArg0 := false
		if shapeOut == OneVregOutAtIn {
			resultInArg0 = true
		}
		// zeroForm returns the regInfo to use for op, whose regInfo is r,
		// if op doesn't overwrite its first input.
		//
		// On amd64, that's the *z form of r, which also permits a vector
		// first input to be in X15. X15 is always zero: the Go ABI requires
		// it, and it's never allocated for a result. So the only values in
		// X15 are zero vectors (Zero128/256/512), and an op that only reads
		// its first input can read a zero vector directly from X15. Ops that
		// overwrite their first input (resultInArg0) can't. The generated
		// function builds the *z forms from the base regInfos.
		zeroForm := func(op Operation, r string) string {
			if archInfo.Arch != "amd64" || !hasVregInput(op) {
				return r
			}
			zeroUsed[r] = true
			return r + "z"
		}
		if CurrentArch().isSVE() {
			switch idx := gOp.sveInPlaceInput(); {
			case idx < 0:
				// Constructive: the destination is independent of the sources.
			case idx == 0:
				// The instruction overwrites its first source. A commutative one is
				// left unconstrained — the ssa-to-prog helper puts the destination in
				// place by swapping the operands or by prefixing a MOVPRFX. A
				// non-commutative one cannot be fixed by swapping, so pin the
				// destination to the first source instead.
				if !gOp.Commutative {
					resultInArg0 = true
				}
			default:
				panic(fmt.Errorf("simdgen: %s overwrites input %d; only the first input is supported: %s",
					gOp.Asm, idx, gOp))
			}
		}
		// baseRegInfo is the regInfo of the op itself, which can be the
		// *z form unless the op overwrites its first input.
		baseRegInfo := regInfo
		if !resultInArg0 {
			baseRegInfo = zeroForm(op, regInfo)
		}
		var memOpData *opData
		// AVX-512 only: x86 spells masked merging as an extra merge-source
		// operand on the K-masked instruction, so it needs a machine-op variant
		// of its own. An SVE predicated instruction merges into its own first
		// source instead — its merging forms are the /M machine ops built from
		// svePredicatedOps below, and avxHasMerging is always false on SVE (see
		// hasMaskedMerging).
		avxRegInfoMerging := regInfo
		avxHasMerging := false
		if op.MemFeatures != nil && *op.MemFeatures == "vbcst" {
			// Right now we only have vbcst case
			// Make a full vec memory variant.
			opMem := rewriteLastVregToMem(op)
			regInfo, err := makeRegInfo(opMem, VregMemIn)
			if err != nil {
				// Just skip it if it's non nill.
				// an error could be triggered by [checkVecAsScalar].
				// TODO: make [checkVecAsScalar] aware of mem ops.
				if *Verbose {
					log.Printf("Seen error: %e", err)
				}
			} else {
				if !resultInArg0 {
					regInfo = zeroForm(opMem, regInfo)
				}
				memOpData = &opData{asm + "load", gOp.Asm, len(gOp.In) + 1, regInfo, false, outType, resultInArg0}
			}
		}
		avxHasMerging = gOp.hasMaskedMerging(maskType, shapeOut)
		if avxHasMerging && !resultInArg0 {
			// We have to copy the slice here because the sort will be visible from other
			// aliases when no reslicing is happening.
			newIn := make([]types.Operand, len(op.In), len(op.In)+1)
			copy(newIn, op.In)
			op.In = newIn
			op.In = append(op.In, op.Out[0])
			op.sortOperand()
			avxRegInfoMerging, err = makeRegInfo(op, NoMem)
			if err != nil {
				panic(err)
			}
		}

		// The inVariant implies machine ops only: one predicated instruction
		// per governing-predicate qualifier the encoding supports, reached by
		// peephole rather than by any API of its own.
		appendSVEPredicated := func(target *[]opData) {
			for _, pred := range gOp.svePredicatedOps() {
				predRegInfo, err := makeRegInfo(pred, NoMem)
				if err != nil {
					panic(err)
				}
				predResultInArg0 := false
				predComm := pred.Commutative
				switch idx := pred.sveInPlaceInput(); {
				case idx < 0:
				case idx == 0:
					// The first input is the merge source: merging predication
					// keeps the destination's inactive lanes, and the destination
					// is the first source. That operand is semantically pinned,
					// so the result must live in its register, and the sources
					// must not be swapped even for a commutative operation — CSE
					// canonicalizes the args of a commutative op, which would
					// change the value the inactive lanes keep.
					predResultInArg0 = true
					predComm = false
				default:
					panic(fmt.Errorf("simdgen: %s overwrites input %d; only the first input is supported: %s",
						pred.Asm, idx, pred))
				}
				*target = append(*target, opData{machineOpName(OneMask, pred), pred.Asm, len(pred.In),
					predRegInfo, predComm, outType, predResultInArg0})
				// There is no zeroing machine op here for Masked to fold into:
				// every ARM64 instruction that has both an unpredicated and a
				// predicated encoding is /M-only. The /Z forms belong to
				// predicated-only instructions (ABS, NEG, NOT, ...), where
				// sveImplicitPredPeepholes folds Masked into them.
				if prefixed := pred.sveMergingPrefixedOp(); prefixed != nil {
					prefixedRegInfo, err := makeRegInfo(*prefixed, NoMem)
					if err != nil {
						panic(err)
					}
					// The extra input is the value the destination starts out
					// holding, so the destination must share its register.
					*target = append(*target, opData{machineOpName(OneMask, *prefixed), prefixed.Asm, len(prefixed.In),
						prefixedRegInfo, false, outType, true})
				}
			}
		}
		if shapeIn == OneImmIn || shapeIn == OneKmaskImmIn {
			opsDataImm = append(opsDataImm, opData{asm, gOp.Asm, len(gOp.In), baseRegInfo, gOp.Commutative, outType, resultInArg0})
			appendSVEPredicated(&opsDataImm)
			if memOpData != nil {
				if *op.MemFeatures != "vbcst" {
					panic("simdgen only knows vbcst for mem ops for now")
				}
				opsDataImmLoad = append(opsDataImmLoad, *memOpData)
			}
			if avxHasMerging {
				mergingLen := len(gOp.In)
				if !resultInArg0 {
					mergingLen++
				}
				opsDataImmMerging = append(opsDataImmMerging, opData{asm, gOp.Asm, mergingLen, avxRegInfoMerging, gOp.Commutative, outType, resultInArg0})
			}
		} else {
			if !sveSeenUnpred[asm] {
				sveSeenUnpred[asm] = true
				opsData = append(opsData, opData{asm, gOp.Asm, len(gOp.In), baseRegInfo, gOp.Commutative, outType, resultInArg0})
			}
			appendSVEPredicated(&opsData)
			if memOpData != nil {
				if *op.MemFeatures != "vbcst" {
					panic("simdgen only knows vbcst for mem ops for now")
				}
				opsDataLoad = append(opsDataLoad, *memOpData)
			}
			if avxHasMerging {
				mergingLen := len(gOp.In)
				if !resultInArg0 {
					mergingLen++
				}
				opsDataMerging = append(opsDataMerging, opData{asm, gOp.Asm, mergingLen, avxRegInfoMerging, gOp.Commutative, outType, resultInArg0})
			}
		}
		// Generate hi-half "2" variant machine op
		if gOp.HiHalfAsm != nil {
			opsDataTarget := &opsData
			if shapeIn == OneImmIn || shapeIn == OneKmaskImmIn {
				opsDataTarget = &opsDataImm
			}
			kind := op.hiHalfKind()
			asm2Name := hiHalfOpName(*gOp.HiHalfAsm, gOp)
			argLen2 := len(gOp.In)
			regInfo2 := regInfo
			resultInArg02 := false
			if kind == "narrow" {
				argLen2++ // extra vreg input for destination
				regInfo2 = hiHalfRegShape2(regInfo, kind)
				resultInArg02 = true
			}
			if _, ok := regInfoSet[regInfo2]; !ok {
				regInfoErrs = append(regInfoErrs, fmt.Errorf("unsupported hi-half register constraint: %s for op %s", regInfo2, asm2Name))
				regInfoMissing[regInfo2] = true
			} else {
				*opsDataTarget = append(*opsDataTarget, opData{asm2Name, *gOp.HiHalfAsm, argLen2, regInfo2, gOp.Commutative, outType, resultInArg02})
			}
		}
	}
	if len(regInfoErrs) != 0 {
		for _, e := range regInfoErrs {
			log.Printf("Errors: %e\n", e)
		}
		panic(fmt.Errorf("these regInfo unseen: %v", regInfoMissing))
	}
	sort.Slice(opsData, func(i, j int) bool {
		return compareNatural(opsData[i].OpName, opsData[j].OpName) < 0
	})
	sort.Slice(opsDataImm, func(i, j int) bool {
		return compareNatural(opsDataImm[i].OpName, opsDataImm[j].OpName) < 0
	})
	sort.Slice(opsDataLoad, func(i, j int) bool {
		return compareNatural(opsDataLoad[i].OpName, opsDataLoad[j].OpName) < 0
	})
	sort.Slice(opsDataImmLoad, func(i, j int) bool {
		return compareNatural(opsDataImmLoad[i].OpName, opsDataImmLoad[j].OpName) < 0
	})
	sort.Slice(opsDataMerging, func(i, j int) bool {
		return compareNatural(opsDataMerging[i].OpName, opsDataMerging[j].OpName) < 0
	})
	sort.Slice(opsDataImmMerging, func(i, j int) bool {
		return compareNatural(opsDataImmMerging[i].OpName, opsDataImmMerging[j].OpName) < 0
	})

	zeroVariants := slices.SortedFunc(maps.Keys(zeroUsed), compareNatural)

	err := t.Execute(buffer, machineOpsData{
		SIMDTag:           archInfo.SIMDTag,
		RegInfoParams:     archInfo.RegInfoParams,
		ZeroVariants:      zeroVariants,
		OpsData:           opsData,
		OpsDataImm:        opsDataImm,
		OpsDataLoad:       opsDataLoad,
		OpsDataImmLoad:    opsDataImmLoad,
		OpsDataMerging:    opsDataMerging,
		OpsDataImmMerging: opsDataImmMerging,
	})
	if err != nil {
		panic(fmt.Errorf("failed to execute template: %w", err))
	}
}
