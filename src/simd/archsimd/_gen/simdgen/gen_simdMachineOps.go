// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"log"
	"simd/archsimd/_gen/simdgen/types"
	"sort"
	"strings"
)

const simdMachineOpsTmpl = `
package main

func simd{{.SIMDTag}}Ops({{.RegInfoParams}}) []opData {
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
	seenUnpred := make(map[string]bool)
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
			// Makes AVX512 operations use upper registers
			if strings.Contains(op.CPUFeature, "AVX512") {
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
				outType = fmt.Sprintf("Vec%d", gOp.Out[0].Bits.N())
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
		var memOpData *opData
		regInfoMerging := regInfo
		hasMerging := false
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
				memOpData = &opData{asm + "load", gOp.Asm, len(gOp.In) + 1, regInfo, false, outType, resultInArg0}
			}
		}
		hasMerging = gOp.hasMaskedMerging(maskType, shapeOut)
		if hasMerging && !resultInArg0 {
			// We have to copy the slice here because the sort will be visible from other
			// aliases when no reslicing is happening.
			newIn := make([]types.Operand, len(op.In), len(op.In)+1)
			copy(newIn, op.In)
			op.In = newIn
			op.In = append(op.In, op.Out[0])
			op.sortOperand()
			regInfoMerging, err = makeRegInfo(op, NoMem)
			if err != nil {
				panic(err)
			}
		}

		if shapeIn == OneImmIn || shapeIn == OneKmaskImmIn {
			opsDataImm = append(opsDataImm, opData{asm, gOp.Asm, len(gOp.In), regInfo, gOp.Commutative, outType, resultInArg0})
			if memOpData != nil {
				if *op.MemFeatures != "vbcst" {
					panic("simdgen only knows vbcst for mem ops for now")
				}
				opsDataImmLoad = append(opsDataImmLoad, *memOpData)
			}
			if hasMerging {
				mergingLen := len(gOp.In)
				if !resultInArg0 {
					mergingLen++
				}
				opsDataImmMerging = append(opsDataImmMerging, opData{asm, gOp.Asm, mergingLen, regInfoMerging, gOp.Commutative, outType, resultInArg0})
			}
		} else {
			if !seenUnpred[asm] {
				seenUnpred[asm] = true
				opsData = append(opsData, opData{asm, gOp.Asm, len(gOp.In), regInfo, gOp.Commutative, outType, resultInArg0})
			}
			// The inVariant implies machine ops only: one predicated instruction
			// per governing-predicate qualifier the encoding supports, reached by
			// peephole rather than by any API of its own.
			for _, pred := range gOp.svePredicatedOps() {
				predRegInfo, err := makeRegInfo(pred, NoMem)
				if err != nil {
					panic(err)
				}
				predResultInArg0 := false
				switch idx := pred.sveInPlaceInput(); {
				case idx < 0:
				case idx == 0:
					// Where the first input is the merge source it is the whole
					// reason the destination is pinned, so commutativity — which
					// is about the two sources — does not enter into it.
					predResultInArg0 = pred.sveMergeSourceIn0 || !pred.Commutative
				default:
					panic(fmt.Errorf("simdgen: %s overwrites input %d; only the first input is supported: %s",
						pred.Asm, idx, pred))
				}
				opsData = append(opsData, opData{machineOpName(OneMask, pred), pred.Asm, len(pred.In),
					predRegInfo, pred.Commutative, outType, predResultInArg0})
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
					opsData = append(opsData, opData{machineOpName(OneMask, *prefixed), prefixed.Asm, len(prefixed.In),
						prefixedRegInfo, false, outType, true})
				}
			}
			if memOpData != nil {
				if *op.MemFeatures != "vbcst" {
					panic("simdgen only knows vbcst for mem ops for now")
				}
				opsDataLoad = append(opsDataLoad, *memOpData)
			}
			if hasMerging {
				mergingLen := len(gOp.In)
				if !resultInArg0 {
					mergingLen++
				}
				opsDataMerging = append(opsDataMerging, opData{asm, gOp.Asm, mergingLen, regInfoMerging, gOp.Commutative, outType, resultInArg0})
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

	err := t.Execute(buffer, machineOpsData{
		SIMDTag:           archInfo.SIMDTag,
		RegInfoParams:     archInfo.RegInfoParams,
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
