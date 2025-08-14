// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"sort"
	"strings"
)

const simdMachineOpsTmpl = `
package main

func simdAMD64Ops(v11, v21, v2k, vkv, v2kv, v2kk, v31, v3kv, vgpv, vgp, vfpv, vfpkv, w11, w21, w2k, wkw, w2kw, w2kk, w31, w3kw, wgpw, wgp, wfpw, wfpkw regInfo) []opData {
	return []opData{
{{- range .OpsData }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}", commutative: {{.Comm}}, typ: "{{.Type}}", resultInArg0: {{.ResultInArg0}}},
{{- end }}
{{- range .OpsDataImm }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, reg: {{.RegInfo}}, asm: "{{.Asm}}", aux: "UInt8", commutative: {{.Comm}}, typ: "{{.Type}}", resultInArg0: {{.ResultInArg0}}},
{{- end }}
	}
}
`

// writeSIMDMachineOps generates the machine ops and writes it to simdAMD64ops.go
// within the specified directory.
func writeSIMDMachineOps(ops []Operation) *bytes.Buffer {
	t := templateOf(simdMachineOpsTmpl, "simdAMD64Ops")
	buffer := new(bytes.Buffer)
	buffer.WriteString(generatedHeader)

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
		OpsData    []opData
		OpsDataImm []opData
	}

	regInfoSet := map[string]bool{
		"v11": true, "v21": true, "v2k": true, "v2kv": true, "v2kk": true, "vkv": true, "v31": true, "v3kv": true, "vgpv": true, "vgp": true, "vfpv": true, "vfpkv": true,
		"w11": true, "w21": true, "w2k": true, "w2kw": true, "w2kk": true, "wkw": true, "w31": true, "w3kw": true, "wgpw": true, "wgp": true, "wfpw": true, "wfpkw": true}
	opsData := make([]opData, 0)
	opsDataImm := make([]opData, 0)

	// Determine the "best" version of an instruction to use
	best := make(map[string]Operation)
	var mOpOrder []string
	countOverrides := func(s []Operand) int {
		a := 0
		for _, o := range s {
			if o.OverwriteBase != nil {
				a++
			}
		}
		return a
	}
	for _, op := range ops {
		_, _, maskType, _, gOp := op.shape()
		asm := machineOpName(maskType, gOp)
		other, ok := best[asm]
		if !ok {
			best[asm] = op
			mOpOrder = append(mOpOrder, asm)
			continue
		}
		// see if "op" is better than "other"
		if countOverrides(op.In)+countOverrides(op.Out) < countOverrides(other.In)+countOverrides(other.Out) {
			best[asm] = op
		}
	}

	for _, asm := range mOpOrder {
		op := best[asm]
		shapeIn, shapeOut, _, _, gOp := op.shape()

		// TODO: all our masked operations are now zeroing, we need to generate machine ops with merging masks, maybe copy
		// one here with a name suffix "Merging". The rewrite rules will need them.

		regInfo, err := op.regShape()
		if err != nil {
			panic(err)
		}
		idx, err := checkVecAsScalar(op)
		if err != nil {
			panic(err)
		}
		if idx != -1 {
			if regInfo == "v21" {
				regInfo = "vfpv"
			} else if regInfo == "v2kv" {
				regInfo = "vfpkv"
			} else {
				panic(fmt.Errorf("simdgen does not recognize uses of treatLikeAScalarOfSize with op regShape %s in op: %s", regInfo, op))
			}
		}
		// Makes AVX512 operations use upper registers
		if strings.Contains(op.CPUFeature, "AVX512") {
			regInfo = strings.ReplaceAll(regInfo, "v", "w")
		}
		if _, ok := regInfoSet[regInfo]; !ok {
			panic(fmt.Errorf("unsupported register constraint, please update the template and AMD64Ops.go: %s.  Op is %s", regInfo, op))
		}
		var outType string
		if shapeOut == OneVregOut || shapeOut == OneVregOutAtIn || gOp.Out[0].OverwriteClass != nil {
			// If class overwrite is happening, that's not really a mask but a vreg.
			outType = fmt.Sprintf("Vec%d", *gOp.Out[0].Bits)
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
		if shapeIn == OneImmIn || shapeIn == OneKmaskImmIn {
			opsDataImm = append(opsDataImm, opData{asm, gOp.Asm, len(gOp.In), regInfo, gOp.Commutative, outType, resultInArg0})
		} else {
			opsData = append(opsData, opData{asm, gOp.Asm, len(gOp.In), regInfo, gOp.Commutative, outType, resultInArg0})
		}
	}
	sort.Slice(opsData, func(i, j int) bool {
		return compareNatural(opsData[i].OpName, opsData[j].OpName) < 0
	})
	sort.Slice(opsDataImm, func(i, j int) bool {
		return compareNatural(opsData[i].OpName, opsData[j].OpName) < 0
	})
	err := t.Execute(buffer, machineOpsData{opsData, opsDataImm})
	if err != nil {
		panic(fmt.Errorf("failed to execute template: %w", err))
	}

	return buffer
}
