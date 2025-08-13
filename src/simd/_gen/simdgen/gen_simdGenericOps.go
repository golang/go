// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"sort"
)

const simdGenericOpsTmpl = `
package main

func simdGenericOps() []opData {
	return []opData{
{{- range .Ops }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, commutative: {{.Comm}}},
{{- end }}
{{- range .OpsImm }}
		{name: "{{.OpName}}", argLength: {{.OpInLen}}, commutative: {{.Comm}}, aux: "UInt8"},
{{- end }}
	}
}
`

// writeSIMDGenericOps generates the generic ops and writes it to simdAMD64ops.go
// within the specified directory.
func writeSIMDGenericOps(ops []Operation) *bytes.Buffer {
	t := templateOf(simdGenericOpsTmpl, "simdgenericOps")
	buffer := new(bytes.Buffer)
	buffer.WriteString(generatedHeader)

	type genericOpsData struct {
		OpName  string
		OpInLen int
		Comm    bool
	}
	type opData struct {
		Ops    []genericOpsData
		OpsImm []genericOpsData
	}
	var opsData opData
	for _, op := range ops {
		if op.NoGenericOps != nil && *op.NoGenericOps == "true" {
			continue
		}
		_, _, _, immType, gOp := op.shape()
		gOpData := genericOpsData{gOp.GenericName(), len(gOp.In), op.Commutative}
		if immType == VarImm || immType == ConstVarImm {
			opsData.OpsImm = append(opsData.OpsImm, gOpData)
		} else {
			opsData.Ops = append(opsData.Ops, gOpData)
		}
	}
	sort.Slice(opsData.Ops, func(i, j int) bool {
		return compareNatural(opsData.Ops[i].OpName, opsData.Ops[j].OpName) < 0
	})
	sort.Slice(opsData.OpsImm, func(i, j int) bool {
		return compareNatural(opsData.OpsImm[i].OpName, opsData.OpsImm[j].OpName) < 0
	})

	err := t.Execute(buffer, opsData)
	if err != nil {
		panic(fmt.Errorf("failed to execute template: %w", err))
	}

	return buffer
}
