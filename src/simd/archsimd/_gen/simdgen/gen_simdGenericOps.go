// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"fmt"
	"simd/archsimd/_gen/sgutil"
)

// writeSIMDGenericOps writes the generic ops for the current architecture to
// buffer.
func writeSIMDGenericOps(buffer *bytes.Buffer, ops []Operation) {
	archInfo := CurrentArch()
	var newOps []sgutil.GenericOpsData
	for _, op := range ops {
		if op.NoGenericOps != nil && *op.NoGenericOps == "true" {
			continue
		}
		if op.SkipMaskedMethod() {
			continue
		}
		_, _, _, immType, gOp, _ := op.shape()

		newOps = append(newOps, sgutil.GenericOpsData{
			OpName: gOp.GenericName(),
			// An implicit-all-true predicate is a machine-op input only; the
			// generic op is unpredicated, so exclude it from the arg count.
			OpInLen: len(gOp.In) - gOp.implicitPredCount(),
			Comm:    op.Commutative,
			HasAux:  immType == VarImm || immType == VarImmLim || immType == ConstVarImm,
		})
	}

	fmt.Fprintf(buffer, "%s\n\n", archInfo.GeneratedHeader)
	sgutil.WriteSIMDGenericOps(buffer, newOps, archInfo.GoTypeArch)
}
