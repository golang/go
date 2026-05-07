// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"simd/archsimd/_gen/sgutil"
)

// writeSIMDGenericOps generates the generic ops and writes it to simdAMD64ops.go
// within the specified directory.
func writeSIMDGenericOps(ops []Operation, genericOpsFilePath string) *bytes.Buffer {

	// Generate fresh ops for current arch.
	const currentArch = "amd64"
	var newOps []sgutil.GenericOpsData
	for _, op := range ops {
		if op.NoGenericOps != nil && *op.NoGenericOps == "true" {
			continue
		}
		if op.SkipMaskedMethod() {
			continue
		}
		_, _, _, immType, gOp := op.shape()

		newOps = append(newOps, sgutil.GenericOpsData{
			OpName:  gOp.GenericName(),
			OpInLen: len(gOp.In),
			Comm:    op.Commutative,
			HasAux:  immType == VarImm || immType == ConstVarImm,
			Archs:   []string{currentArch},
		})
	}

	buf := sgutil.MergeSIMDGenericOps(newOps, genericOpsFilePath, "amd64")

	return buf
}
