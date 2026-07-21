// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"simd/archsimd/_gen/sgutil"
)

// writeSIMDGenericOps generates the generic ops for the current architecture,
// merges them with existing ops from other architectures, and returns the
// result as a buffer ready for writing.
func writeSIMDGenericOps(ops []Operation, genericOpsFilePath string) *bytes.Buffer {
	// Generate fresh ops for current arch.
	currentArch := CurrentArch().Arch
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
			OpName:  gOp.GenericName(),
			OpInLen: len(gOp.In),
			Comm:    op.Commutative,
			HasAux:  immType == VarImm || immType == VarImmLim || immType == ConstVarImm,
			Archs:   []string{currentArch},
		})
	}

	buf := sgutil.MergeSIMDGenericOps(newOps, genericOpsFilePath, currentArch)

	return buf
}
