// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacore

import (
	"fmt"

	"cmd/compile/internal/abi"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa/ssaop"
)

// ArgOpAndRegisterFor converts an abi register index into an ssa Op and corresponding
// arg register index.
func ArgOpAndRegisterFor(r abi.RegIndex, abiConfig *abi.ABIConfig) (ssaop.Op, int64) {
	i := abiConfig.FloatIndexFor(r)
	if i >= 0 { // float PR
		return ssaop.OpArgFloatReg, i
	}
	return ssaop.OpArgIntReg, int64(r)
}

// ParamAssignmentForArgName returns the ABIParamAssignment for f's arg with matching name.
func ParamAssignmentForArgName(f *Func, name *ir.Name) *abi.ABIParamAssignment {
	abiInfo := f.OwnAux.AbiInfo
	ip := abiInfo.InParams()
	for i, a := range ip {
		if a.Name == name {
			return &ip[i]
		}
	}
	panic(fmt.Errorf("Did not match param %v in prInfo %+v", name, abiInfo.InParams()))
}
