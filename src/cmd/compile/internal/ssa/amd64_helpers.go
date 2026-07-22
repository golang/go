// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

// amd64CapAVXShift caps an AMD64 AVX vector shift amount c so that over-shifts
// always result in 0.
//
// These instructions have room for an 8-bit immediate and any value larger than
// the element width will result in 0 or -1 (for an arithmetic right shift).
// Thus, we simply cap this at 255.
func amd64CapAVXShift(auxInt int64) uint8 {
	u := AuxIntToUint64(auxInt)
	if u > 255 {
		return 255
	}
	return uint8(u)
}

// flagify rewrites v which is (X ...) to (Select0 (Xflags ...)).
func flagify(v *ssacore.Value) bool {
	var flagVersion ssaop.Op
	switch v.Op {
	case ssaop.OpAMD64ADDQconst:
		flagVersion = ssaop.OpAMD64ADDQconstflags
	case ssaop.OpAMD64ADDLconst:
		flagVersion = ssaop.OpAMD64ADDLconstflags
	default:
		base.Fatalf("can't flagify op %s", v.Op)
	}
	inner := v.CopyInto(v.Block)
	inner.Op = flagVersion
	inner.Type = types.NewTuple(v.Type, types.TypeFlags)
	v.Reset(ssaop.OpSelect0)
	v.AddArg(inner)
	return true
}

// sequentialAddresses reports true if it can prove that x + n == y
func sequentialAddresses(x, y *ssacore.Value, n int64) bool {
	if x == y && n == 0 {
		return true
	}
	if x.Op == ssaop.Op386ADDL && y.Op == ssaop.Op386LEAL1 && y.AuxInt == n && y.Aux == nil &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	if x.Op == ssaop.Op386LEAL1 && y.Op == ssaop.Op386LEAL1 && y.AuxInt == x.AuxInt+n && x.Aux == y.Aux &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	if x.Op == ssaop.OpAMD64ADDQ && y.Op == ssaop.OpAMD64LEAQ1 && y.AuxInt == n && y.Aux == nil &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	if x.Op == ssaop.OpAMD64LEAQ1 && y.Op == ssaop.OpAMD64LEAQ1 && y.AuxInt == x.AuxInt+n && x.Aux == y.Aux &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	return false
}

// validVal reports whether the value can be used
// as an argument to makeValAndOff.
func validVal(val int64) bool {
	return val == int64(int32(val))
}
