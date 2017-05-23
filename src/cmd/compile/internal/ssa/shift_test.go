// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"testing"
)

func TestShiftConstAMD64(t *testing.T) {
	c := testConfig(t)
	fun := makeConstShiftFunc(c, 18, OpLsh64x64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHLQconst: 1, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})
	fun.f.Free()
	fun = makeConstShiftFunc(c, 66, OpLsh64x64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHLQconst: 0, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})
	fun.f.Free()
	fun = makeConstShiftFunc(c, 18, OpRsh64Ux64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHRQconst: 1, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})
	fun.f.Free()
	fun = makeConstShiftFunc(c, 66, OpRsh64Ux64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHRQconst: 0, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})
	fun.f.Free()
	fun = makeConstShiftFunc(c, 18, OpRsh64x64, TypeInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SARQconst: 1, OpAMD64CMPQconst: 0})
	fun.f.Free()
	fun = makeConstShiftFunc(c, 66, OpRsh64x64, TypeInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SARQconst: 1, OpAMD64CMPQconst: 0})
	fun.f.Free()
}

func makeConstShiftFunc(c *Config, amount int64, op Op, typ Type) fun {
	ptyp := &TypeImpl{Size_: 8, Ptr: true, Name: "ptr"}
	fun := Fun(c, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("SP", OpSP, TypeUInt64, 0, nil),
			Valu("argptr", OpOffPtr, ptyp, 8, nil, "SP"),
			Valu("resptr", OpOffPtr, ptyp, 16, nil, "SP"),
			Valu("load", OpLoad, typ, 0, nil, "argptr", "mem"),
			Valu("c", OpConst64, TypeUInt64, amount, nil),
			Valu("shift", op, typ, 0, nil, "load", "c"),
			Valu("store", OpStore, TypeMem, 8, nil, "resptr", "shift", "mem"),
			Exit("store")))
	Compile(fun.f)
	return fun
}
