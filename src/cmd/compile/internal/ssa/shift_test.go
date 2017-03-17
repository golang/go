// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"testing"
)

func TestShiftConstAMD64(t *testing.T) {
	c := testConfig(t)
	fe := DummyFrontend{t}
	fun := makeConstShiftFunc(c, fe, 18, OpLsh64x64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHLQconst: 1, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, fe, 66, OpLsh64x64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHLQconst: 0, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, fe, 18, OpRsh64Ux64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHRQconst: 1, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, fe, 66, OpRsh64Ux64, TypeUInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SHRQconst: 0, OpAMD64CMPQconst: 0, OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, fe, 18, OpRsh64x64, TypeInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SARQconst: 1, OpAMD64CMPQconst: 0})

	fun = makeConstShiftFunc(c, fe, 66, OpRsh64x64, TypeInt64)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpAMD64SARQconst: 1, OpAMD64CMPQconst: 0})
}

func makeConstShiftFunc(c *Config, fe Frontend, amount int64, op Op, typ Type) fun {
	ptyp := &TypeImpl{Size_: 8, Ptr: true, Name: "ptr"}
	fun := Fun(c, fe, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("SP", OpSP, TypeUInt64, 0, nil),
			Valu("argptr", OpOffPtr, ptyp, 8, nil, "SP"),
			Valu("resptr", OpOffPtr, ptyp, 16, nil, "SP"),
			Valu("load", OpLoad, typ, 0, nil, "argptr", "mem"),
			Valu("c", OpConst64, TypeUInt64, amount, nil),
			Valu("shift", op, typ, 0, nil, "load", "c"),
			Valu("store", OpStore, TypeMem, 0, TypeUInt64, "resptr", "shift", "mem"),
			Exit("store")))
	Compile(fun.f)
	return fun
}

func TestShiftToExtensionAMD64(t *testing.T) {
	// Test that eligible pairs of constant shifts are converted to extensions.
	// For example:
	//   (uint64(x) << 32) >> 32 -> uint64(uint32(x))
	ops := map[Op]int{
		OpAMD64SHLQconst: 0, OpAMD64SHLLconst: 0,
		OpAMD64SHRQconst: 0, OpAMD64SHRLconst: 0,
		OpAMD64SARQconst: 0, OpAMD64SARLconst: 0,
	}
	tests := [...]struct {
		amount      int64
		left, right Op
		typ         Type
	}{
		// unsigned
		{56, OpLsh64x64, OpRsh64Ux64, TypeUInt64},
		{48, OpLsh64x64, OpRsh64Ux64, TypeUInt64},
		{32, OpLsh64x64, OpRsh64Ux64, TypeUInt64},
		{24, OpLsh32x64, OpRsh32Ux64, TypeUInt32},
		{16, OpLsh32x64, OpRsh32Ux64, TypeUInt32},
		{8, OpLsh16x64, OpRsh16Ux64, TypeUInt16},
		// signed
		{56, OpLsh64x64, OpRsh64x64, TypeInt64},
		{48, OpLsh64x64, OpRsh64x64, TypeInt64},
		{32, OpLsh64x64, OpRsh64x64, TypeInt64},
		{24, OpLsh32x64, OpRsh32x64, TypeInt32},
		{16, OpLsh32x64, OpRsh32x64, TypeInt32},
		{8, OpLsh16x64, OpRsh16x64, TypeInt16},
	}
	c := testConfig(t)
	fe := DummyFrontend{t}
	for _, tc := range tests {
		fun := makeShiftExtensionFunc(c, fe, tc.amount, tc.left, tc.right, tc.typ)
		checkOpcodeCounts(t, fun.f, ops)
	}
}

// makeShiftExtensionFunc generates a function containing:
//
//   (rshift (lshift (Const64 [amount])) (Const64 [amount]))
//
// This may be equivalent to a sign or zero extension.
func makeShiftExtensionFunc(c *Config, fe Frontend, amount int64, lshift, rshift Op, typ Type) fun {
	ptyp := &TypeImpl{Size_: 8, Ptr: true, Name: "ptr"}
	fun := Fun(c, fe, "entry",
		Bloc("entry",
			Valu("mem", OpInitMem, TypeMem, 0, nil),
			Valu("SP", OpSP, TypeUInt64, 0, nil),
			Valu("argptr", OpOffPtr, ptyp, 8, nil, "SP"),
			Valu("resptr", OpOffPtr, ptyp, 16, nil, "SP"),
			Valu("load", OpLoad, typ, 0, nil, "argptr", "mem"),
			Valu("c", OpConst64, TypeUInt64, amount, nil),
			Valu("lshift", lshift, typ, 0, nil, "load", "c"),
			Valu("rshift", rshift, typ, 0, nil, "lshift", "c"),
			Valu("store", OpStore, TypeMem, 0, TypeUInt64, "resptr", "rshift", "mem"),
			Exit("store")))
	Compile(fun.f)
	return fun
}
