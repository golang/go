// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"testing"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
)

func TestShiftConstAMD64(t *testing.T) {
	c := testConfig(t)
	fun := makeConstShiftFunc(c, 18, ssaop.OpLsh64x64, c.config.Types.UInt64)
	checkOpcodeCounts(t, fun.f, map[ssaop.Op]int{ssaop.OpAMD64SHLQconst: 1, ssaop.OpAMD64CMPQconst: 0, ssaop.OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, 66, ssaop.OpLsh64x64, c.config.Types.UInt64)
	checkOpcodeCounts(t, fun.f, map[ssaop.Op]int{ssaop.OpAMD64SHLQconst: 0, ssaop.OpAMD64CMPQconst: 0, ssaop.OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, 18, ssaop.OpRsh64Ux64, c.config.Types.UInt64)
	checkOpcodeCounts(t, fun.f, map[ssaop.Op]int{ssaop.OpAMD64SHRQconst: 1, ssaop.OpAMD64CMPQconst: 0, ssaop.OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, 66, ssaop.OpRsh64Ux64, c.config.Types.UInt64)
	checkOpcodeCounts(t, fun.f, map[ssaop.Op]int{ssaop.OpAMD64SHRQconst: 0, ssaop.OpAMD64CMPQconst: 0, ssaop.OpAMD64ANDQconst: 0})

	fun = makeConstShiftFunc(c, 18, ssaop.OpRsh64x64, c.config.Types.Int64)
	checkOpcodeCounts(t, fun.f, map[ssaop.Op]int{ssaop.OpAMD64SARQconst: 1, ssaop.OpAMD64CMPQconst: 0})

	fun = makeConstShiftFunc(c, 66, ssaop.OpRsh64x64, c.config.Types.Int64)
	checkOpcodeCounts(t, fun.f, map[ssaop.Op]int{ssaop.OpAMD64SARQconst: 1, ssaop.OpAMD64CMPQconst: 0})
}

func makeConstShiftFunc(c *Conf, amount int64, op ssaop.Op, typ *types.Type) fun {
	ptyp := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("SP", ssaop.OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("argptr", ssaop.OpOffPtr, ptyp, 8, nil, "SP"),
			Valu("resptr", ssaop.OpOffPtr, ptyp, 16, nil, "SP"),
			Valu("load", ssaop.OpLoad, typ, 0, nil, "argptr", "mem"),
			Valu("c", ssaop.OpConst64, c.config.Types.UInt64, amount, nil),
			Valu("shift", op, typ, 0, nil, "load", "c"),
			Valu("store", ssaop.OpStore, types.TypeMem, 0, c.config.Types.UInt64, "resptr", "shift", "mem"),
			Exit("store")))
	runPasses(fun.f)
	return fun
}

func TestShiftToExtensionAMD64(t *testing.T) {
	c := testConfig(t)
	// Test that eligible pairs of constant shifts are converted to extensions.
	// For example:
	//   (uint64(x) << 32) >> 32 -> uint64(uint32(x))
	ops := map[ssaop.Op]int{
		ssaop.OpAMD64SHLQconst: 0, ssaop.OpAMD64SHLLconst: 0,
		ssaop.OpAMD64SHRQconst: 0, ssaop.OpAMD64SHRLconst: 0,
		ssaop.OpAMD64SARQconst: 0, ssaop.OpAMD64SARLconst: 0,
	}
	tests := [...]struct {
		amount      int64
		left, right ssaop.Op
		typ         *types.Type
	}{
		// unsigned
		{56, ssaop.OpLsh64x64, ssaop.OpRsh64Ux64, c.config.Types.UInt64},
		{48, ssaop.OpLsh64x64, ssaop.OpRsh64Ux64, c.config.Types.UInt64},
		{32, ssaop.OpLsh64x64, ssaop.OpRsh64Ux64, c.config.Types.UInt64},
		{24, ssaop.OpLsh32x64, ssaop.OpRsh32Ux64, c.config.Types.UInt32},
		{16, ssaop.OpLsh32x64, ssaop.OpRsh32Ux64, c.config.Types.UInt32},
		{8, ssaop.OpLsh16x64, ssaop.OpRsh16Ux64, c.config.Types.UInt16},
		// signed
		{56, ssaop.OpLsh64x64, ssaop.OpRsh64x64, c.config.Types.Int64},
		{48, ssaop.OpLsh64x64, ssaop.OpRsh64x64, c.config.Types.Int64},
		{32, ssaop.OpLsh64x64, ssaop.OpRsh64x64, c.config.Types.Int64},
		{24, ssaop.OpLsh32x64, ssaop.OpRsh32x64, c.config.Types.Int32},
		{16, ssaop.OpLsh32x64, ssaop.OpRsh32x64, c.config.Types.Int32},
		{8, ssaop.OpLsh16x64, ssaop.OpRsh16x64, c.config.Types.Int16},
	}
	for _, tc := range tests {
		fun := makeShiftExtensionFunc(c, tc.amount, tc.left, tc.right, tc.typ)
		checkOpcodeCounts(t, fun.f, ops)
	}
}

// makeShiftExtensionFunc generates a function containing:
//
//	(rshift (lshift (Const64 [amount])) (Const64 [amount]))
//
// This may be equivalent to a sign or zero extension.
func makeShiftExtensionFunc(c *Conf, amount int64, lshift, rshift ssaop.Op, typ *types.Type) fun {
	ptyp := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("SP", ssaop.OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("argptr", ssaop.OpOffPtr, ptyp, 8, nil, "SP"),
			Valu("resptr", ssaop.OpOffPtr, ptyp, 16, nil, "SP"),
			Valu("load", ssaop.OpLoad, typ, 0, nil, "argptr", "mem"),
			Valu("c", ssaop.OpConst64, c.config.Types.UInt64, amount, nil),
			Valu("lshift", lshift, typ, 0, nil, "load", "c"),
			Valu("rshift", rshift, typ, 0, nil, "lshift", "c"),
			Valu("store", ssaop.OpStore, types.TypeMem, 0, c.config.Types.UInt64, "resptr", "rshift", "mem"),
			Exit("store")))
	runPasses(fun.f)
	return fun
}

// runPasses is a simplified version of Compile that runs the passes
// for the tests in this file.
func runPasses(f *ssa.Func) {
	for i := range passes {
		p := &passes[i]
		if !f.Config.Optimize && !p.Required || p.Disabled {
			continue
		}
		f.Pass = p
		p.Fn(f)
	}
}
