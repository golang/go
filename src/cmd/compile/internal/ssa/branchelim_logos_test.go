// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestLogosBranchElimZeroSIMD(t *testing.T) {
	c := testConfig(t)
	boolType := c.config.Types.Bool
	i32 := c.config.Types.Int32
	ptr := c.config.Types.BytePtr
	// This true arm has exactly two values, within elimIf's speculation
	// limit. Its scalar result does not make the vector extraction safe
	// to execute before the feature check.
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
			Valu("feature", OpArg, boolType, 0, c.Temp(boolType)),
			Valu("fallback", OpConst32, i32, 7, nil),
			If("feature", "selected", "join")),
		Bloc("selected",
			Valu("zero", OpZeroSIMD, types.TypeVec128, 0, nil),
			Valu("element", OpGetElemInt32x4, i32, 0, nil, "zero"),
			Goto("join")),
		Bloc("join",
			Valu("result", OpPhi, i32, 0, nil, "fallback", "element"),
			Valu("store", OpStore, types.TypeMem, 0, i32, "p", "result", "mem"),
			Exit("store")),
	)

	CheckFunc(fun.f)
	checkLogosSIMDBranch(t, fun, "entry", "feature", "zero", "element")
	branchelim(fun.f)
	CheckFunc(fun.f)
	deadcode(fun.f)
	CheckFunc(fun.f)
	checkLogosSIMDBranch(t, fun, "entry", "feature", "zero", "element")
}

func TestLogosBranchElimSIMDOperand(t *testing.T) {
	for _, tc := range []struct {
		name string
		typ  *types.Type
		op   Op
		args []string
	}{
		{"vector", types.TypeVec256, OpZeroSIMD, nil},
		{"mask", types.TypeMask, OpCvt16toMask16x16, []string{"bits"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			c := testConfig(t)
			boolType := c.config.Types.Bool
			u16 := c.config.Types.UInt16
			ptr := c.config.Types.BytePtr
			// The input already exists before the inner guard. In the
			// vector case AVX can produce the 256-bit zero, while the
			// conversion uses AVX-512 mask instructions. Checking only
			// the result type would miss this scalar-valued operation.
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
					Valu("outerFeature", OpArg, boolType, 0, c.Temp(boolType)),
					Valu("innerFeature", OpArg, boolType, 0, c.Temp(boolType)),
					Valu("bits", OpArg, u16, 0, c.Temp(u16)),
					Valu("fallback", OpConst16, u16, 7, nil),
					If("outerFeature", "outer", "exit")),
				Bloc("outer",
					Valu("input", tc.op, tc.typ, 0, nil, tc.args...),
					If("innerFeature", "selected", "join")),
				Bloc("selected",
					Valu("converted", OpCvtMask16x16to16, u16, 0, nil, "input"),
					Goto("join")),
				Bloc("join",
					Valu("result", OpPhi, u16, 0, nil, "fallback", "converted"),
					Valu("store", OpStore, types.TypeMem, 0, u16, "p", "result", "mem"),
					Exit("store")),
				Bloc("exit", Exit("mem")),
			)

			CheckFunc(fun.f)
			checkLogosSIMDBranch(t, fun, "outer", "innerFeature", "converted")
			branchelim(fun.f)
			CheckFunc(fun.f)
			deadcode(fun.f)
			CheckFunc(fun.f)
			checkLogosSIMDBranch(t, fun, "outer", "innerFeature", "converted")
		})
	}
}

func TestLogosBranchElimMaskResult(t *testing.T) {
	c := testConfig(t)
	boolType := c.config.Types.Bool
	u16 := c.config.Types.UInt16
	ptr := c.config.Types.BytePtr
	// TypeMask is only eight bytes wide. It still represents SIMD state,
	// so neither a size check nor a list of vector widths is sufficient.
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
			Valu("feature", OpArg, boolType, 0, c.Temp(boolType)),
			Valu("bits", OpArg, u16, 0, c.Temp(u16)),
			Valu("fallback", OpConst16, u16, 7, nil),
			If("feature", "selected", "join")),
		Bloc("selected",
			Valu("mask", OpCvt16toMask16x16, types.TypeMask, 0, nil, "bits"),
			Valu("converted", OpCvtMask16x16to16, u16, 0, nil, "mask"),
			Goto("join")),
		Bloc("join",
			Valu("result", OpPhi, u16, 0, nil, "fallback", "converted"),
			Valu("store", OpStore, types.TypeMem, 0, u16, "p", "result", "mem"),
			Exit("store")),
	)

	CheckFunc(fun.f)
	checkLogosSIMDBranch(t, fun, "entry", "feature", "mask", "converted")
	branchelim(fun.f)
	CheckFunc(fun.f)
	deadcode(fun.f)
	CheckFunc(fun.f)
	checkLogosSIMDBranch(t, fun, "entry", "feature", "mask", "converted")
}

func TestLogosBranchElimScalarControl(t *testing.T) {
	c := testConfig(t)
	boolType := c.config.Types.Bool
	u16 := c.config.Types.UInt16
	ptr := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
			Valu("feature", OpArg, boolType, 0, c.Temp(boolType)),
			Valu("input", OpArg, u16, 0, c.Temp(u16)),
			Valu("one", OpConst16, u16, 1, nil),
			Valu("fallback", OpConst16, u16, 7, nil),
			If("feature", "selected", "join")),
		Bloc("selected",
			Valu("added", OpAdd16, u16, 0, nil, "input", "one"),
			Goto("join")),
		Bloc("join",
			Valu("result", OpPhi, u16, 0, nil, "fallback", "added"),
			Valu("store", OpStore, types.TypeMem, 0, u16, "p", "result", "mem"),
			Exit("store")),
	)

	CheckFunc(fun.f)
	branchelim(fun.f)
	CheckFunc(fun.f)
	deadcode(fun.f)
	CheckFunc(fun.f)
	result := fun.values["result"]
	if result.Op != OpCondSelect {
		t.Fatalf("scalar diamond was not eliminated: result is %s", result.LongString())
	}
	if result.Args[2] != fun.values["feature"] || fun.f.Entry.Kind != BlockExit || len(fun.f.Blocks) != 1 {
		t.Errorf("scalar diamond did not become one block with the original selection condition")
	}
}

func checkLogosSIMDBranch(t *testing.T, fun fun, guard, condition string, values ...string) {
	t.Helper()
	b := fun.blocks[guard]
	if b.Kind != BlockIf || b.Controls[0] != fun.values[condition] {
		t.Errorf("CPU feature guard %s was eliminated while its SIMD operation remained", guard)
		return
	}
	selected := b.Succs[0].Block()
	for _, name := range values {
		v := fun.values[name]
		if !fun.f.Sdom().IsAncestorEq(selected, v.Block) {
			t.Errorf("%s in %s escaped the positive edge of CPU feature guard %s", v.LongString(), v.Block, guard)
		}
	}
}
