// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestLogosZCSEPreservesSIMDGuard(t *testing.T) {
	for _, vec := range []*types.Type{types.TypeVec128, types.TypeVec256, types.TypeVec512} {
		t.Run(vec.String(), func(t *testing.T) {
			c := testConfig(t)
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("ptr", OpArg, types.NewPtr(vec), 0, nil),
					Valu("available", OpArg, c.config.Types.Bool, 8, nil),
					Valu("again", OpArg, c.config.Types.Bool, 9, nil),
					Goto("loop")),
				Bloc("loop",
					Valu("loopmem", OpPhi, types.TypeMem, 0, nil, "mem", "joinmem"),
					If("available", "vector", "scalar")),
				Bloc("vector",
					Valu("zero", OpZeroSIMD, vec, 0, nil),
					Valu("store", OpStore, types.TypeMem, 0, vec, "ptr", "zero", "loopmem"),
					Goto("join")),
				Bloc("scalar", Goto("join")),
				Bloc("join",
					Valu("joinmem", OpPhi, types.TypeMem, 0, nil, "store", "loopmem"),
					If("again", "loop", "exit")),
				Bloc("exit", Ret("joinmem")),
			)
			CheckFunc(fun.f)
			zcse(fun.f)
			CheckFunc(fun.f)
			zero := fun.values["store"].Args[1]
			if !fun.f.Sdom().IsAncestorEq(fun.blocks["vector"], zero.Block) {
				t.Fatalf("SIMD zero moved to %s, outside the feature guard's true branch", zero.Block)
			}
			if zero.Block != fun.blocks["vector"] {
				t.Fatalf("SIMD zero moved from its original block to %s", zero.Block)
			}
		})
	}
}

func TestLogosZCSEKeepsSiblingSIMDZeros(t *testing.T) {
	c := testConfig(t)
	vec := types.TypeVec512
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, types.NewPtr(vec), 0, nil),
			Valu("choose", OpArg, c.config.Types.Bool, 8, nil),
			If("choose", "left", "right")),
		Bloc("left",
			Valu("leftzero", OpZeroSIMD, vec, 0, nil),
			Valu("leftstore", OpStore, types.TypeMem, 0, vec, "ptr", "leftzero", "mem"),
			Goto("exit")),
		Bloc("right",
			Valu("rightzero", OpZeroSIMD, vec, 0, nil),
			Valu("rightstore", OpStore, types.TypeMem, 0, vec, "ptr", "rightzero", "mem"),
			Goto("exit")),
		Bloc("exit",
			Valu("joinmem", OpPhi, types.TypeMem, 0, nil, "leftstore", "rightstore"),
			Ret("joinmem")),
	)
	CheckFunc(fun.f)
	zcse(fun.f)
	cse(fun.f)
	CheckFunc(fun.f)
	for _, side := range []string{"left", "right"} {
		zero := fun.values[side+"store"].Args[1]
		if zero.Block != fun.blocks[side] {
			t.Errorf("%s store uses SIMD zero from %s, outside its branch", side, zero.Block)
		}
	}
}

func TestLogosZCSEAllowsDominatedSIMDCommoning(t *testing.T) {
	c := testConfig(t)
	vec := types.TypeVec256
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, types.NewPtr(vec), 0, nil),
			Valu("available", OpArg, c.config.Types.Bool, 8, nil),
			If("available", "vector", "exit")),
		Bloc("vector",
			Valu("zero1", OpZeroSIMD, vec, 0, nil),
			Valu("zero2", OpZeroSIMD, vec, 0, nil),
			Valu("store1", OpStore, types.TypeMem, 0, vec, "ptr", "zero1", "mem"),
			Valu("store2", OpStore, types.TypeMem, 0, vec, "ptr", "zero2", "store1"),
			Goto("exit")),
		Bloc("exit",
			Valu("joinmem", OpPhi, types.TypeMem, 0, nil, "mem", "store2"),
			Ret("joinmem")),
	)
	CheckFunc(fun.f)
	zcse(fun.f)
	cse(fun.f)
	CheckFunc(fun.f)
	zero1, zero2 := fun.values["store1"].Args[1], fun.values["store2"].Args[1]
	if zero1 != zero2 {
		t.Fatal("dominance-aware CSE did not merge identical SIMD zeros in one block")
	}
	if zero1.Block != fun.blocks["vector"] {
		t.Fatalf("common SIMD zero moved outside its guard to %s", zero1.Block)
	}
}

func TestLogosZCSEStillHoistsScalarConstants(t *testing.T) {
	c := testConfig(t)
	i64 := c.config.Types.Int64
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, types.NewPtr(i64), 0, nil),
			Goto("body")),
		Bloc("body",
			Valu("one1", OpConst64, i64, 1, nil),
			Valu("one2", OpConst64, i64, 1, nil),
			Valu("sum", OpAdd64, i64, 0, nil, "one1", "one2"),
			Valu("store", OpStore, types.TypeMem, 0, i64, "ptr", "sum", "mem"),
			Ret("store")),
	)
	CheckFunc(fun.f)
	zcse(fun.f)
	CheckFunc(fun.f)
	args := fun.values["sum"].Args
	if args[0] != args[1] || args[0].Block != fun.f.Entry {
		t.Fatal("scalar constants were not commoned in the entry block")
	}
}
