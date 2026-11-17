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

// Each SIMD store is reachable only through its own guard. The first guard
// also has a dominated duplicate, so ordinary CSE can share that zero without
// crossing either guard. Distinct destination arguments keep all stores live.
func guardedSIMDZeroStores(c *Conf, vec *types.Type) fun {
	return c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("guardA", OpArg, c.config.Types.Bool, 0, c.Temp(c.config.Types.Bool)),
			Valu("guardB", OpArg, c.config.Types.Bool, 0, c.Temp(c.config.Types.Bool)),
			Valu("dstA", OpArg, vec.PtrTo(), 0, c.Temp(vec.PtrTo())),
			Valu("dstChild", OpArg, vec.PtrTo(), 0, c.Temp(vec.PtrTo())),
			Valu("dstB", OpArg, vec.PtrTo(), 0, c.Temp(vec.PtrTo())),
			If("guardA", "a", "skipA")),
		Bloc("a",
			Valu("zeroA", OpZeroSIMD, vec, 0, nil),
			Valu("storeA", OpStore, types.TypeMem, 0, vec, "dstA", "zeroA", "mem"),
			Goto("child")),
		Bloc("child",
			Valu("zeroChild", OpZeroSIMD, vec, 0, nil),
			Valu("storeChild", OpStore, types.TypeMem, 0, vec, "dstChild", "zeroChild", "storeA"),
			Goto("second")),
		Bloc("skipA", Goto("second")),
		Bloc("second",
			Valu("memSecond", OpPhi, types.TypeMem, 0, nil, "storeChild", "mem"),
			If("guardB", "b", "skipB")),
		Bloc("b",
			Valu("zeroB", OpZeroSIMD, vec, 0, nil),
			Valu("storeB", OpStore, types.TypeMem, 0, vec, "dstB", "zeroB", "memSecond"),
			Goto("exit")),
		Bloc("skipB", Goto("exit")),
		Bloc("exit",
			Valu("memExit", OpPhi, types.TypeMem, 0, nil, "storeB", "memSecond"),
			Exit("memExit")))
}

func TestZCSEPreservesSIMDZeroGuards(t *testing.T) {
	for _, tc := range []struct {
		name string
		vec  *types.Type
	}{
		{"128", types.TypeVec128},
		{"256", types.TypeVec256},
		{"512", types.TypeVec512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fun := guardedSIMDZeroStores(testConfig(t), tc.vec)
			CheckFunc(fun.f)
			zcse(fun.f)
			deadcode(fun.f)
			CheckFunc(fun.f)

			a := fun.values["storeA"].Args[1]
			b := fun.values["storeB"].Args[1]
			if a.Block != fun.blocks["a"] {
				t.Errorf("first guarded SIMD zero moved to %s, want %s", a.Block, fun.blocks["a"])
			}
			if b.Block != fun.blocks["b"] {
				t.Errorf("second guarded SIMD zero moved to %s, want %s", b.Block, fun.blocks["b"])
			}
			if a == b {
				t.Error("independently guarded SIMD stores share a zero")
			}
		})
	}
}

func TestCSECommonsSIMDZeroWithinGuard(t *testing.T) {
	for _, tc := range []struct {
		name string
		vec  *types.Type
	}{
		{"128", types.TypeVec128},
		{"256", types.TypeVec256},
		{"512", types.TypeVec512},
	} {
		t.Run(tc.name, func(t *testing.T) {
			fun := guardedSIMDZeroStores(testConfig(t), tc.vec)
			CheckFunc(fun.f)
			cse(fun.f)
			deadcode(fun.f)
			CheckFunc(fun.f)

			a := fun.values["storeA"].Args[1]
			child := fun.values["storeChild"].Args[1]
			b := fun.values["storeB"].Args[1]
			if child != a {
				t.Error("ordinary CSE did not share a zero within the dominating guard")
			}
			if a.Block != fun.blocks["a"] || b.Block != fun.blocks["b"] {
				t.Error("ordinary CSE moved a zero across its guard")
			}
			if a == b {
				t.Error("ordinary CSE shared a zero across independent guards")
			}
		})
	}
}
