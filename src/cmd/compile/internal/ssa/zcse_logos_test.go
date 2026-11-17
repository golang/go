// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestLogosZeroSIMDGuardNestedLoop(t *testing.T) {
	for _, vec := range []*types.Type{types.TypeVec128, types.TypeVec256, types.TypeVec512} {
		t.Run(vec.String(), func(t *testing.T) {
			c := testConfig(t)
			i64 := c.config.Types.Int64
			boolType := c.config.Types.Bool
			ptr := c.config.Types.BytePtr
			// The positive feature edge is inside an outer loop. The zeros
			// initialize a guarded inner loop, so tightening a zero hoisted
			// to entry cannot restore the guard without entering a loop.
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("start", OpInitMem, types.TypeMem, 0, nil),
					Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
					Valu("q", OpArg, ptr, 0, c.Temp(ptr)),
					Valu("feature", OpArg, boolType, 0, c.Temp(boolType)),
					Valu("outerLimit", OpArg, i64, 0, c.Temp(i64)),
					Valu("innerLimit", OpArg, i64, 0, c.Temp(i64)),
					Valu("zero", OpConst64, i64, 0, nil),
					Valu("one", OpConst64, i64, 1, nil),
					Goto("outer")),
				Bloc("outer",
					Valu("i", OpPhi, i64, 0, nil, "zero", "nextI"),
					Valu("outerMem", OpPhi, types.TypeMem, 0, nil, "start", "joinMem"),
					Valu("moreOuter", OpLess64, boolType, 0, nil, "i", "outerLimit"),
					If("moreOuter", "guard", "exit")),
				Bloc("guard", If("feature", "vectorInit", "scalar")),
				Bloc("vectorInit",
					Valu("vectorZero1", OpZeroSIMD, vec, 0, nil),
					Goto("inner")),
				Bloc("inner",
					Valu("j", OpPhi, i64, 0, nil, "zero", "nextJ"),
					Valu("innerMem", OpPhi, types.TypeMem, 0, nil, "outerMem", "store2"),
					Valu("moreInner", OpLess64, boolType, 0, nil, "j", "innerLimit"),
					If("moreInner", "body", "vectorDone")),
				Bloc("body",
					Valu("vectorZero2", OpZeroSIMD, vec, 0, nil),
					Valu("store1", OpStore, types.TypeMem, 0, vec, "p", "vectorZero1", "innerMem"),
					Valu("store2", OpStore, types.TypeMem, 0, vec, "q", "vectorZero2", "store1"),
					Valu("nextJ", OpAdd64, i64, 0, nil, "j", "one"),
					Goto("inner")),
				Bloc("vectorDone", Goto("join")),
				Bloc("scalar",
					Valu("scalarStore", OpStore, types.TypeMem, 0, i64, "p", "zero", "outerMem"),
					Goto("join")),
				Bloc("join",
					Valu("joinMem", OpPhi, types.TypeMem, 0, nil, "innerMem", "scalarStore"),
					Valu("nextI", OpAdd64, i64, 0, nil, "i", "one"),
					Goto("outer")),
				Bloc("exit", Exit("outerMem")),
			)

			CheckFunc(fun.f)
			checkLogosZeroSIMDGuard(t, fun, "input", "store1", "vectorInit")
			checkLogosZeroSIMDGuard(t, fun, "input", "store2", "vectorInit")
			for _, step := range []struct {
				name   string
				run    func(*Func)
				shared bool
			}{
				{"zcse", zcse, false},
				{"cse", cse, true},
				{"tighten", tighten, true},
			} {
				step.run(fun.f)
				deadcode(fun.f)
				CheckFunc(fun.f)
				checkLogosZeroSIMDGuard(t, fun, step.name, "store1", "vectorInit")
				checkLogosZeroSIMDGuard(t, fun, step.name, "store2", "vectorInit")
				// Ordinary dominance-aware CSE must still share the zeros
				// within the feature region; the fix must not disable it.
				if step.shared {
					checkLogosZeroSIMDCount(t, fun.f, step.name, 1)
				}
			}
		})
	}
}

func TestLogosZeroSIMDSiblingGuards(t *testing.T) {
	for _, vec := range []*types.Type{types.TypeVec128, types.TypeVec256, types.TypeVec512} {
		t.Run(vec.String(), func(t *testing.T) {
			c := testConfig(t)
			boolType := c.config.Types.Bool
			ptr := c.config.Types.BytePtr
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
					Valu("choice", OpArg, boolType, 0, c.Temp(boolType)),
					Valu("leftFeature", OpArg, boolType, 0, c.Temp(boolType)),
					Valu("rightFeature", OpArg, boolType, 0, c.Temp(boolType)),
					If("choice", "leftGuard", "rightGuard")),
				Bloc("leftGuard", If("leftFeature", "leftVector", "leftScalar")),
				Bloc("leftVector",
					Valu("leftZero", OpZeroSIMD, vec, 0, nil),
					Valu("leftStore", OpStore, types.TypeMem, 0, vec, "p", "leftZero", "mem"),
					Exit("leftStore")),
				Bloc("leftScalar", Exit("mem")),
				Bloc("rightGuard", If("rightFeature", "rightVector", "rightScalar")),
				Bloc("rightVector",
					Valu("rightZero", OpZeroSIMD, vec, 0, nil),
					Valu("rightStore", OpStore, types.TypeMem, 0, vec, "p", "rightZero", "mem"),
					Exit("rightStore")),
				Bloc("rightScalar", Exit("mem")),
			)

			CheckFunc(fun.f)
			checkLogosZeroSIMDGuard(t, fun, "input", "leftStore", "leftVector")
			checkLogosZeroSIMDGuard(t, fun, "input", "rightStore", "rightVector")
			for _, step := range []struct {
				name string
				run  func(*Func)
			}{
				{"zcse", zcse},
				{"cse", cse},
				{"tighten", tighten},
			} {
				step.run(fun.f)
				deadcode(fun.f)
				CheckFunc(fun.f)
				checkLogosZeroSIMDGuard(t, fun, step.name, "leftStore", "leftVector")
				checkLogosZeroSIMDGuard(t, fun, step.name, "rightStore", "rightVector")
				checkLogosZeroSIMDCount(t, fun.f, step.name, 2)
				if fun.values["leftStore"].Args[1] == fun.values["rightStore"].Args[1] {
					t.Errorf("%s shared a vector zero across sibling feature guards", step.name)
				}
			}
		})
	}
}

func TestLogosZCSEScalarConstants(t *testing.T) {
	c := testConfig(t)
	i64 := c.config.Types.Int64
	boolType := c.config.Types.Bool
	ptr := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpArg, ptr, 0, c.Temp(ptr)),
			Valu("choice", OpArg, boolType, 0, c.Temp(boolType)),
			If("choice", "left", "right")),
		Bloc("left",
			Valu("leftConstant", OpConst64, i64, 42, nil),
			Valu("leftStore", OpStore, types.TypeMem, 0, i64, "p", "leftConstant", "mem"),
			Exit("leftStore")),
		Bloc("right",
			Valu("rightConstant", OpConst64, i64, 42, nil),
			Valu("rightStore", OpStore, types.TypeMem, 0, i64, "p", "rightConstant", "mem"),
			Exit("rightStore")),
	)

	CheckFunc(fun.f)
	zcse(fun.f)
	deadcode(fun.f)
	CheckFunc(fun.f)
	left := fun.values["leftStore"].Args[1]
	right := fun.values["rightStore"].Args[1]
	if left != right || left.Op != OpConst64 || left.AuxInt != 42 || left.Block != fun.f.Entry {
		t.Errorf("scalar constants were not hoisted and shared: left=%s in %s, right=%s in %s", left, left.Block, right, right.Block)
	}
}

// The positive feature edge must dominate the instruction itself, not only
// its users. A zero can lower to an instruction unavailable on the false edge.
func checkLogosZeroSIMDGuard(t *testing.T, fun fun, stage, store, region string) {
	t.Helper()
	zero := fun.values[store].Args[1]
	if zero.Op != OpZeroSIMD {
		t.Fatalf("%s: %s uses %s, want ZeroSIMD", stage, store, zero.LongString())
	}
	if !fun.f.Sdom().IsAncestorEq(fun.blocks[region], zero.Block) {
		t.Errorf("%s: %s uses vector zero in %s outside feature region %s", stage, store, zero.Block, region)
	}
}

func checkLogosZeroSIMDCount(t *testing.T, f *Func, stage string, want int) {
	t.Helper()
	count := 0
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpZeroSIMD {
				count++
			}
		}
	}
	if count != want {
		t.Errorf("%s: got %d live vector zeros, want %d", stage, count, want)
	}
}
