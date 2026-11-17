// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

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
