// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

func TestSchedule(t *testing.T) {
	c := testConfig(t)
	cases := []fun{
		c.Fun("entry",
			Bloc("entry",
				Valu("mem0", OpInitMem, types.TypeMem, 0, nil),
				Valu("ptr", OpConst64, c.config.Types.Int64, 0xABCD, nil),
				Valu("v", OpConst64, c.config.Types.Int64, 12, nil),
				Valu("mem1", OpStore, types.TypeMem, 0, c.config.Types.Int64, "ptr", "v", "mem0"),
				Valu("mem2", OpStore, types.TypeMem, 0, c.config.Types.Int64, "ptr", "v", "mem1"),
				Valu("mem3", OpStore, types.TypeMem, 0, c.config.Types.Int64, "ptr", "sum", "mem2"),
				Valu("l1", OpLoad, c.config.Types.Int64, 0, nil, "ptr", "mem1"),
				Valu("l2", OpLoad, c.config.Types.Int64, 0, nil, "ptr", "mem2"),
				Valu("sum", OpAdd64, c.config.Types.Int64, 0, nil, "l1", "l2"),
				Goto("exit")),
			Bloc("exit",
				Exit("mem3"))),
	}
	for _, c := range cases {
		schedule(c.f)
		if !isSingleLiveMem(c.f) {
			t.Error("single-live-mem restriction not enforced by schedule for func:")
			printFunc(c.f)
		}
	}
}

func isSingleLiveMem(f *Func) bool {
	for _, b := range f.Blocks {
		var liveMem *Value
		for _, v := range b.Values {
			for _, w := range v.Args {
				if w.Type.IsMemory() {
					if liveMem == nil {
						liveMem = w
						continue
					}
					if w != liveMem {
						return false
					}
				}
			}
			if v.Type.IsMemory() {
				liveMem = v
			}
		}
	}
	return true
}

func TestStoreOrder(t *testing.T) {
	// In the function below, v2 depends on v3 and v4, v4 depends on v3, and v3 depends on store v5.
	// storeOrder did not handle this case correctly.
	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem0", OpInitMem, types.TypeMem, 0, nil),
			Valu("a", OpAdd64, c.config.Types.Int64, 0, nil, "b", "c"),                        // v2
			Valu("b", OpLoad, c.config.Types.Int64, 0, nil, "ptr", "mem1"),                    // v3
			Valu("c", OpNeg64, c.config.Types.Int64, 0, nil, "b"),                             // v4
			Valu("mem1", OpStore, types.TypeMem, 0, c.config.Types.Int64, "ptr", "v", "mem0"), // v5
			Valu("mem2", OpStore, types.TypeMem, 0, c.config.Types.Int64, "ptr", "a", "mem1"),
			Valu("ptr", OpConst64, c.config.Types.Int64, 0xABCD, nil),
			Valu("v", OpConst64, c.config.Types.Int64, 12, nil),
			Goto("exit")),
		Bloc("exit",
			Exit("mem2")))

	CheckFunc(fun.f)
	order := storeOrder(fun.f.Blocks[0].Values, fun.f.newSparseSet(fun.f.NumValues()), make([]int32, fun.f.NumValues()))

	// check that v2, v3, v4 is sorted after v5
	var ai, bi, ci, si int
	for i, v := range order {
		switch v.ID {
		case 2:
			ai = i
		case 3:
			bi = i
		case 4:
			ci = i
		case 5:
			si = i
		}
	}
	if ai < si || bi < si || ci < si {
		t.Logf("Func: %s", fun.f)
		t.Errorf("store order is wrong: got %v, want v2 v3 v4 after v5", order)
	}
}

func TestCarryChainOrder(t *testing.T) {
	// In the function below, there are two carry chains that have no dependencies on each other,
	// one is A1 -> A1carry -> A1Carryvalue, the other is A2 -> A2carry -> A2Carryvalue. If they
	// are not scheduled properly, the carry will be clobbered, causing the carry to be regenerated.
	c := testConfigARM64(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem0", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpARM64MOVDconst, c.config.Types.UInt64, 5, nil),
			Valu("y", OpARM64MOVDconst, c.config.Types.UInt64, 6, nil),
			Valu("z", OpARM64MOVDconst, c.config.Types.UInt64, 7, nil),
			Valu("A1", OpARM64ADDSflags, types.NewTuple(c.config.Types.UInt64, types.TypeFlags), 0, nil, "x", "z"), // x+z, set flags
			Valu("A1carry", OpSelect1, types.TypeFlags, 0, nil, "A1"),
			Valu("A2", OpARM64ADDSflags, types.NewTuple(c.config.Types.UInt64, types.TypeFlags), 0, nil, "y", "z"), // y+z, set flags
			Valu("A2carry", OpSelect1, types.TypeFlags, 0, nil, "A2"),
			Valu("A1value", OpSelect0, c.config.Types.UInt64, 0, nil, "A1"),
			Valu("A1Carryvalue", OpARM64ADCzerocarry, c.config.Types.UInt64, 0, nil, "A1carry"), // 0+0+A1carry
			Valu("A2value", OpSelect0, c.config.Types.UInt64, 0, nil, "A2"),
			Valu("A2Carryvalue", OpARM64ADCzerocarry, c.config.Types.UInt64, 0, nil, "A2carry"), // 0+0+A2carry
			Valu("ValueSum", OpARM64ADD, c.config.Types.UInt64, 0, nil, "A1value", "A2value"),
			Valu("CarrySum", OpARM64ADD, c.config.Types.UInt64, 0, nil, "A1Carryvalue", "A2Carryvalue"),
			Valu("Sum", OpARM64AND, c.config.Types.UInt64, 0, nil, "ValueSum", "CarrySum"),
			Goto("exit")),
		Bloc("exit",
			Exit("mem0")),
	)

	CheckFunc(fun.f)
	schedule(fun.f)

	// The expected order is A1 < A1carry < A1Carryvalue < A2 < A2carry < A2Carryvalue.
	// There is no dependency between the two carry chains, so it doesn't matter which
	// comes first and which comes after, but the unsorted position of A1 is before A2,
	// so A1Carryvalue < A2.
	var ai, bi, ci, di, ei, fi int
	for i, v := range fun.f.Blocks[0].Values {
		switch {
		case fun.values["A1"] == v:
			ai = i
		case fun.values["A1carry"] == v:
			bi = i
		case fun.values["A1Carryvalue"] == v:
			ci = i
		case fun.values["A2"] == v:
			di = i
		case fun.values["A2carry"] == v:
			ei = i
		case fun.values["A2Carryvalue"] == v:
			fi = i
		}
	}
	if !(ai < bi && bi < ci && ci < di && di < ei && ei < fi) {
		t.Logf("Func: %s", fun.f)
		t.Errorf("carry chain order is wrong: got %v, want V%d after V%d after V%d after V%d after V%d after V%d,",
			fun.f.Blocks[0], fun.values["A1"].ID, fun.values["A1carry"].ID, fun.values["A1Carryvalue"].ID,
			fun.values["A2"].ID, fun.values["A2carry"].ID, fun.values["A2Carryvalue"].ID)
	}
}
