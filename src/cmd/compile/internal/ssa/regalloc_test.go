// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"testing"
)

func TestLiveControlOps(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpAMD64MOVLconst, c.config.Types.Int8, 1, nil),
			Valu("y", OpAMD64MOVLconst, c.config.Types.Int8, 2, nil),
			Valu("a", OpAMD64TESTB, types.TypeFlags, 0, nil, "x", "y"),
			Valu("b", OpAMD64TESTB, types.TypeFlags, 0, nil, "y", "x"),
			Eq("a", "if", "exit"),
		),
		Bloc("if",
			Eq("b", "plain", "exit"),
		),
		Bloc("plain",
			Goto("exit"),
		),
		Bloc("exit",
			Exit("mem"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
}

// Test to make sure we don't push spills into loops.
// See issue #19595.
func TestSpillWithLoop(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Int64)),
			Valu("cond", OpArg, c.config.Types.Bool, 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Bool)),
			Valu("ld", OpAMD64MOVQload, c.config.Types.Int64, 0, nil, "ptr", "mem"), // this value needs a spill
			Goto("loop"),
		),
		Bloc("loop",
			Valu("memphi", OpPhi, types.TypeMem, 0, nil, "mem", "call"),
			Valu("call", OpAMD64CALLstatic, types.TypeMem, 0, nil, "memphi"),
			Valu("test", OpAMD64CMPBconst, types.TypeFlags, 0, nil, "cond"),
			Eq("test", "next", "exit"),
		),
		Bloc("next",
			Goto("loop"),
		),
		Bloc("exit",
			Valu("store", OpAMD64MOVQstore, types.TypeMem, 0, nil, "ptr", "ld", "call"),
			Exit("store"),
		),
	)
	regalloc(f.f)
	checkFunc(f.f)
	for _, v := range f.blocks["loop"].Values {
		if v.Op == OpStoreReg {
			t.Errorf("spill inside loop %s", v.LongString())
		}
	}
}

func TestSpillMove1(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpArg, c.config.Types.Int64, 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Int64)),
			Valu("p", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Int64.PtrTo())),
			Valu("a", OpAMD64TESTQ, types.TypeFlags, 0, nil, "x", "x"),
			Goto("loop1"),
		),
		Bloc("loop1",
			Valu("y", OpAMD64MULQ, c.config.Types.Int64, 0, nil, "x", "x"),
			Eq("a", "loop2", "exit1"),
		),
		Bloc("loop2",
			Eq("a", "loop1", "exit2"),
		),
		Bloc("exit1",
			// store before call, y is available in a register
			Valu("mem2", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem"),
			Valu("mem3", OpAMD64CALLstatic, types.TypeMem, 0, nil, "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, types.TypeMem, 0, nil, "mem"),
			Valu("mem5", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem4"),
			Exit("mem5"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// Spill should be moved to exit2.
	if numSpills(f.blocks["loop1"]) != 0 {
		t.Errorf("spill present from loop1")
	}
	if numSpills(f.blocks["loop2"]) != 0 {
		t.Errorf("spill present in loop2")
	}
	if numSpills(f.blocks["exit1"]) != 0 {
		t.Errorf("spill present in exit1")
	}
	if numSpills(f.blocks["exit2"]) != 1 {
		t.Errorf("spill missing in exit2")
	}

}

func TestSpillMove2(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpArg, c.config.Types.Int64, 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Int64)),
			Valu("p", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Frontend().Auto(src.NoXPos, c.config.Types.Int64.PtrTo())),
			Valu("a", OpAMD64TESTQ, types.TypeFlags, 0, nil, "x", "x"),
			Goto("loop1"),
		),
		Bloc("loop1",
			Valu("y", OpAMD64MULQ, c.config.Types.Int64, 0, nil, "x", "x"),
			Eq("a", "loop2", "exit1"),
		),
		Bloc("loop2",
			Eq("a", "loop1", "exit2"),
		),
		Bloc("exit1",
			// store after call, y must be loaded from a spill location
			Valu("mem2", OpAMD64CALLstatic, types.TypeMem, 0, nil, "mem"),
			Valu("mem3", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, types.TypeMem, 0, nil, "mem"),
			Valu("mem5", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem4"),
			Exit("mem5"),
		),
	)
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// There should be a spill in loop1, and nowhere else.
	// TODO: resurrect moving spills out of loops? We could put spills at the start of both exit1 and exit2.
	if numSpills(f.blocks["loop1"]) != 1 {
		t.Errorf("spill missing from loop1")
	}
	if numSpills(f.blocks["loop2"]) != 0 {
		t.Errorf("spill present in loop2")
	}
	if numSpills(f.blocks["exit1"]) != 0 {
		t.Errorf("spill present in exit1")
	}
	if numSpills(f.blocks["exit2"]) != 0 {
		t.Errorf("spill present in exit2")
	}

}

func numSpills(b *Block) int {
	n := 0
	for _, v := range b.Values {
		if v.Op == OpStoreReg {
			n++
		}
	}
	return n
}
