// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"cmd/internal/obj/x86"
	"fmt"
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

// Test to make sure G register is never reloaded from spill (spill of G is okay)
// See #25504
func TestNoGetgLoadReg(t *testing.T) {
	/*
		Original:
		func fff3(i int) *g {
			gee := getg()
			if i == 0 {
				fff()
			}
			return gee // here
		}
	*/
	c := testConfigARM64(t)
	f := c.Fun("b1",
		Bloc("b1",
			Valu("v1", OpInitMem, types.TypeMem, 0, nil),
			Valu("v6", OpArg, c.config.Types.Int64, 0, c.Temp(c.config.Types.Int64)),
			Valu("v8", OpGetG, c.config.Types.Int64.PtrTo(), 0, nil, "v1"),
			Valu("v11", OpARM64CMPconst, types.TypeFlags, 0, nil, "v6"),
			Eq("v11", "b2", "b4"),
		),
		Bloc("b4",
			Goto("b3"),
		),
		Bloc("b3",
			Valu("v14", OpPhi, types.TypeMem, 0, nil, "v1", "v12"),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v16", OpARM64MOVDstore, types.TypeMem, 0, nil, "v8", "sb", "v14"),
			Exit("v16"),
		),
		Bloc("b2",
			Valu("v12", OpARM64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "v1"),
			Goto("b3"),
		),
	)
	regalloc(f.f)
	checkFunc(f.f)
	// Double-check that we never restore to the G register. Regalloc should catch it, but check again anyway.
	r := f.f.RegAlloc
	for _, b := range f.blocks {
		for _, v := range b.Values {
			if v.Op == OpLoadReg && r[v.ID].String() == "g" {
				t.Errorf("Saw OpLoadReg targeting g register: %s", v.LongString())
			}
		}
	}
}

// Test to make sure we don't push spills into loops.
// See issue #19595.
func TestSpillWithLoop(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64)),
			Valu("cond", OpArg, c.config.Types.Bool, 0, c.Temp(c.config.Types.Bool)),
			Valu("ld", OpAMD64MOVQload, c.config.Types.Int64, 0, nil, "ptr", "mem"), // this value needs a spill
			Goto("loop"),
		),
		Bloc("loop",
			Valu("memphi", OpPhi, types.TypeMem, 0, nil, "mem", "call"),
			Valu("call", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "memphi"),
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
			Valu("x", OpArg, c.config.Types.Int64, 0, c.Temp(c.config.Types.Int64)),
			Valu("p", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
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
			Valu("mem3", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
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
			Valu("x", OpArg, c.config.Types.Int64, 0, c.Temp(c.config.Types.Int64)),
			Valu("p", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
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
			Valu("mem2", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
			Valu("mem3", OpAMD64MOVQstore, types.TypeMem, 0, nil, "p", "y", "mem2"),
			Exit("mem3"),
		),
		Bloc("exit2",
			// store after call, y must be loaded from a spill location
			Valu("mem4", OpAMD64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "mem"),
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

func TestClobbersArg0(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("ptr", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("dst", OpArg, c.config.Types.Int64.PtrTo().PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo().PtrTo())),
			Valu("zero", OpAMD64LoweredZeroLoop, types.TypeMem, 256, nil, "ptr", "mem"),
			Valu("store", OpAMD64MOVQstore, types.TypeMem, 0, nil, "dst", "ptr", "zero"),
			Exit("store")))
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// LoweredZeroLoop clobbers its argument, so there must be a copy of "ptr" somewhere
	// so we still have that value available at "store".
	if n := numCopies(f.blocks["entry"]); n != 1 {
		fmt.Printf("%s\n", f.f.String())
		t.Errorf("got %d copies, want 1", n)
	}
}

func TestClobbersArg1(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("src", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("dst", OpArg, c.config.Types.Int64.PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo())),
			Valu("use1", OpArg, c.config.Types.Int64.PtrTo().PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo().PtrTo())),
			Valu("use2", OpArg, c.config.Types.Int64.PtrTo().PtrTo(), 0, c.Temp(c.config.Types.Int64.PtrTo().PtrTo())),
			Valu("move", OpAMD64LoweredMoveLoop, types.TypeMem, 256, nil, "dst", "src", "mem"),
			Valu("store1", OpAMD64MOVQstore, types.TypeMem, 0, nil, "use1", "src", "move"),
			Valu("store2", OpAMD64MOVQstore, types.TypeMem, 0, nil, "use2", "dst", "store1"),
			Exit("store2")))
	flagalloc(f.f)
	regalloc(f.f)
	checkFunc(f.f)
	// LoweredMoveLoop clobbers its arguments, so there must be a copy of "src" and "dst" somewhere
	// so we still have that value available at the stores.
	if n := numCopies(f.blocks["entry"]); n != 2 {
		fmt.Printf("%s\n", f.f.String())
		t.Errorf("got %d copies, want 2", n)
	}
}

func TestNoRematerializeDeadConstant(t *testing.T) {
	c := testConfigARM64(t)
	f := c.Fun("b1",
		Bloc("b1",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("addr", OpArg, c.config.Types.Int32.PtrTo(), 0, c.Temp(c.config.Types.Int32.PtrTo())),
			Valu("const", OpARM64MOVDconst, c.config.Types.Int32, -1, nil), // Original constant
			Valu("cmp", OpARM64CMPconst, types.TypeFlags, 0, nil, "const"),
			Goto("b2"),
		),
		Bloc("b2",
			Valu("phi_mem", OpPhi, types.TypeMem, 0, nil, "mem", "callmem"),
			Eq("cmp", "b6", "b3"),
		),
		Bloc("b3",
			Valu("call", OpARM64CALLstatic, types.TypeMem, 0, AuxCallLSym("_"), "phi_mem"),
			Valu("callmem", OpSelectN, types.TypeMem, 0, nil, "call"),
			Eq("cmp", "b5", "b4"),
		),
		Bloc("b4", // A block where we don't really need to rematerialize the constant -1
			Goto("b2"),
		),
		Bloc("b5",
			Valu("user", OpAMD64MOVQstore, types.TypeMem, 0, nil, "addr", "const", "callmem"),
			Exit("user"),
		),
		Bloc("b6",
			Exit("phi_mem"),
		),
	)

	regalloc(f.f)
	checkFunc(f.f)

	// Check that in block b4, there's no dead rematerialization of the constant -1
	for _, v := range f.blocks["b4"].Values {
		if v.Op == OpARM64MOVDconst && v.AuxInt == -1 {
			t.Errorf("constant -1 rematerialized in loop block b4: %s", v.LongString())
		}
	}
}

func numSpills(b *Block) int {
	return numOps(b, OpStoreReg)
}
func numCopies(b *Block) int {
	return numOps(b, OpCopy)
}
func numOps(b *Block, op Op) int {
	n := 0
	for _, v := range b.Values {
		if v.Op == op {
			n++
		}
	}
	return n
}

func TestRematerializeableRegCompatible(t *testing.T) {
	c := testConfig(t)
	f := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("x", OpAMD64MOVLconst, c.config.Types.Int32, 1, nil),
			Valu("a", OpAMD64POR, c.config.Types.Float32, 0, nil, "x", "x"),
			Valu("res", OpMakeResult, types.NewResults([]*types.Type{c.config.Types.Float32, types.TypeMem}), 0, nil, "a", "mem"),
			Ret("res"),
		),
	)
	regalloc(f.f)
	checkFunc(f.f)
	moveFound := false
	for _, v := range f.f.Blocks[0].Values {
		if v.Op == OpCopy && x86.REG_X0 <= v.Reg() && v.Reg() <= x86.REG_X31 {
			moveFound = true
		}
	}
	if !moveFound {
		t.Errorf("Expects an Copy to be issued, but got: %+v", f.f)
	}
}
