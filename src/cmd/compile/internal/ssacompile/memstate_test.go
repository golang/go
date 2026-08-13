// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
package ssacompile

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"testing"
)

func assertBlockNilMem(block string, state []*ssa.Value, f fun, t *testing.T) {
	b, ok := f.blocks[block]
	if !ok {
		t.Fatalf("Didn't find block %v", block)
	}

	if state[b.ID] != nil {
		t.Fatalf("Expected nil memory state for block %v, got %v", b, state[b.ID])
	}
}

func assertBlockMem(block, value string, state []*ssa.Value, f fun, t *testing.T) {
	expected, ok := f.values[value]
	if !ok {
		t.Fatalf("No memory value")
	}
	if !expected.Type.IsMemory() {
		t.Fatalf("State is not a memory %v", expected)
	}

	b, ok := f.blocks[block]
	if !ok {
		t.Fatalf("Didn't find block %v", block)
	}

	if state[b.ID] != expected {
		t.Fatalf("Expected %v memory state for block %v, got %v", expected, b, state[b.ID])
	}
}
func TestMemStateEmptyFunc(t *testing.T) {
	// entry:
	// Plain -> loop
	//
	// loop:
	// Plain -> loop

	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry", Goto("loop")),
		Bloc("loop", Goto("loop")),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockNilMem("entry", startMem, fun, t)
	assertBlockNilMem("loop", endMem, fun, t)
}

func TestMemStateSingleBlock(t *testing.T) {
	// entry:
	// start = OpInitMem
	// Exit start

	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Exit("start")),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockMem("entry", "start", startMem, fun, t)
	assertBlockMem("entry", "start", endMem, fun, t)
}

func TestMemStateSingleBlockMultipleMem(t *testing.T) {
	// entry:
	// start = OpInitMem <- startMem
	// ...
	// zero1
	// store1
	// store2
	// store3
	// store4 <- endMem
	// Exit

	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			// Boilerplate -- prepare arguments for stores
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("addr2", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("addr3", ssaop.OpAddr, ptrType, 0, nil, "sb"),

			Valu("zero1", ssaop.OpZero, types.TypeMem, 1, c.config.Types.Bool, "addr3", "start"),
			Valu("store1", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "zero1"),
			Valu("store2", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr2", "v", "store1"),
			Valu("store3", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "store2"),
			Valu("store4", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr3", "v", "store3"),

			Exit("store4")))

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockMem("entry", "start", startMem, fun, t)
	assertBlockMem("entry", "store4", endMem, fun, t)
}

func TestMemStateMemoryFreeBlock(t *testing.T) {
	// entry:
	// start = OpInitMem
	// Goto middle
	//
	// middle:
	// OpLoad(..., start)
	// Goto exit
	//
	// exit:
	// Exit
	//
	// The load consumes memory but doesn't define a new memory state.

	c := testConfig(t)
	ptrType := c.config.Types.BytePtr

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Goto("middle")),

		Bloc("middle",
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("addr", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("load", ssaop.OpLoad, c.config.Types.Bool, 0, nil, "addr", "start"),
			Goto("exit")),

		Bloc("exit",
			Exit("start")),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockMem("entry", "start", startMem, fun, t)
	assertBlockMem("entry", "start", endMem, fun, t)

	assertBlockMem("middle", "start", startMem, fun, t)
	assertBlockMem("middle", "start", endMem, fun, t)

	assertBlockMem("exit", "start", startMem, fun, t)
	assertBlockMem("exit", "start", endMem, fun, t)
}

func TestMemStateMemoryPhi(t *testing.T) {
	// entry:
	// start = OpInitMem
	// If cond -> left, right
	//
	// left:
	// store1(..., start)
	// Goto merge
	//
	// right:
	// store2(..., start)
	// Goto merge
	//
	// merge:
	// memphi = Phi(store1, store2)
	// Exit memphi

	c := testConfig(t)
	ptrType := c.config.Types.BytePtr

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("cond", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			If("cond", "left", "right")),

		Bloc("left",
			Valu("sb1", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v1", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", ssaop.OpAddr, ptrType, 0, nil, "sb1"),
			Valu("store1", ssaop.OpStore, types.TypeMem, 0,
				c.config.Types.Bool, "addr1", "v1", "start"),
			Goto("merge")),

		Bloc("right",
			Valu("sb2", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v2", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr2", ssaop.OpAddr, ptrType, 0, nil, "sb2"),
			Valu("store2", ssaop.OpStore, types.TypeMem, 0,
				c.config.Types.Bool, "addr2", "v2", "start"),
			Goto("merge")),

		Bloc("merge",
			Valu("memphi", ssaop.OpPhi, types.TypeMem, 0, nil,
				"store1", "store2"),
			Exit("memphi"),
		),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockMem("entry", "start", startMem, fun, t)
	assertBlockMem("entry", "start", endMem, fun, t)

	assertBlockMem("left", "start", startMem, fun, t)
	assertBlockMem("left", "store1", endMem, fun, t)

	assertBlockMem("right", "start", startMem, fun, t)
	assertBlockMem("right", "store2", endMem, fun, t)

	assertBlockMem("merge", "memphi", startMem, fun, t)
	assertBlockMem("merge", "memphi", endMem, fun, t)
}

func TestMemStateMemoryFreeMerge(t *testing.T) {
	// entry:
	// start = OpInitMem
	// If cond -> left, right
	//
	// left:
	// Goto merge
	//
	// right:
	// Goto merge
	//
	// merge:
	// Goto exit
	//
	// exit:
	// Exit start

	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("cond", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			If("cond", "left", "right")),

		Bloc("left",
			Goto("merge")),

		Bloc("right",
			Goto("merge")),

		Bloc("merge",
			Goto("exit")),

		Bloc("exit",
			Exit("start")),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	for _, block := range []string{"entry", "left", "right", "merge", "exit"} {
		assertBlockMem(block, "start", startMem, fun, t)
		assertBlockMem(block, "start", endMem, fun, t)
	}
}

func TestMemStateLoopWithoutMemoryOperations(t *testing.T) {
	// entry:
	// start = OpInitMem
	// If cond -> loop, exit
	//
	// loop:
	// If cond -> loop, exit
	//
	// exit:
	// Exit start

	c := testConfig(t)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("cond", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			If("cond", "loop", "exit")),

		Bloc("loop",
			If("cond", "loop", "exit")),

		Bloc("exit",
			Exit("start")),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockMem("entry", "start", startMem, fun, t)
	assertBlockMem("entry", "start", endMem, fun, t)

	assertBlockMem("loop", "start", startMem, fun, t)
	assertBlockMem("loop", "start", endMem, fun, t)

	assertBlockMem("exit", "start", startMem, fun, t)
	assertBlockMem("exit", "start", endMem, fun, t)
}

func TestMemStateLoopWithMemoryOperation(t *testing.T) {
	// entry:
	// start = OpInitMem
	// If cond -> loop, exit
	//
	// loop:
	// memphi = Phi(start, store)
	// store = OpStore(..., memphi)
	// If cond -> loop, exit
	//
	// exit:
	// Exit store

	c := testConfig(t)
	ptrType := c.config.Types.BytePtr

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("cond", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			If("cond", "loop", "exit")),

		Bloc("loop",
			Valu("memphi", ssaop.OpPhi, types.TypeMem, 0, nil,
				"start", "store"),
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("store", ssaop.OpStore, types.TypeMem, 0,
				c.config.Types.Bool, "addr", "v", "memphi"),
			If("cond", "loop", "exit")),

		Bloc("exit",
			Valu("memexit", ssaop.OpPhi, types.TypeMem, 0, nil,
				"start", "store"),
			Exit("memexit")),
	)

	CheckFunc(fun.f)
	startMem, endMem := memState(fun.f)
	defer fun.f.Cache.FreeValueSlice(startMem)
	defer fun.f.Cache.FreeValueSlice(endMem)

	assertBlockMem("entry", "start", startMem, fun, t)
	assertBlockMem("entry", "start", endMem, fun, t)

	assertBlockMem("loop", "memphi", startMem, fun, t)
	assertBlockMem("loop", "store", endMem, fun, t)

	assertBlockMem("exit", "memexit", startMem, fun, t)
	assertBlockMem("exit", "memexit", endMem, fun, t)
}
