// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"internal/goarch"
	"testing"
)

func TestLogosArenaBumpOpContract(t *testing.T) {
	if !OpLogosArenaBump.UnsafePoint() {
		t.Fatal("OpLogosArenaBump.UnsafePoint() = false, want true")
	}
	info := opcodeTable[OpLogosArenaBump]
	if info.argLen != 2 {
		t.Errorf("OpLogosArenaBump argLen = %d, want 2 (chunk, mem)", info.argLen)
	}
	if info.auxType != auxInt64 {
		t.Errorf("OpLogosArenaBump auxType = %v, want auxInt64", info.auxType)
	}
	if !info.hasSideEffects {
		t.Error("OpLogosArenaBump hasSideEffects = false, want true")
	}
}

func TestLogosArenaBumpLowersAMD64(t *testing.T) {
	c := testConfig(t)
	bumpTyp := types.NewTuple(types.Types[types.TUNSAFEPTR], types.TypeMem)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("chunk", OpOffPtr, c.config.Types.BytePtr, 0, nil, "sp"),
			Valu("bump", OpLogosArenaBump, bumpTyp, 16, nil, "chunk", "mem"),
			Valu("ptr", OpSelect0, types.Types[types.TUNSAFEPTR], 0, nil, "bump"),
			Valu("mem2", OpSelect1, types.TypeMem, 0, nil, "bump"),
			Ret("mem2")))
	CheckFunc(fun.f)
	lower(fun.f)
	CheckFunc(fun.f)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpLogosArenaBump: 0, OpAMD64LoweredLogosArenaBump: 1})
	if !OpAMD64LoweredLogosArenaBump.UnsafePoint() {
		t.Fatal("OpAMD64LoweredLogosArenaBump.UnsafePoint() = false, want true")
	}
}

func TestLogosArenaGOffset(t *testing.T) {
	if logosArenaGOffset != 4*goarch.PtrSize {
		t.Fatalf("logosArenaGOffset = %d, want 4*goarch.PtrSize = %d", logosArenaGOffset, 4*goarch.PtrSize)
	}
}
