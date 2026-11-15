// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"internal/goarch"
	"testing"

	"cmd/compile/internal/types"
)

func TestLogosRegionBumpOpContract(t *testing.T) {
	if !OpLogosRegionBump.UnsafePoint() {
		t.Fatal("OpLogosRegionBump.UnsafePoint() = false, want true")
	}
	info := opcodeTable[OpLogosRegionBump]
	if info.argLen != 3 {
		t.Errorf("OpLogosRegionBump argLen = %d, want 3 (record, generation, mem)", info.argLen)
	}
	if info.auxType != auxInt64 {
		t.Errorf("OpLogosRegionBump auxType = %v, want auxInt64", info.auxType)
	}
	if !info.hasSideEffects {
		t.Error("OpLogosRegionBump hasSideEffects = false, want true")
	}
}

func TestLogosRegionBumpLowersAMD64(t *testing.T) {
	c := testConfig(t)
	bumpTyp := types.NewTuple(types.Types[types.TUNSAFEPTR], types.TypeMem)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("rec", OpOffPtr, types.Types[types.TUNSAFEPTR], 0, nil, "sp"),
			Valu("gen", OpOffPtr, c.config.Types.Uintptr, 0, nil, "sp"),
			Valu("bump", OpLogosRegionBump, bumpTyp, 16, nil, "rec", "gen", "mem"),
			Valu("ptr", OpSelect0, types.Types[types.TUNSAFEPTR], 0, nil, "bump"),
			Valu("mem2", OpSelect1, types.TypeMem, 0, nil, "bump"),
			Ret("mem2")))
	CheckFunc(fun.f)
	lower(fun.f)
	CheckFunc(fun.f)
	checkOpcodeCounts(t, fun.f, map[Op]int{OpLogosRegionBump: 0, OpAMD64LoweredLogosRegionBump: 1})
	if !OpAMD64LoweredLogosRegionBump.UnsafePoint() {
		t.Fatal("OpAMD64LoweredLogosRegionBump.UnsafePoint() = false, want true")
	}
}

func TestLogosRegionOffsets(t *testing.T) {
	if LogosRegionChunkOffset != 0 {
		t.Fatalf("LogosRegionChunkOffset = %d, want 0", LogosRegionChunkOffset)
	}
	if LogosRegionGenOffset != goarch.PtrSize {
		t.Fatalf("LogosRegionGenOffset = %d, want %d", LogosRegionGenOffset, goarch.PtrSize)
	}
	if LogosRegionOwnerOffset != 3*goarch.PtrSize {
		t.Fatalf("LogosRegionOwnerOffset = %d, want %d", LogosRegionOwnerOffset, 3*goarch.PtrSize)
	}
	if LogosRegionNoscanCurOffset != 2*goarch.PtrSize {
		t.Fatalf("LogosRegionNoscanCurOffset = %d, want %d", LogosRegionNoscanCurOffset, 2*goarch.PtrSize)
	}
	if LogosRegionScanCurOffset != 3*goarch.PtrSize {
		t.Fatalf("LogosRegionScanCurOffset = %d, want %d", LogosRegionScanCurOffset, 3*goarch.PtrSize)
	}
}
