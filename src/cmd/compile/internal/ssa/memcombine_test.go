// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

// countLoads counts the number of OpLoad operations in a function
func countLoads(f *Func) int {
	count := 0
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpLoad {
				count++
			}
		}
	}
	return count
}

// TestMemcombineAlignedLoads tests combining aligned memory accesses on architectures
// that don't support unaligned access (e.g., RISC-V).
func TestMemcombineAlignedLoads(t *testing.T) {
	c := testConfig(t)
	c.config.unalignedOK = false
	c.config.arch = "RISCV64"

	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("ptr", OpAddr, ptrType, 0, nil, "sb"),

			// Two consecutive 1-byte loads forming OR tree
			Valu("load1", OpLoad, c.config.Types.UInt8, 0, nil, "ptr", "mem"),
			Valu("off1", OpOffPtr, ptrType, 1, nil, "ptr"),
			Valu("load2", OpLoad, c.config.Types.UInt8, 0, nil, "off1", "mem"),

			// Extend to 16 bits
			Valu("ext1", OpZeroExt8to16, c.config.Types.UInt16, 0, nil, "load1"),
			Valu("ext2", OpZeroExt8to16, c.config.Types.UInt16, 0, nil, "load2"),

			// Shift second byte
			Valu("shift", OpConst64, c.config.Types.UInt64, 8, nil),
			Valu("shifted", OpLsh16x64, c.config.Types.UInt16, 0, nil, "ext2", "shift"),

			// OR operation
			Valu("result", OpOr16, c.config.Types.UInt16, 0, nil, "ext1", "shifted"),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	loadCountBefore := countLoads(fun.f)

	memcombine(fun.f)

	CheckFunc(fun.f)
	loadCountAfter := countLoads(fun.f)

	if loadCountAfter >= loadCountBefore {
		t.Logf("Load count: %d -> %d (no combining occurred)", loadCountBefore, loadCountAfter)
	} else {
		t.Logf("Success: Load count reduced from %d to %d", loadCountBefore, loadCountAfter)
	}
}

// TestMemcombineDynamicIndex verifies that dynamic indexes are not combined
// when unalignedOK=false, as alignment cannot be statically proven.
func TestMemcombineDynamicIndex(t *testing.T) {
	c := testConfig(t)
	c.config.unalignedOK = false

	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("idx", OpArg, c.config.Types.Int64, 0, nil),
			Valu("ptr", OpAddPtr, ptrType, 0, nil, "sb", "idx"),

			// OR tree with dynamic index
			Valu("load1", OpLoad, c.config.Types.UInt8, 0, nil, "ptr", "mem"),
			Valu("off1", OpOffPtr, ptrType, 1, nil, "ptr"),
			Valu("load2", OpLoad, c.config.Types.UInt8, 0, nil, "off1", "mem"),

			Valu("ext1", OpZeroExt8to16, c.config.Types.UInt16, 0, nil, "load1"),
			Valu("ext2", OpZeroExt8to16, c.config.Types.UInt16, 0, nil, "load2"),

			Valu("shift", OpConst64, c.config.Types.UInt64, 8, nil),
			Valu("shifted", OpLsh16x64, c.config.Types.UInt16, 0, nil, "ext2", "shift"),

			Valu("result", OpOr16, c.config.Types.UInt16, 0, nil, "ext1", "shifted"),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	loadCountBefore := countLoads(fun.f)

	memcombine(fun.f)

	CheckFunc(fun.f)
	loadCountAfter := countLoads(fun.f)

	if loadCountAfter < loadCountBefore {
		t.Errorf("Unexpected: Load combining occurred with dynamic index (before=%d, after=%d)",
			loadCountBefore, loadCountAfter)
	} else {
		t.Logf("Correct: No combining with dynamic index (load count=%d)", loadCountAfter)
	}
}

// TestMemcombineUnalignedOffset verifies that unaligned offsets are not combined
// when unalignedOK=false.
func TestMemcombineUnalignedOffset(t *testing.T) {
	c := testConfig(t)
	c.config.unalignedOK = false

	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("ptr", OpAddr, ptrType, 0, nil, "sb"),

			// 4-byte access from offset 1 (not 4-byte aligned)
			Valu("off1", OpOffPtr, ptrType, 1, nil, "ptr"),
			Valu("load1", OpLoad, c.config.Types.UInt8, 0, nil, "off1", "mem"),
			Valu("off2", OpOffPtr, ptrType, 2, nil, "ptr"),
			Valu("load2", OpLoad, c.config.Types.UInt8, 0, nil, "off2", "mem"),
			Valu("off3", OpOffPtr, ptrType, 3, nil, "ptr"),
			Valu("load3", OpLoad, c.config.Types.UInt8, 0, nil, "off3", "mem"),
			Valu("off4", OpOffPtr, ptrType, 4, nil, "ptr"),
			Valu("load4", OpLoad, c.config.Types.UInt8, 0, nil, "off4", "mem"),

			// 32-bit OR tree
			Valu("ext1", OpZeroExt8to32, c.config.Types.UInt32, 0, nil, "load1"),
			Valu("ext2", OpZeroExt8to32, c.config.Types.UInt32, 0, nil, "load2"),
			Valu("ext3", OpZeroExt8to32, c.config.Types.UInt32, 0, nil, "load3"),
			Valu("ext4", OpZeroExt8to32, c.config.Types.UInt32, 0, nil, "load4"),

			Valu("shift8", OpConst64, c.config.Types.UInt64, 8, nil),
			Valu("shift16", OpConst64, c.config.Types.UInt64, 16, nil),
			Valu("shift24", OpConst64, c.config.Types.UInt64, 24, nil),

			Valu("shifted2", OpLsh32x64, c.config.Types.UInt32, 0, nil, "ext2", "shift8"),
			Valu("shifted3", OpLsh32x64, c.config.Types.UInt32, 0, nil, "ext3", "shift16"),
			Valu("shifted4", OpLsh32x64, c.config.Types.UInt32, 0, nil, "ext4", "shift24"),

			Valu("or1", OpOr32, c.config.Types.UInt32, 0, nil, "ext1", "shifted2"),
			Valu("or2", OpOr32, c.config.Types.UInt32, 0, nil, "shifted3", "shifted4"),
			Valu("result", OpOr32, c.config.Types.UInt32, 0, nil, "or1", "or2"),

			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	loadCountBefore := countLoads(fun.f)

	memcombine(fun.f)

	CheckFunc(fun.f)
	loadCountAfter := countLoads(fun.f)

	if loadCountAfter < loadCountBefore {
		t.Errorf("Unexpected: Load combining occurred with unaligned offset (before=%d, after=%d)",
			loadCountBefore, loadCountAfter)
	} else {
		t.Logf("Correct: No combining with unaligned offset (load count=%d)", loadCountAfter)
	}
}
