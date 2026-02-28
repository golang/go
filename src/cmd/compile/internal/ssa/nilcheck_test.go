// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"strconv"
	"testing"
)

func BenchmarkNilCheckDeep1(b *testing.B)     { benchmarkNilCheckDeep(b, 1) }
func BenchmarkNilCheckDeep10(b *testing.B)    { benchmarkNilCheckDeep(b, 10) }
func BenchmarkNilCheckDeep100(b *testing.B)   { benchmarkNilCheckDeep(b, 100) }
func BenchmarkNilCheckDeep1000(b *testing.B)  { benchmarkNilCheckDeep(b, 1000) }
func BenchmarkNilCheckDeep10000(b *testing.B) { benchmarkNilCheckDeep(b, 10000) }

// benchmarkNilCheckDeep is a stress test of nilcheckelim.
// It uses the worst possible input: A linear string of
// nil checks, none of which can be eliminated.
// Run with multiple depths to observe big-O behavior.
func benchmarkNilCheckDeep(b *testing.B, depth int) {
	c := testConfig(b)
	ptrType := c.config.Types.BytePtr

	var blocs []bloc
	blocs = append(blocs,
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto(blockn(0)),
		),
	)
	for i := 0; i < depth; i++ {
		blocs = append(blocs,
			Bloc(blockn(i),
				Valu(ptrn(i), OpAddr, ptrType, 0, nil, "sb"),
				Valu(booln(i), OpIsNonNil, c.config.Types.Bool, 0, nil, ptrn(i)),
				If(booln(i), blockn(i+1), "exit"),
			),
		)
	}
	blocs = append(blocs,
		Bloc(blockn(depth), Goto("exit")),
		Bloc("exit", Exit("mem")),
	)

	fun := c.Fun("entry", blocs...)

	CheckFunc(fun.f)
	b.SetBytes(int64(depth)) // helps for eyeballing linearity
	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		nilcheckelim(fun.f)
	}
}

func blockn(n int) string { return "b" + strconv.Itoa(n) }
func ptrn(n int) string   { return "p" + strconv.Itoa(n) }
func booln(n int) string  { return "c" + strconv.Itoa(n) }

func isNilCheck(b *Block) bool {
	return b.Kind == BlockIf && b.Controls[0].Op == OpIsNonNil
}

// TestNilcheckSimple verifies that a second repeated nilcheck is removed.
func TestNilcheckSimple(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("bool1", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool1", "secondCheck", "exit")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
	}
}

// TestNilcheckDomOrder ensures that the nil check elimination isn't dependent
// on the order of the dominees.
func TestNilcheckDomOrder(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("bool1", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool1", "secondCheck", "exit")),
		Bloc("exit",
			Exit("mem")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			Goto("exit")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
	}
}

// TestNilcheckAddr verifies that nilchecks of OpAddr constructed values are removed.
func TestNilcheckAddr(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpAddr, ptrType, 0, nil, "sb"),
			Valu("bool1", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool1", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["checkPtr"] && isNilCheck(b) {
			t.Errorf("checkPtr was not eliminated")
		}
	}
}

// TestNilcheckAddPtr verifies that nilchecks of OpAddPtr constructed values are removed.
func TestNilcheckAddPtr(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("off", OpConst64, c.config.Types.Int64, 20, nil),
			Valu("ptr1", OpAddPtr, ptrType, 0, nil, "sb", "off"),
			Valu("bool1", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool1", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["checkPtr"] && isNilCheck(b) {
			t.Errorf("checkPtr was not eliminated")
		}
	}
}

// TestNilcheckPhi tests that nil checks of phis, for which all values are known to be
// non-nil are removed.
func TestNilcheckPhi(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("sp", OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("baddr", OpLocalAddr, c.config.Types.Bool, 0, StringToAux("b"), "sp", "mem"),
			Valu("bool1", OpLoad, c.config.Types.Bool, 0, nil, "baddr", "mem"),
			If("bool1", "b1", "b2")),
		Bloc("b1",
			Valu("ptr1", OpAddr, ptrType, 0, nil, "sb"),
			Goto("checkPtr")),
		Bloc("b2",
			Valu("ptr2", OpAddr, ptrType, 0, nil, "sb"),
			Goto("checkPtr")),
		// both ptr1 and ptr2 are guaranteed non-nil here
		Bloc("checkPtr",
			Valu("phi", OpPhi, ptrType, 0, nil, "ptr1", "ptr2"),
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "phi"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["checkPtr"] && isNilCheck(b) {
			t.Errorf("checkPtr was not eliminated")
		}
	}
}

// TestNilcheckKeepRemove verifies that duplicate checks of the same pointer
// are removed, but checks of different pointers are not.
func TestNilcheckKeepRemove(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("bool1", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool1", "differentCheck", "exit")),
		Bloc("differentCheck",
			Valu("ptr2", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr2"),
			If("bool2", "secondCheck", "exit")),
		Bloc("secondCheck",
			Valu("bool3", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool3", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	foundDifferentCheck := false
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
		if b == fun.blocks["differentCheck"] && isNilCheck(b) {
			foundDifferentCheck = true
		}
	}
	if !foundDifferentCheck {
		t.Errorf("removed differentCheck, but shouldn't have")
	}
}

// TestNilcheckInFalseBranch tests that nil checks in the false branch of a nilcheck
// block are *not* removed.
func TestNilcheckInFalseBranch(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("bool1", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool1", "extra", "secondCheck")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool2", "extra", "thirdCheck")),
		Bloc("thirdCheck",
			Valu("bool3", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool3", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	foundSecondCheck := false
	foundThirdCheck := false
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			foundSecondCheck = true
		}
		if b == fun.blocks["thirdCheck"] && isNilCheck(b) {
			foundThirdCheck = true
		}
	}
	if !foundSecondCheck {
		t.Errorf("removed secondCheck, but shouldn't have [false branch]")
	}
	if !foundThirdCheck {
		t.Errorf("removed thirdCheck, but shouldn't have [false branch]")
	}
}

// TestNilcheckUser verifies that a user nil check that dominates a generated nil check
// wil remove the generated nil check.
func TestNilcheckUser(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, c.config.Types.Bool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "secondCheck", "exit")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			Goto("exit")),
		Bloc("exit",
			Exit("mem")))

	CheckFunc(fun.f)
	// we need the opt here to rewrite the user nilcheck
	opt(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			t.Errorf("secondCheck was not eliminated")
		}
	}
}

// TestNilcheckBug reproduces a bug in nilcheckelim found by compiling math/big
func TestNilcheckBug(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", OpSB, c.config.Types.Uintptr, 0, nil),
			Goto("checkPtr")),
		Bloc("checkPtr",
			Valu("ptr1", OpLoad, ptrType, 0, nil, "sb", "mem"),
			Valu("nilptr", OpConstNil, ptrType, 0, nil),
			Valu("bool1", OpNeqPtr, c.config.Types.Bool, 0, nil, "ptr1", "nilptr"),
			If("bool1", "secondCheck", "couldBeNil")),
		Bloc("couldBeNil",
			Goto("secondCheck")),
		Bloc("secondCheck",
			Valu("bool2", OpIsNonNil, c.config.Types.Bool, 0, nil, "ptr1"),
			If("bool2", "extra", "exit")),
		Bloc("extra",
			// prevent fuse from eliminating this block
			Valu("store", OpStore, types.TypeMem, 0, ptrType, "ptr1", "nilptr", "mem"),
			Goto("exit")),
		Bloc("exit",
			Valu("phi", OpPhi, types.TypeMem, 0, nil, "mem", "store"),
			Exit("phi")))

	CheckFunc(fun.f)
	// we need the opt here to rewrite the user nilcheck
	opt(fun.f)
	nilcheckelim(fun.f)

	// clean up the removed nil check
	fuse(fun.f, fuseTypePlain)
	deadcode(fun.f)

	CheckFunc(fun.f)
	foundSecondCheck := false
	for _, b := range fun.f.Blocks {
		if b == fun.blocks["secondCheck"] && isNilCheck(b) {
			foundSecondCheck = true
		}
	}
	if !foundSecondCheck {
		t.Errorf("secondCheck was eliminated, but shouldn't have")
	}
}
