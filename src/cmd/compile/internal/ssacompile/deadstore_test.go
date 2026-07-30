// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"fmt"
	"sort"
	"testing"

	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

func TestDeadStore(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	t.Logf("PTRTYPE %v", ptrType)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
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
			Goto("exit")),
		Bloc("exit",
			Exit("store3")))

	CheckFunc(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v1 := fun.values["store1"]
	if v1.Op != ssaop.OpCopy {
		t.Errorf("dead store not removed")
	}

	v2 := fun.values["zero1"]
	if v2.Op != ssaop.OpCopy {
		t.Errorf("dead store (zero) not removed")
	}
}

func TestDeadStorePhi(t *testing.T) {
	// make sure we don't get into an infinite loop with phi values.
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Goto("loop")),
		Bloc("loop",
			Valu("phi", ssaop.OpPhi, types.TypeMem, 0, nil, "start", "store"),
			Valu("store", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr", "v", "phi"),
			If("v", "loop", "exit")),
		Bloc("exit",
			Exit("store")))

	CheckFunc(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)
}

func TestDeadStoreTypes(t *testing.T) {
	// Make sure a narrow store can't shadow a wider one. We test an even
	// stronger restriction, that one store can't shadow another unless the
	// types of the address fields are identical (where identicalness is
	// decided by the CSE pass).
	c := testConfig(t)
	t1 := c.config.Types.UInt64.PtrTo()
	t2 := c.config.Types.UInt32.PtrTo()
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", ssaop.OpAddr, t1, 0, nil, "sb"),
			Valu("addr2", ssaop.OpAddr, t2, 0, nil, "sb"),
			Valu("store1", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "start"),
			Valu("store2", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr2", "v", "store1"),
			Goto("exit")),
		Bloc("exit",
			Exit("store2")))

	CheckFunc(fun.f)
	cse(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v := fun.values["store1"]
	if v.Op == ssaop.OpCopy {
		t.Errorf("store %s incorrectly removed", v)
	}
}

func TestDeadStoreUnsafe(t *testing.T) {
	// Make sure a narrow store can't shadow a wider one. The test above
	// covers the case of two different types, but unsafe pointer casting
	// can get to a point where the size is changed but type unchanged.
	c := testConfig(t)
	ptrType := c.config.Types.UInt64.PtrTo()
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, c.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, c.config.Types.Bool, 1, nil),
			Valu("addr1", ssaop.OpAddr, ptrType, 0, nil, "sb"),
			Valu("store1", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Int64, "addr1", "v", "start"), // store 8 bytes
			Valu("store2", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Bool, "addr1", "v", "store1"), // store 1 byte
			Goto("exit")),
		Bloc("exit",
			Exit("store2")))

	CheckFunc(fun.f)
	cse(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v := fun.values["store1"]
	if v.Op == ssaop.OpCopy {
		t.Errorf("store %s incorrectly removed", v)
	}
}

func TestDeadStoreSmallStructInit(t *testing.T) {
	c := testConfig(t)
	ptrType := c.config.Types.BytePtr
	typ := types.NewStruct([]*types.Field{
		types.NewField(src.NoXPos, &types.Sym{Name: "A"}, c.config.Types.Int),
		types.NewField(src.NoXPos, &types.Sym{Name: "B"}, c.config.Types.Int),
	})
	name := c.Temp(typ)
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", ssaop.OpSP, c.config.Types.Uintptr, 0, nil),
			Valu("zero", ssaop.OpConst64, c.config.Types.Int, 0, nil),
			Valu("v6", ssaop.OpLocalAddr, ptrType, 0, name, "sp", "start"),
			Valu("v3", ssaop.OpOffPtr, ptrType, 8, nil, "v6"),
			Valu("v22", ssaop.OpOffPtr, ptrType, 0, nil, "v6"),
			Valu("zerostore1", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Int, "v22", "zero", "start"),
			Valu("zerostore2", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Int, "v3", "zero", "zerostore1"),
			Valu("v8", ssaop.OpLocalAddr, ptrType, 0, name, "sp", "zerostore2"),
			Valu("v23", ssaop.OpOffPtr, ptrType, 8, nil, "v8"),
			Valu("v25", ssaop.OpOffPtr, ptrType, 0, nil, "v8"),
			Valu("zerostore3", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Int, "v25", "zero", "zerostore2"),
			Valu("zerostore4", ssaop.OpStore, types.TypeMem, 0, c.config.Types.Int, "v23", "zero", "zerostore3"),
			Goto("exit")),
		Bloc("exit",
			Exit("zerostore4")))

	fun.f.Name = "smallstructinit"
	CheckFunc(fun.f)
	cse(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	v1 := fun.values["zerostore1"]
	if v1.Op != ssaop.OpCopy {
		t.Errorf("dead store not removed")
	}
	v2 := fun.values["zerostore2"]
	if v2.Op != ssaop.OpCopy {
		t.Errorf("dead store not removed")
	}
}

func TestDeadStoreArrayGap(t *testing.T) {
	c := testConfig(t)
	ptr := c.config.Types.BytePtr
	i64 := c.config.Types.Int64

	typ := types.NewArray(i64, 5)
	tmp := c.Temp(typ)

	fun := c.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", ssaop.OpSP, c.config.Types.Uintptr, 0, nil),

			Valu("base", ssaop.OpLocalAddr, ptr, 0, tmp, "sp", "start"),

			Valu("p0", ssaop.OpOffPtr, ptr, 0, nil, "base"),
			Valu("p1", ssaop.OpOffPtr, ptr, 8, nil, "base"),
			Valu("p2", ssaop.OpOffPtr, ptr, 16, nil, "base"),
			Valu("p3", ssaop.OpOffPtr, ptr, 24, nil, "base"),
			Valu("p4", ssaop.OpOffPtr, ptr, 32, nil, "base"),

			Valu("one", ssaop.OpConst64, i64, 1, nil),
			Valu("seven", ssaop.OpConst64, i64, 7, nil),
			Valu("zero", ssaop.OpConst64, i64, 0, nil),

			Valu("mem0", ssaop.OpZero, types.TypeMem, 40, typ, "base", "start"),

			Valu("s0", ssaop.OpStore, types.TypeMem, 0, i64, "p0", "one", "mem0"),
			Valu("s1", ssaop.OpStore, types.TypeMem, 0, i64, "p1", "seven", "s0"),
			Valu("s2", ssaop.OpStore, types.TypeMem, 0, i64, "p3", "one", "s1"),
			Valu("s3", ssaop.OpStore, types.TypeMem, 0, i64, "p4", "one", "s2"),
			Valu("s4", ssaop.OpStore, types.TypeMem, 0, i64, "p2", "zero", "s3"),

			Goto("exit")),
		Bloc("exit",
			Exit("s4")))

	CheckFunc(fun.f)
	dse(fun.f)
	CheckFunc(fun.f)

	if op := fun.values["mem0"].Op; op != ssaop.OpCopy {
		t.Fatalf("dead Zero not removed: got %s, want OpCopy", op)
	}
}

func TestShadowRanges(t *testing.T) {
	t.Run("simple insert & contains", func(t *testing.T) {
		var sr shadowRanges
		sr.add(10, 20)

		wantRanges(t, sr.ranges, [][2]uint16{{10, 20}})
		if !sr.contains(12, 18) || !sr.contains(10, 20) {
			t.Fatalf("contains failed after simple add")
		}
		if sr.contains(9, 11) || sr.contains(11, 21) {
			t.Fatalf("contains erroneously true for non-contained range")
		}
	})

	t.Run("merge overlapping", func(t *testing.T) {
		var sr shadowRanges
		sr.add(10, 20)
		sr.add(15, 25)

		wantRanges(t, sr.ranges, [][2]uint16{{10, 25}})
		if !sr.contains(13, 24) {
			t.Fatalf("contains should be true after merge")
		}
	})

	t.Run("merge touching boundary", func(t *testing.T) {
		var sr shadowRanges
		sr.add(100, 150)
		// touches at 150 - should coalesce
		sr.add(150, 180)

		wantRanges(t, sr.ranges, [][2]uint16{{100, 180}})
	})

	t.Run("union across several ranges", func(t *testing.T) {
		var sr shadowRanges
		sr.add(10, 20)
		sr.add(30, 40)
		// bridges second, not first
		sr.add(25, 35)

		wantRanges(t, sr.ranges, [][2]uint16{{10, 20}, {25, 40}})

		// envelops everything
		sr.add(5, 50)
		wantRanges(t, sr.ranges, [][2]uint16{{5, 50}})
	})

	t.Run("disjoint intervals stay separate", func(t *testing.T) {
		var sr shadowRanges
		sr.add(10, 20)
		sr.add(22, 30)

		wantRanges(t, sr.ranges, [][2]uint16{{10, 20}, {22, 30}})
		// spans both
		if sr.contains(15, 25) {
			t.Fatalf("contains across two disjoint ranges should be false")
		}
	})

	t.Run("large uint16 offsets still work", func(t *testing.T) {
		var sr shadowRanges
		sr.add(40000, 45000)

		if !sr.contains(42000, 43000) {
			t.Fatalf("contains failed for large uint16 values")
		}
	})

	t.Run("out-of-bounds inserts ignored", func(t *testing.T) {
		var sr shadowRanges
		sr.add(10, 20)
		sr.add(-5, 5)
		sr.add(70000, 70010)

		wantRanges(t, sr.ranges, [][2]uint16{{10, 20}})
	})
}

// canonicalise order for comparisons
func sortRanges(r []shadowRange) {
	sort.Slice(r, func(i, j int) bool { return r[i].lo < r[j].lo })
}

// compare actual slice with expected pairs
func wantRanges(t *testing.T, got []shadowRange, want [][2]uint16) {
	t.Helper()
	sortRanges(got)

	if len(got) != len(want) {
		t.Fatalf("len(ranges)=%d, want %d (got=%v)", len(got), len(want), got)
	}

	for i, w := range want {
		if got[i].lo != w[0] || got[i].hi != w[1] {
			t.Fatalf("range %d = [%d,%d], want [%d,%d] (full=%v)",
				i, got[i].lo, got[i].hi, w[0], w[1], got)
		}
	}
}

func BenchmarkDeadStore(b *testing.B) {
	cfg := testConfig(b)
	ptr := cfg.config.Types.BytePtr

	f := cfg.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, cfg.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, cfg.config.Types.Bool, 1, nil),
			Valu("a1", ssaop.OpAddr, ptr, 0, nil, "sb"),
			Valu("a2", ssaop.OpAddr, ptr, 0, nil, "sb"),
			Valu("a3", ssaop.OpAddr, ptr, 0, nil, "sb"),
			Valu("z1", ssaop.OpZero, types.TypeMem, 1, cfg.config.Types.Bool, "a3", "start"),
			Valu("s1", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a1", "v", "z1"),
			Valu("s2", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a2", "v", "s1"),
			Valu("s3", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a1", "v", "s2"),
			Valu("s4", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a3", "v", "s3"),
			Goto("exit")),
		Bloc("exit",
			Exit("s3")))

	runBench(b, func() {
		dse(f.f)
	})
}

func BenchmarkDeadStorePhi(b *testing.B) {
	cfg := testConfig(b)
	ptr := cfg.config.Types.BytePtr

	f := cfg.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, cfg.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, cfg.config.Types.Bool, 1, nil),
			Valu("addr", ssaop.OpAddr, ptr, 0, nil, "sb"),
			Goto("loop")),
		Bloc("loop",
			Valu("phi", ssaop.OpPhi, types.TypeMem, 0, nil, "start", "store"),
			Valu("store", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "addr", "v", "phi"),
			If("v", "loop", "exit")),
		Bloc("exit",
			Exit("store")))

	runBench(b, func() {
		dse(f.f)
	})
}

func BenchmarkDeadStoreTypes(b *testing.B) {
	cfg := testConfig(b)

	t1 := cfg.config.Types.UInt64.PtrTo()
	t2 := cfg.config.Types.UInt32.PtrTo()

	f := cfg.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, cfg.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, cfg.config.Types.Bool, 1, nil),
			Valu("a1", ssaop.OpAddr, t1, 0, nil, "sb"),
			Valu("a2", ssaop.OpAddr, t2, 0, nil, "sb"),
			Valu("s1", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a1", "v", "start"),
			Valu("s2", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a2", "v", "s1"),
			Goto("exit")),
		Bloc("exit",
			Exit("s2")))
	cse(f.f)

	runBench(b, func() {
		dse(f.f)
	})
}

func BenchmarkDeadStoreUnsafe(b *testing.B) {
	cfg := testConfig(b)
	ptr := cfg.config.Types.UInt64.PtrTo()
	f := cfg.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sb", ssaop.OpSB, cfg.config.Types.Uintptr, 0, nil),
			Valu("v", ssaop.OpConstBool, cfg.config.Types.Bool, 1, nil),
			Valu("a1", ssaop.OpAddr, ptr, 0, nil, "sb"),
			Valu("s1", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Int64, "a1", "v", "start"),
			Valu("s2", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Bool, "a1", "v", "s1"),
			Goto("exit")),
		Bloc("exit",
			Exit("s2")))
	cse(f.f)
	runBench(b, func() {
		dse(f.f)
	})
}

func BenchmarkDeadStoreSmallStructInit(b *testing.B) {
	cfg := testConfig(b)
	ptr := cfg.config.Types.BytePtr

	typ := types.NewStruct([]*types.Field{
		types.NewField(src.NoXPos, &types.Sym{Name: "A"}, cfg.config.Types.Int),
		types.NewField(src.NoXPos, &types.Sym{Name: "B"}, cfg.config.Types.Int),
	})
	tmp := cfg.Temp(typ)

	f := cfg.Fun("entry",
		Bloc("entry",
			Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
			Valu("sp", ssaop.OpSP, cfg.config.Types.Uintptr, 0, nil),
			Valu("zero", ssaop.OpConst64, cfg.config.Types.Int, 0, nil),

			Valu("v6", ssaop.OpLocalAddr, ptr, 0, tmp, "sp", "start"),
			Valu("v3", ssaop.OpOffPtr, ptr, 8, nil, "v6"),
			Valu("v22", ssaop.OpOffPtr, ptr, 0, nil, "v6"),
			Valu("s1", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Int, "v22", "zero", "start"),
			Valu("s2", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Int, "v3", "zero", "s1"),

			Valu("v8", ssaop.OpLocalAddr, ptr, 0, tmp, "sp", "s2"),
			Valu("v23", ssaop.OpOffPtr, ptr, 8, nil, "v8"),
			Valu("v25", ssaop.OpOffPtr, ptr, 0, nil, "v8"),
			Valu("s3", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Int, "v25", "zero", "s2"),
			Valu("s4", ssaop.OpStore, types.TypeMem, 0, cfg.config.Types.Int, "v23", "zero", "s3"),
			Goto("exit")),
		Bloc("exit",
			Exit("s4")))
	cse(f.f)

	runBench(b, func() {
		dse(f.f)
	})
}

func BenchmarkDeadStoreLargeBlock(b *testing.B) {
	// create a very large block with many shadowed stores
	const (
		addrCount = 128
		// first 7 are dead
		storesPerAddr = 8
	)
	cfg := testConfig(b)
	ptrType := cfg.config.Types.BytePtr
	boolType := cfg.config.Types.Bool

	items := []interface{}{
		Valu("start", ssaop.OpInitMem, types.TypeMem, 0, nil),
		Valu("sb", ssaop.OpSB, cfg.config.Types.Uintptr, 0, nil),
		Valu("v", ssaop.OpConstBool, boolType, 1, nil),
	}

	for i := 0; i < addrCount; i++ {
		items = append(items,
			Valu(fmt.Sprintf("addr%d", i), ssaop.OpAddr, ptrType, 0, nil, "sb"),
		)
	}

	prev := "start"
	for round := 0; round < storesPerAddr; round++ {
		for i := 0; i < addrCount; i++ {
			store := fmt.Sprintf("s_%03d_%d", i, round)
			addr := fmt.Sprintf("addr%d", i)
			items = append(items,
				Valu(store, ssaop.OpStore, types.TypeMem, 0, boolType, addr, "v", prev),
			)
			prev = store
		}
	}

	items = append(items, Goto("exit"))
	entryBlk := Bloc("entry", items...)
	exitBlk := Bloc("exit", Exit(prev))

	f := cfg.Fun("stress", entryBlk, exitBlk)

	runBench(b, func() {
		dse(f.f)
	})
}

func runBench(b *testing.B, build func()) {
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		build()
	}
}
