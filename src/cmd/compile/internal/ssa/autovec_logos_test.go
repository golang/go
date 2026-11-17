// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/types"
	"cmd/internal/src"
	"internal/buildcfg"
	"testing"
)

func autovecTestSettings(t *testing.T) {
	t.Helper()
	oldSIMD, oldAMD64 := buildcfg.Experiment.SIMD, buildcfg.GOAMD64
	oldRace, oldMSan, oldASan := base.Flag.Race, base.Flag.MSan, base.Flag.ASan
	buildcfg.Experiment.SIMD, buildcfg.GOAMD64 = true, 3
	base.Flag.Race, base.Flag.MSan, base.Flag.ASan = false, false, false
	t.Cleanup(func() {
		buildcfg.Experiment.SIMD, buildcfg.GOAMD64 = oldSIMD, oldAMD64
		base.Flag.Race, base.Flag.MSan, base.Flag.ASan = oldRace, oldMSan, oldASan
	})
}

func autovecTestLoop(t *testing.T, arch string) fun {
	t.Helper()
	c := testConfigArch(t, arch)
	typ := &c.config.Types
	fn := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("dst", OpArg, typ.UInt32Ptr, 0, nil),
			Valu("src", OpArg, typ.UInt32Ptr, 0, nil),
			Valu("limit", OpArg, typ.Int64, 0, nil),
			Valu("delta", OpArg, typ.UInt32, 0, nil),
			Valu("zero", OpConst64, typ.Int64, 0, nil),
			Valu("one", OpConst64, typ.Int64, 1, nil),
			Valu("shift", OpConst64, typ.UInt64, 2, nil),
			Goto("header")),
		Bloc("header",
			Valu("index", OpPhi, typ.Int64, 0, nil, "zero", "next"),
			Valu("loopmem", OpPhi, types.TypeMem, 0, nil, "mem", "store"),
			Valu("cond", OpLess64, typ.Bool, 0, nil, "index", "limit"),
			If("cond", "body", "exit")),
		Bloc("body",
			Valu("offset", OpLsh64x64, typ.Int64, 0, nil, "index", "shift"),
			Valu("dstaddr", OpAddPtr, typ.UInt32Ptr, 0, nil, "dst", "offset"),
			Valu("srcaddr", OpAddPtr, typ.UInt32Ptr, 0, nil, "src", "offset"),
			Valu("load", OpLoad, typ.UInt32, 0, nil, "srcaddr", "loopmem"),
			Valu("sum", OpAdd32, typ.UInt32, 0, nil, "load", "delta"),
			Valu("store", OpStore, types.TypeMem, 0, typ.UInt32, "dstaddr", "sum", "loopmem"),
			Valu("next", OpAdd64, typ.Int64, 0, nil, "index", "one"),
			Goto("header")),
		Bloc("exit", Exit("loopmem")))
	fn.f.ABIDefault = c.config.ABI1
	return fn
}

func autovecVectorStores(f *Func) int {
	n := 0
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op == OpStore && v.Args[1].Type.IsSIMD() {
				n++
			}
		}
	}
	return n
}

func TestLogosAutovecPreservesScalarLoop(t *testing.T) {
	autovecTestSettings(t)
	for _, arch := range []string{"amd64", "arm64"} {
		for _, reversed := range []bool{false, true} {
			t.Run(arch+map[bool]string{false: "/forward", true: "/reversed"}[reversed], func(t *testing.T) {
				fun := autovecTestLoop(t, arch)
				h, b := fun.blocks["header"], fun.blocks["body"]
				entryIndex, backIndex := 0, 1
				if reversed {
					h.Preds[0], h.Preds[1] = h.Preds[1], h.Preds[0]
					for i, e := range h.Preds {
						e.b.Succs[e.i].i = i
					}
					for _, name := range []string{"index", "loopmem"} {
						v := fun.values[name]
						v.Args[0], v.Args[1] = v.Args[1], v.Args[0]
					}
					entryIndex, backIndex = 1, 0
				}
				CheckFunc(fun.f)
				autovec(fun.f)
				CheckFunc(fun.f)
				if got := autovecVectorStores(fun.f); got != 1 {
					t.Fatalf("vector stores = %d, want 1", got)
				}
				if h.Succs[0].b != b || h.Succs[1].b != fun.blocks["exit"] || b.Succs[0].b != h {
					t.Fatal("scalar loop body, backedge, or exit changed")
				}
				for _, pair := range [][2]string{{"index", "next"}, {"loopmem", "store"}} {
					v := fun.values[pair[0]]
					if v.Args[backIndex] != fun.values[pair[1]] {
						t.Fatalf("scalar %s backedge changed", pair[0])
					}
					if v.Args[entryIndex].Op != OpPhi || v.Args[entryIndex].Block == h {
						t.Fatalf("scalar %s does not receive the vector loop result", pair[0])
					}
				}
				if fun.blocks["exit"].Controls[0] != fun.values["loopmem"] {
					t.Fatal("exit no longer observes the scalar loop's final memory")
				}
			})
		}
	}
}

func TestLogosAutovecPredecessorBranch(t *testing.T) {
	autovecTestSettings(t)
	fun := autovecTestLoop(t, "amd64")
	entry := fun.blocks["entry"]
	skip := fun.f.NewBlock(BlockExit)
	skip.SetControl(fun.values["mem"])
	entry.Kind = BlockIf
	entry.SetControl(entry.NewValue0(src.NoXPos, OpArg, fun.f.Config.Types.Bool))
	entry.AddEdgeTo(skip)
	CheckFunc(fun.f)
	autovec(fun.f)
	CheckFunc(fun.f)
	if autovecVectorStores(fun.f) != 1 {
		t.Fatal("loop with a branching predecessor was not vectorized")
	}
	if entry.Succs[1].b != skip || skip.Controls[0] != fun.values["mem"] {
		t.Fatal("unrelated predecessor edge or memory changed")
	}
}

func TestLogosAutovecSequentialLoops(t *testing.T) {
	autovecTestSettings(t)
	fun := autovecTestLoop(t, "amd64")
	f, typ := fun.f, &fun.f.Config.Types
	between := fun.blocks["exit"]
	between.Kind = BlockPlain
	between.ResetControls()
	h, b, exit := f.NewBlock(BlockIf), f.NewBlock(BlockPlain), f.NewBlock(BlockExit)
	between.AddEdgeTo(h)
	h.AddEdgeTo(b)
	h.AddEdgeTo(exit)
	b.AddEdgeTo(h)
	index := h.NewValue1(src.NoXPos, OpPhi, typ.Int64, fun.values["zero"])
	mem := h.NewValue1(src.NoXPos, OpPhi, types.TypeMem, fun.values["loopmem"])
	h.SetControl(h.NewValue2(src.NoXPos, OpLess64, typ.Bool, index, fun.values["limit"]))
	offset := b.NewValue2(src.NoXPos, OpLsh64x64, typ.Int64, index, fun.values["shift"])
	ptr := b.NewValue2(src.NoXPos, OpAddPtr, typ.UInt32Ptr, fun.values["dst"], offset)
	load := b.NewValue2(src.NoXPos, OpLoad, typ.UInt32, ptr, mem)
	xor := b.NewValue2(src.NoXPos, OpXor32, typ.UInt32, load, fun.values["delta"])
	store := b.NewValue3A(src.NoXPos, OpStore, types.TypeMem, typ.UInt32, ptr, xor, mem)
	index.AddArg(b.NewValue2(src.NoXPos, OpAdd64, typ.Int64, index, fun.values["one"]))
	mem.AddArg(store)
	exit.SetControl(mem)
	CheckFunc(f)
	autovec(f)
	CheckFunc(f)
	if got := autovecVectorStores(f); got != 2 {
		t.Fatalf("vector stores across sequential loops = %d, want 2", got)
	}
}

func TestLogosAutovecRejectsUnsafeLoops(t *testing.T) {
	autovecTestSettings(t)
	tests := []struct {
		name   string
		change func(fun)
	}{
		{"division", func(f fun) { f.values["sum"].Op = OpDiv32u }},
		{"nonunit stride", func(f fun) { f.values["one"].AuxInt = 2 }},
		{"inclusive limit", func(f fun) { f.values["cond"].Op = OpLeq64 }},
		{"shifted source", func(f fun) {
			b := f.blocks["body"]
			shifted := b.NewValue2(src.NoXPos, OpAdd64, f.f.Config.Types.Int64, f.values["index"], f.values["one"])
			offset := b.NewValue2(src.NoXPos, OpLsh64x64, f.f.Config.Types.Int64, shifted, f.values["shift"])
			f.values["srcaddr"].SetArg(1, offset)
		}},
		{"mutable scalar load", func(f fun) {
			b := f.blocks["body"]
			load := b.NewValue2(src.NoXPos, OpLoad, f.f.Config.Types.UInt32, f.values["dst"], f.values["loopmem"])
			f.values["sum"].SetArg(1, load)
		}},
		{"index as data", func(f fun) {
			b := f.blocks["body"]
			index := b.NewValue1(src.NoXPos, OpTrunc64to32, f.f.Config.Types.UInt32, f.values["index"])
			f.values["sum"].SetArg(1, index)
		}},
		{"extra store", func(f fun) {
			b := f.blocks["body"]
			store := b.NewValue3A(src.NoXPos, OpStore, types.TypeMem, f.f.Config.Types.UInt32, f.values["srcaddr"], f.values["delta"], f.values["store"])
			f.values["loopmem"].SetArg(1, store)
		}},
		{"nil check", func(f fun) {
			f.blocks["body"].NewValue2(src.NoXPos, OpNilCheck, f.f.Config.Types.UInt32Ptr, f.values["src"], f.values["loopmem"])
		}},
		{"header load", func(f fun) {
			load := f.blocks["header"].NewValue2(src.NoXPos, OpLoad, f.f.Config.Types.UInt32, f.values["dst"], f.values["loopmem"])
			f.values["sum"].SetArg(1, load)
		}},
		{"reduction phi", func(f fun) {
			phi := f.blocks["header"].NewValue2(src.NoXPos, OpPhi, f.f.Config.Types.UInt32, f.values["delta"], f.values["sum"])
			f.values["sum"].SetArg(1, phi)
		}},
		{"extra live out", func(f fun) {
			h, exit := f.blocks["header"], f.blocks["exit"]
			visible := h.NewValue2(src.NoXPos, OpAdd64, f.f.Config.Types.Int64, f.values["index"], f.values["one"])
			low := exit.NewValue1(src.NoXPos, OpTrunc64to32, f.f.Config.Types.UInt32, visible)
			store := exit.NewValue3A(src.NoXPos, OpStore, types.TypeMem, f.f.Config.Types.UInt32, f.values["dst"], low, f.values["loopmem"])
			exit.SetControl(store)
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			fun := autovecTestLoop(t, "amd64")
			test.change(fun)
			CheckFunc(fun.f)
			before := fun.f.String()
			autovec(fun.f)
			CheckFunc(fun.f)
			if after := fun.f.String(); after != before {
				t.Fatalf("rejected loop changed\nbefore:\n%s\nafter:\n%s", before, after)
			}
		})
	}
}

func TestLogosAutovecTargetAndInstrumentationGates(t *testing.T) {
	tests := []struct {
		name   string
		change func(*Config)
	}{
		{"no SIMD", func(c *Config) { buildcfg.Experiment.SIMD = false }},
		{"amd64 v1", func(c *Config) { buildcfg.GOAMD64 = 1 }},
		{"amd64 v2", func(c *Config) { buildcfg.GOAMD64 = 2 }},
		{"unsupported target", func(c *Config) { c.arch = "riscv64" }},
		{"optimizations off", func(c *Config) { c.optimize = false }},
		{"race configuration", func(c *Config) { c.Race = true }},
		{"race flag", func(c *Config) { base.Flag.Race = true }},
		{"memory sanitizer", func(c *Config) { base.Flag.MSan = true }},
		{"address sanitizer", func(c *Config) { base.Flag.ASan = true }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			autovecTestSettings(t)
			fun := autovecTestLoop(t, "amd64")
			test.change(fun.f.Config)
			before := fun.f.String()
			autovec(fun.f)
			CheckFunc(fun.f)
			if fun.f.String() != before {
				t.Fatal("disabled autovectorization changed the function")
			}
		})
	}
}
