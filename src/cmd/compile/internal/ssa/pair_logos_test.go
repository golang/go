// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

// TestLogosPairARM64 pins the arm64 load/store pairing pass (SPEC 15.1,
// row 111) at the level where it runs. Sixteen-byte vector traffic
// cannot be coaxed out of Go source with adjacent memory arguments:
// the ABI homes every 16-byte temporary on the stack and the spill
// stores split the shared memory value the pass requires. These cases
// therefore build the SSA values directly.
func TestLogosPairARM64(t *testing.T) {
	vec := types.TypeVec128
	u64 := types.Types[types.TUINT64]

	loadCases := []struct {
		name   string
		op     Op
		typ    *types.Type
		off    int64
		pair   Op
		fused  bool
		offset int64
	}{
		{name: "load8", op: OpARM64MOVDload, typ: u64, off: 0, pair: OpARM64LDP, fused: true, offset: 0},
		{name: "load8off504", op: OpARM64MOVDload, typ: u64, off: 504, pair: OpARM64LDP, fused: true, offset: 504},
		{name: "load8off512", op: OpARM64MOVDload, typ: u64, off: 512, pair: OpARM64LDP, fused: true, offset: 512},
		{name: "load16off0", op: OpARM64FMOVQload, typ: vec, off: 0, pair: OpARM64FLDPQ, fused: true, offset: 0},
		{name: "load16off992", op: OpARM64FMOVQload, typ: vec, off: 992, pair: OpARM64FLDPQ, fused: true, offset: 992},
		{name: "load16off1008", op: OpARM64FMOVQload, typ: vec, off: 1008, pair: OpARM64FLDPQ, fused: true, offset: 1008},
		{name: "load16off1024", op: OpARM64FMOVQload, typ: vec, off: 1024, pair: OpARM64FLDPQ, fused: false},
		{name: "load16misaligned", op: OpARM64FMOVQload, typ: vec, off: 8, pair: OpARM64FLDPQ, fused: false},
	}

	for _, tc := range loadCases {
		t.Run(tc.name, func(t *testing.T) {
			c := testConfigARM64(t)
			width := int64(8)
			if tc.op == OpARM64FMOVQload {
				width = 16
			}
			off1 := tc.off
			off2 := tc.off
			if tc.name == "load16misaligned" {
				off1 = 0
				off2 = 8
			} else if tc.fused || tc.name == "load16off1024" {
				off2 = off1 + width
			}
			fun := c.Fun("entry",
				Bloc("entry",
					Valu("mem", OpInitMem, types.TypeMem, 0, nil),
					Valu("p", OpARM64MOVDconst, u64, 0, nil),
					Valu("l1", tc.op, tc.typ, off1, nil, "p", "mem"),
					Valu("l2", tc.op, tc.typ, off2, nil, "p", "mem"),
					Goto("exit")),
				Bloc("exit",
					Exit("mem")),
			)
			CheckFunc(fun.f)
			pair(fun.f)
			CheckFunc(fun.f)
			l1 := fun.values["l1"]
			l2 := fun.values["l2"]
			if !tc.fused {
				if l1.Op != tc.op || l2.Op != tc.op {
					t.Errorf("loads fused unexpectedly: l1=%v l2=%v", l1.Op, l2.Op)
				}
				return
			}
			checkPair(t, fun, tc.pair, tc.offset, l1, l2)
		})
	}
}

// TestLogosPairARM64SplitMem pins that any memory operation between two
// candidate loads keeps them separate. Calls and atomics thread memory
// the same way; the end-to-end cases pin those spellings.
func TestLogosPairARM64SplitMem(t *testing.T) {
	c := testConfigARM64(t)
	vec := types.TypeVec128
	u64 := types.Types[types.TUINT64]
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpARM64MOVDconst, u64, 0, nil),
			Valu("v8", OpARM64MOVDconst, u64, 0, nil),
			Valu("l1", OpARM64FMOVQload, vec, 0, nil, "p", "mem"),
			Valu("st", OpARM64MOVDstore, types.TypeMem, 0, nil, "p", "v8", "mem"),
			Valu("l2", OpARM64FMOVQload, vec, 16, nil, "p", "st"),
			Goto("exit")),
		Bloc("exit",
			Exit("st")),
	)
	CheckFunc(fun.f)
	pair(fun.f)
	CheckFunc(fun.f)
	if fun.values["l1"].Op != OpARM64FMOVQload || fun.values["l2"].Op != OpARM64FMOVQload {
		t.Errorf("loads fused across a store: l1=%v l2=%v", fun.values["l1"].Op, fun.values["l2"].Op)
	}
}

// TestLogosPairARM64JoinBlock pins that a load in a join block never
// fuses with a load from a predecessor.
func TestLogosPairARM64JoinBlock(t *testing.T) {
	c := testConfigARM64(t)
	vec := types.TypeVec128
	u64 := types.Types[types.TUINT64]
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpARM64MOVDconst, u64, 0, nil),
			Valu("l1", OpARM64FMOVQload, vec, 0, nil, "p", "mem"),
			Valu("c", OpConstBool, c.config.Types.Bool, 1, nil),
			If("c", "left", "right")),
		Bloc("left",
			Goto("join")),
		Bloc("right",
			Goto("join")),
		Bloc("join",
			Valu("l2", OpARM64FMOVQload, vec, 16, nil, "p", "mem"),
			Exit("mem")),
	)
	CheckFunc(fun.f)
	pair(fun.f)
	CheckFunc(fun.f)
	if fun.values["l1"].Op != OpARM64FMOVQload || fun.values["l2"].Op != OpARM64FMOVQload {
		t.Errorf("loads fused across a join: l1=%v l2=%v", fun.values["l1"].Op, fun.values["l2"].Op)
	}
}

// TestLogosPairARM64DiffPtr pins that a load feeding the base of a later
// load never fuses with it.
func TestLogosPairARM64DiffPtr(t *testing.T) {
	c := testConfigARM64(t)
	vec := types.TypeVec128
	u64 := types.Types[types.TUINT64]
	fun := c.Fun("entry",
		Bloc("entry",
			Valu("mem", OpInitMem, types.TypeMem, 0, nil),
			Valu("p", OpARM64MOVDconst, u64, 0, nil),
			Valu("q", OpARM64MOVDconst, u64, 0, nil),
			Valu("l1", OpARM64FMOVQload, vec, 0, nil, "p", "mem"),
			Valu("l2", OpARM64FMOVQload, vec, 16, nil, "q", "mem"),
			Goto("exit")),
		Bloc("exit",
			Exit("mem")),
	)
	CheckFunc(fun.f)
	pair(fun.f)
	CheckFunc(fun.f)
	if fun.values["l1"].Op != OpARM64FMOVQload || fun.values["l2"].Op != OpARM64FMOVQload {
		t.Errorf("loads fused across bases: l1=%v l2=%v", fun.values["l1"].Op, fun.values["l2"].Op)
	}
}

// TestLogosPairARM64Stores pins 16-byte store fusion and its barriers.
func TestLogosPairARM64Stores(t *testing.T) {
	vec := types.TypeVec128
	u64 := types.Types[types.TUINT64]
	cases := []struct {
		name  string
		off1  int64
		off2  int64
		flush bool
		fused bool
		place int64
	}{
		{name: "store16off0", off1: 0, off2: 16, fused: true, place: 0},
		{name: "store16off992", off1: 992, off2: 1008, fused: true, place: 992},
		{name: "store16off1024", off1: 1024, off2: 1040, fused: false},
		{name: "store16misaligned", off1: 0, off2: 8, fused: false},
		{name: "store16flush", off1: 0, off2: 16, flush: true, fused: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			c := testConfigARM64(t)
			entry := []any{
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("p", OpARM64MOVDconst, u64, 0, nil),
				Valu("v1", OpARM64FMOVQload, vec, 64, nil, "p", "mem"),
				Valu("v2", OpARM64FMOVQload, vec, 80, nil, "p", "mem"),
				Valu("s1", OpARM64FMOVQstore, types.TypeMem, tc.off1, nil, "p", "v1", "mem"),
			}
			tail := "s1"
			if tc.flush {
				entry = append(entry, Valu("mid", OpCopy, types.TypeMem, 0, nil, "s1"))
				tail = "mid"
			}
			entry = append(entry,
				Valu("s2", OpARM64FMOVQstore, types.TypeMem, tc.off2, nil, "p", "v2", tail),
				Goto("exit"))
			fun := c.Fun("entry",
				Bloc("entry", entry...),
				Bloc("exit",
					Exit("s2")),
			)
			CheckFunc(fun.f)
			pair(fun.f)
			CheckFunc(fun.f)
			s1 := fun.values["s1"]
			s2 := fun.values["s2"]
			if !tc.fused {
				if s1.Op != OpARM64FMOVQstore || s2.Op != OpARM64FMOVQstore {
					t.Errorf("stores fused unexpectedly: s1=%v s2=%v", s1.Op, s2.Op)
				}
				return
			}
			if s1.Op != OpARM64FSTPQ {
				t.Fatalf("stores did not fuse: s1=%v s2=%v", s1.Op, s2.Op)
			}
			if s1.AuxInt != tc.place {
				t.Errorf("fused store at %d, want %d", s1.AuxInt, tc.place)
			}
			if s2.Op != OpCopy {
				t.Errorf("second store not replaced by copy: %v", s2.Op)
			}
		})
	}
}

// checkPair verifies that l1 and l2 became the two distinct selections
// of one pair instruction at the expected offset.
func checkPair(t *testing.T, fun fun, want Op, offset int64, l1, l2 *Value) {
	t.Helper()
	if l1.Op != OpSelect0 || l2.Op != OpSelect1 {
		t.Fatalf("loads did not fuse: l1=%v l2=%v", l1.Op, l2.Op)
	}
	p := l1.Args[0]
	if len(l2.Args) != 1 || l2.Args[0] != p {
		t.Fatalf("selections share no pair value: l1=%v l2=%v", l1, l2)
	}
	if p.Op != want {
		t.Errorf("pair op is %v, want %v", p.Op, want)
	}
	if p.AuxInt != offset {
		t.Errorf("pair offset is %d, want %d", p.AuxInt, offset)
	}
}
