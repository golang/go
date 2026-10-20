// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/types"
	"testing"
)

// M13-S2 synthetic PGO (SPEC T-105): the Cold hint marks the branch into
// the cold block BranchUnlikely, likelyadjust keeps the explicit mark, and
// blocks without a single two-way predecessor stay unmarked.

// TestLogosColdBlockLaidOutLast pins SPEC #104 and #105: a marked cold
// successor stays inside its function and layout places it after the hot
// successor. The unmarked control keeps the first-successor order, so the
// mark is what moves the cold block.
func TestLogosColdBlockLaidOutLast(t *testing.T) {
	order := func(t *testing.T, mark bool) []*Block {
		t.Helper()
		c := testConfig(t)
		fun := c.Fun("entry",
			Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("cond", OpConstBool, c.config.Types.Bool, 1, nil),
				If("cond", "cold", "hot")),
			Bloc("cold",
				Ret("mem")),
			Bloc("hot",
				Ret("mem")))
		CheckFunc(fun.f)
		if mark && !MarkColdBranch(fun.blocks["cold"]) {
			t.Fatal("MarkColdBranch(cold) = false, want true")
		}
		layout(fun.f)
		CheckFunc(fun.f)
		return []*Block{fun.blocks["entry"], fun.blocks["hot"], fun.blocks["cold"]}
	}
	index := func(f []*Block, b *Block) int {
		for i, x := range f {
			if x == b {
				return i
			}
		}
		return -1
	}

	t.Run("marked cold block is last", func(t *testing.T) {
		want := order(t, true)
		got := want[0].Func.Blocks
		if len(got) != 3 || got[len(got)-1] != want[2] {
			t.Fatalf("layout = %v, want cold block %v last", got, want[2])
		}
		if index(got, want[1]) >= index(got, want[2]) {
			t.Fatalf("hot block not before cold block in %v", got)
		}
	})

	t.Run("unmarked control keeps first successor first", func(t *testing.T) {
		want := order(t, false)
		got := want[0].Func.Blocks
		if index(got, want[2]) >= index(got, want[1]) {
			t.Fatalf("unmarked cold block not before hot block in %v", got)
		}
	})
}

func TestLogosColdBranchMark(t *testing.T) {
	build := func(t *testing.T, first, second string) (*Conf, *Func, map[string]*Block) {
		t.Helper()
		c := testConfig(t)
		fun := c.Fun("entry",
			Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("cond", OpConstBool, c.config.Types.Bool, 1, nil),
				If("cond", first, second)),
			Bloc("hot",
				Goto("join")),
			Bloc("cold",
				Goto("join")),
			Bloc("join",
				Exit("mem")))
		CheckFunc(fun.f)
		return c, fun.f, fun.blocks
	}

	t.Run("cold first successor prefers the other", func(t *testing.T) {
		_, f, blocks := build(t, "cold", "hot")
		if !MarkColdBranch(blocks["cold"]) {
			t.Fatal("MarkColdBranch(cold) = false, want true")
		}
		if got := blocks["entry"].Likely; got != BranchUnlikely {
			t.Fatalf("entry.Likely = %v, want BranchUnlikely", got)
		}
		likelyadjust(f)
		CheckFunc(f)
		if got := blocks["entry"].Likely; got != BranchUnlikely {
			t.Fatalf("after likelyadjust entry.Likely = %v, want BranchUnlikely", got)
		}
	})

	t.Run("cold second successor prefers the other", func(t *testing.T) {
		_, f, blocks := build(t, "hot", "cold")
		if !MarkColdBranch(blocks["cold"]) {
			t.Fatal("MarkColdBranch(cold) = false, want true")
		}
		if got := blocks["entry"].Likely; got != BranchLikely {
			t.Fatalf("entry.Likely = %v, want BranchLikely", got)
		}
		likelyadjust(f)
		CheckFunc(f)
		if got := blocks["entry"].Likely; got != BranchLikely {
			t.Fatalf("after likelyadjust entry.Likely = %v, want BranchLikely", got)
		}
	})

	t.Run("shared cold block stays unmarked", func(t *testing.T) {
		c := testConfig(t)
		fun := c.Fun("entry",
			Bloc("entry",
				Valu("mem", OpInitMem, types.TypeMem, 0, nil),
				Valu("cond", OpConstBool, c.config.Types.Bool, 1, nil),
				If("cond", "hot", "cold")),
			Bloc("hot",
				Goto("cold")),
			Bloc("cold",
				Goto("join")),
			Bloc("join",
				Exit("mem")))
		CheckFunc(fun.f)
		if MarkColdBranch(fun.blocks["cold"]) {
			t.Fatal("MarkColdBranch(shared cold) = true, want false")
		}
		if got := fun.blocks["entry"].Likely; got != BranchUnknown {
			t.Fatalf("entry.Likely = %v, want BranchUnknown", got)
		}
	})

	t.Run("explicit mark is never overridden", func(t *testing.T) {
		_, _, blocks := build(t, "hot", "cold")
		blocks["entry"].Likely = BranchUnlikely
		if MarkColdBranch(blocks["cold"]) {
			t.Fatal("MarkColdBranch against explicit mark = true, want false")
		}
		if got := blocks["entry"].Likely; got != BranchUnlikely {
			t.Fatalf("entry.Likely = %v, want BranchUnlikely", got)
		}
	})
}
