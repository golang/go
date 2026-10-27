// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"testing"

	"cmd/compile/internal/abt"
	"cmd/compile/internal/ssa/ssaop"
)

// TestVarLocHighRegister checks that a variable in a register numbered 64 or
// above (e.g. an arm64 P register) keeps its location.
func TestVarLocHighRegister(t *testing.T) {
	const lo, hi = 3, 70

	loc := VarLoc{Registers: RegisterSet{}.AddReg(hi)}
	if loc.absent() {
		t.Fatalf("location in register %d is absent", hi)
	}
	if got := firstReg(loc.Registers); got != hi {
		t.Errorf("firstReg = %d, want %d", got, hi)
	}

	both := VarLoc{Registers: loc.Registers.AddReg(lo)}
	if got := both.intersect(loc); got.Registers != loc.Registers {
		t.Errorf("intersect = %v, want %v", got.Registers, loc.Registers)
	}
	if !canMerge(loc, both) {
		t.Errorf("canMerge(%v, %v) = false, want true", loc.Registers, both.Registers)
	}
	if canMerge(both, loc) {
		t.Errorf("canMerge(%v, %v) = true, want false", both.Registers, loc.Registers)
	}
	if got := both.Registers.RemoveReg(ssaop.Register(hi)); got.HasReg(hi) || !got.HasReg(lo) {
		t.Errorf("RemoveReg(%d) = %v", hi, got)
	}

	var live abt.T
	live.Insert(0, &liveSlot{both})
	state := StateAtPC{slots: make([]VarLoc, 1), registers: make([][]SlotID, hi+1)}
	state.reset(live)
	for _, r := range []int{lo, hi} {
		if len(state.registers[r]) != 1 || state.registers[r][0] != 0 {
			t.Errorf("registers[%d] = %v, want [0]", r, state.registers[r])
		}
	}
}
