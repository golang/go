// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: live at start of block instead?

package ssacompile

import (
	"fmt"

	"cmd/compile/internal/ssa"
)

// stackalloc allocates storage in the stack frame for
// all Values that did not get a register.
// Returns a map from block ID to the stack values live at the end of that block.
func stackalloc(f *ssa.Func, spillLive [][]ssa.ID) [][]ssa.ID {
	if f.Pass.Debug > ssa.StackDebug {
		fmt.Println("before stackalloc")
		fmt.Println(f.String())
	}
	s := ssa.NewStackAllocState(f)
	s.Init(f, spillLive)
	defer ssa.PutStackAllocState(s)

	s.Stackalloc()
	if f.Pass.Stats > 0 {
		f.LogStat("stack_alloc_stats",
			s.NArgSlot, "arg_slots", s.NNotNeed, "slot_not_needed",
			s.NNamedSlot, "named_slots", s.NAuto, "auto_slots",
			s.NReuse, "reused_slots", s.NSelfInterfere, "self_interfering")
	}

	return s.Live
}
