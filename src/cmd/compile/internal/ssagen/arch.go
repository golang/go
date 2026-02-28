// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/objw"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
)

var Arch ArchInfo

// interface to back end

type ArchInfo struct {
	LinkArch *obj.LinkArch

	REGSP     int
	MAXWIDTH  int64
	SoftFloat bool

	PadFrame func(int64) int64

	// ZeroRange zeroes a range of memory the on stack.
	//  - it is only called at function entry
	//  - it is ok to clobber (non-arg) registers.
	//  - currently used only for small things, so it can be simple.
	//    - pointers to heap-allocated return values
	//    - open-coded deferred functions
	// (Max size in make.bash is 40 bytes.)
	ZeroRange func(*objw.Progs, *obj.Prog, int64, int64, *uint32) *obj.Prog

	Ginsnop func(*objw.Progs) *obj.Prog

	// SSAMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
	SSAMarkMoves func(*State, *ssa.Block)

	// SSAGenValue emits Prog(s) for the Value.
	SSAGenValue func(*State, *ssa.Value)

	// SSAGenBlock emits end-of-block Progs. SSAGenValue should be called
	// for all values in the block before SSAGenBlock.
	SSAGenBlock func(s *State, b, next *ssa.Block)

	// LoadRegResult emits instructions that loads register-assigned result
	// at n+off (n is PPARAMOUT) to register reg. The result is already in
	// memory. Used in open-coded defer return path.
	LoadRegResult func(s *State, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog

	// SpillArgReg emits instructions that spill reg to n+off.
	SpillArgReg func(pp *objw.Progs, p *obj.Prog, f *ssa.Func, t *types.Type, reg int16, n *ir.Name, off int64) *obj.Prog
}
