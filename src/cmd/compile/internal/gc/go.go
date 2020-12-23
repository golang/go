// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

import (
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"sync"
)

var pragcgobuf [][]string

var zerosize int64

var (
	funcsymsmu sync.Mutex // protects funcsyms and associated package lookups (see func funcsym)
	funcsyms   []*types.Sym
)

// interface to back end

type Arch struct {
	LinkArch *obj.LinkArch

	REGSP     int
	MAXWIDTH  int64
	SoftFloat bool

	PadFrame func(int64) int64

	// ZeroRange zeroes a range of memory on stack. It is only inserted
	// at function entry, and it is ok to clobber registers.
	ZeroRange func(*Progs, *obj.Prog, int64, int64, *uint32) *obj.Prog

	Ginsnop      func(*Progs) *obj.Prog
	Ginsnopdefer func(*Progs) *obj.Prog // special ginsnop for deferreturn

	// SSAMarkMoves marks any MOVXconst ops that need to avoid clobbering flags.
	SSAMarkMoves func(*SSAGenState, *ssa.Block)

	// SSAGenValue emits Prog(s) for the Value.
	SSAGenValue func(*SSAGenState, *ssa.Value)

	// SSAGenBlock emits end-of-block Progs. SSAGenValue should be called
	// for all values in the block before SSAGenBlock.
	SSAGenBlock func(s *SSAGenState, b, next *ssa.Block)
}

var thearch Arch

var (
	BoundsCheckFunc [ssa.BoundsKindCount]*obj.LSym
	ExtendCheckFunc [ssa.BoundsKindCount]*obj.LSym
)

// GCWriteBarrierReg maps from registers to gcWriteBarrier implementation LSyms.
var GCWriteBarrierReg map[int16]*obj.LSym
