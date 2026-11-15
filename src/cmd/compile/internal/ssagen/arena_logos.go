// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/escape"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
	"go/constant"
	"internal/buildcfg"
)

const LogosArenaBumpMaxBytes = 512

func logosArenaBumpBuilder(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	if !buildcfg.Experiment.LogosArena || Arch.LinkArch.Family != sys.AMD64 || base.Flag.Race {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	var size int64
	if len(args) > 0 && (args[0].Op == ssa.OpConst64 || args[0].Op == ssa.OpConst32) {
		size = args[0].AuxInt
	} else if len(n.Args) > 0 && ir.IsConst(n.Args[0], constant.Int) {
		size = ir.Int64Val(n.Args[0])
	} else {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	if size <= 0 || size > LogosArenaBumpMaxBytes {
		return s.constNil(types.Types[types.TUNSAFEPTR])
	}
	rnd := (size + 7) &^ 7

	mem := s.mem()
	g := s.newValue1(ssa.OpGetG, s.f.Config.Types.BytePtr, mem)
	offset := int64(ssa.LogosArenaGOffset)
	chunkAddr := s.newValue1I(ssa.OpOffPtr, types.NewPtr(s.f.Config.Types.BytePtr), offset, g)
	chunk := s.newValue2(ssa.OpLoad, s.f.Config.Types.BytePtr, chunkAddr, mem)

	bumpTyp := types.NewTuple(types.Types[types.TUNSAFEPTR], types.TypeMem)
	bump := s.newValue2I(ssa.OpLogosArenaBump, bumpTyp, rnd, chunk, mem)
	ptr := s.newValue1(ssa.OpSelect0, types.Types[types.TUNSAFEPTR], bump)
	newMem := s.newValue1(ssa.OpSelect1, types.TypeMem, bump)
	s.vars[memVar] = newMem
	return ptr
}

func hasArenaBump(fn *ir.Func) bool {
	if fn == nil {
		return false
	}
	found := false
	ir.VisitList(fn.Body, func(n ir.Node) {
		if found {
			return
		}
		if call, ok := n.(*ir.CallExpr); ok {
			if name, ok := call.Fun.(*ir.Name); ok && name.Sym() != nil && name.Sym().Pkg == ir.Pkgs.Runtime {
				switch name.Sym().Name {
				case "logosArenaBump", "logosArenaAlloc", "logosArenaMakeSlice", "logosArenaAllocScan":
					found = true
				}
			}
		}
	})
	return found
}

func logosRegisterArena(add func(string, string, intrinsicBuilder, ...*sys.Arch)) {
	add("runtime", "logosArenaBump", logosArenaBumpBuilder, sys.ArchAMD64)
}

func (s *state) logosArenaFrameEnter() {
	if !buildcfg.Experiment.LogosArena || escape.LogosArenaFrameDeferred(s.curfn) || !hasArenaBump(s.curfn) {
		return
	}
	saveFn := typecheck.LookupRuntimeFunc("logosArenaMarkSave")
	res := s.rtcall(saveFn, true, []*types.Type{types.Types[types.TUINTPTR], types.Types[types.TUINTPTR], types.Types[types.TUINTPTR]})
	s.arenaMarkChunk = res[0]
	s.arenaMarkNoscan = res[1]
	s.arenaMarkScan = res[2]
	s.hasArenaSave = true
}

func (s *state) logosArenaFrameExit() {
	if !s.hasArenaSave {
		return
	}
	restoreFn := typecheck.LookupRuntimeFunc("logosArenaMarkRestore")
	s.rtcall(restoreFn, true, nil, s.arenaMarkChunk, s.arenaMarkNoscan, s.arenaMarkScan)
}

func logosArenaLoopEligible() bool {
	return buildcfg.Experiment.LogosArena && Arch.LinkArch.Family == sys.AMD64
}

func (s *state) logosArenaLoopSave(loop ir.Node) {
	if !logosArenaLoopEligible() || loop == nil {
		return
	}
	if s.arenaLoopMarks == nil {
		s.arenaLoopMarks = make(map[ir.Node][3]*ssa.Value)
	} else if _, ok := s.arenaLoopMarks[loop]; ok {
		return
	}
	saveFn := typecheck.LookupRuntimeFunc("logosArenaMarkSave")
	res := s.rtcall(saveFn, true, []*types.Type{types.Types[types.TUINTPTR], types.Types[types.TUINTPTR], types.Types[types.TUINTPTR]})
	s.arenaLoopMarks[loop] = [3]*ssa.Value{res[0], res[1], res[2]}
}

func (s *state) logosArenaLoopRestore(loop ir.Node) {
	if !logosArenaLoopEligible() || loop == nil {
		return
	}
	if s.arenaLoopMarks == nil {
		return
	}
	m, ok := s.arenaLoopMarks[loop]
	if !ok {
		return
	}
	restoreFn := typecheck.LookupRuntimeFunc("logosArenaMarkRestore")
	s.rtcall(restoreFn, true, nil, m[0], m[1], m[2])
}

func (s *state) logosArenaLoopRestoreDepth(depth int) {
	if depth < 0 || depth > len(s.arenaLoopStack) {
		return
	}
	for i := len(s.arenaLoopStack) - 1; i >= depth; i-- {
		s.logosArenaLoopRestore(s.arenaLoopStack[i].loop)
	}
}

func (s *state) logosArenaNew(n *ir.UnaryExpr) *ssa.Value {
	if x, ok := n.X.(*ir.DynamicType); ok && x.Op() == ir.ODYNAMICTYPE {
		return s.logosArenaNewObject(n.Type().Elem(), s.expr(x.RType))
	}
	return s.logosArenaNewObject(n.Type().Elem(), s.reflectType(n.Type().Elem()))
}

func (s *state) logosArenaNewObject(typ *types.Type, rtype *ssa.Value) *ssa.Value {
	one := s.constInt(types.Types[types.TINT64], 1)
	return s.rtcall(typecheck.LookupRuntimeFunc("logosArenaMakeSlice"), true,
		[]*types.Type{types.NewPtr(typ)}, rtype, one, one)[0]
}
