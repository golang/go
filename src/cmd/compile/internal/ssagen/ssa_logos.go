// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssagen

import (
	"cmd/internal/sys"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
)

// logosSynthTier reads the synthetic PGO level from -d=logossynthpgo:
// 0 disables marking, 1 marks os.Exit paths, 2 also marks explicit
// panics. The flag arrives through cmd/go -O tiers or an explicit
// -gcflags override, for Go and loGo/s packages alike.
func logosSynthTier() int {
	switch base.Debug.LogosSynthPGO {
	case "standard":
		return 1
	case "aggressive":
		return 2
	}
	return 0
}

// logosColdBuilder implements the logos/rt.Cold intrinsic: it emits no
// code and marks the branch into the current block unlikely.
func logosColdBuilder(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
	ssa.MarkColdBranch(s.curBlock)
	return nil
}

// logosRegisterCold registers the Cold intrinsic on every architecture.
func logosRegisterCold(add func(string, string, intrinsicBuilder, ...*sys.Arch)) {
	add("logos/rt", "Cold", logosColdBuilder, sys.Archs[:]...)
}

// logosSynthMark marks synthetic-PGO cold terminators: os.Exit calls at
// the standard level and above, explicit panics (walked to
// runtime.gopanic) at the aggressive level. It inserts no call and no
// import; the mark only guides branch layout.
func logosSynthMark(s *state, n *ir.CallExpr) {
	tier := logosSynthTier()
	if tier == 0 {
		return
	}
	if logosExitCall(n) {
		ssa.MarkColdBranch(s.curBlock)
		return
	}
	if tier >= 2 && logosPanicCall(n) {
		ssa.MarkColdBranch(s.curBlock)
	}
}

// logosExitCall reports a static call to os.Exit.
func logosExitCall(n *ir.CallExpr) bool {
	name, ok := n.Fun.(*ir.Name)
	if !ok || name.Sym() == nil || name.Sym().Pkg == nil {
		return false
	}
	return name.Sym().Pkg.Path == "os" && name.Sym().Name == "Exit"
}

// logosPanicCall reports a static call to runtime.gopanic, the walked
// form of an explicit panic.
func logosPanicCall(n *ir.CallExpr) bool {
	name, ok := n.Fun.(*ir.Name)
	if !ok || name.Sym() == nil || name.Sym().Pkg == nil {
		return false
	}
	return name.Sym().Pkg == ir.Pkgs.Runtime && name.Sym().Name == "gopanic"
}
