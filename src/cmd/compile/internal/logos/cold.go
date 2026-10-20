// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// emitColdExit reports whether ? error blocks gain a Cold mark. Only the
// standard and aggressive synthetic PGO levels emit, and never into eject
// output: ejected code builds with an upstream toolchain.
func emitColdExit() bool {
	if base.LogosEject != "" {
		return false
	}
	switch base.Debug.LogosSynthPGO {
	case "standard", "aggressive":
		return true
	}
	return false
}

// coldStmtOf builds _logos_rt.Cold() as the first statement of a ? error
// block. The Cold intrinsic removes the call and marks the branch into
// the block unlikely.
func coldStmtOf(pos syntax.Pos) syntax.Stmt {
	return exprStmtOf(pos, callOf(pos, rtSel(pos, "Cold"), nil))
}

// coldTerminator reports whether s ends its block through a synthetic-PGO
// cold path: an os.Exit call at the standard level and above, or an
// explicit panic call at the aggressive level. Deferred and spawned calls
// keep running, so only expression statements qualify.
func (fc *funcLower) coldTerminator(s syntax.Stmt) bool {
	e, ok := s.(*syntax.ExprStmt)
	if !ok || !emitColdExit() {
		return false
	}
	call, ok := e.X.(*syntax.CallExpr)
	if !ok {
		return false
	}
	if sel, ok := call.Fun.(*syntax.SelectorExpr); ok {
		return isOsExitCall(fc.fl.pkg.info, sel)
	}
	if name, ok := call.Fun.(*syntax.Name); ok {
		return isPanicCall(fc.fl.pkg.info, name)
	}
	return false
}

// isOsExitCall reports a selector call to os.Exit, resolved through the
// type info so shadowed names never match.
func isOsExitCall(info *types2.Info, sel *syntax.SelectorExpr) bool {
	if sel.Sel == nil || sel.Sel.Value != "Exit" {
		return false
	}
	name, ok := syntax.Unparen(sel.X).(*syntax.Name)
	if !ok {
		return false
	}
	pkg, ok := info.Uses[name].(*types2.PkgName)
	return ok && pkg.Imported().Path() == "os"
}

// isPanicCall reports a call to the panic builtin, which the aggressive
// level marks. Shadowed names never match.
func isPanicCall(info *types2.Info, name *syntax.Name) bool {
	if name.Value != "panic" {
		return false
	}
	if base.Debug.LogosSynthPGO != "aggressive" {
		return false
	}
	_, ok := info.Uses[name].(*types2.Builtin)
	return ok
}
