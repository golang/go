// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package escape

import (
	"cmd/compile/internal/ir"
	"cmp"
	"strings"
)

const leakForeign = leakResult0 + numEscResults

func (l leaks) Foreign() int           { return l.get(leakForeign) }
func (l *leaks) AddForeign(derefs int) { l.add(leakForeign, derefs) }

func (l leaks) foreignPrefix() string {
	if l.Foreign() >= 0 {
		return "escf:"
	}
	return "esc:"
}

func (l *leaks) parseForeign(s string) {
	switch {
	case strings.HasPrefix(s, "escf:"):
		copy(l[:], s[5:])
	case strings.HasPrefix(s, "esc:"):
		copy(l[:leakForeign], s[4:])
	default:
		l.AddHeap(0)
	}
}

func foreignBacking(l *location) bool {
	if l.n == nil || l.n.Op() != ir.OMAKESLICE {
		return false
	}
	n := l.n.(*ir.MakeExpr)
	return !ir.IsSmallIntConst(cmp.Or(n.Cap, n.Len))
}

func (e *escape) compilerKeepAliveArgs(call ir.Node, recvArg ir.Node, args ir.Nodes) {
	wasCompilerKeepAlive := e.compilerKeepAlive
	e.compilerKeepAlive = true
	if recvArg != nil {
		e.expr(e.discardHole().note(call, "call parameter"), recvArg)
	}
	for _, arg := range args {
		e.expr(e.discardHole().note(call, "call parameter"), arg)
	}
	e.compilerKeepAlive = wasCompilerKeepAlive
}

func (e *escape) foreignConversion(k hole, n *ir.ConvExpr) hole {
	if !n.Type().IsUnsafePtr() && !n.X.Type().IsUnsafePtr() {
		return k
	}
	if e.compilerKeepAlive {
		return k
	}
	foreign := e.foreignLoc.asHole().note(n, "foreign conversion")
	if k.dst == &e.blankLoc {
		return foreign
	}
	return e.teeHole(k, foreign)
}

func (e *escape) foreignCgoFrame(call *ir.CallExpr) {
	if e.curfn.Pragma&ir.CgoUnsafeArgs == 0 || len(call.Args) != 2 {
		return
	}
	callee := ir.StaticCalleeName(call.Fun)
	if callee == nil || !foreignCgoTarget(callee) || !e.foreignFrameAddress(call.Args[1]) {
		return
	}
	for _, param := range e.curfn.Type().RecvParams() {
		if param.Nname != nil && param.Type.HasPointers() {
			e.flow(e.foreignLoc.asHole().note(call, "cgo argument frame"), e.oldLoc(param.Nname.(*ir.Name)))
		}
	}
}

func foreignCgoTarget(callee *ir.Name) bool {
	sym := callee.Sym()
	if sym == nil {
		return false
	}
	return sym.Linkname == "runtime.cgocall" || sym.Linkname == "runtime.logosCgocallFast"
}

func (e *escape) foreignFrameAddress(n ir.Node) bool {
	conv, ok := n.(*ir.ConvExpr)
	if !ok || !conv.Type().IsUintptr() || !conv.X.Type().IsUnsafePtr() {
		return false
	}
	conv, ok = conv.X.(*ir.ConvExpr)
	if !ok || !conv.Type().IsUnsafePtr() {
		return false
	}
	addr, ok := conv.X.(*ir.AddrExpr)
	if !ok {
		return false
	}
	name, ok := addr.X.(*ir.Name)
	return ok && name.Curfn == e.curfn && (name.Class == ir.PPARAM || name.Class == ir.PPARAMOUT)
}
