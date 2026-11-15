// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package staticinit

import (
	"strconv"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/staticdata"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

func logosBackingArray(l *ir.Name, loff int64, t *types.Type) *ir.Name {
	if !l.Readonly() {
		return StaticName(t)
	}
	sym := types.LocalPkg.Lookup(l.Sym().Name + "·logos_backing" + strconv.FormatInt(loff, 10))
	if sym.Def != nil {
		base.FatalfAt(base.Pos, "logos: backing %v already defined", sym)
	}
	n := ir.NewNameAt(base.Pos, sym, t)
	sym.Def = n
	n.Class = ir.PEXTERN
	typecheck.Target.Externs = append(typecheck.Target.Externs, n)
	n.Linksym().Align = int16(t.Alignment())
	n.MarkReadonly()
	return n
}

func logosStaticInit(s *Schedule, n ir.Node) bool {
	markLogosRodata(n)
	if logosStaticSlice(s, n) {
		return true
	}
	nOut := len(s.Out)
	ok := s.tryStaticInit(n)
	if target, name := logosRodataTarget(n); target && (!ok || len(s.Out) != nOut) {
		if !logos.IsWrittenRodata(types.LocalPkg.Path, name) {
			base.FatalfAt(n.Pos(), "logos: const composite %s needs dynamic initialization", name)
		}
		base.ErrorfAt(n.Pos(), 0, "[correctness::rodata_dynamic_init] variable %s marked //logos:rodata needs dynamic initialization", name)
	}
	return ok
}

func logosRodataTarget(n ir.Node) (bool, string) {
	switch n.Op() {
	case ir.OAS:
		as := n.(*ir.AssignStmt)
		nm, ok := as.X.(*ir.Name)
		if !ok || nm.Class != ir.PEXTERN {
			return false, ""
		}
		return logosRodataName(nm)
	case ir.OCALLFUNC:
		call, ok := n.(*ir.CallExpr)
		if !ok {
			return false, ""
		}
		fn, ok := call.Fun.(*ir.Name)
		if !ok || fn.Func == nil {
			return false, ""
		}
		v := MapInitToVar[fn.Func]
		if v == nil {
			return false, ""
		}
		return logosRodataName(v)
	}
	return false, ""
}

func logosRodataName(nm *ir.Name) (bool, string) {
	name := nm.Sym().Name
	if !logos.IsRodata(types.LocalPkg.Path, name) {
		return false, ""
	}
	return true, name
}

func markLogosRodata(n ir.Node) {
	if ok, _ := logosRodataTarget(n); !ok {
		return
	}
	as, ok := n.(*ir.AssignStmt)
	if !ok {
		return
	}
	if nm, ok := as.X.(*ir.Name); ok {
		nm.MarkReadonly()
	}
}

func logosMarkTemp(a, l *ir.Name) {
	if l != nil && l.Readonly() {
		a.MarkReadonly()
	}
}

func logosStaticSlice(s *Schedule, n ir.Node) bool {
	target, _ := logosRodataTarget(n)
	if !target {
		return false
	}
	as, ok := n.(*ir.AssignStmt)
	if !ok {
		return false
	}
	nm, ok := as.X.(*ir.Name)
	if !ok || nm.Class != ir.PEXTERN || !nm.Readonly() {
		return false
	}
	if base.Ctxt.IsFIPS() {
		return false
	}
	sl, ok := as.Y.(*ir.SliceExpr)
	if !ok || sl.Op() != ir.OSLICE3 {
		return false
	}
	lo, ok := staticSliceBound(sl.Low, 0)
	if !ok {
		return false
	}
	hi, ok := staticSliceBound(sl.High, -1)
	if !ok || hi < 0 {
		return false
	}
	mx, ok := staticSliceBound(sl.Max, -1)
	if !ok || mx < 0 {
		return false
	}
	back, ok := sl.X.(*ir.CompLitExpr)
	if !ok || back.Op() != ir.OSLICELIT {
		return false
	}
	ln, cp := hi-lo, mx-lo
	if lo < 0 || ln < 0 || cp < ln || mx > int64(back.Len) {
		return false
	}
	l, loff := nm, int64(0)
	ta := types.NewArray(back.Type().Elem(), back.Len)
	ta.SetNoalg(true)
	a := logosBackingArray(l, loff, ta)
	s.initplan(back)
	p := s.Plans[back]
	for i := range p.E {
		e := &p.E[i]
		if e.Expr.Op() != ir.OLITERAL && e.Expr.Op() != ir.ONIL {
			return false
		}
	}
	for i := range p.E {
		e := &p.E[i]
		staticdata.InitConst(a, e.Xoffset, e.Expr, int(e.Expr.Type().Size()))
	}
	s.Temps[back] = a
	sym := l.Linksym()
	sym.WriteAddr(base.Ctxt, loff, types.PtrSize, a.Linksym(), lo*int64(back.Type().Elem().Size()))
	sym.WriteInt(base.Ctxt, loff+types.SliceLenOffset, types.PtrSize, ln)
	sym.WriteInt(base.Ctxt, loff+types.SliceCapOffset, types.PtrSize, cp)
	return true
}

func staticSliceBound(e ir.Node, def int64) (int64, bool) {
	if e == nil {
		return def, true
	}
	if e.Op() != ir.OLITERAL {
		return 0, false
	}
	return ir.Int64Val(e), true
}
