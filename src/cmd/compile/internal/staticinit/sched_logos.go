// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package staticinit

// M5-S3 read-only placement backend (SPEC Sec 2.1, T-201): registered
// package vars are marked readonly while scheduled so ggloblnod emits
// obj.RODATA; slice backing temps inherit the mark; a registered var
// that still needs dynamic code stops the compile, because the lowering
// must never produce one.

import (
	"strconv"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/logos"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
)

// logosBackingArray returns the array backing a slice literal of the
// initializer for l at offset loff. Under a read-only parent the array is
// a named, global, read-only symbol (<parent>·logos_backing<off>), so the
// linked binary lists it with type R. Otherwise it is the usual static
// temp, which is file-local and cannot be a global read-only symbol.
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
	nOut := len(s.Out)
	ok := s.tryStaticInit(n)
	if target, name := logosRodataTarget(n); target && (!ok || len(s.Out) != nOut) {
		base.FatalfAt(n.Pos(), "logos: const composite %s needs dynamic initialization", name)
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
