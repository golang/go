// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"
	"strings"

	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// logosTypeQueryBody folds a logos/rt type-query instance while its
// body is read. Implements alone requires !dict.shaped: shaping erases
// method sets, while kind and struct layout survive it.
func logosTypeQueryBody(r *reader, pos src.XPos) bool {
	fn := r.curfn
	if r.inlFunc != nil {
		fn = r.inlFunc
	}
	if fn == nil {
		return false
	}
	sym := fn.Sym()
	if sym == nil || sym.Pkg == nil || sym.Pkg.Path != "logos/rt" {
		return false
	}
	name := sym.Name
	if i := strings.IndexByte(name, '['); i >= 0 {
		name = name[:i]
	}
	dict := r.dict
	if dict == nil || len(dict.targs) == 0 {
		return false
	}
	switch name {
	case "TildeIs":
		if len(dict.targs) != 2 {
			return false
		}
		v, ok := logosTildeIs(dict.targs[0], dict.targs[1])
		if !ok {
			return false
		}
		logosBoolResult(r.curfn, pos, v)
		return true
	case "Implements":
		if dict.shaped || len(dict.targs) != 2 {
			return false
		}
		v, ok := logosImplements(dict.targs[0], dict.targs[1])
		if !ok {
			return false
		}
		logosBoolResult(r.curfn, pos, v)
		return true
	case "viewAsEnforcePOD":
		if len(dict.targs) != 1 {
			return false
		}
		if !logosViewAsPOD(dict.targs[0]) {
			return false
		}
		r.curfn.Body = []ir.Node{typecheck.Stmt(ir.NewBlockStmt(src.NoXPos, nil))}
		return true
	}
	return false
}

func logosConcreteImplements(pos src.XPos, fn ir.Node, targs []*types.Type) ir.Node {
	sym := fn.Sym()
	if sym == nil || sym.Pkg == nil || sym.Pkg.Path != "logos/rt" || !strings.HasPrefix(sym.Name, "Implements[") || len(targs) != 2 {
		return nil
	}
	for _, t := range targs {
		if t.HasShape() {
			return nil
		}
	}
	v, ok := logosImplements(targs[0], targs[1])
	if !ok {
		return nil
	}
	return ir.NewBasicLit(pos, types.Types[types.TBOOL], constant.MakeBool(v))
}

func logosBoolResult(fn *ir.Func, pos src.XPos, v bool) {
	lit := ir.NewBasicLit(pos, types.Types[types.TBOOL], constant.MakeBool(v))
	fn.Body = []ir.Node{typecheck.Stmt(ir.NewReturnStmt(pos, []ir.Node{lit}))}
}

func logosTildeIs(t, b *types.Type) (bool, bool) {
	kt, ok := logosKind(t)
	if !ok {
		return false, false
	}
	kb, ok := logosKind(b)
	if !ok {
		return false, false
	}
	return kt == kb, true
}

func logosKind(t *types.Type) (types.Kind, bool) {
	if t == nil {
		return 0, false
	}
	u := t.Underlying()
	if u == nil {
		return 0, false
	}
	switch u.Kind() {
	case types.TBOOL,
		types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64,
		types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR,
		types.TFLOAT32, types.TFLOAT64,
		types.TCOMPLEX64, types.TCOMPLEX128,
		types.TSTRING, types.TUNSAFEPTR,
		types.TPTR, types.TFUNC, types.TARRAY, types.TSLICE,
		types.TSTRUCT, types.TCHAN, types.TMAP, types.TINTER:
		return u.Kind(), true
	}
	return 0, false
}

func logosImplements(t, iface *types.Type) (bool, bool) {
	if t == nil || t.Kind() == types.TFORW || iface == nil || !iface.IsInterface() {
		return false, false
	}
	if t.Underlying() == nil || iface.Underlying() == nil {
		return false, false
	}
	return typecheck.Implements(t, iface), true
}

func logosViewAsPOD(t *types.Type) bool {
	u := t.Underlying()
	if u == nil || u.Kind() != types.TSTRUCT {
		return false
	}
	_, kind := logosViewAsScan(t, 0)
	return kind == ""
}

func logosViewAsScan(t *types.Type, base int64) (int64, string) {
	u := t.Underlying()
	if u == nil {
		return base, "unknown"
	}
	switch u.Kind() {
	case types.TSTRUCT:
		prev := base
		for i := range u.NumFields() {
			abs := base + u.FieldOff(i)
			if abs > prev {
				return prev, "padding"
			}
			ft := u.FieldType(i)
			if off, kind := logosViewAsScan(ft, abs); kind != "" {
				return off, kind
			}
			prev = abs + ft.Size()
		}
		if base+u.Size() > prev {
			return prev, "padding"
		}
		return 0, ""
	case types.TARRAY:
		return logosViewAsScan(u.Elem(), base)
	case types.TBOOL:
		return base, "bool"
	case types.TSTRING:
		return base, "string"
	case types.TSLICE:
		return base, "slice"
	case types.TMAP:
		return base, "map"
	case types.TINTER, types.TPTR, types.TUNSAFEPTR:
		return base, "pointer"
	case types.TCHAN:
		return base, "channel"
	case types.TFUNC:
		return base, "func"
	case types.TINT, types.TINT8, types.TINT16, types.TINT32, types.TINT64,
		types.TUINT, types.TUINT8, types.TUINT16, types.TUINT32, types.TUINT64, types.TUINTPTR,
		types.TFLOAT32, types.TFLOAT64,
		types.TCOMPLEX64, types.TCOMPLEX128:
		return 0, ""
	}
	return base, "unknown"
}
