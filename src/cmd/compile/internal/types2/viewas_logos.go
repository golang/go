// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M7-S5 checking for comptime.ViewAs (SPEC Sec 2.5, T-205).
//
// ViewAs reinterprets bytes as a POD struct. The virtual comptime
// package is provided by comptime_logos.go in Logos mode, so the types2
// tests, the eject oracle, and the compiler all resolve
// `import "comptime"` the same way; it declares ViewAs through
// logosEnsureComptimeViewAs below.

const logosViewAsNotPOD = "[comptime::viewas_not_pod]"

// logosEnsureComptimeViewAs inserts the ViewAs declaration unless the
// package already has one.
func logosEnsureComptimeViewAs(pkg *Package) {
	if pkg.Scope().Lookup("ViewAs") != nil {
		return
	}
	pkg.Scope().Insert(logosMakeComptimeViewAs(pkg))
}

// logosMakeComptimeViewAs builds `func ViewAs[T any](b const []byte) const *T`.
func logosMakeComptimeViewAs(pkg *Package) *Func {
	tname := NewTypeName(nopos, pkg, "T", nil)
	tparam := NewTypeParam(tname, Universe.Lookup("any").Type())
	param := NewParam(nopos, pkg, "b", &ConstView{Elem: NewSlice(Typ[Byte])})
	result := NewParam(nopos, pkg, "", &ConstView{Elem: NewPointer(tparam)})
	sig := NewSignatureType(nil, nil, []*TypeParam{tparam}, NewTuple(param), NewTuple(result), false)
	return NewFunc(nopos, pkg, "ViewAs", sig)
}

// logosViewAsInstCheck validates the type argument of every
// comptime.ViewAs instantiation, called or not: T must be a struct
// with no padding and no reference-carrying fields, else
// [comptime::viewas_not_pod] naming the offset. A bare
// `f := comptime.ViewAs[Bad]` validates here; the call form needs no
// second check.
func (check *Checker) logosViewAsInstCheck(expr syntax.Expr, targs []Type) {
	if !check.conf.Logos || len(targs) != 1 {
		return
	}
	fn, ok := check.Info.Uses[instantiatedIdent(expr)].(*Func)
	if !ok || fn.Name() != "ViewAs" {
		return
	}
	if pkg := fn.Pkg(); pkg == nil || pkg.Path() != "comptime" {
		return
	}
	t := targs[0]
	if t == nil || t == Typ[Invalid] {
		return
	}
	at := expr
	if idx, ok := expr.(*syntax.IndexExpr); ok {
		at = idx.Index
	}
	name := TypeString(t, nil)
	if named, ok := t.(*Named); ok {
		name = named.Obj().Name()
	}
	if _, ok := Unalias(t).Underlying().(*Struct); !ok {
		check.errorf(at, InvalidTypeArg, "%s %s is not a struct", logosViewAsNotPOD, name)
		return
	}
	if off := logosViewasScan(check, t); off != nil {
		if off.kind == "padding" {
			check.errorf(at, InvalidTypeArg, "%s %s has padding at offset %d", logosViewAsNotPOD, name, off.offset)
			return
		}
		check.errorf(at, InvalidTypeArg, "%s %s has %s field at offset %d", logosViewAsNotPOD, name, off.kind, off.offset)
	}
}

// logosViewasOffense describes one POD violation and its byte offset.
type logosViewasOffense struct {
	kind   string
	offset int64
}

// logosViewasScan scans T for the first POD violation: padding, bool,
// pointer, interface, slice, map, string, channel, or func fields, at
// absolute struct offsets.
func logosViewasScan(check *Checker, T Type) *logosViewasOffense {
	st, _ := Unalias(T).Underlying().(*Struct)
	offsets := check.conf.offsetsof(st)
	size := check.conf.sizeof(st)
	prevEnd := int64(0)
	for i := range st.NumFields() {
		f := st.Field(i)
		if off := offsets[i]; off > prevEnd {
			return &logosViewasOffense{"padding", prevEnd}
		} else if o := logosViewasField(check, f.Type(), off); o != nil {
			return o
		} else {
			prevEnd = off + check.conf.sizeof(f.Type())
		}
	}
	if size > prevEnd {
		return &logosViewasOffense{"padding", prevEnd}
	}
	return nil
}

// logosViewasField checks one field at absolute offset off, recursing
// into structs and arrays.
func logosViewasField(check *Checker, T Type, off int64) *logosViewasOffense {
	if named, ok := Unalias(T).(*Named); ok {
		T = named.Underlying()
	}
	switch t := Unalias(T).(type) {
	case *Struct:
		offsets := check.conf.offsetsof(t)
		size := check.conf.sizeof(t)
		prevEnd := off
		for i := range t.NumFields() {
			f := t.Field(i)
			foff := off + offsets[i]
			if foff > prevEnd {
				return &logosViewasOffense{"padding", prevEnd}
			} else if o := logosViewasField(check, f.Type(), foff); o != nil {
				return o
			} else {
				prevEnd = foff + check.conf.sizeof(f.Type())
			}
		}
		if off+size > prevEnd {
			return &logosViewasOffense{"padding", prevEnd}
		}
		return nil
	case *Array:
		return logosViewasField(check, t.Elem(), off)
	case *Basic:
		switch t.Kind() {
		case Bool:
			return &logosViewasOffense{"bool", off}
		case String:
			return &logosViewasOffense{"string", off}
		case UnsafePointer:
			return &logosViewasOffense{"pointer", off}
		}
		return nil
	case *Pointer:
		return &logosViewasOffense{"pointer", off}
	case *Interface:
		return &logosViewasOffense{"interface", off}
	case *Slice:
		return &logosViewasOffense{"slice", off}
	case *Map:
		return &logosViewasOffense{"map", off}
	case *Chan:
		return &logosViewasOffense{"channel", off}
	case *Signature:
		return &logosViewasOffense{"func", off}
	default:
		return &logosViewasOffense{"pointer", off}
	}
}
