// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M3 non-nil pointer zero-value rules (SPEC Sec 2.2, T-202).
// Upstream files call in through hook lines only: decl.go, literals.go,
// builtins.go, assignments.go. Every rule applies only with Config.Logos.
// S5 (platform pointers) builds on logosIsNonNilPointer and
// logosMissingNonNilField; platform values are never compile-time nil,
// so logosNilToNonNil stays valid under S5.

const (
	logosNilToNonNil  = "[correctness::nil_to_nonnil]"
	logosNonNilUninit = "[correctness::nonnil_uninitialized]"
)

// logosIsNonNilPointer reports whether t is a non-nil pointer type in
// .lgo code: a locally-spelled *Pointer. Type parameters are exempt
// (`var t T` is allowed); imported named types are platform pointers
// (T-202) and stay opaque; option types (M3-S2) are never a bare
// *Pointer.
func (check *Checker) logosIsNonNilPointer(t Type) bool {
	if t == nil || isTypeParam(t) || check.logosOpaqueType(t) {
		return false
	}
	_, ok := t.Underlying().(*Pointer)
	return ok
}

// logosOpaqueType reports whether t is a named type declared outside the
// current package. Its interior (e.g. Go pointer fields) is platform
// territory: nilable, with no zero-value rules.
func (check *Checker) logosOpaqueType(t Type) bool {
	n, _ := t.(*Named)
	return n != nil && n.obj != nil && n.obj.pkg != check.pkg
}

// logosZeroNeedsInit reports whether the zero value of t is illegal in
// .lgo code: t is or transitively holds a non-nil pointer.
func (check *Checker) logosZeroNeedsInit(t Type, seen map[Type]bool) bool {
	if t == nil || isTypeParam(t) || check.logosOpaqueType(t) {
		return false
	}
	if check.logosIsNonNilPointer(t) {
		return true
	}
	u := t.Underlying()
	if u == nil || isTypeParam(u) {
		return false
	}
	switch u := u.(type) {
	case *Array:
		return check.logosZeroNeedsInit(u.elem, seen)
	case *Struct:
		return check.logosMissingNonNilField(u, seen) != nil
	}
	return false
}

// logosMissingNonNilField returns the first direct field of st that
// transitively needs initialization, or nil. Only direct fields are
// reported, so the diagnostic names the field and its type.
func (check *Checker) logosMissingNonNilField(st *Struct, seen map[Type]bool) *Var {
	if seen == nil {
		seen = make(map[Type]bool)
	}
	for _, fld := range st.fields {
		ft := fld.typ
		if !isValid(ft) || isTypeParam(ft) || check.logosOpaqueType(ft) {
			continue
		}
		if check.logosIsNonNilPointer(ft) {
			return fld
		}
		if su, _ := ft.Underlying().(*Struct); su != nil {
			if seen[ft] {
				continue
			}
			seen[ft] = true
			if check.logosMissingNonNilField(su, seen) != nil {
				return fld
			}
		} else if au, _ := ft.Underlying().(*Array); au != nil {
			if check.logosZeroNeedsInit(au.elem, seen) {
				return fld
			}
		}
	}
	return nil
}

// logosZeroDecl rejects an uninitialized variable declaration whose type
// has no legal zero value. Errors report at the declared name.
func (check *Checker) logosZeroDecl(obj *Var) {
	if !check.conf.Logos || obj.name == "_" || !isValid(obj.typ) {
		return
	}
	if isTypeParam(obj.typ) {
		return // `var t T` for a type parameter is allowed.
	}
	if check.logosIsNonNilPointer(obj.typ) {
		check.errorf(obj.pos, BadDecl, "%s var %s %s needs a value; non-nil pointers have no zero value", logosNonNilUninit, obj.name, obj.typ)
		return
	}
	var missing *Var
	if u, _ := obj.typ.Underlying().(*Struct); u != nil {
		missing = check.logosMissingNonNilField(u, nil)
	} else if u, _ := obj.typ.Underlying().(*Array); u != nil {
		if check.logosZeroNeedsInit(u.elem, nil) {
			check.errorf(obj.pos, BadDecl, "%s var %s %s needs initializers; non-nil pointer elements have no zero value", logosNonNilUninit, obj.name, obj.typ)
			return
		}
	}
	if missing != nil {
		check.errorf(obj.pos, BadDecl, "%s field %s.%s (type %s) needs a value", logosNonNilUninit, obj.typ, missing.name, missing.typ)
	}
}

// logosZeroLiteral rejects a struct composite literal that leaves a
// non-nil pointer field empty. Positional forms are left to the upstream
// arity check. Errors report at the literal.
func (check *Checker) logosZeroLiteral(e *syntax.CompositeLit, typ, base Type) {
	if !check.conf.Logos || typ == nil || base == nil {
		return
	}
	u, _ := commonUnder(base, nil)
	st, _ := u.(*Struct)
	if st == nil || !isValid(base) {
		return
	}
	if len(e.ElemList) > 0 {
		if _, ok := e.ElemList[0].(*syntax.KeyValueExpr); !ok {
			return
		}
	}
	provided := make(map[string]bool, len(e.ElemList))
	for _, e := range e.ElemList {
		if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
			if key, _ := kv.Key.(*syntax.Name); key != nil {
				provided[key.Value] = true
			}
		}
	}
	seen := make(map[Type]bool)
	for _, fld := range st.fields {
		if !provided[fld.name] && isValid(fld.typ) && !isTypeParam(fld.typ) && check.logosZeroNeedsInit(fld.typ, seen) {
			check.errorf(e, InvalidLit, "%s field %s.%s (type %s) needs a value", logosNonNilUninit, base, fld.name, fld.typ)
			return
		}
	}
}

// logosZeroMakeElem reports whether T is a slice whose elements have no
// legal zero value: the shared gate for the make hooks below.
func (check *Checker) logosZeroMakeElem(T Type) bool {
	if !check.conf.Logos || !isValid(T) {
		return false
	}
	u, _ := T.Underlying().(*Slice)
	return u != nil && check.logosZeroNeedsInit(u.elem, nil)
}

// logosZeroMake rejects `make([]*T, n)` or `make([]*T, c)` when the
// length exceeds 0: constant length reports the bound, non-constant
// length reports unconditionally. `make([]*T, 0, n)` is allowed.
// Errors report at the length argument.
func (check *Checker) logosZeroMake(call *syntax.CallExpr, T Type) {
	if !check.logosZeroMakeElem(T) || len(call.ArgList) < 2 {
		return
	}
	_, length := check.index(call.ArgList[1], -1)
	if length > 0 {
		check.errorf(call.ArgList[1], InvalidMake, "%s make of %s with length %d would leave non-nil elements without values; use length 0", logosNonNilUninit, T, length)
		return
	}
	if length == 0 {
		return
	}
	check.errorf(call.ArgList[1], InvalidMake, "%s make of %s with non-constant length would leave non-nil elements without values; use length 0", logosNonNilUninit, T)
}

// logosZeroNew rejects `new(T)` when T has no legal zero value. Only the
// type form is hooked; `new(expr)` initializes from its argument. Errors
// report at the type argument.
func (check *Checker) logosZeroNew(arg syntax.Expr, T Type) {
	if !check.conf.Logos || !isValid(T) || isTypeParam(T) {
		return
	}
	if check.logosIsNonNilPointer(T) {
		check.errorf(arg, BadDecl, "%s new(%s) would leave a non-nil pointer without a value", logosNonNilUninit, T)
		return
	}
	if st, _ := T.Underlying().(*Struct); st != nil {
		if missing := check.logosMissingNonNilField(st, nil); missing != nil {
			check.errorf(arg, BadDecl, "%s field %s.%s (type %s) needs a value", logosNonNilUninit, T, missing.name, missing.typ)
		}
	} else if au, _ := T.Underlying().(*Array); au != nil {
		if check.logosZeroNeedsInit(au.elem, nil) {
			check.errorf(arg, BadDecl, "%s new(%s) would leave non-nil elements without values", logosNonNilUninit, T)
		}
	}
}

// logosNilToNonNil rejects storing the nil literal into a declared
// non-nil location. Only compile-time nils report; typed values stay
// silent for the S5 run-time check. It returns true when it reported.
func (check *Checker) logosNilToNonNil(x *operand, T Type) bool {
	if !check.conf.Logos || !x.isNil() || !isValid(T) {
		return false
	}
	if !check.logosIsNonNilPointer(T) {
		return false
	}
	check.errorf(x, IncompatibleAssign, "%s cannot assign nil to non-nil %s", logosNilToNonNil, T)
	x.invalidate()
	return true
}
