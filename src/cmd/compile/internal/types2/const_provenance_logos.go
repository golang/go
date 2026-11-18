// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

type ReadonlyDescendants struct {
	Elem Type
}

func (t *ReadonlyDescendants) Underlying() Type { return t }

func (t *ReadonlyDescendants) String() string { return TypeString(t, nil) }

func logosAsReadonlyDescendants(T Type) *ReadonlyDescendants {
	t, _ := Unalias(T).(*ReadonlyDescendants)
	return t
}

func logosWithReadonlyDescendants(T Type) Type {
	if T == nil || logosAsConstView(T) != nil || logosAsReadonlyDescendants(T) != nil || logosReferenceFree(T) {
		return T
	}
	return &ReadonlyDescendants{Elem: T}
}

func logosReadonlyBase(T Type) (base Type, descendants bool, readonly bool) {
	if t := logosAsConstView(T); t != nil {
		return t.Elem, false, true
	}
	if t := logosAsReadonlyDescendants(T); t != nil {
		return t.Elem, true, false
	}
	return T, false, false
}

func logosElementsAreCopies(container Type) bool {
	c := logosAsConstView(container)
	return c != nil && c.ConstMap
}

func logosReadonlyProject(container, projected Type, throughReference bool) Type {
	if logosElementsAreCopies(container) {
		return projected
	}
	_, descendants, readonly := logosReadonlyBase(container)
	if readonly || descendants && throughReference {
		if logosAsConstView(projected) != nil {
			return projected
		}
		if t := logosAsReadonlyDescendants(projected); t != nil {
			projected = t.Elem
		}
		return &ConstView{Elem: projected}
	}
	if descendants {
		return logosWithReadonlyDescendants(projected)
	}
	return projected
}

func (check *Checker) logosReadonlyIndexBase(x *operand) Type {
	readonlyContainer := x.typ()
	if check.conf.Logos {
		x.typ_, _, _ = logosReadonlyBase(x.typ())
		if p, ok := x.typ().Underlying().(*Pointer); ok {
			if b, d, _ := logosReadonlyBase(p.base); d {
				readonlyContainer = p.base
				x.typ_ = &Pointer{base: b}
			}
		}
	}
	check.logosConstUnwrapBase(x)
	return readonlyContainer
}

func (check *Checker) logosReadonlyMapIndex(x *operand, readonlyContainer Type) {
	if check.conf.Logos {
		x.typ_ = logosReadonlyProject(readonlyContainer, x.typ(), true)
	}
	check.logosMapIndexOption(x)
}

func (check *Checker) logosReadonlyIndexResult(x *operand, baseExpr syntax.Expr, readonlyContainer Type) {
	check.logosConstWrapBase(x, baseExpr)
	if check.conf.Logos {
		base, _, _ := logosReadonlyBase(readonlyContainer)
		_, inline := base.Underlying().(*Array)
		x.typ_ = logosReadonlyProject(readonlyContainer, x.typ(), !inline)
	}
}

func (check *Checker) logosReadonlySliceArray(x *operand, e *syntax.SliceExpr) bool {
	if check.logosConstSliceArray(e) {
		return true
	}
	if x.mode() != variable {
		check.errorf(x, NonSliceableOperand, "cannot slice unaddressable value %s", x)
		return true
	}
	return false
}

func (check *Checker) logosReadonlySliceResult(x *operand, e *syntax.SliceExpr, readonlyContainer Type) {
	check.logosConstSliceBound(x, e)
	check.logosConstWrapBase(x, e.X)
	if check.conf.Logos {
		base, descendants, readonly := logosReadonlyBase(readonlyContainer)
		if _, inline := base.Underlying().(*Array); inline && descendants && !readonly {
			if s, ok := x.typ().Underlying().(*Slice); ok {
				x.typ_ = &Slice{elem: logosWithReadonlyDescendants(s.elem)}
			}
		} else {
			x.typ_ = logosReadonlyProject(readonlyContainer, x.typ(), true)
		}
	}
}

func logosIndexStoreProtected(container Type) bool {
	base, descendants, readonly := logosReadonlyBase(container)
	if !descendants {
		return readonly
	}
	_, inline := base.Underlying().(*Array)
	return !inline
}

func (check *Checker) logosFieldStoreProtected(container Type, name string) bool {
	base, descendants, readonly := logosReadonlyBase(container)
	if !descendants {
		return readonly
	}
	_, _, indirect := lookupFieldOrMethod(base, true, check.pkg, name, false)
	return indirect
}

func logosDerefStoreProtected(container Type) bool {
	_, descendants, readonly := logosReadonlyBase(container)
	return readonly || descendants
}

func logosContainsReadonlyDescendants(T Type) bool {
	found := false
	logosWalkType(T, func(T Type) {
		if _, ok := Unalias(T).(*ReadonlyDescendants); ok {
			found = true
		}
	}, make(map[Type]bool))
	return found
}

func logosAnyContainsReadonlyDescendants(V, T Type) bool {
	return logosContainsReadonlyDescendants(V) || logosContainsReadonlyDescendants(T)
}

func logosReadonlyDescendantsAssignable(x *operand, T Type) (bool, Code) {
	if x.isNil() && hasNil(T) {
		return true, 0
	}
	return false, IncompatibleAssign
}

func logosAnyIsReadonlyDescendants(x, y Type) bool {
	return logosAsReadonlyDescendants(x) != nil || logosAsReadonlyDescendants(y) != nil
}

func (u *unifier) logosNifyReadonlyDescendants(x, y Type, mode unifyMode, p *ifacePair) bool {
	rx, ry := logosAsReadonlyDescendants(x), logosAsReadonlyDescendants(y)
	return rx != nil && ry != nil && u.nify(rx.Elem, ry.Elem, mode|exact, p)
}

func logosUnwrapReadonlyDescendants(T *Type) {
	if r, ok := (*T).Underlying().(*ReadonlyDescendants); ok {
		*T = r.Elem
	}
}

func (check *Checker) logosRejectReadonlyInference(pos syntax.Pos, args []*operand, targs []Type, err *error_) bool {
	for _, arg := range args {
		if arg.isValid() && logosContainsReadonlyDescendants(arg.typ()) {
			err.addf(arg, "cannot use %s in generic inference: readonly descendants would be discarded", arg)
			return true
		}
	}
	for _, targ := range targs {
		if targ != nil && logosContainsReadonlyDescendants(targ) {
			err.addf(atPos(pos), "cannot use type %s as generic type argument: readonly descendants would be discarded", targ)
			return true
		}
	}
	return false
}

func (check *Checker) logosRejectReadonlyBuiltinArgs(x *operand, args []*operand, id builtinId) bool {
	if !check.conf.Logos {
		return false
	}
	for _, a := range args {
		if !check.logosReadonlyBuiltinArg(a, id) {
			x.invalidate()
			return true
		}
	}
	if len(args) > 0 {
		*x = *args[0]
	}
	return false
}

func logosBuiltinObservesOnly(id builtinId) bool {
	switch id {
	case _Len, _Cap, _Sizeof, _Alignof, _Offsetof, _Print, _Println, _Min, _Max, _Complex, _Real, _Imag:
		return true
	}
	return false
}

func logosBuiltinWritesThroughFirst(id builtinId) bool {
	switch id {
	case _Append, _Copy, _Delete, _Clear:
		return true
	}
	return false
}

func logosObservedType(T Type) Type {
	base, _, _ := logosReadonlyBase(T)
	if p, ok := base.Underlying().(*Pointer); ok {
		if b, d, _ := logosReadonlyBase(p.base); d {
			return &Pointer{base: b}
		}
	}
	return base
}

func (check *Checker) logosObservedOperand(x *operand) {
	if check.conf.Logos {
		x.typ_ = logosObservedType(x.typ())
	}
}

func (check *Checker) logosComparisonReadonly(x, y *operand, op syntax.Operator) {
	if !check.conf.Logos || op != syntax.Eql && op != syntax.Neq {
		return
	}
	for _, o := range []*operand{x, y} {
		if base, _, _ := logosReadonlyBase(o.typ()); base != o.typ() {
			o.typ_ = base
		}
	}
}

func (check *Checker) logosReadonlyBuiltinArg(a *operand, id builtinId) bool {
	if !logosContainsReadonlyDescendants(a.typ()) {
		return true
	}
	if logosBuiltinObservesOnly(id) {
		a.typ_ = logosObservedType(a.typ())
		return true
	}
	base, _, _ := logosReadonlyBase(a.typ())
	if logosBuiltinWritesThroughFirst(id) && base != a.typ() {
		a.typ_ = &ConstView{Elem: base}
		return true
	}
	check.errorf(a, IncompatibleAssign, "%s cannot pass readonly descendants to builtin %s", logosConstEscape, predeclaredFuncs[id].name)
	return false
}

func (check *Checker) logosReadonlyBoundary(x *operand, T Type, context string) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	src := logosContainsReadonlyDescendants(x.typ())
	if T == nil {
		return false
	}
	dst := logosContainsReadonlyDescendants(T)
	if !src && !dst {
		return false
	}
	if x.isNil() && hasNil(T) {
		x.typ_ = T
		return true
	}
	if Identical(x.typ(), T) {
		return true
	}
	base, descendants, readonly := logosReadonlyBase(T)
	if descendants || readonly {
		S, _, _ := logosReadonlyBase(x.typ())
		if p, ok := S.Underlying().(*Pointer); ok && readonly {
			if b, d, _ := logosReadonlyBase(p.base); d {
				S = &Pointer{base: b}
			}
		}
		copy := *x
		copy.typ_ = S
		if ok, _ := copy.assignableTo(check, base, nil); ok {
			x.typ_ = T
			return true
		}
	}
	check.errorf(x, IncompatibleAssign, "%s cannot erase readonly descendants of %s in %s", logosConstEscape, ExprString(x.expr), context)
	x.invalidate()
	return true
}

func (check *Checker) logosConstTypeExpr(x *operand, e *syntax.ConstType) exprKind {
	x.mode_ = typexpr
	x.typ_ = check.logosConstType(e, nil)
	x.expr = e
	return expression
}

func (check *Checker) logosZeroCopyConvert(x *operand, T Type) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	cdst := logosAsConstView(T)
	if cdst == nil {
		return false
	}
	if csrc := logosAsConstView(x.typ()); csrc != nil {
		if !Identical(csrc.Elem, cdst.Elem) {
			return false
		}
		x.typ_ = T
		return true
	}
	check.conversion(x, cdst.Elem)
	if !x.isValid() {
		return true
	}
	if x.mode() == constant_ {
		x.mode_ = value
		x.val = nil
	}
	x.typ_ = T
	return true
}
