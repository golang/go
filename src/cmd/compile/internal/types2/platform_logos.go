// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
	"strings"
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
// current package or in one of its cgo-generated Go files (M20-S0: the
// C.struct_x types of _cgo_gotypes.go). Its interior (e.g. Go pointer
// fields) is platform territory: nilable, with no zero-value rules.
func (check *Checker) logosOpaqueType(t Type) bool {
	n, _ := t.(*Named)
	return n != nil && n.obj != nil && (n.obj.pkg != check.pkg || logosGoModePos(n.obj.pos))
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
	if !check.isLogosFile(obj) || obj.name == "_" || !isValid(obj.typ) {
		return
	}
	if isTypeParam(obj.typ) || check.logosOpaqueType(obj.typ) {
		return // `var t T` for a type parameter or an imported named type is allowed.
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
	if !check.isLogosFile(e) || typ == nil || base == nil {
		return
	}
	u, _ := commonUnder(base, nil)
	st, _ := u.(*Struct)
	if st == nil || !isValid(base) || check.logosOpaqueType(base) {
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
	// Reached only from logosZeroMake, which already gates on the call
	// position; the shape check below carries no position of its own.
	if !isValid(T) {
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
	if !check.isLogosFile(call) || !check.logosZeroMakeElem(T) || len(call.ArgList) < 2 {
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
	if !check.isLogosFile(arg) || !isValid(T) || isTypeParam(T) || check.logosOpaqueType(T) {
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
	if !check.isLogosFile(x) || !x.isNil() || !isValid(T) {
		return false
	}
	if !check.logosIsNonNilPointer(T) {
		return false
	}
	check.errorf(x, IncompatibleAssign, "%s cannot assign nil to non-nil %s", logosNilToNonNil, T)
	x.invalidate()
	return true
}

// ---- M3-S5 platform pointers (SPEC Sec 2.2, T-202) ----
//
// A pointer-typed value produced by a Go-mode declaration (the standard
// library, every other Go package, cgo) is a platform pointer:
// dereference and == nil/!= nil compile with no proof, and storing it
// into a declared non-nil *T location wraps the stored expression in
// logos/rt.MustNonNil (panics with file:line on nil). Provenance is
// operand derivation, not a type: Checker maps track Go-derived vars
// and synthetic tuple dummies, while calls, fields, globals, and
// elements resolve on demand. Info.LogosChecks records one entry per
// store site for the lowering (logos/platform.go).
//
// Hooks: assignments.go (assignment store check, initVar inherit,
// assignVar LHS scope), expr.go (multiExpr tuple dummies), call.go
// (Go-callee argument scope), option_logos.go (option to Go pointer).

// LogosCheck records one MustNonNil site: the stored expression (the
// key of Info.LogosChecks) held a platform pointer for declared
// non-nil Target. The lowering wraps the key in
// _logos_rt.MustNonNil(expr, &site).
type LogosCheck struct {
	Target Type
}

// logosIsCgoName reports whether name is a cgo-generated declaration
// (from _cgo_gotypes.go, living in the current package): such objects
// are platform declarations despite their local package.
func logosIsCgoName(name string) bool {
	for _, p := range cgoPrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

// logosIsGoObj reports whether obj is declared outside the current
// package (a Go-mode declaration), or is cgo-generated: named like a cgo
// declaration or declared in a cgo-generated Go file of the package.
// Universe objects (builtins, nil) are never Go-mode.
func (check *Checker) logosIsGoObj(obj Object) bool {
	if obj == nil {
		return false
	}
	if pkg := obj.Pkg(); pkg != nil && pkg != check.pkg {
		return true
	}
	return logosIsCgoName(obj.Name()) || logosGoModePos(obj.Pos())
}

// logosFuncOfExpr resolves e to the *Func it denotes, looking through
// parentheses and generic instantiations, or nil for func values,
// builtins, and anything else without a declared function object.
func (check *Checker) logosFuncOfExpr(e syntax.Expr) *Func {
	switch e := syntax.Unparen(e).(type) {
	case *syntax.Name:
		if obj := check.lookup(e.Value); obj != nil {
			if fn, ok := obj.(*Func); ok {
				return fn
			}
		}
		return nil
	case *syntax.IndexExpr:
		return check.logosFuncOfExpr(e.X)
	case *syntax.SelectorExpr:
		if xn, ok := syntax.Unparen(e.X).(*syntax.Name); ok {
			if obj := check.lookup(xn.Value); obj != nil {
				if pn, ok := obj.(*PkgName); ok && pn.imported != nil {
					if exp := pn.imported.scope.Lookup(e.Sel.Value); exp != nil {
						if fn, ok := exp.(*Func); ok {
							return fn
						}
					}
					return nil
				}
			}
		}
		if check.Info != nil {
			if sel := check.Selections[e]; sel != nil {
				if fn, ok := sel.Obj().(*Func); ok {
					return fn
				}
			}
		}
		return nil
	default:
		return nil
	}
}

// logosCalleeIsGo reports whether call targets a Go-mode declaration:
// a cgo call, a package-qualified call into another package, a method
// declared outside .lgo, or a func value held by a Go-derived variable.
// Unresolvable callees count as .lgo (checks apply), matching
// logosIsLgoCall's default.
func (check *Checker) logosCalleeIsGo(call *syntax.CallExpr) bool {
	if call == nil || !check.isLogosFile(call) {
		return false
	}
	if sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr); ok {
		if xn, ok := syntax.Unparen(sel.X).(*syntax.Name); ok {
			if obj := check.lookup(xn.Value); obj != nil {
				if pn, ok := obj.(*PkgName); ok {
					if pn.imported == nil {
						return false
					}
					return pn.imported.cgo || pn.imported != check.pkg
				}
			}
		}
	}
	if fn := check.logosFuncOfExpr(call.Fun); fn != nil {
		return check.logosIsGoObj(fn)
	}
	if name, ok := syntax.Unparen(call.Fun).(*syntax.Name); ok {
		if obj := check.lookup(name.Value); obj != nil {
			if v, ok := obj.(*Var); ok && check.logosDerivedVar[v] {
				return true
			}
		}
	}
	return false
}

// logosCalleeWasGeneric reports whether call targets a generic function
// (explicitly or implicitly instantiated). Type-parameter results
// instantiated with a non-nil pointer are platform pointers.
func (check *Checker) logosCalleeWasGeneric(call *syntax.CallExpr) bool {
	if call == nil || !check.isLogosFile(call) {
		return false
	}
	fn := check.logosFuncOfExpr(call.Fun)
	if fn == nil || fn.typ == nil {
		return false
	}
	sig, ok := fn.typ.(*Signature)
	return ok && sig.TypeParams().Len() > 0
}

// logosCallIsConversion reports whether e is a type conversion (its
// function position denotes a type name), as opposed to a call.
// Conversions propagate derivation from their single argument.
func (check *Checker) logosCallIsConversion(e *syntax.CallExpr) bool {
	fun := syntax.Unparen(e.Fun)
	if name, ok := fun.(*syntax.Name); ok {
		if obj := check.lookup(name.Value); obj != nil {
			if _, ok := obj.(*TypeName); ok {
				return true
			}
		}
		return false
	}
	if sel, ok := fun.(*syntax.SelectorExpr); ok {
		if xn, ok := syntax.Unparen(sel.X).(*syntax.Name); ok {
			if obj := check.lookup(xn.Value); obj != nil {
				if pn, ok := obj.(*PkgName); ok && pn.imported != nil {
					if exp := pn.imported.scope.Lookup(sel.Sel.Value); exp != nil {
						if _, ok := exp.(*TypeName); ok {
							return true
						}
					}
				}
			}
		}
	}
	return false
}

// logosMarkDerivedVar records v as holding a Go-derived value.
func (check *Checker) logosMarkDerivedVar(v *Var) {
	if v == nil || v.name == "_" {
		return
	}
	if check.logosDerivedVar == nil {
		check.logosDerivedVar = make(map[*Var]bool)
	}
	check.logosDerivedVar[v] = true
}

// logosMarkDerivedExpr records a synthetic tuple dummy as Go-derived.
func (check *Checker) logosMarkDerivedExpr(e syntax.Expr) {
	if e == nil {
		return
	}
	if check.logosDerivedExpr == nil {
		check.logosDerivedExpr = make(map[syntax.Expr]bool)
	}
	check.logosDerivedExpr[e] = true
}

// logosExprDerived reports whether e denotes a value produced (possibly
// transitively) by a Go-mode declaration: calls into Go, fields and
// globals of Go packages, elements of Go-derived containers, and
// conversions thereof. Inferred variables and tuple dummies consult
// the Checker maps; everything else resolves on demand.
func (check *Checker) logosExprDerived(e syntax.Expr) bool {
	if e == nil || !check.isLogosFile(e) {
		return false
	}
	if check.logosDerivedExpr[e] {
		return true
	}
	switch e := syntax.Unparen(e).(type) {
	case nil:
		return false
	case *syntax.Name:
		if e.Value == "_" {
			return false
		}
		obj := check.lookup(e.Value)
		v, ok := obj.(*Var)
		if !ok {
			return false
		}
		if check.logosDerivedVar[v] {
			return true
		}
		return v.pkg != nil && v.pkg != check.pkg
	case *syntax.SelectorExpr:
		if check.Info != nil {
			if sel := check.Selections[e]; sel != nil {
				if obj := sel.Obj(); obj != nil && check.logosIsGoObj(obj) {
					return true
				}
			}
		}
		if xn, ok := syntax.Unparen(e.X).(*syntax.Name); ok {
			if obj := check.lookup(xn.Value); obj != nil {
				if pn, ok := obj.(*PkgName); ok && pn.imported != nil {
					if exp := pn.imported.scope.Lookup(e.Sel.Value); exp != nil {
						return true
					}
					return false
				}
			}
		}
		return check.logosExprDerived(e.X)
	case *syntax.IndexExpr:
		return check.logosExprDerived(e.X)
	case *syntax.SliceExpr:
		return check.logosExprDerived(e.X)
	case *syntax.AssertExpr:
		return check.logosExprDerived(e.X)
	case *syntax.KeyValueExpr:
		return check.logosExprDerived(e.Value)
	case *syntax.CallExpr:
		if check.logosCalleeIsGo(e) {
			return true
		}
		if check.logosCallIsConversion(e) {
			if len(e.ArgList) == 1 {
				return check.logosExprDerived(e.ArgList[0])
			}
			return false
		}
		return false
	case *syntax.Operation:
		if e.Op == syntax.And && e.Y == nil {
			return false
		}
		if e.X != nil && check.logosExprDerived(e.X) {
			return true
		}
		return e.Y != nil && check.logosExprDerived(e.Y)
	case *syntax.CoalesceExpr:
		return check.logosExprDerived(e.X) || check.logosExprDerived(e.Y)
	case *syntax.OptSelectExpr:
		return check.logosExprDerived(e.X)
	default:
		return false
	}
}

// logosIsDerived reports whether the operand value is Go-derived.
func (check *Checker) logosIsDerived(x *operand) bool {
	if x == nil || !x.isValid() || x.expr == nil || !check.isLogosFile(x) {
		return false
	}
	if x.mode() == typexpr {
		return false
	}
	return check.logosExprDerived(x.expr)
}

// logosIsPlatform reports whether the operand is a platform pointer: a
// Go-derived (or generic-instantiated) value of non-nil pointer type.
// Such values dereference and compare with nil freely; stores into
// declared non-nil locations need the run-time check.
func (check *Checker) logosIsPlatform(x *operand) bool {
	if !check.logosIsDerived(x) {
		// Generic callees are local, so derivation misses them: a
		// type-parameter result instantiated with a non-nil pointer
		// is still a platform pointer.
		if x == nil || !x.isValid() || x.expr == nil || x.mode() == typexpr {
			return false
		}
		call, ok := syntax.Unparen(x.expr).(*syntax.CallExpr)
		if !ok || !check.logosCalleeWasGeneric(call) {
			return false
		}
		return check.logosIsNonNilPointer(x.typ())
	}
	return check.logosIsNonNilPointer(x.typ())
}

// logosInheritPlatform marks an inferred variable from its initializer:
// `x := <derived>` and tuple dummies inherit derivation, and a derived
// initializer needs no store check (its type was just inferred). A
// generic call with pointer result inherits too: the variable holds a
// platform pointer. It reports true when the caller (initVar) may skip
// the assignment.
func (check *Checker) logosInheritPlatform(lhs *Var, x *operand) bool {
	if lhs == nil || x == nil || !x.isValid() || !check.isLogosFile(lhs) {
		return false
	}
	if !check.logosIsDerived(x) && !check.logosIsPlatform(x) {
		return false
	}
	check.logosMarkDerivedVar(lhs)
	return true
}

// logosMarkTupleDummies marks the synthetic result operands of a
// multi-valued call: every result of a Go-mode call is Go-derived, and
// pointer results of a generic call are platform sources. Hooked from
// multiExpr, which owns the dummy nodes.
func (check *Checker) logosMarkTupleDummies(e syntax.Expr, list []*operand) {
	if e == nil || len(list) == 0 || !check.isLogosFile(e) {
		return
	}
	call, ok := syntax.Unparen(e).(*syntax.CallExpr)
	if !ok {
		return
	}
	isGo := check.logosCalleeIsGo(call)
	wasGeneric := false
	if !isGo {
		wasGeneric = check.logosCalleeWasGeneric(call)
		if !wasGeneric {
			return
		}
	}
	for _, item := range list {
		if item == nil || !item.isValid() || item.expr == nil {
			continue
		}
		if isGo {
			check.logosMarkDerivedExpr(item.expr)
		} else if check.logosIsNonNilPointer(item.typ()) {
			check.logosMarkDerivedExpr(item.expr)
		}
	}
}

// logosLHSIsGo reports whether an assignment target denotes Go memory
// (a field or element of a Go-derived value, or through a derived
// pointer): such stores keep Go's nullable contract and need no check.
func (check *Checker) logosLHSIsGo(lhs syntax.Expr) bool {
	if lhs == nil || !check.isLogosFile(lhs) {
		return false
	}
	switch lhs := syntax.Unparen(lhs).(type) {
	case *syntax.Name:
		return false
	case *syntax.SelectorExpr:
		if check.Info != nil {
			if sel := check.Selections[lhs]; sel != nil {
				if obj := sel.Obj(); obj != nil && check.logosIsGoObj(obj) {
					return true
				}
			}
		}
		return check.logosExprDerived(lhs.X)
	case *syntax.IndexExpr:
		return check.logosExprDerived(lhs.X)
	case *syntax.Operation:
		if lhs.Op == syntax.Mul && lhs.Y == nil {
			return check.logosExprDerived(lhs.X)
		}
		return false
	default:
		return false
	}
}

// logosPushGoStore opens an assignment-LHS scope for assignVar.
// logosPopGoStore closes it. Hooked as bare calls around assignment.
func (check *Checker) logosPushGoStore(lhs syntax.Expr) {
	if !check.conf.Logos {
		return
	}
	check.logosGoStore = append(check.logosGoStore, check.logosLHSIsGo(lhs))
}

// logosPopGoStore closes the scope opened by logosPushGoStore.
func (check *Checker) logosPopGoStore() {
	if !check.conf.Logos {
		return
	}
	if n := len(check.logosGoStore); n > 0 {
		check.logosGoStore = check.logosGoStore[:n-1]
	}
}

// logosGenericScope is one generic-callee argument scope: whether the
// callee had type parameters and its final type arguments. A store
// whose target is identical to a type argument fills a T-position
// (nullable like the parameter) and needs no check.
type logosGenericScope struct {
	generic bool
	targs   []Type
}

// logosPushGenericArgs opens a generic-callee argument scope holding
// the final type arguments. Hooked around the argument loop.
// logosPushConcreteArgs opens a non-generic scope, which suppresses
// nothing. logosPopGenericCall closes either.
func (check *Checker) logosPushGenericArgs(targs []Type) {
	check.logosPushGenericScope(logosGenericScope{generic: true, targs: targs})
}

// logosPushConcreteArgs opens a non-generic argument scope.
func (check *Checker) logosPushConcreteArgs() {
	check.logosPushGenericScope(logosGenericScope{})
}

func (check *Checker) logosPushGenericScope(scope logosGenericScope) {
	if !check.conf.Logos {
		return
	}
	check.logosGenericCall = append(check.logosGenericCall, scope)
}

// logosPopGenericCall closes the scope opened by logosPushGenericCall.
func (check *Checker) logosPopGenericCall() {
	if !check.conf.Logos {
		return
	}
	if n := len(check.logosGenericCall); n > 0 {
		check.logosGenericCall = check.logosGenericCall[:n-1]
	}
}

// logosGenericArg reports whether T fills a type-parameter position of
// an enclosing generic call: such parameters are nullable and take
// platform arguments without a check.
func (check *Checker) logosGenericArg(T Type) bool {
	// Reached only from logosPlatformStore, which already gates on the
	// store position; the scope walk below carries no position of its own.
	if !isValid(T) {
		return false
	}
	for _, scope := range check.logosGenericCall {
		if !scope.generic {
			continue
		}
		for _, targ := range scope.targs {
			if targ != nil && Identical(T, targ) {
				return true
			}
		}
	}
	return false
}

// logosPushGoCall opens a Go-callee argument scope for arguments.
// logosPopGoCall closes it. Go parameters keep Go's nullable contract:
// option-to-pointer needs no check and platform stores need none.
func (check *Checker) logosPushGoCall(call *syntax.CallExpr) {
	if !check.conf.Logos {
		return
	}
	check.logosGoCall = append(check.logosGoCall, check.logosCalleeIsGo(call))
}

// logosPopGoCall closes the scope opened by logosPushGoCall.
func (check *Checker) logosPopGoCall() {
	if !check.conf.Logos {
		return
	}
	if n := len(check.logosGoCall); n > 0 {
		check.logosGoCall = check.logosGoCall[:n-1]
	}
}

// logosInGoCall reports whether argument checking runs for a Go-mode
// callee (any enclosing scope).
func (check *Checker) logosInGoCall() bool {
	for _, b := range check.logosGoCall {
		if b {
			return true
		}
	}
	return false
}

// logosIsSharedTupleCall reports whether x is one element decomposed
// from a multi-valued call sharing the call node: such operands have
// no syntax of their own and cannot be wrapped.
func (check *Checker) logosIsSharedTupleCall(x *operand) bool {
	if x == nil || x.expr == nil {
		return false
	}
	e := syntax.Unparen(x.expr)
	if _, ok := e.(*syntax.CallExpr); !ok {
		return false
	}
	if check.Info != nil {
		if t := check.Info.TypeOf(x.expr); t != nil {
			if tup, ok := t.(*Tuple); ok && tup.Len() > 1 {
				return true
			}
		}
	}
	if call, ok := e.(*syntax.CallExpr); ok {
		if fn := check.logosFuncOfExpr(call.Fun); fn != nil && fn.typ != nil {
			if sig, ok := fn.typ.(*Signature); ok && sig.results != nil && sig.results.Len() > 1 {
				return true
			}
		}
	}
	return false
}

// logosContainsNoWrap reports whether e holds a node the MustNonNil
// lowering cannot enclose: TryExpr (statement lowering owns it),
// SwitchExpr and AssertedStruct (union lowering owns them), and
// FuncLit (later passes splice inside it). Such stores stay holes.
func (check *Checker) logosContainsNoWrap(e syntax.Expr) bool {
	found := false
	syntax.Inspect(e, func(n syntax.Node) bool {
		if found {
			return false
		}
		switch n.(type) {
		case *syntax.TryExpr, *syntax.SwitchExpr, *syntax.AssertedStruct, *syntax.FuncLit:
			found = true
			return false
		}
		return true
	})
	return found
}

// logosRecordCheck records one MustNonNil site, idempotently.
func (check *Checker) logosRecordCheck(e syntax.Expr, T Type) {
	if e == nil || check.Info == nil {
		return
	}
	if check.Info.LogosChecks == nil {
		check.Info.LogosChecks = make(map[syntax.Expr]LogosCheck)
	}
	if _, ok := check.Info.LogosChecks[e]; ok {
		return
	}
	check.Info.LogosChecks[e] = LogosCheck{Target: T}
}

// logosPlatformStore records a MustNonNil site for a store of a
// platform pointer into a declared non-nil location. It never reports
// errors: nil literals keep the compile-time rule, options keep
// theirs, and proven non-nil values need no check. Hooked as a bare
// call from assignment.
func (check *Checker) logosPlatformStore(x *operand, T Type) {
	if x == nil || !check.isLogosFile(x) {
		return
	}
	if check.logosInGoCall() {
		return
	}
	if n := len(check.logosGoStore); n > 0 && check.logosGoStore[n-1] {
		return
	}
	if !isValid(T) || x == nil || !x.isValid() || x.expr == nil {
		return
	}
	if x.isNil() {
		return
	}
	if logosAsOption(T) != nil {
		return
	}
	if !check.logosIsNonNilPointer(T) {
		return
	}
	if !check.logosIsPlatform(x) {
		return
	}
	if check.logosGenericArg(T) {
		return
	}
	if check.logosContainsNoWrap(x.expr) {
		return
	}
	if check.logosIsSharedTupleCall(x) {
		return
	}
	check.logosRecordCheck(x.expr, T)
}
