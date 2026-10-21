// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

// LoGo/s M3-S2 type rules for option types `T?`, `??`, and `?.` (SPEC
// Sec 2.2, T-202, and Sec 1.4, T-104.3). Upstream files call in through
// hook lines only: typexpr.go (option types), expr.go (`??` and `?.`
// expressions, untyped conversions, and the comma-ok split), call.go
// (plain selections on options and optional calls), assignments.go
// (assignments with option sources or targets, and map-index stores),
// index.go (single-value map reads), and predicates.go (`nil`
// comparisons). Every rule below applies only to a package of .lgo
// files (Config.Logos), so plain Go keeps its types exactly as upstream.
//
// An option type is a distinct types2 type: `T` and `nil` are assignable
// to `T?`, `T?` is assignable to `T?` with an identical element, and
// nothing else crosses the boundary. Lowering (a later slice) maps `*T?`
// over a non-nil `*T` to the bare pointer and every other `T?` to
// `logos/rt.Option[T]`; nothing here depends on that mapping. Narrowing
// after `!= nil` checks and the error-first guard are M3-S4: until then,
// every plain selection on an option reports nil_deref_unproven.

const (
	logosNotAnOption = "[correctness::not_an_option]"
	logosNilDeref    = "[correctness::nil_deref_unproven]"
)

// Option is the types2 representation of a loGo/s option type `T?`
// (SPEC T-202). Elem is the wrapped type T; a nested `(T?)?` stays a
// nested *Option, never flattened. Underlying is itself so no upstream
// type switch mistakes an option for its element.
type Option struct {
	Elem Type
}

func (o *Option) Underlying() Type { return o }

// logosCompositeKind names a loGo/s type kind for comparison and
// conversion diagnostics (compositeKind in operand.go).
func logosCompositeKind(typ Type) string {
	if _, ok := typ.Underlying().(*ConstView); ok {
		return "const view"
	}
	return "option"
}

// String prints the option through the qualified writer (like Pointer),
// so nested options parenthesize and package qualification has one owner
// in typestring.go.
func (o *Option) String() string { return TypeString(o, nil) }

// logosAsOption returns the *Option denoted by T, looking through
// aliases and through a defined type whose underlying is an option, or
// nil when T is not an option type.
func logosAsOption(T Type) *Option {
	if o, ok := Unalias(T).(*Option); ok {
		return o
	}
	if n, ok := Unalias(T).(*Named); ok {
		if u := n.Underlying(); u != nil && u != Typ[Invalid] {
			if o, ok := u.(*Option); ok {
				return o
			}
		}
	}
	return nil
}

// logosAsOptionThroughConst returns the *Option denoted by T, looking
// through const views as well as aliases and defined types. A field read
// through a const view (SPEC T-201) types as a const view of the field
// type, so `const string?` still denotes an option for `?.` and `??`
// (SPEC T-104.3); the qualifier is erased before code generation. Plain
// selections, assignments, and narrowing keep using logosAsOption, so
// const-ness is preserved everywhere except these explicit option
// operators.
func logosAsOptionThroughConst(T Type) *Option {
	if c, ok := Unalias(T).(*ConstView); ok {
		return logosAsOptionThroughConst(c.Elem)
	}
	return logosAsOption(T)
}

// logosWrapOption builds the option of t. An already-optional t stays
// nested: `?.` over an option field flattens at the selection, not here.
func logosWrapOption(t Type) *Option { return &Option{Elem: t} }

// logosWrapOne wraps t in an option unless t already is one, flattening
// `F?` selected or returned through `?.` (SPEC T-104.3).
func logosWrapOne(t Type) Type {
	if logosAsOption(t) != nil {
		return t
	}
	return logosWrapOption(t)
}

// logosOptionType type-checks an *syntax.OptionType node (`T?`). The
// element is an ordinary type in the scope of the declaration, so field,
// parameter, result, and alias positions all work through the one hook.
// It mirrors varType (declaredType plus the constraint-interface check)
// while propagating def, so an asserted struct under `?` keeps its
// declaration for the conformance check.
func (check *Checker) logosOptionType(e *syntax.OptionType, def *TypeName) Type {
	elem := check.declaredType(e.Elem, def)
	if !isValid(elem) {
		return Typ[Invalid]
	}
	check.validVarType(e.Elem, elem)
	return logosWrapOption(elem)
}

// logosOptionAssign checks an assignment whose target or source involves
// an option (SPEC T-202): `nil` and `T` are assignable to `T?`, and `T?`
// is assignable to `T?` with an identical element. It reports true when
// it handled the assignment (even on failure); otherwise it reports
// false and the caller runs the upstream path.
func (check *Checker) logosOptionAssign(x *operand, T Type, context string) bool {
	if !check.conf.Logos {
		return false
	}
	Topt := logosAsOption(T)
	Vopt := logosAsOption(x.typ())
	if Topt == nil && Vopt == nil {
		return false
	}
	if isUntyped(x.typ()) {
		if x.isNil() {
			if Topt == nil {
				return false
			}
			x.typ_ = T
			check.updateExprType(x.expr, T, false)
			return true
		}
		if Topt == nil {
			return false
		}
		t, val, code := check.implicitTypeAndValue(x, Topt.Elem)
		if code != 0 || t == nil {
			check.errorf(x, IncompatibleAssign, "cannot use %s as %s value in %s", x, T, context)
			x.invalidate()
			return true
		}
		// Like the upstream untyped path, the operand takes the
		// converted type (t, element-flavored), never the option
		// itself: a constant recorded with a non-const type trips
		// the const-type assert in recordTypeAndValue.
		if val != nil {
			x.val = val
			check.updateExprVal(x.expr, val)
		}
		x.typ_ = t
		check.updateExprType(x.expr, t, false)
		return true
	}
	if Topt != nil {
		// A value wraps into an option only for the identical element
		// type, so `T?` wraps into `(T?)?`; options stay invariant like
		// slices (see comparer).
		return Identical(x.typ(), Topt.Elem) || Identical(x.typ(), T)
	}
	if Vopt != nil && Topt == nil && check.logosInGoCall() {
		// A .lgo *T? lowers to the bare pointer, identical to a Go *T
		// parameter: it compiles with no check (SPEC T-202).
		if check.logosIsNonNilPointer(Vopt.Elem) && Identical(Vopt.Elem, T) {
			return true
		}
	}
	return false
}

// logosOptionImplicit converts an untyped operand against an option
// target through its element, so untyped constants, `nil`, and option
// comparisons keep working wherever implicitTypeAndValue runs (SPEC
// T-202). It reports true with the conversion when the target is an
// option, even on failure (nil type, non-zero code).
func (check *Checker) logosOptionImplicit(x *operand, target Type) (Type, constant.Value, Code, bool) {
	if check == nil || !check.conf.Logos {
		// A nil checker reaches here through the exported AssignableTo
		// API (see the assignableTo contract); only upstream rules apply.
		return nil, nil, 0, false
	}
	Topt := logosAsOption(target)
	if Topt == nil {
		return nil, nil, 0, false
	}
	if x.isNil() {
		return target, nil, 0, true
	}
	t, val, code := check.implicitTypeAndValue(x, Topt.Elem)
	if code != 0 || t == nil {
		return nil, nil, code, true
	}
	// Return the converted type t, never the option target: t is what
	// the recursive call resolved (stripping nesting level by level, so
	// `42` against `(int?)?` yields int), and recording a constant with
	// an option type trips the const-type assert in recordTypeAndValue.
	return t, val, 0, true
}

// logosOptionSelect reports nil_deref_unproven for a plain selection on
// an option-typed receiver (SPEC T-202, T-102): the value may be none,
// so an ordinary field or method selection needs a nil guard, `?.`, or
// `??` first. Missing members fall through to the upstream "undefined"
// diagnostic. It reports true when it handled e.
func (check *Checker) logosOptionSelect(x *operand, e *syntax.SelectorExpr, sel string) bool {
	if !check.conf.Logos || x.mode() == typexpr {
		return false
	}
	opt := logosAsOption(x.typ())
	if opt == nil {
		return false
	}
	obj, _, _ := lookupFieldOrMethod(opt.Elem, x.mode() == variable, check.pkg, sel, false)
	if obj == nil {
		return false
	}
	if _, ok := obj.(*Func); ok {
		check.errorf(e.Sel, InvalidMethodExpr, "%s cannot call method on %s without proving != nil", logosNilDeref, x.typ())
	} else {
		check.errorf(e.Sel, MissingFieldOrMethod, "%s cannot access field %s of %s without proving != nil", logosNilDeref, sel, x.typ())
	}
	x.invalidate()
	x.typ_ = Typ[Invalid]
	x.expr = e
	return true
}

// logosOptSelect type-checks `x?.Sel` (SPEC T-104.3, T-202): x must be
// an option, else not_an_option. The result is the selected field type
// or method signature as an option, flattening when it already is one.
func (check *Checker) logosOptSelect(x *operand, e *syntax.OptSelectExpr) exprKind {
	var rx operand
	check.expr(nil, &rx, e.X)
	if !rx.isValid() {
		x.invalidate()
		x.expr = e
		return expression
	}
	check.logosUnnarrowOperand(&rx) // logos M3-S4 hook: `?.` on a narrowed var uses its declared option.
	opt := logosAsOptionThroughConst(rx.typ())
	if opt == nil {
		check.errorf(e, MissingFieldOrMethod, "%s ?. on non-option type %s", logosNotAnOption, rx.typ())
		x.invalidate()
		x.expr = e
		return expression
	}
	obj, _, _ := lookupFieldOrMethod(opt.Elem, rx.mode() == variable, check.pkg, e.Sel.Value, false)
	if obj == nil {
		alt, _, _ := lookupFieldOrMethod(opt.Elem, rx.mode() == variable, check.pkg, e.Sel.Value, true)
		why := check.lookupError(opt.Elem, e.Sel.Value, alt, false)
		check.errorf(e.Sel, MissingFieldOrMethod, "%s?.%s undefined (%s)", e.X, e.Sel.Value, why)
		x.invalidate()
		x.expr = e
		return expression
	}
	var rt Type
	switch obj := obj.(type) {
	case *Var:
		rt = obj.typ
	case *Func:
		check.objDecl(obj)
		check.addDeclDep(obj)
		// Method value: drop the receiver like the upstream
		// method-value branch, leaving an ordinary signature
		// for the optional-call hook to unwrap.
		sig, ok := obj.typ.(*Signature)
		if !ok {
			check.errorf(e.Sel, MissingFieldOrMethod, "%s?.%s undefined (not a field or method)", logosNotAnOption, e.Sel.Value)
			x.invalidate()
			x.expr = e
			return expression
		}
		s := *sig
		s.recvold = s.recv
		s.recv = nil
		rt = &s
	default:
		check.errorf(e.Sel, MissingFieldOrMethod, "%s?.%s undefined (not a field or method)", logosNotAnOption, e.Sel.Value)
		x.invalidate()
		x.expr = e
		return expression
	}
	x.mode_ = value
	x.typ_ = logosWrapOne(rt)
	x.expr = e
	return expression
}

// logosCoalesce type-checks `x ?? y` (SPEC T-104.3, T-202): x must be an
// option `T?`, else not_an_option; y must be assignable to T; the result
// is T. `m[k] ?? d` types as V through the option-wrapped map read.
func (check *Checker) logosCoalesce(x *operand, e *syntax.CoalesceExpr) exprKind {
	check.logosRejectTryChainUnder(e.Y)
	var lx operand
	check.expr(nil, &lx, e.X)
	var rx operand
	check.expr(nil, &rx, e.Y)
	if !lx.isValid() || !rx.isValid() {
		x.invalidate()
		x.expr = e
		return expression
	}
	check.logosUnnarrowOperand(&lx) // logos M3-S4 hook: `??` on a narrowed var uses its declared option.
	opt := logosAsOptionThroughConst(lx.typ())
	if opt == nil {
		check.errorf(e, UndefinedOp, "%s ?? with non-option left type %s", logosNotAnOption, lx.typ())
		x.invalidate()
		x.expr = e
		return expression
	}
	check.assignment(&rx, opt.Elem, "?? right-hand side")
	if !rx.isValid() {
		x.invalidate()
		x.expr = e
		return expression
	}
	x.mode_ = value
	x.typ_ = opt.Elem
	x.expr = e
	return expression
}

// logosOptionalCall type-checks `x?.M(args)` (SPEC T-104.3): the function
// operand carries an option of a signature (from logosOptSelect), so the
// call runs against the unwrapped signature and each result is wrapped
// back into an option, since none propagates. A plain call on an
// unnarrowed option-typed function value is rejected with
// nil_deref_unproven (SPEC T-202). It reports true when it handled the
// call.
func (check *Checker) logosOptionalCall(x *operand, call *syntax.CallExpr) bool {
	if !check.conf.Logos {
		return false
	}
	if _, ok := syntax.Unparen(call.Fun).(*syntax.OptSelectExpr); !ok {
		if opt := logosAsOption(x.typ()); opt != nil {
			if _, ok := Unalias(opt.Elem).(*Signature); ok {
				check.errorf(syntax.Unparen(call.Fun), InvalidCall, "%s cannot call %s without proving != nil", logosNilDeref, x.typ())
				x.invalidate()
				x.expr = call
				return true
			}
		}
		return false
	}
	opt := logosAsOption(x.typ())
	if opt == nil {
		return false
	}
	sig, ok := Unalias(opt.Elem).(*Signature)
	if !ok || sig.TypeParams().Len() > 0 {
		// A non-function field or a generic method under `?.`:
		// leave the upstream "cannot call" diagnostic.
		return false
	}
	for _, arg := range call.ArgList {
		check.logosRejectTryChainUnder(arg)
	}
	args, atargs := check.genericExprList(call.ArgList)
	rsig := check.arguments(call, sig, nil, nil, args, atargs)
	switch rsig.results.Len() {
	case 0:
		x.mode_ = novalue
	case 1:
		typ := rsig.results.vars[0].typ
		if !check.isComplete(typ) {
			x.invalidate()
			x.expr = call
			return true
		}
		x.mode_ = value
		x.typ_ = logosWrapOne(typ)
	default:
		vars := make([]*Var, rsig.results.Len())
		for i, v := range rsig.results.vars {
			vars[i] = newVar(ResultVar, v.Pos(), v.Pkg(), v.Name(), logosWrapOne(v.typ))
		}
		x.mode_ = value
		x.typ_ = NewTuple(vars...)
	}
	x.expr = call
	check.hasCallOrRecv = true
	return true
}

// logosMapIndexOption wraps a single-value .lgo map read in an option
// (SPEC T-104.3): an absent key reads as none, so `m[k]` types as `V?`.
// Comma-ok reads and assignment targets unwrap through
// logosUnwrapMapIndex and keep Go types.
func (check *Checker) logosMapIndexOption(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	x.typ_ = logosWrapOption(x.typ())
}

// logosUnwrapMapIndex restores the plain element type of a map-index
// operand for comma-ok and assignment-target positions: `v, ok := m[k]`
// keeps Go types and `m[k] = v` stores V (SPEC T-104.3, T-202). The
// expression record (written by rawExpr before these positions run)
// follows the operand back through a re-record; the comma-ok path
// overwrites it once more with the pair type via recordCommaOkTypes.
func (check *Checker) logosUnwrapMapIndex(x *operand) {
	if !check.conf.Logos || x.mode() != mapindex {
		return
	}
	if opt := logosAsOption(x.typ()); opt != nil {
		x.typ_ = opt.Elem
		check.recordTypeAndValue(x.expr, mapindex, opt.Elem, nil)
	}
}
