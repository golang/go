// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
	"strings"
)

// LoGo/s M5-S1 type rules for const views `const T` (SPEC Sec 2.1,
// T-201). A const view is a compile-time-only qualifier, erased by
// lowering: reading through one yields a const view of the element
// (transitive), a plain value converts into its view, and a view
// converts back only as a reference-free copy. Every mutating use
// reports [correctness::const_view_mutation], and a
// reference-carrying assignment out of a view reports
// [correctness::const_view_escape].
//
// Upstream files call in through hook lines only: typexpr.go (const
// view types), typestring.go (printing), predicates.go (identity and
// nil), subst.go and unify.go (generics), resolver.go (const pointer
// receivers), call.go (selectors and argument passing), expr.go
// (index, slice, dereference, and binary reads), range.go (const
// elements), stmt.go (numeric unwrapping for ++), assignments.go
// (stores and conversions back), and builtins.go (append, copy,
// delete, clear, len, cap). The parser only produces
// *syntax.ConstType in LogosMode, so the checker hooks are unreachable
// in plain Go. Imported callees fail closed: without a visible
// ParamMutated=0 fact the checker refuses a reference-carrying view,
// and a fact-0 parameter will read as a const parameter once the const
// overlay reader lands.
//
// Chain lambdas (M6, #41) reuse the declared-type machinery below: a
// lambda parameter declared with a const view type is const inside its
// body, exactly like a named function parameter, so no separate chain
// rule is needed here.

const (
	logosConstMutation = "[correctness::const_view_mutation]"
	logosConstEscape   = "[correctness::const_view_escape]"
)

// ConstView is the types2 representation of a loGo/s const view
// `const T` (SPEC T-201). Elem is the viewed type T. Underlying is
// itself so no upstream type switch mistakes a view for its element.
// Lowering erases views to their Go-facing form.
type ConstView struct {
	Elem Type
	// ConstMap marks the view of a const map declaration (M5-S4, SPEC
	// T-Sec9). Its plain uses materialize one regular map, so it converts
	// to map[K]V and passes to standard-library parameters that do not mutate
	// their map argument; every store fails as for any const view.
	ConstMap bool
}

func (c *ConstView) Underlying() Type { return c }

// String prints the view through the qualified writer (like Pointer),
// so package qualification has one owner in typestring.go.
func (c *ConstView) String() string { return TypeString(c, nil) }

// logosAsConstView returns the *ConstView denoted by T, looking through
// aliases, or nil when T is not a const view.
func logosAsConstView(T Type) *ConstView {
	t, _ := Unalias(T).(*ConstView)
	return t
}

// logosReferenceFree reports whether a value of type t can hold no
// reference: no pointers, slices, maps, channels, functions, or
// interfaces inside (SPEC T-201). Only a reference-free copy may leave
// a const view. Type parameters and unsafe pointers are conservatively
// not free.
func logosReferenceFree(t Type) bool {
	switch t := t.Underlying().(type) {
	case *Basic:
		return t.Kind() != UnsafePointer
	case *Array:
		return logosReferenceFree(t.Elem())
	case *Struct:
		for i := range t.NumFields() {
			if !logosReferenceFree(t.Field(i).Type()) {
				return false
			}
		}
		return true
	}
	return false
}

// logosConstType type-checks a *syntax.ConstType node (`const T`). The
// element is an ordinary type in the scope of the declaration, so
// parameter, receiver, result, variable, and field positions all work
// through the one hook. A doubled `const const T` flattens to one view.
func (check *Checker) logosConstType(e *syntax.ConstType, def *TypeName) Type {
	typ := check.varType(e.Elem)
	if !isValid(typ) {
		return typ
	}
	if inner := logosAsConstView(typ); inner != nil {
		return inner
	}
	return &ConstView{Elem: typ}
}

// logosConstAssign checks an assignment whose source or target involves
// a const view (SPEC T-201): plain values and nil convert into a view,
// a view converts back only as a reference-free copy, and anything else
// leaving a view escapes. It reports true when it handled the
// assignment (even on failure); otherwise it reports false and the
// caller runs the upstream path.
func (check *Checker) logosConstAssign(x *operand, T Type, context string) bool {
	if !check.conf.Logos {
		return false
	}
	csrc := logosAsConstView(x.typ())
	cdst := logosAsConstView(T)
	if csrc == nil && cdst == nil {
		return false
	}
	if x.isNil() {
		if cdst == nil || !hasNil(cdst.Elem) {
			return false
		}
		x.typ_ = T
		check.updateExprType(x.expr, T, false)
		return true
	}
	if cdst != nil {
		return check.logosConstWrap(x, csrc, cdst, T, context)
	}
	if T == nil {
		return false
	}
	if logosReferenceFree(csrc.Elem) || csrc.ConstMap {
		x.typ_ = csrc.Elem
		check.updateExprType(x.expr, csrc.Elem, false)
		return false
	}
	op := "assign"
	if strings.HasPrefix(context, "argument to ") {
		op = "pass argument"
	}
	check.errorf(x, IncompatibleAssign, "%s cannot %s through const view %s in %s", logosConstEscape, op, ExprString(x.expr), context)
	x.invalidate()
	return true
}

// logosConstWrap checks a plain value wrapping into a const view: the
// value must convert to the element type. Untyped constants convert
// through the element; anything else escapes.
func (check *Checker) logosConstWrap(x *operand, csrc *ConstView, cdst *ConstView, T Type, context string) bool {
	if csrc != nil {
		return Identical(x.typ(), T)
	}
	if isUntyped(x.typ()) {
		t, val, code := check.implicitTypeAndValue(x, cdst.Elem)
		if code != 0 || t == nil {
			check.constWrapError(x, T, context)
			return true
		}
		if val != nil {
			x.val = val
			check.updateExprVal(x.expr, val)
		}
		x.typ_ = t
		check.updateExprType(x.expr, t, false)
		return true
	}
	if ok, _ := x.assignableTo(check, cdst.Elem, nil); !ok {
		check.constWrapError(x, T, context)
		return true
	}
	return true
}

// constWrapError reports a failed wrap into a const view.
func (check *Checker) constWrapError(x *operand, T Type, context string) {
	check.errorf(x, IncompatibleAssign, "%s cannot use %s as %s value in %s", logosConstEscape, x, T, context)
	x.invalidate()
}

// logosConstStore reports a store through a const view: any proper
// subexpression of the assignment target with a const view type means
// the store reads through the view (index, field, and pointer stores).
// A bare name rebinds the view itself and stays allowed. It reports
// true on a violation.
func (check *Checker) logosConstStore(lhs syntax.Expr) bool {
	if !check.conf.Logos {
		return false
	}
	var op string
	switch e := syntax.Unparen(lhs).(type) {
	case *syntax.IndexExpr:
		op = "assign index"
	case *syntax.SelectorExpr:
		op = "assign field"
	case *syntax.Operation:
		if e.Op != syntax.Mul || e.Y != nil {
			return false
		}
		op = "assign"
	default:
		return false
	}
	base := check.logosConstStoreBase(lhs)
	if base == nil {
		return false
	}
	check.errorf(base, IncompatibleAssign, "%s cannot %s through const view %s", logosConstMutation, op, ExprString(base))
	return true
}

// logosConstStoreBase returns the innermost proper subexpression of an
// assignment target with a const view type, or nil when the store reads
// through no view.
func (check *Checker) logosConstStoreBase(lhs syntax.Expr) syntax.Expr {
	switch e := syntax.Unparen(lhs).(type) {
	case *syntax.IndexExpr:
		if logosAsConstView(check.Info.TypeOf(e.X)) != nil {
			if inner := check.logosConstStoreBase(e.X); inner != nil {
				return inner
			}
			return e.X
		}
		return check.logosConstStoreBase(e.X)
	case *syntax.SelectorExpr:
		if logosAsConstView(check.Info.TypeOf(e.X)) != nil {
			if inner := check.logosConstStoreBase(e.X); inner != nil {
				return inner
			}
			return e.X
		}
		return check.logosConstStoreBase(e.X)
	case *syntax.Operation:
		if e.Op != syntax.Mul || e.Y != nil {
			return nil
		}
		if logosAsConstView(check.Info.TypeOf(e.X)) != nil {
			return e.X
		}
		return check.logosConstStoreBase(e.X)
	}
	return nil
}

// logosConstSelect type-checks selections on a const view base (SPEC
// T-201): fields read through to a const view of the field type, while
// methods follow the receiver rule (a plain pointer receiver is
// refused; value and const receivers unwrap and continue upstream). It
// reports true when it handled the selection.
func (check *Checker) logosConstSelect(x *operand, e *syntax.SelectorExpr, sel string) bool {
	if !check.conf.Logos || x.mode() == typexpr {
		return false
	}
	base := logosAsConstView(x.typ())
	if base == nil {
		return false
	}
	obj, index, indirect := lookupFieldOrMethod(base.Elem, x.mode() == variable, check.pkg, sel, false)
	if obj == nil {
		return false
	}
	switch obj := obj.(type) {
	case *Var:
		check.recordSelection(e, FieldVal, base.Elem, obj, index, indirect)
		if x.mode() == variable || indirect {
			x.mode_ = variable
		} else {
			x.mode_ = value
		}
		x.typ_ = &ConstView{Elem: obj.Type()}
		x.expr = e
		return true
	case *Func:
		check.objDecl(obj)
		sig, _ := obj.Type().(*Signature)
		if sig == nil || sig.Recv() == nil {
			return false
		}
		if _, ok := sig.Recv().Type().(*Pointer); ok {
			check.errorf(e.Sel, IncompatibleAssign, "%s cannot call %s through const view %s", logosConstMutation, sel, ExprString(e.X))
			x.invalidate()
			return true
		}
		x.typ_ = base.Elem
		return false
	}
	return false
}

// logosConstUnwrapBase unwraps a const view base for index, slice, and
// dereference checking, so the upstream kind switches see the element
// type. Results wrap again through logosConstWrapBase below.
func (check *Checker) logosConstUnwrapBase(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	if c := logosAsConstView(x.typ()); c != nil {
		x.typ_ = c.Elem
	}
}

// logosConstWrapBase wraps a fresh index, slice, or dereference result
// in a const view when its base reads through a view (transitive
// reads, SPEC T-201). Map-index results keep their M3-S6 option form
// for the comma-ok positions.
func (check *Checker) logosConstWrapBase(x *operand, base syntax.Expr) {
	if !check.conf.Logos || !x.isValid() || x.mode() == mapindex {
		return
	}
	if logosAsConstView(check.Info.TypeOf(syntax.Unparen(base))) == nil {
		return
	}
	x.typ_ = &ConstView{Elem: x.typ()}
}

// logosConstNumeric unwraps a reference-free const operand for the
// numeric checks behind ++ and numeric operators, so `total += v` and
// `v++` on a const int read the value. Stores still fail through the
// syntax-based store check.
func (check *Checker) logosConstNumeric(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	if c := logosAsConstView(x.typ()); c != nil && logosReferenceFree(c.Elem) {
		x.typ_ = c.Elem
		check.updateExprType(x.expr, c.Elem, false)
	}
}

// logosConstBinary unwraps reference-free const operands of a binary
// expression. Non-reference-free views keep their type and fail in the
// operator checks.
func (check *Checker) logosConstBinary(x, y *operand) {
	check.logosConstNumeric(x)
	check.logosConstNumeric(y)
}

// logosConstConvert checks an explicit conversion out of a const view
// (SPEC T-201): string and []byte convert in both directions
// (zero-copy provenance is a lowering decision), reference-free
// targets copy out, and anything else escapes.
func (check *Checker) logosConstConvert(x *operand, T Type) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	c := logosAsConstView(x.typ())
	if c == nil {
		return false
	}
	_, elemSlice := Unalias(c.Elem).(*Slice)
	_, targSlice := Unalias(T).(*Slice)
	if isString(c.Elem) && targSlice || elemSlice && isString(T) {
		x.typ_ = c.Elem
		check.updateExprType(x.expr, c.Elem, false)
		return false
	}
	if logosReferenceFree(T) {
		x.typ_ = c.Elem
		check.updateExprType(x.expr, c.Elem, false)
		return false
	}
	check.errorf(x, IncompatibleAssign, "%s cannot convert const view %s to %s", logosConstEscape, ExprString(x.expr), T)
	x.invalidate()
	return true
}

// logosConstBuiltin checks built-in calls with const view arguments
// (SPEC T-201): append to, copy into, delete from, and clear of a view
// are refused; every other built-in reads, so const operands unwrap.
func (check *Checker) logosConstBuiltin(x *operand, call *syntax.CallExpr, id builtinId, args []*operand) bool {
	if !check.conf.Logos {
		return false
	}
	unwrap := func(a *operand) {
		if c := logosAsConstView(a.typ()); c != nil {
			a.typ_ = c.Elem
			check.updateExprType(a.expr, c.Elem, false)
		}
	}
	constView := func(a *operand) bool {
		return logosAsConstView(a.typ()) != nil
	}
	switch id {
	case _Append:
		if len(args) > 0 {
			if constView(args[0]) {
				check.errorf(args[0], IncompatibleAssign, "%s cannot append through const view %s", logosConstMutation, ExprString(args[0].expr))
				return true
			}
			for _, a := range args[1:] {
				unwrap(a)
			}
		}
	case _Copy:
		if len(args) > 0 {
			if constView(args[0]) {
				check.errorf(args[0], IncompatibleAssign, "%s cannot copy through const view %s", logosConstMutation, ExprString(args[0].expr))
				return true
			}
			for _, a := range args[1:] {
				unwrap(a)
			}
		}
	case _Delete:
		if len(args) > 0 && constView(args[0]) {
			check.errorf(args[0], IncompatibleAssign, "%s cannot delete through const view %s", logosConstMutation, ExprString(args[0].expr))
			return true
		}
	case _Clear:
		if len(args) > 0 && constView(args[0]) {
			check.errorf(args[0], IncompatibleAssign, "%s cannot clear through const view %s", logosConstMutation, ExprString(args[0].expr))
			return true
		}
	default:
		for _, a := range args {
			unwrap(a)
		}
		if len(args) > 0 {
			if c := logosAsConstView(x.typ()); c != nil {
				x.typ_ = c.Elem
			}
		}
	}
	return false
}

// logosConstArgs checks const view call arguments against the callee
// parameters (SPEC T-201): const and type-parameter parameters wrap the
// argument, reference-free elements copy out, and anything else fails.
// Callees outside the current package fail closed: without a visible
// ParamMutated=0 fact the checker refuses a reference-carrying view.
func (check *Checker) logosConstArgs(call *syntax.CallExpr, sigParams *Tuple, args []*operand) {
	if !check.conf.Logos {
		return
	}
	if name, _ := call.Fun.(*syntax.Name); name != nil {
		if _, ok := check.lookup(name.Value).(*Builtin); ok {
			return
		}
	}
	for i, a := range args {
		if sigParams == nil || i >= sigParams.Len() {
			break
		}
		if !a.isValid() {
			continue
		}
		c := logosAsConstView(a.typ())
		if c == nil {
			continue
		}
		pt := sigParams.At(i).Type()
		if logosAsConstView(pt) != nil || isTypeParam(pt) {
			a.typ_ = c.Elem
			check.updateExprType(a.expr, c.Elem, false)
			continue
		}
		if logosReferenceFree(c.Elem) {
			a.typ_ = c.Elem
			check.updateExprType(a.expr, c.Elem, false)
			continue
		}
		check.errorf(a, IncompatibleAssign, "%s cannot pass argument through const view %s in argument to %s", logosConstMutation, ExprString(a.expr), ExprString(call.Fun))
		a.invalidate()
	}
}

// logosConstRecvVar returns a copy of a method receiver variable with a
// const view receiver unwrapped for validation, so `func (r const *T)`
// validates as `*T` while the body keeps the declared view.
func (check *Checker) logosConstRecvVar(recv *Var) *Var {
	if !check.conf.Logos {
		return recv
	}
	if c := logosAsConstView(recv.Type()); c != nil {
		cp := *recv
		cp.setType(c.Elem)
		return &cp
	}
	return recv
}

// logosUnpackConstRecv strips one `const` layer from a method receiver
// type expression, so `func (r const *T) M()` collects on T like `*T`
// while the declared receiver type stays a view (SPEC T-201).
func (check *Checker) logosUnpackConstRecv(rtyp syntax.Expr) syntax.Expr {
	if !check.conf.Logos {
		return rtyp
	}
	if t, _ := syntax.Unparen(rtyp).(*syntax.ConstType); t != nil {
		return t.Elem
	}
	return rtyp
}

// logosConstRange unwraps a const range expression to its element type
// for the iteration checks. The original view stays recorded on the
// range expression for the value wrap below.
func (check *Checker) logosConstRange(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	if c := logosAsConstView(x.typ()); c != nil {
		x.typ_ = c.Elem
	}
}

// logosConstRangeVal wraps a range value in a const view when the range
// expression reads through a view (SPEC T-201: range over a const
// slice yields const elements). It is a bare call so the go/types
// mirror drops it.
func (check *Checker) logosConstRangeVal(rangeVar syntax.Expr, valp *Type) {
	if !check.conf.Logos || !isValid(*valp) {
		return
	}
	if logosAsConstView(check.Info.TypeOf(rangeVar)) == nil {
		return
	}
	*valp = &ConstView{Elem: *valp}
}

// logosConstSliceBound checks the high and max bounds of a slice
// expression over a const view base (SPEC T-201). `b[:cap(b)]` fails:
// slicing to the capacity is legal Go but reads past the const length.
// Constant bounds past an array length stay with the upstream
// out-of-bounds error, which already rejects what Go rejects. Every
// other present bound (a `len` call over the same base is provably
// fine) is recorded in Info.LogosConstSlices for the lowering-time
// runtime check.
func (check *Checker) logosConstSliceBound(x *operand, e *syntax.SliceExpr) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	base := logosAsConstView(check.Info.TypeOf(syntax.Unparen(e.X)))
	if base == nil {
		return
	}
	for _, hi := range []syntax.Expr{e.Index[1], e.Index[2]} {
		if hi == nil {
			continue
		}
		if check.logosIsCapOf(hi, e.X) {
			check.errorf(e.X, IncompatibleAssign, "%s cannot slice through const view %s", logosConstMutation, ExprString(e.X))
			x.invalidate()
			return
		}
		if check.logosIsLenOf(hi, e.X) {
			continue
		}
		if c, ok := check.constBoundValue(hi); ok {
			if arr, ok := Unalias(base.Elem).(*Array); ok && c > arr.Len() {
				check.errorf(e.X, IncompatibleAssign, "%s cannot slice through const view %s", logosConstMutation, ExprString(e.X))
				x.invalidate()
				return
			}
			continue
		}
		if check.Info.LogosConstSlices == nil {
			check.Info.LogosConstSlices = make(map[syntax.Expr]bool)
		}
		if _, ok := check.constBoundValue(hi); !ok {
			check.Info.LogosConstSlices[e] = true
		}
	}
}

// logosIsCapOf reports whether bound is a `cap(base)` call over the
// same variable as base.
func (check *Checker) logosIsCapOf(bound, base syntax.Expr) bool {
	call, _ := syntax.Unparen(bound).(*syntax.CallExpr)
	if call == nil || len(call.ArgList) != 1 {
		return false
	}
	fn, _ := syntax.Unparen(call.Fun).(*syntax.Name)
	if fn == nil {
		return false
	}
	obj, _ := check.lookup(fn.Value).(*Builtin)
	if obj == nil || obj.Name() != "cap" {
		return false
	}
	bVar, _ := syntax.Unparen(base).(*syntax.Name)
	aVar, _ := syntax.Unparen(call.ArgList[0]).(*syntax.Name)
	if bVar == nil || aVar == nil {
		return false
	}
	bObj, bOk := check.Info.Uses[bVar]
	aObj, aOk := check.Info.Uses[aVar]
	return bOk && aOk && bObj == aObj
}

// logosIsLenOf reports whether bound is a `len(base)` call over the
// same variable as base. Slicing to the length is provably fine.
func (check *Checker) logosIsLenOf(bound, base syntax.Expr) bool {
	call, _ := syntax.Unparen(bound).(*syntax.CallExpr)
	if call == nil || len(call.ArgList) != 1 {
		return false
	}
	fn, _ := syntax.Unparen(call.Fun).(*syntax.Name)
	if fn == nil {
		return false
	}
	obj, _ := check.lookup(fn.Value).(*Builtin)
	if obj == nil || obj.Name() != "len" {
		return false
	}
	bVar, _ := syntax.Unparen(base).(*syntax.Name)
	aVar, _ := syntax.Unparen(call.ArgList[0]).(*syntax.Name)
	if bVar == nil || aVar == nil {
		return false
	}
	bObj, bOk := check.Info.Uses[bVar]
	aObj, aOk := check.Info.Uses[aVar]
	return bOk && aOk && bObj == aObj
}

// constBoundValue returns the constant value of a slice bound.
func (check *Checker) constBoundValue(bound syntax.Expr) (int64, bool) {
	un := syntax.Unparen(bound)
	if tv, ok := check.Info.Types[un]; ok && tv.Value != nil {
		if v, ok := constant.Int64Val(tv.Value); ok {
			return v, true
		}
		return 0, false
	}
	if tv := un.GetTypeInfo(); tv.Value != nil {
		if v, ok := constant.Int64Val(tv.Value); ok {
			return v, true
		}
	}
	return 0, false
}
