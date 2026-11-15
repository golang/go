// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
)

const (
	logosNotAnOption = "[correctness::not_an_option]"
	logosNilDeref    = "[correctness::nil_deref_unproven]"
)

type Option struct {
	Elem Type
}

func (o *Option) Underlying() Type { return o }

func logosCompositeKind(typ Type) string {
	switch typ.Underlying().(type) {
	case *ConstView:
		return "const view"
	case *ReadonlyDescendants:
		return "readonly descendants"
	}
	return "option"
}

func (o *Option) String() string { return TypeString(o, nil) }

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

func logosAsOptionThroughConst(T Type) *Option {
	base, descendants, readonly := logosReadonlyBase(T)
	if descendants || readonly {
		if o := logosAsOptionThroughConst(base); o != nil {
			return &Option{Elem: logosReadonlyProject(T, o.Elem, false)}
		}
		return nil
	}
	return logosAsOption(T)
}

func LogosOptionIsBare(elem Type) bool {
	if elem == nil {
		return false
	}
	elem, _, _ = logosReadonlyBase(elem)
	if _, ok := elem.Underlying().(*Pointer); ok {
		return true
	}
	members, isUnion := LogosUnionTerms(elem)
	if !isUnion {
		return logosImportedBareUnion(elem)
	}
	for _, m := range members {
		if logosImportedBareUnion(m) {
			continue
		}
		mu := Unalias(m)
		mu, _, _ = logosReadonlyBase(mu)
		if n, ok := mu.(*Named); ok {
			mu = n.Underlying()
		}
		switch mu := mu.(type) {
		case *Slice, *Map, *Signature, *Chan, *Interface, *Union:
			return false
		case *Basic:
			if mu.Kind() == UnsafePointer {
				return false
			}
		case *TypeParam:
			return false
		}
	}
	return true
}

func logosOptionRepr(o *Option) Type {
	if LogosOptionIsBare(o.Elem) {
		return o.Elem
	}
	return NewStruct([]*Var{
		NewField(nopos, nil, "Value", o.Elem, false),
		NewField(nopos, nil, "Present", Typ[Bool], false),
	}, nil)
}

func logosWrapOption(t Type) *Option { return &Option{Elem: t} }

func logosWrapOne(t Type) Type {
	if logosAsOption(t) != nil {
		return t
	}
	return logosWrapOption(t)
}

func (check *Checker) logosOptionType(e *syntax.OptionType, def *TypeName) Type {
	elem := check.declaredType(e.Elem, def)
	if !isValid(elem) {
		return Typ[Invalid]
	}
	check.validVarType(e.Elem, elem)
	return logosWrapOption(elem)
}

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
		if val != nil {
			x.val = val
			check.updateExprVal(x.expr, val)
		}
		x.typ_ = t
		check.updateExprType(x.expr, t, false)
		return true
	}
	if Topt != nil {
		if _, ok := syntax.Unparen(x.expr).(*syntax.TupleLit); ok && Vopt == nil {
			if check.logosTupleAssign(x, Topt.Elem, context) {
				return true
			}
		}
		return Identical(x.typ(), Topt.Elem) || Identical(x.typ(), T)
	}
	if Vopt != nil && check.logosInGoCall() {
		if check.logosIsNonNilPointer(Vopt.Elem) && Identical(Vopt.Elem, T) {
			return true
		}
	}
	return false
}

func (check *Checker) logosOptionImplicit(x *operand, target Type) (Type, constant.Value, Code, bool) {
	if check == nil || !check.conf.Logos {
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
	return t, val, 0, true
}

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

func (check *Checker) logosOptionUnwrap(x *operand, T Type, context string) bool {
	if !check.conf.Logos {
		return false
	}
	opt := logosAsOption(x.typ())
	if opt == nil || !Identical(opt.Elem, T) {
		return false
	}
	check.errorf(x, IncompatibleAssign, "%s cannot use %s as %s value in %s without proving != nil", logosNilDeref, x, T, context)
	x.invalidate()
	return true
}

func (check *Checker) logosOptionIndirect(x *operand) bool {
	if !check.conf.Logos {
		return false
	}
	opt := logosAsOption(x.typ())
	if opt == nil {
		return false
	}
	if _, ok := opt.Elem.Underlying().(*Pointer); !ok {
		return false
	}
	check.errorf(x, InvalidIndirection, "%s cannot indirect %s without proving != nil", logosNilDeref, x)
	return true
}

func (check *Checker) logosOptSelect(x *operand, e *syntax.OptSelectExpr) exprKind {
	var rx operand
	check.expr(nil, &rx, e.X)
	if !rx.isValid() {
		x.invalidate()
		x.expr = e
		return expression
	}
	check.logosUnnarrowOperand(&rx)
	opt := logosAsOptionThroughConst(rx.typ())
	if opt == nil {
		check.errorf(e, MissingFieldOrMethod, "%s ?. on non-option type %s", logosNotAnOption, rx.typ())
		x.invalidate()
		x.expr = e
		return expression
	}
	obj, _, _ := lookupFieldOrMethod(opt.Elem, rx.mode() == variable, check.pkg, e.Sel.Value, false)
	if obj == nil {
		if f := logosTupleField(opt.Elem, e.Sel.Value); f != nil {
			obj = f
		}
	}
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
		base, _, _ := logosReadonlyBase(opt.Elem)
		_, throughReference := base.Underlying().(*Pointer)
		rt = logosReadonlyProject(opt.Elem, obj.typ, throughReference)
	case *Func:
		check.objDecl(obj)
		if _, descendants, readonly := logosReadonlyBase(opt.Elem); (descendants || readonly) && !check.logosReceiverReadonly(obj) {
			check.errorf(e.Sel, InvalidMethodExpr, "%s cannot call mutating method %s through %s", logosConstEscape, e.Sel.Value, opt.Elem)
			x.invalidate()
			x.expr = e
			return expression
		}
		check.addDeclDep(obj)
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

func (check *Checker) logosCoalesce(x *operand, e *syntax.CoalesceExpr) exprKind {
	check.logosChainNestedTry(e.Y)
	var lx operand
	check.expr(nil, &lx, e.X)
	var rx operand
	if !check.logosLambdaDefer(&rx, e.Y) {
		check.expr(nil, &rx, e.Y)
	}
	if !lx.isValid() || !rx.isValid() {
		x.invalidate()
		x.expr = e
		return expression
	}
	check.logosUnnarrowOperand(&lx)
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
		return false
	}
	for _, arg := range call.ArgList {
		check.logosChainNestedTry(arg)
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

func (check *Checker) logosMapIndexOption(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	x.typ_ = logosWrapOption(x.typ())
}

func (check *Checker) logosUnwrapMapIndex(x *operand) {
	if !check.conf.Logos || x.mode() != mapindex {
		return
	}
	if opt := logosAsOption(x.typ()); opt != nil {
		x.typ_ = opt.Elem
		check.recordTypeAndValue(x.expr, mapindex, opt.Elem, nil)
	}
}
