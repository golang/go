// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/logos/ctfe"
	"cmd/compile/internal/syntax"
	"go/constant"
	. "internal/types/errors"
	"slices"
	"strconv"
	"strings"
)

const (
	logosConstMutation = "[correctness::const_view_mutation]"
	logosConstEscape   = "[correctness::const_view_escape]"
)

type ConstView struct {
	Elem     Type
	ConstMap bool
}

func (c *ConstView) Underlying() Type { return c }

func (c *ConstView) String() string { return TypeString(c, nil) }

func logosAsConstView(T Type) *ConstView {
	t, _ := Unalias(T).(*ConstView)
	return t
}

func logosReferenceFree(t Type) bool {
	return logosReferenceFreeAlong(t, nil)
}

func logosReferenceFreeAlong(t Type, expanding []*TypeParam) bool {
	if tp, ok := Unalias(t).(*TypeParam); ok {
		if slices.Contains(expanding, tp) {
			return false
		}
		expanding = append(expanding, tp)
		return tp.is(func(tm *term) bool {
			return tm != nil && logosReferenceFreeAlong(tm.typ, expanding)
		})
	}
	switch t := t.Underlying().(type) {
	case *Basic:
		return t.Kind() != UnsafePointer
	case *Array:
		return logosReferenceFreeAlong(t.Elem(), expanding)
	case *Struct:
		for i := range t.NumFields() {
			if !logosReferenceFreeAlong(t.Field(i).Type(), expanding) {
				return false
			}
		}
		return true
	case *Option:
		return logosReferenceFreeAlong(t.Elem, expanding)
	}
	return false
}

func LogosAliasFreeReturn(value, result Type) bool {
	if result != nil && logosAsConstView(result) != nil {
		return true
	}
	return value != nil && logosReferenceFree(value)
}

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

func (check *Checker) logosConstAssign(x *operand, T Type, context string) bool {
	if !check.conf.Logos {
		return false
	}
	if !strings.HasPrefix(context, "argument to ") {
		check.logosConstOrigin(x)
	}
	if check.logosReadonlyBoundary(x, T, context) {
		return true
	}
	if csrc0 := logosAsConstView(x.typ()); csrc0 == nil && logosAsConstView(T) == nil {
		if qiface := logosQualifiedIface(T); qiface != nil && x.isValid() && !x.isNil() && T != nil {
			if Vu := Unalias(x.typ()); Vu != nil && !isTypeParam(Vu) {
				var cause string
				ok := false
				if Ti := logosPropertyInterface(T); Ti != nil {
					ok = check.logosImplementsProperty(Vu, T, Ti, &cause, x.Pos())
				} else {
					ok = check.hasAllMethodsAt(Vu, T, true, Identical, &cause, x.Pos())
				}
				if !ok {
					if cause != "" {
						check.errorf(x, InvalidIfaceAssign, "cannot use %s as %s value in %s: %s", x, T, context, cause)
					} else {
						check.errorf(x, InvalidIfaceAssign, "cannot use %s as %s value in %s", x, T, context)
					}
					x.invalidate()
				}
				return true
			}
		}
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
	if logosReferenceFree(csrc.Elem) || logosConstMapCopy(csrc) {
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

func (check *Checker) logosConstInferVar(lhs *Var) {
	if !check.conf.Logos {
		return
	}
	csrc := logosAsConstView(lhs.typ)
	if csrc == nil {
		return
	}
	switch csrc.Elem.Underlying().(type) {
	case *Struct, *Array:
		lhs.typ = logosWithReadonlyDescendants(csrc.Elem)
	}
	if logosReferenceFree(csrc.Elem) || logosConstMapCopy(csrc) {
		lhs.typ = csrc.Elem
	}
}

func logosConstMapCopy(c *ConstView) bool {
	_, ok := c.Elem.Underlying().(*Map)
	return c.ConstMap && ok
}

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
		if check.logosConstCovariant(x, cdst, context) {
			return true
		}
		check.constWrapError(x, T, context)
		return true
	}
	return true
}

func (check *Checker) logosConstCovariant(x *operand, cdst *ConstView, context string) bool {
	if !strings.HasPrefix(context, "argument to ") {
		return false
	}
	iface := logosCovariantTarget(x.typ(), cdst.Elem, func(c, i Type) bool {
		return check.logosElemImplementsChecked(c, i, x.Pos())
	})
	if iface == nil || !check.logosNameable(iface) {
		return false
	}
	if _, spread := check.Info.TypeOf(x.expr).(*Tuple); spread {
		return false
	}
	if check.Info.LogosConstCov == nil {
		check.Info.LogosConstCov = make(map[syntax.Expr]Type)
	}
	check.Info.LogosConstCov[x.expr] = iface
	return true
}

func logosCovariantTarget(src, dst Type, implements func(c, i Type) bool) Type {
	if logosAsConstView(src) != nil {
		return nil
	}
	s := Unalias(src)
	if s == nil {
		return nil
	}
	from, _ := s.Underlying().(*Slice)
	to, _ := Unalias(dst).(*Slice)
	if from == nil || to == nil {
		return nil
	}
	c, i := from.Elem(), to.Elem()
	if Identical(c, i) || !implements(c, i) {
		return nil
	}
	return i
}

func (check *Checker) logosNameable(t Type) bool {
	var obj *TypeName
	var targs *TypeList
	switch t := t.(type) {
	case *Alias:
		obj, targs = t.Obj(), t.TypeArgs()
	case *Named:
		obj, targs = t.Obj(), t.TypeArgs()
	default:
		return true
	}
	if obj.pkg != nil && obj.pkg != check.pkg && !obj.Exported() {
		return false
	}
	for i := range targs.Len() {
		if !check.logosNameable(targs.At(i)) {
			return false
		}
	}
	return true
}

func logosElemTarget(c, i Type) *Interface {
	if isTypeParam(i) || logosAsConstView(c) != nil || logosAsConstView(i) != nil {
		return nil
	}
	target, _ := i.Underlying().(*Interface)
	return target
}

func logosElemImplements(c, i Type) bool {
	target := logosElemTarget(c, i)
	return target != nil && Implements(c, target)
}

func (check *Checker) logosElemImplementsChecked(c, i Type, pos syntax.Pos) bool {
	target := logosElemTarget(c, i)
	if target == nil {
		return false
	}
	if logosPropertyInterface(i) != nil {
		return check.logosImplementsProperty(c, i, target, nil, pos)
	}
	return check.hasAllMethodsAt(c, target, true, Identical, nil, pos)
}

func (check *Checker) logosCovSliceCall(call *syntax.CallExpr, args []*operand) {
	inst, _ := syntax.Unparen(call.Fun).(*syntax.IndexExpr)
	if inst == nil || len(args) != 1 || !args[0].isValid() {
		return
	}
	var name *syntax.Name
	switch f := syntax.Unparen(inst.X).(type) {
	case *syntax.Name:
		name = f
	case *syntax.SelectorExpr:
		name = f.Sel
	default:
		return
	}
	fn, _ := check.Info.Uses[name].(*Func)
	if fn == nil || fn.Pkg() == nil || fn.Pkg().Path() != "logos/rt" || fn.Name() != "CovSlice" {
		return
	}
	first := inst.Index
	var explicitElem Type
	if list, ok := first.(*syntax.ListExpr); ok {
		if len(list.ElemList) == 0 {
			return
		}
		first = list.ElemList[0]
		if len(list.ElemList) > 1 {
			explicitElem = check.Info.TypeOf(list.ElemList[1])
		}
	}
	i := check.Info.TypeOf(first)
	from, _ := args[0].typ().Underlying().(*Slice)
	if i == nil {
		return
	}
	if target, _ := i.Underlying().(*Interface); isTypeParam(i) || target == nil {
		check.errorf(args[0], IncompatibleAssign, "rt.CovSlice target %s is not an interface", i)
		return
	}
	elem := explicitElem
	if from != nil {
		elem = from.Elem()
	}
	if elem != nil && !check.logosElemImplementsChecked(elem, i, args[0].Pos()) {
		check.errorf(args[0], IncompatibleAssign, "rt.CovSlice cannot convert %s to []%s: element type %s does not implement interface %s", args[0].typ(), i, elem, i)
	}
}

func (check *Checker) constWrapError(x *operand, T Type, context string) {
	check.errorf(x, IncompatibleAssign, "%s cannot use %s as %s value in %s", logosConstEscape, x, T, context)
	x.invalidate()
}

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
		base, _ = check.logosCompositeRoot(lhs)
	}
	if base == nil {
		return false
	}
	check.errorf(base, IncompatibleAssign, "%s cannot %s through const view %s", logosConstMutation, op, ExprString(base))
	return true
}

func (check *Checker) logosConstStoreBase(lhs syntax.Expr) syntax.Expr {
	switch e := syntax.Unparen(lhs).(type) {
	case *syntax.IndexExpr:
		if logosIndexStoreProtected(check.Info.TypeOf(e.X)) {
			if inner := check.logosConstStoreBase(e.X); inner != nil {
				return inner
			}
			return e.X
		}
		return check.logosConstStoreBase(e.X)
	case *syntax.SelectorExpr:
		if check.logosFieldStoreProtected(check.Info.TypeOf(e.X), e.Sel.Value) {
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
		if logosDerefStoreProtected(check.Info.TypeOf(e.X)) {
			return e.X
		}
		return check.logosConstStoreBase(e.X)
	}
	return nil
}

func (check *Checker) logosCompositeRoot(e syntax.Expr) (syntax.Expr, bool) {
	root, _, _ := check.logosWriteBase(e)
	if !check.logosCompositeRootNode(root) {
		return nil, false
	}
	return root, true
}

func (check *Checker) logosConstBase(e syntax.Expr) (base syntax.Expr, viaReference, constRooted bool) {
	base, _, viaReference = check.logosWriteBase(e)
	if n, ok := base.(*syntax.Name); ok {
		if _, ok := check.Info.Uses[n].(*Const); ok {
			return base, viaReference, true
		}
	}
	return base, viaReference, check.logosCompositeRootNode(base)
}

func (check *Checker) logosConstSliceArray(e *syntax.SliceExpr) bool {
	if !check.conf.Logos {
		return false
	}
	base, _, ok := check.logosConstBase(e.X)
	if !ok {
		return false
	}
	check.errorf(e, IncompatibleAssign, "%s cannot slice an array through const view %s", logosConstMutation, ExprString(base))
	return true
}

func (check *Checker) logosCompositeRootNode(e syntax.Expr) bool {
	switch n := e.(type) {
	case *syntax.Name:
		switch obj := check.Info.Uses[n].(type) {
		case *Const:
			v, ok := check.logosRecorded(obj)
			return ok && v.IsComposite()
		case *Var:
			return check.logosImportedComposite(obj)
		}
	case *syntax.SelectorExpr:
		obj, _ := check.Info.Uses[n.Sel].(*Var)
		return obj != nil && check.logosQualified(n) && check.logosImportedComposite(obj)
	}
	return false
}

func (check *Checker) logosImportedComposite(obj *Var) bool {
	return obj.Pkg() != nil && obj.Pkg() != check.pkg && !obj.IsField() &&
		obj.Pkg().Scope().Lookup(obj.Name()) == obj &&
		LogosImportedConstComposite != nil && LogosImportedConstComposite(obj.Pkg().Path(), obj.Name())
}

func (check *Checker) logosConstAddressable(x *operand, e *syntax.SelectorExpr) bool {
	if !check.conf.Logos || x.mode() == typexpr {
		return false
	}
	_, _, constRooted := check.logosConstBase(e.X)
	return constRooted
}

func (check *Checker) logosCompositeCallProven(e *syntax.SelectorExpr, fn *Func) bool {
	sig, _ := fn.Type().(*Signature)
	if sig == nil || sig.Recv() == nil {
		return true
	}
	rt := sig.Recv().Type()
	if _, isInterface := Unalias(rt).Underlying().(*Interface); isInterface {
		return true
	}
	if logosAsConstView(rt) != nil || logosReferenceFree(rt) {
		return true
	}
	if _, pointer := Unalias(rt).(*Pointer); !pointer && !check.logosValueReceiverGuarded(e.X) {
		return true
	}
	if check.logosLocalValueReceiverSafe(e, sig) {
		return true
	}
	if fn.Origin().Pkg() == check.pkg {
		check.logosDeferReadonlyCheck(fn, e.Sel.Pos(), e.X, e.Sel.Value)
		return true
	}
	return check.logosReceiverReadonly(fn)
}

func (check *Checker) logosValueReceiverGuarded(e syntax.Expr) bool {
	root, composite := check.logosCompositeRoot(e)
	if !composite {
		return false
	}
	if n, ok := root.(*syntax.Name); ok {
		if obj, ok := check.Info.Uses[n].(*Const); ok {
			return !check.logosIsBareMap(obj)
		}
	}
	return true
}

func (check *Checker) logosLocalValueReceiverSafe(e *syntax.SelectorExpr, sig *Signature) bool {
	rt := sig.Recv().Type()
	if _, pointer := Unalias(rt).(*Pointer); pointer || logosTypeHasMap(rt) || !logosReferenceFreeResults(sig.Results()) {
		return false
	}
	root, _, rooted := check.logosConstBase(e.X)
	if !rooted {
		return false
	}
	n, ok := root.(*syntax.Name)
	if !ok {
		return false
	}
	obj, ok := check.Info.Uses[n].(*Const)
	if !ok || obj.Pkg() != check.pkg {
		return false
	}
	v, ok := check.logosRecorded(obj)
	if !ok {
		return false
	}
	t := check.TypeOf(e.X)
	if base := logosAsConstView(t); base != nil {
		t = base.Elem
	}
	method, index, _ := lookupFieldOrMethod(t, false, check.pkg, e.Sel.Value, false)
	if _, ok := method.(*Func); !ok {
		return false
	}
	path := logosReceiverPath{expr: e, indices: index[:len(index)-1]}
	return check.logosReceiverValueSafe(e.X, n, v, &path)
}

func (check *Checker) logosCompositeMutatingCall(e *syntax.SelectorExpr, fn *Func) bool {
	if !check.conf.Logos {
		return false
	}
	base, _, constRooted := check.logosConstBase(e.X)
	if !constRooted || check.logosCompositeCallProven(e, fn) {
		return false
	}
	check.errorf(e.Sel, IncompatibleAssign, "%s cannot call %s through const view %s", logosConstMutation, e.Sel.Value, ExprString(base))
	return true
}

func (check *Checker) logosIsBareMap(obj *Const) bool {
	v, ok := check.Info.LogosCTFE[obj]
	return ok && v.Type() != nil && v.Type().Kind == ctfe.KMap
}

func (check *Checker) logosInConstInit() bool {
	return check.iota != nil
}

func (check *Checker) logosCompositeWrite(x *operand, id builtinId, args []*operand) bool {
	if !check.conf.Logos {
		return false
	}
	switch id {
	case _Append, _Copy, _Clear, _Delete:
	default:
		return false
	}
	if len(args) == 0 || id == _Append && check.logosInConstInit() {
		return false
	}
	root, ok := check.logosCompositeRoot(args[0].expr)
	if !ok {
		return false
	}
	check.errorf(args[0], IncompatibleAssign, "%s cannot %s through const view %s", logosConstMutation, predeclaredFuncs[id].name, ExprString(root))
	x.invalidate()
	return true
}

func (check *Checker) logosConstSelect(x *operand, e *syntax.SelectorExpr, sel string) bool {
	if !check.conf.Logos || x.mode() == typexpr {
		return false
	}
	check.logosConstOrigin(x)
	base := logosAsConstView(x.typ())
	if base == nil {
		return false
	}
	obj, index, indirect := lookupFieldOrMethod(base.Elem, x.mode() == variable || check.logosConstAddressable(x, e), check.pkg, sel, false)
	if obj == nil {
		if logosReferenceFree(base.Elem) {
			x.typ_ = base.Elem
		}
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
		if _, _, rooted := check.logosConstBase(e.X); rooted {
			x.typ_ = base.Elem
			return false
		}
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
		if _, isIface := Unalias(base.Elem).Underlying().(*Interface); isIface {
			if logosIfaceMethodReadonly(obj) {
				x.typ_ = base.Elem
				return false
			}
			check.errorf(e.Sel, IncompatibleAssign, "%s cannot call %s through const view %s", logosConstMutation, sel, ExprString(e.X))
			x.invalidate()
			return true
		}
		rt := sig.Recv().Type()
		if logosAsConstView(rt) != nil {
			x.typ_ = base.Elem
			return false
		}
		if logosReferenceFree(rt) {
			x.typ_ = base.Elem
			return false
		}
		if check.logosLocalValueReceiverSafe(e, sig) {
			x.typ_ = base.Elem
			return false
		}
		if obj.Origin().Pkg() != check.pkg {
			if check.logosReceiverReadonly(obj) {
				x.typ_ = base.Elem
				return false
			}
			check.errorf(e.Sel, IncompatibleAssign, "%s cannot call %s through const view %s", logosConstMutation, sel, ExprString(e.X))
			x.invalidate()
			return true
		}
		check.logosDeferReadonlyCheck(obj, e.Sel.Pos(), e.X, sel)
		x.typ_ = base.Elem
		return false
	}
	return false
}

func (check *Checker) logosConstUnwrapBase(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	if c := logosAsConstView(x.typ()); c != nil {
		x.typ_ = c.Elem
	}
}

func (check *Checker) logosConstOrigin(x *operand) {
	if !check.conf.Logos || check.logosInConstInit() || !x.isValid() || x.expr == nil || logosAsConstView(x.typ()) != nil || logosReferenceFree(x.typ()) || check.logosBareConstMapCopy(x) {
		return
	}
	if _, fn := x.typ().Underlying().(*Signature); fn {
		return
	}
	if _, _, rooted := check.logosConstBase(x.expr); rooted {
		x.typ_ = &ConstView{Elem: x.typ()}
	}
}

func (check *Checker) logosBareConstMapCopy(x *operand) bool {
	if c := logosAsConstView(check.Info.TypeOf(x.expr)); c != nil && c.ConstMap {
		return true
	}
	base, _, _ := check.logosWriteBase(x.expr)
	name, ok := base.(*syntax.Name)
	if !ok {
		return false
	}
	obj, ok := check.Info.Uses[name].(*Const)
	return ok && check.logosIsBareMap(obj)
}

func (check *Checker) logosConstWrapBase(x *operand, base syntax.Expr) {
	if !check.conf.Logos || !x.isValid() || x.mode() == mapindex {
		return
	}
	if logosAsConstView(check.Info.TypeOf(syntax.Unparen(base))) == nil {
		if _, _, rooted := check.logosConstBase(base); !rooted {
			return
		}
	}
	if logosAsConstView(x.typ()) != nil {
		return
	}
	x.typ_ = &ConstView{Elem: x.typ()}
}

func (check *Checker) logosConstAddr(x *operand) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	if c := logosAsConstView(x.typ()); c != nil {
		x.typ_ = &ConstView{Elem: &Pointer{base: c.Elem}}
		return true
	}
	return false
}

func (check *Checker) logosConstNumeric(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	if c := logosAsConstView(x.typ()); c != nil && logosReferenceFree(c.Elem) {
		x.typ_ = c.Elem
		check.updateExprType(x.expr, c.Elem, false)
	}
}

func (check *Checker) logosConstBinary(x, y *operand) {
	check.logosConstNumeric(x)
	check.logosConstNumeric(y)
}

func (check *Checker) logosConstConvert(x *operand, T Type) bool {
	if !check.conf.Logos || !x.isValid() {
		return false
	}
	check.logosConstOrigin(x)
	if check.logosReadonlyBoundary(x, T, "conversion") {
		return true
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

func (check *Checker) logosConstCopyElements(a *operand) bool {
	c := logosAsConstView(a.typ())
	if c == nil {
		return false
	}
	source := *a
	source.typ_ = c.Elem
	elem, err := sliceElem(&source)
	if err != nil {
		return false
	}
	return check.logosConstCopyReference(a, elem)
}

func (check *Checker) logosConstCopyReference(a *operand, copied Type) bool {
	c := logosAsConstView(a.typ())
	if c == nil || logosReferenceFree(copied) || logosConstMapCopy(c) {
		return false
	}
	check.errorf(a, IncompatibleAssign, "%s cannot copy reference-bearing elements through const view %s", logosConstMutation, ExprString(a.expr))
	return true
}

func (check *Checker) logosConstBuiltinCopySources(call *syntax.CallExpr, id builtinId, args []*operand) bool {
	if (id != _Copy && id != _Append) || len(args) < 2 {
		return false
	}
	for _, a := range args[1:] {
		if id == _Copy || hasDots(call) {
			if check.logosConstCopyElements(a) {
				return true
			}
			continue
		}
		if c := logosAsConstView(a.typ()); c != nil && check.logosConstCopyReference(a, c.Elem) {
			return true
		}
	}
	return false
}

func (check *Checker) logosConstBuiltin(x *operand, call *syntax.CallExpr, id builtinId, args []*operand) bool {
	if !check.conf.Logos {
		return false
	}
	if check.logosCompositeWrite(x, id, args) {
		return true
	}
	if check.logosConstBuiltinCopySources(call, id, args) {
		x.invalidate()
		return true
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
	case _Len:
		for _, a := range args {
			unwrap(a)
		}
		if c := logosAsConstView(x.typ()); c != nil {
			x.typ_ = c.Elem
		}
		if len(args) > 0 {
			if n, ok := check.logosConstMapLen(args[0]); ok {
				x.mode_ = constant_
				x.typ_ = Typ[Int]
				x.val = constant.MakeInt64(n)
				return true
			}
		}
		return false
	case _Append:
		if len(args) > 0 {
			if constView(args[0]) {
				check.errorf(args[0], IncompatibleAssign, "%s cannot append through const view %s", logosConstMutation, ExprString(args[0].expr))
				x.invalidate()
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
				x.invalidate()
				return true
			}
			for _, a := range args[1:] {
				unwrap(a)
			}
		}
	case _Delete:
		if len(args) > 0 && constView(args[0]) {
			check.errorf(args[0], IncompatibleAssign, "%s cannot delete through const view %s", logosConstMutation, ExprString(args[0].expr))
			x.invalidate()
			return true
		}
	case _Clear:
		if len(args) > 0 && constView(args[0]) {
			check.errorf(args[0], IncompatibleAssign, "%s cannot clear through const view %s", logosConstMutation, ExprString(args[0].expr))
			x.invalidate()
			return true
		}
	case _Close:
		if len(args) > 0 && constView(args[0]) {
			check.errorf(args[0], IncompatibleAssign, "%s cannot close through const view %s", logosConstMutation, ExprString(args[0].expr))
			x.invalidate()
			return true
		}
	case _Panic:
		return false
	case _SliceData, _Slice, _StringData, _String, _Add:
		return check.logosConstUnsafeDeriv(x, call, id, args)
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

func logosConstUnsafeData(a *operand) Type {
	c := logosAsConstView(a.typ())
	if c == nil {
		return nil
	}
	return c.Elem
}

func (check *Checker) logosConstUnsafeLen(a *operand) {
	if c := logosAsConstView(a.typ()); c != nil && logosReferenceFree(c.Elem) {
		a.typ_ = c.Elem
		check.updateExprType(a.expr, c.Elem, false)
	}
}

func (check *Checker) logosUnwrapUnsafeData(x *operand, args []*operand) Type {
	data := logosConstUnsafeData(args[0])
	args[0].typ_ = data
	check.updateExprType(args[0].expr, data, false)
	x.typ_ = data
	return data
}

func (check *Checker) logosWrapUnsafeResult(x *operand, call *syntax.CallExpr, elem Type, params ...Type) {
	x.typ_ = &ConstView{Elem: elem}
	x.mode_ = value
	if check.recordTypes() {
		check.recordBuiltinType(call.Fun, makeSig(x.typ(), params...))
	}
}

func (check *Checker) logosConstUnsafeDeriv(x *operand, call *syntax.CallExpr, id builtinId, args []*operand) bool {
	nargs := 1
	switch id {
	case _Slice, _String, _Add:
		nargs = 2
	case _SliceData, _StringData:
	default:
		return false
	}
	if len(args) < nargs {
		return false
	}
	if nargs == 2 {
		check.logosConstUnsafeLen(args[1])
	}
	check.logosConstOrigin(args[0])
	if logosAsConstView(args[0].typ()) == nil && args[0].isValid() {
		if _, _, rooted := check.logosConstBase(args[0].expr); rooted {
			args[0].typ_ = &ConstView{Elem: args[0].typ()}
		}
	}
	if logosConstUnsafeData(args[0]) == nil {
		return false
	}
	switch id {
	case _SliceData:
		return check.logosConstUnsafeSliceData(x, call, args)
	case _Slice:
		return check.logosConstUnsafeSlice(x, call, args)
	case _StringData:
		return check.logosConstUnsafeStringData(x, call, args)
	case _String:
		return check.logosConstUnsafeString(x, call, args)
	case _Add:
		return check.logosConstUnsafeAdd(x, call, args)
	}
	return false
}

func (check *Checker) logosConstUnsafeSliceData(x *operand, call *syntax.CallExpr, args []*operand) bool {
	data := check.logosUnwrapUnsafeData(x, args)
	u, _ := commonUnder(data, nil)
	sl, _ := u.(*Slice)
	if sl == nil {
		return false
	}
	check.verifyVersionf(call.Fun, go1_20, "unsafe.SliceData")
	check.logosWrapUnsafeResult(x, call, NewPointer(sl.elem), data)
	return true
}

func (check *Checker) logosConstUnsafeSlice(x *operand, call *syntax.CallExpr, args []*operand) bool {
	data := check.logosUnwrapUnsafeData(x, args)
	u, _ := commonUnder(data, nil)
	ptr, _ := u.(*Pointer)
	if ptr == nil {
		return false
	}
	check.verifyVersionf(call.Fun, go1_17, "unsafe.Slice")
	if !check.isValidIndex(args[1], InvalidUnsafeSlice, "length", false) {
		x.invalidate()
		return true
	}
	check.logosWrapUnsafeResult(x, call, NewSlice(ptr.base), data, args[1].typ())
	return true
}

func (check *Checker) logosConstUnsafeStringData(x *operand, call *syntax.CallExpr, args []*operand) bool {
	data := check.logosUnwrapUnsafeData(x, args)
	check.verifyVersionf(call.Fun, go1_20, "unsafe.StringData")
	check.assignment(x, Typ[String], "argument to unsafe.StringData")
	if !x.isValid() {
		return true
	}
	check.logosWrapUnsafeResult(x, call, NewPointer(universeByte), data)
	return true
}

func (check *Checker) logosConstUnsafeString(x *operand, call *syntax.CallExpr, args []*operand) bool {
	check.logosUnwrapUnsafeData(x, args)
	check.verifyVersionf(call.Fun, go1_20, "unsafe.String")
	check.assignment(x, NewPointer(universeByte), "argument to unsafe.String")
	if !x.isValid() {
		return true
	}
	if !check.isValidIndex(args[1], InvalidUnsafeString, "length", false) {
		x.invalidate()
		return true
	}
	check.logosWrapUnsafeResult(x, call, Typ[String], NewPointer(universeByte), args[1].typ())
	return true
}

func (check *Checker) logosConstUnsafeAdd(x *operand, call *syntax.CallExpr, args []*operand) bool {
	check.logosUnwrapUnsafeData(x, args)
	check.verifyVersionf(call.Fun, go1_17, "unsafe.Add")
	check.assignment(x, Typ[UnsafePointer], "argument to unsafe.Add")
	if !x.isValid() {
		return true
	}
	if !check.isValidIndex(args[1], InvalidUnsafeAdd, "length", true) {
		x.invalidate()
		return true
	}
	check.logosWrapUnsafeResult(x, call, Typ[UnsafePointer], Typ[UnsafePointer], args[1].typ())
	return true
}

func (check *Checker) logosConstMapLen(a *operand) (int64, bool) {
	if !a.isValid() || a.expr == nil {
		return 0, false
	}
	switch e := syntax.Unparen(a.expr).(type) {
	case *syntax.Name:
		if obj, ok := check.Info.Uses[e].(*Const); ok {
			return check.logosLocalConstMapLen(obj)
		}
		if obj, ok := check.Info.Uses[e].(*Var); ok {
			return check.logosImportedConstMapLen(obj)
		}
	case *syntax.SelectorExpr:
		if e.Sel == nil {
			return 0, false
		}
		if obj, ok := check.Info.Uses[e.Sel].(*Var); ok {
			return check.logosImportedConstMapLen(obj)
		}
	}
	return 0, false
}

func (check *Checker) logosLocalConstMapLen(obj *Const) (int64, bool) {
	v, ok := check.logosRecorded(obj)
	if !ok || !v.IsValid() || v.Type() == nil || v.Type().Kind != ctfe.KMap {
		return 0, false
	}
	keys, _ := v.MapEntries()
	return int64(len(keys)), true
}

func (check *Checker) logosImportedConstMapLen(obj *Var) (int64, bool) {
	if obj.Pkg() == nil || obj.Pkg() == check.pkg {
		return 0, false
	}
	view := logosAsConstView(obj.Type())
	if view == nil || !view.ConstMap {
		return 0, false
	}
	if LogosImportedConstMapLen == nil {
		return 0, false
	}
	n, ok := LogosImportedConstMapLen(obj.Pkg().Path(), obj.Name())
	if !ok {
		return 0, false
	}
	return int64(n), true
}

var LogosImportedConstMapLen func(path, object string) (int, bool)

var LogosImportedConstComposite func(path, object string) bool

func (check *Checker) logosInferOperands(par, arg *Type) {
	if c := logosAsConstView(*par); c != nil {
		*par = c.Elem
	}
	if c := logosAsConstView(*arg); c != nil {
		*arg = c.Elem
	}
}

var LogosImportedParamProvenZero func(path, object string, param int) bool

type logosGoConstGate struct {
	active   bool
	variadic bool
	methodEx bool
	fn       *Func
	sig      *Signature
}

func (check *Checker) logosGoConstGate(call *syntax.CallExpr) logosGoConstGate {
	var g logosGoConstGate
	if !check.logosCalleeIsGo(call) {
		return g
	}
	g.active = true
	fn := check.logosFuncOfExpr(call.Fun)
	if fn == nil || fn.Pkg() == nil || fn.Pkg() == check.pkg {
		return g
	}
	sig, ok := fn.Type().(*Signature)
	if !ok || sig == nil {
		return g
	}
	g.fn = fn
	g.sig = sig
	g.variadic = sig.Variadic()
	if sig.Recv() != nil {
		if sel, ok := syntax.Unparen(call.Fun).(*syntax.SelectorExpr); ok {
			if check.Info != nil {
				if s, ok := check.Selections[sel]; ok && s != nil && s.Kind() == MethodExpr {
					g.methodEx = true
				}
			}
		}
	}
	return g
}
func (check *Checker) logosGoParamSlot(g logosGoConstGate, sigParams *Tuple, index int) int {
	if g.sig != nil {
		npars := g.sig.Params().Len()
		if g.variadic && index >= npars-1 {
			return npars - 1
		}
		if index < npars {
			return index
		}
	}
	if g.variadic && index >= sigParams.Len()-1 {
		return sigParams.Len() - 1
	}
	return index
}

func (check *Checker) logosGoParamName(g logosGoConstGate, sigParams *Tuple, index int) string {
	if v := sigParams.At(check.logosGoParamSlot(g, sigParams, index)).Name(); v != "" {
		return v
	}
	return "#" + strconv.Itoa(index)
}

func logosConstMethodObject(fn *Func) string {
	sig, ok := fn.Type().(*Signature)
	if !ok || sig == nil || sig.Recv() == nil {
		return fn.Name()
	}
	t := Unalias(sig.Recv().Type())
	if c, ok := t.(*ConstView); ok {
		t = Unalias(c.Elem)
	}
	if p, ok := t.(*Pointer); ok {
		t = Unalias(p.Elem())
		if c, ok := t.(*ConstView); ok {
			t = Unalias(c.Elem)
		}
	}
	n, ok := t.(*Named)
	if !ok {
		return fn.Name()
	}
	if _, ok := Unalias(sig.Recv().Type()).(*Pointer); ok {
		return "(*" + n.Obj().Name() + ")." + fn.Name()
	}
	return n.Obj().Name() + "." + fn.Name()
}

func (check *Checker) logosGoParamProvenZero(g logosGoConstGate, sigParams *Tuple, index int) bool {
	if !g.active || g.fn == nil || g.methodEx || LogosImportedParamProvenZero == nil {
		return false
	}
	object := g.fn.Name()
	if g.sig.Recv() != nil {
		object = logosConstMethodObject(g.fn)
	}
	return LogosImportedParamProvenZero(g.fn.Pkg().Path(), object, check.logosGoParamSlot(g, sigParams, index))
}

func (check *Checker) logosConstArgs(call *syntax.CallExpr, sigParams *Tuple, args []*operand) {
	if !check.conf.Logos {
		return
	}
	check.logosCovSliceCall(call, args)
	if name, _ := call.Fun.(*syntax.Name); name != nil {
		if _, ok := check.lookup(name.Value).(*Builtin); ok {
			return
		}
	}
	gate := check.logosGoConstGate(call)
	for i, a := range args {
		if sigParams == nil || sigParams.Len() == 0 {
			break
		}
		j := i
		if j >= sigParams.Len() {
			if !gate.variadic {
				break
			}
			j = sigParams.Len() - 1
		}
		if !a.isValid() {
			continue
		}
		check.logosConstOrigin(a)
		c := logosAsConstView(a.typ())
		if c == nil {
			continue
		}
		pt := sigParams.At(j).Type()
		if gate.variadic && i > j {
			if s, ok := pt.(*Slice); ok {
				pt = s.Elem()
			}
		}
		if cp := logosAsConstView(pt); cp != nil && logosCovariantTarget(c.Elem, cp.Elem, logosElemImplements) != nil {
			check.errorf(a, IncompatibleAssign, "%s cannot pass const view %s as %s in argument to %s: its elements would convert through an interface", logosConstEscape, ExprString(a.expr), pt, ExprString(call.Fun))
			a.invalidate()
			continue
		}
		if logosAsConstView(pt) != nil {
			a.typ_ = c.Elem
			check.updateExprType(a.expr, c.Elem, false)
			continue
		}
		if logosReferenceFree(c.Elem) {
			a.typ_ = c.Elem
			check.updateExprType(a.expr, c.Elem, false)
			continue
		}
		if gate.active {
			if check.logosObservedConstArg(call, a, c) {
				a.typ_ = c.Elem
				check.updateExprType(a.expr, c.Elem, false)
				continue
			}
			if check.logosGoParamProvenZero(gate, sigParams, i) {
				if gate.sig == nil || !logosReferenceFreeResults(gate.sig.Results()) {
					check.errorf(a, IncompatibleAssign, "%s cannot pass const view %s to %s: non-mutation does not prove result alias independence", logosConstEscape, ExprString(a.expr), ExprString(call.Fun))
					a.invalidate()
					continue
				}
				a.typ_ = c.Elem
				check.updateExprType(a.expr, c.Elem, false)
				continue
			}
			check.errorf(a, IncompatibleAssign, "%s cannot pass const view %s to parameter %s of %s", logosConstMutation, ExprString(a.expr), check.logosGoParamName(gate, sigParams, i), ExprString(call.Fun))
			a.invalidate()
			continue
		}
		check.errorf(a, IncompatibleAssign, "%s cannot pass argument through const view %s in argument to %s", logosConstMutation, ExprString(a.expr), ExprString(call.Fun))
		a.invalidate()
	}
}

func logosReferenceFreeResults(results *Tuple) bool {
	for i := range results.Len() {
		if !logosReferenceFree(results.At(i).Type()) {
			return false
		}
	}
	return true
}

var logosStdObservers = map[string]bool{
	"fmt.Print":         true,
	"fmt.Printf":        true,
	"fmt.Println":       true,
	"fmt.Sprint":        true,
	"fmt.Sprintf":       true,
	"fmt.Sprintln":      true,
	"reflect.DeepEqual": true,
}

var logosFormatMethods = [...]string{"Error", "Format", "GoString", "String"}

func (check *Checker) logosObservedConstArg(call *syntax.CallExpr, a *operand, c *ConstView) bool {
	path, name, ok := logosImportedCallee(check, call)
	if !ok || !logosStdObservers[path+"."+name] {
		return false
	}
	if _, _, rooted := check.logosConstBase(a.expr); !rooted {
		return false
	}
	return path != "fmt" || check.logosFormatReadonly(c.Elem)
}

func (check *Checker) logosFormatReadonly(T Type) bool {
	readonly := true
	seen := make(map[Type]bool)
	var visit func(Type)
	visit = func(t Type) {
		switch u := Unalias(t).(type) {
		case *Named:
			readonly = readonly && check.logosFormatMethodsReadonly(u)
			logosWalkType(u.Underlying(), visit, seen)
		case *Interface, *TypeParam:
			readonly = false
		}
	}
	logosWalkType(T, visit, seen)
	return readonly
}

func (check *Checker) logosFormatMethodsReadonly(n *Named) bool {
	ptr := NewPointer(n)
	for _, name := range logosFormatMethods {
		obj, _, _ := lookupFieldOrMethod(ptr, false, check.pkg, name, false)
		if fn, ok := obj.(*Func); ok && !check.logosReceiverReadonly(fn) {
			return false
		}
	}
	return true
}

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

func (check *Checker) logosUnpackConstRecv(rtyp syntax.Expr) syntax.Expr {
	if !check.conf.Logos {
		return rtyp
	}
	if t, _ := syntax.Unparen(rtyp).(*syntax.ConstType); t != nil {
		return t.Elem
	}
	return rtyp
}

func (check *Checker) logosConstRange(x *operand) {
	if !check.conf.Logos || !x.isValid() {
		return
	}
	x.typ_, _, _ = logosReadonlyBase(x.typ())
}

func (check *Checker) logosConstRangeVal(rangeVar syntax.Expr, valp *Type) {
	if !check.conf.Logos || *valp == nil || !isValid(*valp) {
		return
	}
	_, descendants, readonly := logosReadonlyBase(check.Info.TypeOf(rangeVar))
	if descendants {
		*valp = logosWithReadonlyDescendants(*valp)
		return
	}
	if !readonly {
		if _, _, rooted := check.logosConstBase(rangeVar); !rooted {
			return
		}
	}
	*valp = &ConstView{Elem: *valp}
}

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
