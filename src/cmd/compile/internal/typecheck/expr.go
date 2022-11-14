// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"fmt"
	"go/constant"
	"go/token"
	"strings"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// tcAddr typechecks an OADDR node.
func tcAddr(n *ir.AddrExpr) ir.Node {
	n.X = Expr(n.X)
	if n.X.Type() == nil {
		n.SetType(nil)
		return n
	}

	switch n.X.Op() {
	case ir.OARRAYLIT, ir.OMAPLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
		n.SetOp(ir.OPTRLIT)

	default:
		checklvalue(n.X, "take the address of")
		r := ir.OuterValue(n.X)
		if r.Op() == ir.ONAME {
			r := r.(*ir.Name)
			if ir.Orig(r) != r {
				base.Fatalf("found non-orig name node %v", r) // TODO(mdempsky): What does this mean?
			}
		}
		n.X = DefaultLit(n.X, nil)
		if n.X.Type() == nil {
			n.SetType(nil)
			return n
		}
	}

	n.SetType(types.NewPtr(n.X.Type()))
	return n
}

func tcShift(n, l, r ir.Node) (ir.Node, ir.Node, *types.Type) {
	if l.Type() == nil || r.Type() == nil {
		return l, r, nil
	}

	r = DefaultLit(r, types.Types[types.TUINT])
	t := r.Type()
	if !t.IsInteger() {
		base.Errorf("invalid operation: %v (shift count type %v, must be integer)", n, r.Type())
		return l, r, nil
	}
	t = l.Type()
	if t != nil && t.Kind() != types.TIDEAL && !t.IsInteger() {
		base.Errorf("invalid operation: %v (shift of type %v)", n, t)
		return l, r, nil
	}

	// no DefaultLit for left
	// the outer context gives the type
	t = l.Type()
	if (l.Type() == types.UntypedFloat || l.Type() == types.UntypedComplex) && r.Op() == ir.OLITERAL {
		t = types.UntypedInt
	}
	return l, r, t
}

// tcArith typechecks operands of a binary arithmetic expression.
// The result of tcArith MUST be assigned back to original operands,
// t is the type of the expression, and should be set by the caller. e.g:
//
//	n.X, n.Y, t = tcArith(n, op, n.X, n.Y)
//	n.SetType(t)
func tcArith(n ir.Node, op ir.Op, l, r ir.Node) (ir.Node, ir.Node, *types.Type) {
	l, r = defaultlit2(l, r, false)
	if l.Type() == nil || r.Type() == nil {
		return l, r, nil
	}
	t := l.Type()
	if t.Kind() == types.TIDEAL {
		t = r.Type()
	}
	aop := ir.OXXX
	if n.Op().IsCmp() && t.Kind() != types.TIDEAL && !types.Identical(l.Type(), r.Type()) {
		// comparison is okay as long as one side is
		// assignable to the other.  convert so they have
		// the same type.
		//
		// the only conversion that isn't a no-op is concrete == interface.
		// in that case, check comparability of the concrete type.
		// The conversion allocates, so only do it if the concrete type is huge.
		converted := false
		if r.Type().Kind() != types.TBLANK {
			aop, _ = Assignop(l.Type(), r.Type())
			if aop != ir.OXXX {
				if r.Type().IsInterface() && !l.Type().IsInterface() && !types.IsComparable(l.Type()) {
					base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(l.Type()))
					return l, r, nil
				}

				types.CalcSize(l.Type())
				if r.Type().IsInterface() == l.Type().IsInterface() || l.Type().Size() >= 1<<16 {
					l = ir.NewConvExpr(base.Pos, aop, r.Type(), l)
					l.SetTypecheck(1)
				}

				t = r.Type()
				converted = true
			}
		}

		if !converted && l.Type().Kind() != types.TBLANK {
			aop, _ = Assignop(r.Type(), l.Type())
			if aop != ir.OXXX {
				if l.Type().IsInterface() && !r.Type().IsInterface() && !types.IsComparable(r.Type()) {
					base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(r.Type()))
					return l, r, nil
				}

				types.CalcSize(r.Type())
				if r.Type().IsInterface() == l.Type().IsInterface() || r.Type().Size() >= 1<<16 {
					r = ir.NewConvExpr(base.Pos, aop, l.Type(), r)
					r.SetTypecheck(1)
				}

				t = l.Type()
			}
		}
	}

	if t.Kind() != types.TIDEAL && !types.Identical(l.Type(), r.Type()) {
		l, r = defaultlit2(l, r, true)
		if l.Type() == nil || r.Type() == nil {
			return l, r, nil
		}
		if l.Type().IsInterface() == r.Type().IsInterface() || aop == 0 {
			base.Errorf("invalid operation: %v (mismatched types %v and %v)", n, l.Type(), r.Type())
			return l, r, nil
		}
	}

	if t.Kind() == types.TIDEAL {
		t = mixUntyped(l.Type(), r.Type())
	}
	if dt := defaultType(t); !okfor[op][dt.Kind()] {
		base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, op, typekind(t))
		return l, r, nil
	}

	// okfor allows any array == array, map == map, func == func.
	// restrict to slice/map/func == nil and nil == slice/map/func.
	if l.Type().IsArray() && !types.IsComparable(l.Type()) {
		base.Errorf("invalid operation: %v (%v cannot be compared)", n, l.Type())
		return l, r, nil
	}

	if l.Type().IsSlice() && !ir.IsNil(l) && !ir.IsNil(r) {
		base.Errorf("invalid operation: %v (slice can only be compared to nil)", n)
		return l, r, nil
	}

	if l.Type().IsMap() && !ir.IsNil(l) && !ir.IsNil(r) {
		base.Errorf("invalid operation: %v (map can only be compared to nil)", n)
		return l, r, nil
	}

	if l.Type().Kind() == types.TFUNC && !ir.IsNil(l) && !ir.IsNil(r) {
		base.Errorf("invalid operation: %v (func can only be compared to nil)", n)
		return l, r, nil
	}

	if l.Type().IsStruct() {
		if f := types.IncomparableField(l.Type()); f != nil {
			base.Errorf("invalid operation: %v (struct containing %v cannot be compared)", n, f.Type)
			return l, r, nil
		}
	}

	if (op == ir.ODIV || op == ir.OMOD) && ir.IsConst(r, constant.Int) {
		if constant.Sign(r.Val()) == 0 {
			base.Errorf("division by zero")
			return l, r, nil
		}
	}

	return l, r, t
}

// The result of tcCompLit MUST be assigned back to n, e.g.
//
//	n.Left = tcCompLit(n.Left)
func tcCompLit(n *ir.CompLitExpr) (res ir.Node) {
	if base.EnableTrace && base.Flag.LowerT {
		defer tracePrint("tcCompLit", n)(&res)
	}

	lno := base.Pos
	defer func() {
		base.Pos = lno
	}()

	// Save original node (including n.Right)
	n.SetOrig(ir.Copy(n))

	ir.SetPos(n)

	t := n.Type()
	base.AssertfAt(t != nil, n.Pos(), "missing type in composite literal")

	switch t.Kind() {
	default:
		base.Errorf("invalid composite literal type %v", t)
		n.SetType(nil)

	case types.TARRAY:
		typecheckarraylit(t.Elem(), t.NumElem(), n.List, "array literal")
		n.SetOp(ir.OARRAYLIT)

	case types.TSLICE:
		length := typecheckarraylit(t.Elem(), -1, n.List, "slice literal")
		n.SetOp(ir.OSLICELIT)
		n.Len = length

	case types.TMAP:
		for i3, l := range n.List {
			ir.SetPos(l)
			if l.Op() != ir.OKEY {
				n.List[i3] = Expr(l)
				base.Errorf("missing key in map literal")
				continue
			}
			l := l.(*ir.KeyExpr)

			r := l.Key
			r = Expr(r)
			l.Key = AssignConv(r, t.Key(), "map key")

			r = l.Value
			r = Expr(r)
			l.Value = AssignConv(r, t.Elem(), "map value")
		}

		n.SetOp(ir.OMAPLIT)

	case types.TSTRUCT:
		// Need valid field offsets for Xoffset below.
		types.CalcSize(t)

		errored := false
		if len(n.List) != 0 && nokeys(n.List) {
			// simple list of variables
			ls := n.List
			for i, n1 := range ls {
				ir.SetPos(n1)
				n1 = Expr(n1)
				ls[i] = n1
				if i >= t.NumFields() {
					if !errored {
						base.Errorf("too many values in %v", n)
						errored = true
					}
					continue
				}

				f := t.Field(i)
				s := f.Sym

				// Do the test for assigning to unexported fields.
				// But if this is an instantiated function, then
				// the function has already been typechecked. In
				// that case, don't do the test, since it can fail
				// for the closure structs created in
				// walkClosure(), because the instantiated
				// function is compiled as if in the source
				// package of the generic function.
				if !(ir.CurFunc != nil && strings.Index(ir.CurFunc.Nname.Sym().Name, "[") >= 0) {
					if s != nil && !types.IsExported(s.Name) && s.Pkg != types.LocalPkg {
						base.Errorf("implicit assignment of unexported field '%s' in %v literal", s.Name, t)
					}
				}
				// No pushtype allowed here. Must name fields for that.
				n1 = AssignConv(n1, f.Type, "field value")
				ls[i] = ir.NewStructKeyExpr(base.Pos, f, n1)
			}
			if len(ls) < t.NumFields() {
				base.Errorf("too few values in %v", n)
			}
		} else {
			hash := make(map[string]bool)

			// keyed list
			ls := n.List
			for i, n := range ls {
				ir.SetPos(n)

				sk, ok := n.(*ir.StructKeyExpr)
				if !ok {
					kv, ok := n.(*ir.KeyExpr)
					if !ok {
						if !errored {
							base.Errorf("mixture of field:value and value initializers")
							errored = true
						}
						ls[i] = Expr(n)
						continue
					}

					sk = tcStructLitKey(t, kv)
					if sk == nil {
						continue
					}

					fielddup(sk.Sym().Name, hash)
				}

				// No pushtype allowed here. Tried and rejected.
				sk.Value = Expr(sk.Value)
				sk.Value = AssignConv(sk.Value, sk.Field.Type, "field value")
				ls[i] = sk
			}
		}

		n.SetOp(ir.OSTRUCTLIT)
	}

	return n
}

// tcStructLitKey typechecks an OKEY node that appeared within a
// struct literal.
func tcStructLitKey(typ *types.Type, kv *ir.KeyExpr) *ir.StructKeyExpr {
	key := kv.Key

	sym := key.Sym()

	// An OXDOT uses the Sym field to hold
	// the field to the right of the dot,
	// so s will be non-nil, but an OXDOT
	// is never a valid struct literal key.
	if sym == nil || sym.Pkg != types.LocalPkg || key.Op() == ir.OXDOT || sym.IsBlank() {
		base.Errorf("invalid field name %v in struct initializer", key)
		return nil
	}

	if f := Lookdot1(nil, sym, typ, typ.Fields(), 0); f != nil {
		return ir.NewStructKeyExpr(kv.Pos(), f, kv.Value)
	}

	if ci := Lookdot1(nil, sym, typ, typ.Fields(), 2); ci != nil { // Case-insensitive lookup.
		if visible(ci.Sym) {
			base.Errorf("unknown field '%v' in struct literal of type %v (but does have %v)", sym, typ, ci.Sym)
		} else if nonexported(sym) && sym.Name == ci.Sym.Name { // Ensure exactness before the suggestion.
			base.Errorf("cannot refer to unexported field '%v' in struct literal of type %v", sym, typ)
		} else {
			base.Errorf("unknown field '%v' in struct literal of type %v", sym, typ)
		}
		return nil
	}

	var f *types.Field
	p, _ := dotpath(sym, typ, &f, true)
	if p == nil || f.IsMethod() {
		base.Errorf("unknown field '%v' in struct literal of type %v", sym, typ)
		return nil
	}

	// dotpath returns the parent embedded types in reverse order.
	var ep []string
	for ei := len(p) - 1; ei >= 0; ei-- {
		ep = append(ep, p[ei].field.Sym.Name)
	}
	ep = append(ep, sym.Name)
	base.Errorf("cannot use promoted field %v in struct literal of type %v", strings.Join(ep, "."), typ)
	return nil
}

// tcConv typechecks an OCONV node.
func tcConv(n *ir.ConvExpr) ir.Node {
	types.CheckSize(n.Type()) // ensure width is calculated for backend
	n.X = Expr(n.X)
	n.X = convlit1(n.X, n.Type(), true, nil)
	t := n.X.Type()
	if t == nil || n.Type() == nil {
		n.SetType(nil)
		return n
	}
	op, why := Convertop(n.X.Op() == ir.OLITERAL, t, n.Type())
	if op == ir.OXXX {
		base.Fatalf("cannot convert %L to type %v%s", n.X, n.Type(), why)
	}

	n.SetOp(op)
	switch n.Op() {
	case ir.OCONVNOP:
		if t.Kind() == n.Type().Kind() {
			switch t.Kind() {
			case types.TFLOAT32, types.TFLOAT64, types.TCOMPLEX64, types.TCOMPLEX128:
				// Floating point casts imply rounding and
				// so the conversion must be kept.
				n.SetOp(ir.OCONV)
			}
		}

	// do not convert to []byte literal. See CL 125796.
	// generated code and compiler memory footprint is better without it.
	case ir.OSTR2BYTES:
		// ok

	case ir.OSTR2RUNES:
		if n.X.Op() == ir.OLITERAL {
			return stringtoruneslit(n)
		}

	case ir.OBYTES2STR:
		if t.Elem() != types.ByteType && t.Elem() != types.Types[types.TUINT8] {
			// If t is a slice of a user-defined byte type B (not uint8
			// or byte), then add an extra CONVNOP from []B to []byte, so
			// that the call to slicebytetostring() added in walk will
			// typecheck correctly.
			n.X = ir.NewConvExpr(n.X.Pos(), ir.OCONVNOP, types.NewSlice(types.ByteType), n.X)
			n.X.SetTypecheck(1)
		}

	case ir.ORUNES2STR:
		if t.Elem() != types.RuneType && t.Elem() != types.Types[types.TINT32] {
			// If t is a slice of a user-defined rune type B (not uint32
			// or rune), then add an extra CONVNOP from []B to []rune, so
			// that the call to slicerunetostring() added in walk will
			// typecheck correctly.
			n.X = ir.NewConvExpr(n.X.Pos(), ir.OCONVNOP, types.NewSlice(types.RuneType), n.X)
			n.X.SetTypecheck(1)
		}

	}
	return n
}

// tcDot typechecks an OXDOT or ODOT node.
func tcDot(n *ir.SelectorExpr, top int) ir.Node {
	if n.Op() == ir.OXDOT {
		n = AddImplicitDots(n)
		n.SetOp(ir.ODOT)
		if n.X == nil {
			n.SetType(nil)
			return n
		}
	}

	n.X = typecheck(n.X, ctxExpr|ctxType)
	n.X = DefaultLit(n.X, nil)

	t := n.X.Type()
	if t == nil {
		base.UpdateErrorDot(ir.Line(n), fmt.Sprint(n.X), fmt.Sprint(n))
		n.SetType(nil)
		return n
	}

	if n.X.Op() == ir.OTYPE {
		return typecheckMethodExpr(n)
	}

	if t.IsPtr() && !t.Elem().IsInterface() {
		t = t.Elem()
		if t == nil {
			n.SetType(nil)
			return n
		}
		n.SetOp(ir.ODOTPTR)
		types.CheckSize(t)
	}

	if n.Sel.IsBlank() {
		base.Errorf("cannot refer to blank field or method")
		n.SetType(nil)
		return n
	}

	if Lookdot(n, t, 0) == nil {
		// Legitimate field or method lookup failed, try to explain the error
		switch {
		case t.IsEmptyInterface():
			base.Errorf("%v undefined (type %v is interface with no methods)", n, n.X.Type())

		case t.IsPtr() && t.Elem().IsInterface():
			// Pointer to interface is almost always a mistake.
			base.Errorf("%v undefined (type %v is pointer to interface, not interface)", n, n.X.Type())

		case Lookdot(n, t, 1) != nil:
			// Field or method matches by name, but it is not exported.
			base.Errorf("%v undefined (cannot refer to unexported field or method %v)", n, n.Sel)

		default:
			if mt := Lookdot(n, t, 2); mt != nil && visible(mt.Sym) { // Case-insensitive lookup.
				base.Errorf("%v undefined (type %v has no field or method %v, but does have %v)", n, n.X.Type(), n.Sel, mt.Sym)
			} else {
				base.Errorf("%v undefined (type %v has no field or method %v)", n, n.X.Type(), n.Sel)
			}
		}
		n.SetType(nil)
		return n
	}

	if (n.Op() == ir.ODOTINTER || n.Op() == ir.ODOTMETH) && top&ctxCallee == 0 {
		n.SetOp(ir.OMETHVALUE)
		n.SetType(NewMethodType(n.Type(), nil))
	}
	return n
}

// tcDotType typechecks an ODOTTYPE node.
func tcDotType(n *ir.TypeAssertExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	if !t.IsInterface() {
		base.Errorf("invalid type assertion: %v (non-interface type %v on left)", n, t)
		n.SetType(nil)
		return n
	}

	base.AssertfAt(n.Type() != nil, n.Pos(), "missing type: %v", n)

	if n.Type() != nil && !n.Type().IsInterface() {
		var missing, have *types.Field
		var ptr int
		if !implements(n.Type(), t, &missing, &have, &ptr) {
			if have != nil && have.Sym == missing.Sym {
				base.Errorf("impossible type assertion:\n\t%v does not implement %v (wrong type for %v method)\n"+
					"\t\thave %v%S\n\t\twant %v%S", n.Type(), t, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
			} else if ptr != 0 {
				base.Errorf("impossible type assertion:\n\t%v does not implement %v (%v method has pointer receiver)", n.Type(), t, missing.Sym)
			} else if have != nil {
				base.Errorf("impossible type assertion:\n\t%v does not implement %v (missing %v method)\n"+
					"\t\thave %v%S\n\t\twant %v%S", n.Type(), t, missing.Sym, have.Sym, have.Type, missing.Sym, missing.Type)
			} else {
				base.Errorf("impossible type assertion:\n\t%v does not implement %v (missing %v method)", n.Type(), t, missing.Sym)
			}
			n.SetType(nil)
			return n
		}
	}
	return n
}

// tcITab typechecks an OITAB node.
func tcITab(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	t := n.X.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	if !t.IsInterface() {
		base.Fatalf("OITAB of %v", t)
	}
	n.SetType(types.NewPtr(types.Types[types.TUINTPTR]))
	return n
}

// tcIndex typechecks an OINDEX node.
func tcIndex(n *ir.IndexExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	n.X = implicitstar(n.X)
	l := n.X
	n.Index = Expr(n.Index)
	r := n.Index
	t := l.Type()
	if t == nil || r.Type() == nil {
		n.SetType(nil)
		return n
	}
	switch t.Kind() {
	default:
		base.Errorf("invalid operation: %v (type %v does not support indexing)", n, t)
		n.SetType(nil)
		return n

	case types.TSTRING, types.TARRAY, types.TSLICE:
		n.Index = indexlit(n.Index)
		if t.IsString() {
			n.SetType(types.ByteType)
		} else {
			n.SetType(t.Elem())
		}
		why := "string"
		if t.IsArray() {
			why = "array"
		} else if t.IsSlice() {
			why = "slice"
		}

		if n.Index.Type() != nil && !n.Index.Type().IsInteger() {
			base.Errorf("non-integer %s index %v", why, n.Index)
			return n
		}

		if !n.Bounded() && ir.IsConst(n.Index, constant.Int) {
			x := n.Index.Val()
			if constant.Sign(x) < 0 {
				base.Errorf("invalid %s index %v (index must be non-negative)", why, n.Index)
			} else if t.IsArray() && constant.Compare(x, token.GEQ, constant.MakeInt64(t.NumElem())) {
				base.Errorf("invalid array index %v (out of bounds for %d-element array)", n.Index, t.NumElem())
			} else if ir.IsConst(n.X, constant.String) && constant.Compare(x, token.GEQ, constant.MakeInt64(int64(len(ir.StringVal(n.X))))) {
				base.Errorf("invalid string index %v (out of bounds for %d-byte string)", n.Index, len(ir.StringVal(n.X)))
			} else if ir.ConstOverflow(x, types.Types[types.TINT]) {
				base.Errorf("invalid %s index %v (index too large)", why, n.Index)
			}
		}

	case types.TMAP:
		n.Index = AssignConv(n.Index, t.Key(), "map index")
		n.SetType(t.Elem())
		n.SetOp(ir.OINDEXMAP)
		n.Assigned = false
	}
	return n
}

// tcLenCap typechecks an OLEN or OCAP node.
func tcLenCap(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	n.X = implicitstar(n.X)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}

	var ok bool
	if n.Op() == ir.OLEN {
		ok = okforlen[t.Kind()]
	} else {
		ok = okforcap[t.Kind()]
	}
	if !ok {
		base.Errorf("invalid argument %L for %v", l, n.Op())
		n.SetType(nil)
		return n
	}

	n.SetType(types.Types[types.TINT])
	return n
}

// tcUnsafeData typechecks an OUNSAFESLICEDATA or OUNSAFESTRINGDATA node.
func tcUnsafeData(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}

	var kind types.Kind
	if n.Op() == ir.OUNSAFESLICEDATA {
		kind = types.TSLICE
	} else {
		/* kind is string */
		kind = types.TSTRING
	}

	if t.Kind() != kind {
		base.Errorf("invalid argument %L for %v", l, n.Op())
		n.SetType(nil)
		return n
	}

	if kind == types.TSTRING {
		t = types.ByteType
	} else {
		t = t.Elem()
	}
	n.SetType(types.NewPtr(t))
	return n
}

// tcRecv typechecks an ORECV node.
func tcRecv(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	n.X = DefaultLit(n.X, nil)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	if !t.IsChan() {
		base.Errorf("invalid operation: %v (receive from non-chan type %v)", n, t)
		n.SetType(nil)
		return n
	}

	if !t.ChanDir().CanRecv() {
		base.Errorf("invalid operation: %v (receive from send-only type %v)", n, t)
		n.SetType(nil)
		return n
	}

	n.SetType(t.Elem())
	return n
}

// tcSPtr typechecks an OSPTR node.
func tcSPtr(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	t := n.X.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	if !t.IsSlice() && !t.IsString() {
		base.Fatalf("OSPTR of %v", t)
	}
	if t.IsString() {
		n.SetType(types.NewPtr(types.Types[types.TUINT8]))
	} else {
		n.SetType(types.NewPtr(t.Elem()))
	}
	return n
}

// tcSlice typechecks an OSLICE or OSLICE3 node.
func tcSlice(n *ir.SliceExpr) ir.Node {
	n.X = DefaultLit(Expr(n.X), nil)
	n.Low = indexlit(Expr(n.Low))
	n.High = indexlit(Expr(n.High))
	n.Max = indexlit(Expr(n.Max))
	hasmax := n.Op().IsSlice3()
	l := n.X
	if l.Type() == nil {
		n.SetType(nil)
		return n
	}
	if l.Type().IsArray() {
		if !ir.IsAddressable(n.X) {
			base.Errorf("invalid operation %v (slice of unaddressable value)", n)
			n.SetType(nil)
			return n
		}

		addr := NodAddr(n.X)
		addr.SetImplicit(true)
		n.X = Expr(addr)
		l = n.X
	}
	t := l.Type()
	var tp *types.Type
	if t.IsString() {
		if hasmax {
			base.Errorf("invalid operation %v (3-index slice of string)", n)
			n.SetType(nil)
			return n
		}
		n.SetType(t)
		n.SetOp(ir.OSLICESTR)
	} else if t.IsPtr() && t.Elem().IsArray() {
		tp = t.Elem()
		n.SetType(types.NewSlice(tp.Elem()))
		types.CalcSize(n.Type())
		if hasmax {
			n.SetOp(ir.OSLICE3ARR)
		} else {
			n.SetOp(ir.OSLICEARR)
		}
	} else if t.IsSlice() {
		n.SetType(t)
	} else {
		base.Errorf("cannot slice %v (type %v)", l, t)
		n.SetType(nil)
		return n
	}

	if n.Low != nil && !checksliceindex(l, n.Low, tp) {
		n.SetType(nil)
		return n
	}
	if n.High != nil && !checksliceindex(l, n.High, tp) {
		n.SetType(nil)
		return n
	}
	if n.Max != nil && !checksliceindex(l, n.Max, tp) {
		n.SetType(nil)
		return n
	}
	if !checksliceconst(n.Low, n.High) || !checksliceconst(n.Low, n.Max) || !checksliceconst(n.High, n.Max) {
		n.SetType(nil)
		return n
	}
	return n
}

// tcSliceHeader typechecks an OSLICEHEADER node.
func tcSliceHeader(n *ir.SliceHeaderExpr) ir.Node {
	// Errors here are Fatalf instead of Errorf because only the compiler
	// can construct an OSLICEHEADER node.
	// Components used in OSLICEHEADER that are supplied by parsed source code
	// have already been typechecked in e.g. OMAKESLICE earlier.
	t := n.Type()
	if t == nil {
		base.Fatalf("no type specified for OSLICEHEADER")
	}

	if !t.IsSlice() {
		base.Fatalf("invalid type %v for OSLICEHEADER", n.Type())
	}

	if n.Ptr == nil || n.Ptr.Type() == nil || !n.Ptr.Type().IsUnsafePtr() {
		base.Fatalf("need unsafe.Pointer for OSLICEHEADER")
	}

	n.Ptr = Expr(n.Ptr)
	n.Len = DefaultLit(Expr(n.Len), types.Types[types.TINT])
	n.Cap = DefaultLit(Expr(n.Cap), types.Types[types.TINT])

	if ir.IsConst(n.Len, constant.Int) && ir.Int64Val(n.Len) < 0 {
		base.Fatalf("len for OSLICEHEADER must be non-negative")
	}

	if ir.IsConst(n.Cap, constant.Int) && ir.Int64Val(n.Cap) < 0 {
		base.Fatalf("cap for OSLICEHEADER must be non-negative")
	}

	if ir.IsConst(n.Len, constant.Int) && ir.IsConst(n.Cap, constant.Int) && constant.Compare(n.Len.Val(), token.GTR, n.Cap.Val()) {
		base.Fatalf("len larger than cap for OSLICEHEADER")
	}

	return n
}

// tcStringHeader typechecks an OSTRINGHEADER node.
func tcStringHeader(n *ir.StringHeaderExpr) ir.Node {
	t := n.Type()
	if t == nil {
		base.Fatalf("no type specified for OSTRINGHEADER")
	}

	if !t.IsString() {
		base.Fatalf("invalid type %v for OSTRINGHEADER", n.Type())
	}

	if n.Ptr == nil || n.Ptr.Type() == nil || !n.Ptr.Type().IsUnsafePtr() {
		base.Fatalf("need unsafe.Pointer for OSTRINGHEADER")
	}

	n.Ptr = Expr(n.Ptr)
	n.Len = DefaultLit(Expr(n.Len), types.Types[types.TINT])

	if ir.IsConst(n.Len, constant.Int) && ir.Int64Val(n.Len) < 0 {
		base.Fatalf("len for OSTRINGHEADER must be non-negative")
	}

	return n
}

// tcStar typechecks an ODEREF node, which may be an expression or a type.
func tcStar(n *ir.StarExpr, top int) ir.Node {
	n.X = typecheck(n.X, ctxExpr|ctxType)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}

	// TODO(mdempsky): Remove (along with ctxType above) once I'm
	// confident this code path isn't needed any more.
	if l.Op() == ir.OTYPE {
		base.Fatalf("unexpected type in deref expression: %v", l)
	}

	if !t.IsPtr() {
		if top&(ctxExpr|ctxStmt) != 0 {
			base.Errorf("invalid indirect of %L", n.X)
			n.SetType(nil)
			return n
		}
		base.Errorf("%v is not a type", l)
		return n
	}

	n.SetType(t.Elem())
	return n
}

// tcUnaryArith typechecks a unary arithmetic expression.
func tcUnaryArith(n *ir.UnaryExpr) ir.Node {
	n.X = Expr(n.X)
	l := n.X
	t := l.Type()
	if t == nil {
		n.SetType(nil)
		return n
	}
	if !okfor[n.Op()][defaultType(t).Kind()] {
		base.Errorf("invalid operation: %v (operator %v not defined on %s)", n, n.Op(), typekind(t))
		n.SetType(nil)
		return n
	}

	n.SetType(t)
	return n
}
