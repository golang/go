// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package noder

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
	"cmd/compile/internal/types"
	"cmd/internal/src"
)

// Helpers for constructing typed IR nodes.
//
// TODO(mdempsky): Move into their own package so they can be easily
// reused by iimport and frontend optimizations.
//
// TODO(mdempsky): Update to consistently return already typechecked
// results, rather than leaving the caller responsible for using
// typecheck.Expr or typecheck.Stmt.

type ImplicitNode interface {
	ir.Node
	SetImplicit(x bool)
}

// Implicit returns n after marking it as Implicit.
func Implicit(n ImplicitNode) ImplicitNode {
	n.SetImplicit(true)
	return n
}

// typed returns n after setting its type to typ.
func typed(typ *types.Type, n ir.Node) ir.Node {
	n.SetType(typ)
	n.SetTypecheck(1)
	return n
}

// Values

func Const(pos src.XPos, typ *types.Type, val constant.Value) ir.Node {
	return typed(typ, ir.NewBasicLit(pos, val))
}

func Nil(pos src.XPos, typ *types.Type) ir.Node {
	return typed(typ, ir.NewNilExpr(pos))
}

// Expressions

func Addr(pos src.XPos, x ir.Node) *ir.AddrExpr {
	n := typecheck.NodAddrAt(pos, x)
	switch x.Op() {
	case ir.OARRAYLIT, ir.OMAPLIT, ir.OSLICELIT, ir.OSTRUCTLIT:
		n.SetOp(ir.OPTRLIT)
	}
	typed(types.NewPtr(x.Type()), n)
	return n
}

func Assert(pos src.XPos, x ir.Node, typ *types.Type) ir.Node {
	return typed(typ, ir.NewTypeAssertExpr(pos, x, nil))
}

// transformAdd transforms an addition operation (currently just addition of
// strings). Equivalent to the "binary operators" case in typecheck.typecheck1.
func transformAdd(n *ir.BinaryExpr) ir.Node {
	l := n.X
	if l.Type().IsString() {
		var add *ir.AddStringExpr
		if l.Op() == ir.OADDSTR {
			add = l.(*ir.AddStringExpr)
			add.SetPos(n.Pos())
		} else {
			add = ir.NewAddStringExpr(n.Pos(), []ir.Node{l})
		}
		r := n.Y
		if r.Op() == ir.OADDSTR {
			r := r.(*ir.AddStringExpr)
			add.List.Append(r.List.Take()...)
		} else {
			add.List.Append(r)
		}
		add.SetType(l.Type())
		return add
	}
	return n
}

func Binary(pos src.XPos, op ir.Op, typ *types.Type, x, y ir.Node) ir.Node {
	switch op {
	case ir.OANDAND, ir.OOROR:
		return typed(x.Type(), ir.NewLogicalExpr(pos, op, x, y))
	case ir.OADD:
		n := ir.NewBinaryExpr(pos, op, x, y)
		if x.Type().HasTParam() || y.Type().HasTParam() {
			// Delay transformAdd() if either arg has a type param,
			// since it needs to know the exact types to decide whether
			// to transform OADD to OADDSTR.
			n.SetType(typ)
			n.SetTypecheck(3)
			return n
		}
		n1 := transformAdd(n)
		return typed(typ, n1)
	default:
		return typed(x.Type(), ir.NewBinaryExpr(pos, op, x, y))
	}
}

func Call(pos src.XPos, typ *types.Type, fun ir.Node, args []ir.Node, dots bool) ir.Node {
	n := ir.NewCallExpr(pos, ir.OCALL, fun, args)
	n.IsDDD = dots

	if fun.Op() == ir.OTYPE {
		// Actually a type conversion, not a function call.
		if fun.Type().HasTParam() || args[0].Type().HasTParam() {
			// For type params, don't typecheck until we actually know
			// the type.
			return typed(typ, n)
		}
		return typecheck.Expr(n)
	}

	if fun, ok := fun.(*ir.Name); ok && fun.BuiltinOp != 0 {
		// For Builtin ops, we currently stay with using the old
		// typechecker to transform the call to a more specific expression
		// and possibly use more specific ops. However, for a bunch of the
		// ops, we delay doing the old typechecker if any of the args have
		// type params, for a variety of reasons:
		//
		// OMAKE: hard to choose specific ops OMAKESLICE, etc. until arg type is known
		// OREAL/OIMAG: can't determine type float32/float64 until arg type know
		// OLEN/OCAP: old typechecker will complain if arg is not obviously a slice/array.
		// OAPPEND: old typechecker will complain if arg is not obviously slice, etc.
		//
		// We will eventually break out the transforming functionality
		// needed for builtin's, and call it here or during stenciling, as
		// appropriate.
		switch fun.BuiltinOp {
		case ir.OMAKE, ir.OREAL, ir.OIMAG, ir.OLEN, ir.OCAP, ir.OAPPEND:
			hasTParam := false
			for _, arg := range args {
				if arg.Type().HasTParam() {
					hasTParam = true
					break
				}
			}
			if hasTParam {
				return typed(typ, n)
			}
		}

		switch fun.BuiltinOp {
		case ir.OCLOSE, ir.ODELETE, ir.OPANIC, ir.OPRINT, ir.OPRINTN:
			return typecheck.Stmt(n)
		default:
			return typecheck.Expr(n)
		}
	}

	// Add information, now that we know that fun is actually being called.
	switch fun := fun.(type) {
	case *ir.ClosureExpr:
		fun.Func.SetClosureCalled(true)
	case *ir.SelectorExpr:
		if fun.Op() == ir.OCALLPART {
			op := ir.ODOTMETH
			if fun.X.Type().IsInterface() {
				op = ir.ODOTINTER
			}
			fun.SetOp(op)
			// Set the type to include the receiver, since that's what
			// later parts of the compiler expect
			fun.SetType(fun.Selection.Type)
		}
	}

	if fun.Op() == ir.OXDOT {
		if !fun.(*ir.SelectorExpr).X.Type().HasTParam() {
			base.FatalfAt(pos, "Expecting type param receiver in %v", fun)
		}
		// For methods called in a generic function, don't do any extra
		// transformations. We will do those later when we create the
		// instantiated function and have the correct receiver type.
		typed(typ, n)
		return n
	}
	if fun.Op() != ir.OFUNCINST {
		// If no type params, do normal typechecking, since we're
		// still missing some things done by tcCall (mainly
		// typecheckaste/assignconvfn - implementing assignability of args
		// to params).  This will convert OCALL to OCALLFUNC.
		typecheck.Call(n)
		return n
	}

	// Leave the op as OCALL, which indicates the call still needs typechecking.
	n.Use = ir.CallUseExpr
	if fun.Type().NumResults() == 0 {
		n.Use = ir.CallUseStmt
	}
	typed(typ, n)
	return n
}

// transformCompare transforms a compare operation (currently just equals/not
// equals). Equivalent to the "comparison operators" case in
// typecheck.typecheck1, including tcArith.
func transformCompare(n *ir.BinaryExpr) {
	if (n.Op() == ir.OEQ || n.Op() == ir.ONE) && !types.Identical(n.X.Type(), n.Y.Type()) {
		// Comparison is okay as long as one side is assignable to the
		// other. The only allowed case where the conversion is not CONVNOP is
		// "concrete == interface". In that case, check comparability of
		// the concrete type. The conversion allocates, so only do it if
		// the concrete type is huge.
		l, r := n.X, n.Y
		lt, rt := l.Type(), r.Type()
		converted := false
		if rt.Kind() != types.TBLANK {
			aop, _ := typecheck.Assignop(lt, rt)
			if aop != ir.OXXX {
				types.CalcSize(lt)
				if rt.IsInterface() == lt.IsInterface() || lt.Width >= 1<<16 {
					l = ir.NewConvExpr(base.Pos, aop, rt, l)
					l.SetTypecheck(1)
				}

				converted = true
			}
		}

		if !converted && lt.Kind() != types.TBLANK {
			aop, _ := typecheck.Assignop(rt, lt)
			if aop != ir.OXXX {
				types.CalcSize(rt)
				if rt.IsInterface() == lt.IsInterface() || rt.Width >= 1<<16 {
					r = ir.NewConvExpr(base.Pos, aop, lt, r)
					r.SetTypecheck(1)
				}
			}
		}
		n.X, n.Y = l, r
	}
}

func Compare(pos src.XPos, typ *types.Type, op ir.Op, x, y ir.Node) ir.Node {
	n := ir.NewBinaryExpr(pos, op, x, y)
	if x.Type().HasTParam() || y.Type().HasTParam() {
		// Delay transformCompare() if either arg has a type param, since
		// it needs to know the exact types to decide on any needed conversions.
		n.SetType(typ)
		n.SetTypecheck(3)
		return n
	}
	transformCompare(n)
	return typed(typ, n)
}

func Deref(pos src.XPos, x ir.Node) *ir.StarExpr {
	n := ir.NewStarExpr(pos, x)
	typed(x.Type().Elem(), n)
	return n
}

func DotField(pos src.XPos, x ir.Node, index int) *ir.SelectorExpr {
	op, typ := ir.ODOT, x.Type()
	if typ.IsPtr() {
		op, typ = ir.ODOTPTR, typ.Elem()
	}
	if !typ.IsStruct() {
		base.FatalfAt(pos, "DotField of non-struct: %L", x)
	}

	// TODO(mdempsky): This is the backend's responsibility.
	types.CalcSize(typ)

	field := typ.Field(index)
	return dot(pos, field.Type, op, x, field)
}

func DotMethod(pos src.XPos, x ir.Node, index int) *ir.SelectorExpr {
	method := method(x.Type(), index)

	// Method value.
	typ := typecheck.NewMethodType(method.Type, nil)
	return dot(pos, typ, ir.OCALLPART, x, method)
}

// MethodExpr returns a OMETHEXPR node with the indicated index into the methods
// of typ. The receiver type is set from recv, which is different from typ if the
// method was accessed via embedded fields. Similarly, the X value of the
// ir.SelectorExpr is recv, the original OTYPE node before passing through the
// embedded fields.
func MethodExpr(pos src.XPos, recv ir.Node, embed *types.Type, index int) *ir.SelectorExpr {
	method := method(embed, index)
	typ := typecheck.NewMethodType(method.Type, recv.Type())
	// The method expression T.m requires a wrapper when T
	// is different from m's declared receiver type. We
	// normally generate these wrappers while writing out
	// runtime type descriptors, which is always done for
	// types declared at package scope. However, we need
	// to make sure to generate wrappers for anonymous
	// receiver types too.
	if recv.Sym() == nil {
		typecheck.NeedRuntimeType(recv.Type())
	}
	return dot(pos, typ, ir.OMETHEXPR, recv, method)
}

func dot(pos src.XPos, typ *types.Type, op ir.Op, x ir.Node, selection *types.Field) *ir.SelectorExpr {
	n := ir.NewSelectorExpr(pos, op, x, selection.Sym)
	n.Selection = selection
	typed(typ, n)
	return n
}

// TODO(mdempsky): Move to package types.
func method(typ *types.Type, index int) *types.Field {
	if typ.IsInterface() {
		return typ.Field(index)
	}
	return types.ReceiverBaseType(typ).Methods().Index(index)
}

func Index(pos src.XPos, typ *types.Type, x, index ir.Node) ir.Node {
	n := ir.NewIndexExpr(pos, x, index)
	// TODO(danscales): Temporary fix. Need to separate out the
	// transformations done by the old typechecker (in tcIndex()), to be
	// called here or after stenciling.
	if x.Type().HasTParam() && x.Type().Kind() != types.TMAP &&
		x.Type().Kind() != types.TSLICE && x.Type().Kind() != types.TARRAY {
		// Old typechecker will complain if arg is not obviously a slice/array/map.
		typed(typ, n)
		return n
	}
	return typecheck.Expr(n)
}

// transformSlice transforms a slice operation.  Equivalent to typecheck.tcSlice.
func transformSlice(n *ir.SliceExpr) {
	l := n.X
	if l.Type().IsArray() {
		addr := typecheck.NodAddr(n.X)
		addr.SetImplicit(true)
		typed(types.NewPtr(n.X.Type()), addr)
		n.X = addr
		l = addr
	}
	t := l.Type()
	if t.IsString() {
		n.SetOp(ir.OSLICESTR)
	} else if t.IsPtr() && t.Elem().IsArray() {
		if n.Op().IsSlice3() {
			n.SetOp(ir.OSLICE3ARR)
		} else {
			n.SetOp(ir.OSLICEARR)
		}
	}
}

func Slice(pos src.XPos, typ *types.Type, x, low, high, max ir.Node) ir.Node {
	op := ir.OSLICE
	if max != nil {
		op = ir.OSLICE3
	}
	n := ir.NewSliceExpr(pos, op, x, low, high, max)
	if x.Type().HasTParam() {
		// transformSlice needs to know if x.Type() is a string or an array or a slice.
		n.SetType(typ)
		n.SetTypecheck(3)
		return n
	}
	transformSlice(n)
	return typed(typ, n)
}

func Unary(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	switch op {
	case ir.OADDR:
		return Addr(pos, x)
	case ir.ODEREF:
		return Deref(pos, x)
	}

	typ := x.Type()
	if op == ir.ORECV {
		typ = typ.Elem()
	}
	return typed(typ, ir.NewUnaryExpr(pos, op, x))
}

// Statements

var one = constant.MakeInt64(1)

func IncDec(pos src.XPos, op ir.Op, x ir.Node) ir.Node {
	x = typecheck.AssignExpr(x)
	return ir.NewAssignOpStmt(pos, op, x, typecheck.DefaultLit(ir.NewBasicLit(pos, one), x.Type()))
}
