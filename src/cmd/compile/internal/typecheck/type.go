// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"go/constant"

	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

// tcArrayType typechecks an OTARRAY node.
func tcArrayType(n *ir.ArrayType) ir.Node {
	n.Elem = typecheckNtype(n.Elem)
	if n.Elem.Type() == nil {
		return n
	}
	if n.Len == nil { // [...]T
		if !n.Diag() {
			n.SetDiag(true)
			base.Errorf("use of [...] array outside of array literal")
		}
		return n
	}
	n.Len = indexlit(Expr(n.Len))
	size := n.Len
	if ir.ConstType(size) != constant.Int {
		switch {
		case size.Type() == nil:
			// Error already reported elsewhere.
		case size.Type().IsInteger() && size.Op() != ir.OLITERAL:
			base.Errorf("non-constant array bound %v", size)
		default:
			base.Errorf("invalid array bound %v", size)
		}
		return n
	}

	v := size.Val()
	if ir.ConstOverflow(v, types.Types[types.TINT]) {
		base.Errorf("array bound is too large")
		return n
	}

	if constant.Sign(v) < 0 {
		base.Errorf("array bound must be non-negative")
		return n
	}

	bound, _ := constant.Int64Val(v)
	t := types.NewArray(n.Elem.Type(), bound)
	n.SetOTYPE(t)
	types.CheckSize(t)
	return n
}

// tcChanType typechecks an OTCHAN node.
func tcChanType(n *ir.ChanType) ir.Node {
	n.Elem = typecheckNtype(n.Elem)
	l := n.Elem
	if l.Type() == nil {
		return n
	}
	if l.Type().NotInHeap() {
		base.Errorf("chan of incomplete (or unallocatable) type not allowed")
	}
	n.SetOTYPE(types.NewChan(l.Type(), n.Dir))
	return n
}

// tcFuncType typechecks an OTFUNC node.
func tcFuncType(n *ir.FuncType) ir.Node {
	misc := func(f *types.Field, nf *ir.Field) {
		f.SetIsDDD(nf.IsDDD)
		if nf.Decl != nil {
			nf.Decl.SetType(f.Type)
			f.Nname = nf.Decl
		}
	}

	lno := base.Pos

	var recv *types.Field
	if n.Recv != nil {
		recv = tcField(n.Recv, misc)
	}

	t := types.NewSignature(types.LocalPkg, recv, nil, tcFields(n.Params, misc), tcFields(n.Results, misc))
	checkdupfields("argument", t.Recvs().FieldSlice(), t.Params().FieldSlice(), t.Results().FieldSlice())

	base.Pos = lno

	n.SetOTYPE(t)
	return n
}

// tcInterfaceType typechecks an OTINTER node.
func tcInterfaceType(n *ir.InterfaceType) ir.Node {
	if len(n.Methods) == 0 {
		n.SetOTYPE(types.Types[types.TINTER])
		return n
	}

	lno := base.Pos
	methods := tcFields(n.Methods, nil)
	base.Pos = lno

	n.SetOTYPE(types.NewInterface(types.LocalPkg, methods, false))
	return n
}

// tcMapType typechecks an OTMAP node.
func tcMapType(n *ir.MapType) ir.Node {
	n.Key = typecheckNtype(n.Key)
	n.Elem = typecheckNtype(n.Elem)
	l := n.Key
	r := n.Elem
	if l.Type() == nil || r.Type() == nil {
		return n
	}
	if l.Type().NotInHeap() {
		base.Errorf("incomplete (or unallocatable) map key not allowed")
	}
	if r.Type().NotInHeap() {
		base.Errorf("incomplete (or unallocatable) map value not allowed")
	}
	n.SetOTYPE(types.NewMap(l.Type(), r.Type()))
	mapqueue = append(mapqueue, n) // check map keys when all types are settled
	return n
}

// tcSliceType typechecks an OTSLICE node.
func tcSliceType(n *ir.SliceType) ir.Node {
	n.Elem = typecheckNtype(n.Elem)
	if n.Elem.Type() == nil {
		return n
	}
	t := types.NewSlice(n.Elem.Type())
	n.SetOTYPE(t)
	types.CheckSize(t)
	return n
}

// tcStructType typechecks an OTSTRUCT node.
func tcStructType(n *ir.StructType) ir.Node {
	lno := base.Pos

	fields := tcFields(n.Fields, func(f *types.Field, nf *ir.Field) {
		if nf.Embedded {
			checkembeddedtype(f.Type)
			f.Embedded = 1
		}
		f.Note = nf.Note
	})
	checkdupfields("field", fields)

	base.Pos = lno
	n.SetOTYPE(types.NewStruct(types.LocalPkg, fields))
	return n
}

// tcField typechecks a generic Field.
// misc can be provided to handle specialized typechecking.
func tcField(n *ir.Field, misc func(*types.Field, *ir.Field)) *types.Field {
	base.Pos = n.Pos
	if n.Ntype != nil {
		n.Type = typecheckNtype(n.Ntype).Type()
		n.Ntype = nil
	}
	f := types.NewField(n.Pos, n.Sym, n.Type)
	if misc != nil {
		misc(f, n)
	}
	return f
}

// tcFields typechecks a slice of generic Fields.
// misc can be provided to handle specialized typechecking.
func tcFields(l []*ir.Field, misc func(*types.Field, *ir.Field)) []*types.Field {
	fields := make([]*types.Field, len(l))
	for i, n := range l {
		fields[i] = tcField(n, misc)
	}
	return fields
}
