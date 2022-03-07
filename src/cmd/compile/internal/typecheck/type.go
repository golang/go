// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
)

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
