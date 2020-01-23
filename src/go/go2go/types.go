// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/ast"
	"go/types"
)

// lookupType returns the types.Type for an AST expression.
func (t *translator) lookupType(e ast.Expr) types.Type {
	if t, ok := t.info.Types[e]; ok {
		return t.Type
	}
	if t, ok := t.types[e]; ok {
		return t
	}
	panic(fmt.Sprintf("no type found for %T %v", e, e))
}

// setType records the type for an AST expression. This is only used for
// AST expressions created during function instantiation.
// Uninstantiated AST expressions will be listed in t.info.Types.
func (t *translator) setType(e ast.Expr, nt types.Type) {
	if ot, ok := t.info.Types[e]; ok {
		if !types.Identical(ot.Type, nt) {
			panic("expression type changed")
		}
		return
	}
	if ot, ok := t.types[e]; ok {
		if !types.Identical(ot, nt) {
			panic("expression type changed")
		}
		return
	}
	t.types[e] = nt
}

// instantiateType instantiates typ using ta.
func (t *translator) instantiateType(ta *typeArgs, typ types.Type) types.Type {
	switch typ := typ.(type) {
	case *types.TypeParam:
		if instType, ok := ta.typ(typ); ok {
			return instType
		}
		return typ
	case *types.Slice:
		elem := typ.Elem()
		instElem := t.instantiateType(ta, elem)
		if elem == instElem {
			return typ
		}
		return types.NewSlice(elem)
	case *types.Signature:
		params := t.instantiateTypeTuple(ta, typ.Params())
		results := t.instantiateTypeTuple(ta, typ.Results())
		if params == typ.Params() && results == typ.Results() {
			return typ
		}
		r := types.NewSignature(typ.Recv(), params, results, typ.Variadic())
		if tparams := typ.TParams(); tparams != nil {
			r.SetTParams(tparams)
		}
		return r
	case *types.Tuple:
		return t.instantiateTypeTuple(ta, typ)
	default:
		panic(fmt.Sprintf("unimplemented Type %T", typ))
	}
}

// instantiateTypeTuple instantiates a types.Tuple.
func (t *translator) instantiateTypeTuple(ta *typeArgs, tuple *types.Tuple) *types.Tuple {
	if tuple == nil {
		return nil
	}
	l := tuple.Len()
	instTypes := make([]types.Type, l)
	changed := false
	for i := 0; i < l; i++ {
		typ := tuple.At(i).Type()
		instType := t.instantiateType(ta, typ)
		if typ != instType {
			changed = true
		}
		instTypes[i] = instType
	}
	if !changed {
		return tuple
	}
	vars := make([]*types.Var, l)
	for i := 0; i < l; i++ {
		v := tuple.At(i)
		vars[i] = types.NewVar(v.Pos(), v.Pkg(), v.Name(), instTypes[i])
	}
	return types.NewTuple(vars...)
}
