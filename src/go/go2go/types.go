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
// Returns nil if the type is not known.
func (t *translator) lookupType(e ast.Expr) types.Type {
	if typ, ok := t.importer.info.Types[e]; ok {
		return typ.Type
	}
	if typ, ok := t.types[e]; ok {
		return typ
	}
	return nil
}

// setType records the type for an AST expression. This is only used for
// AST expressions created during function instantiation.
// Uninstantiated AST expressions will be listed in t.importer.info.Types.
func (t *translator) setType(e ast.Expr, nt types.Type) {
	if ot, ok := t.importer.info.Types[e]; ok {
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
	if insts, ok := t.typeInstantiations[typ]; ok {
		for _, inst := range insts {
			if t.sameTypes(ta.types, inst.types) {
				return inst.typ
			}
		}
	}

	ityp := t.doInstantiateType(ta, typ)
	typinst := &typeInstantiation{
		types: ta.types,
		typ:   ityp,
	}
	t.typeInstantiations[typ] = append(t.typeInstantiations[typ], typinst)
	return ityp
}

// doInstantiateType does the work of instantiating typ using ta.
// This should only be called from instantiateType.
func (t *translator) doInstantiateType(ta *typeArgs, typ types.Type) types.Type {
	switch typ := typ.(type) {
	case *types.Named:
		return typ
	case *types.Basic:
		return typ
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
		return types.NewSlice(instElem)
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
	case *types.Struct:
		n := typ.NumFields()
		fields := make([]*types.Var, n)
		changed := false
		tags := make([]string, n)
		hasTag := false
		for i := 0; i < n; i++ {
			v := typ.Field(i)
			instType := t.instantiateType(ta, v.Type())
			if v.Type() != instType {
				changed = true
			}
			fields[i] = types.NewVar(v.Pos(), v.Pkg(), v.Name(), instType)

			tag := typ.Tag(i)
			if tag != "" {
				tags[i] = tag
				hasTag = true
			}
		}
		if !changed {
			return typ
		}
		if !hasTag {
			tags = nil
		}
		return types.NewStruct(fields, tags)
	case *types.Map:
		key := t.instantiateType(ta, typ.Key())
		elem := t.instantiateType(ta, typ.Elem())
		if key == typ.Key() && elem == typ.Elem() {
			return typ
		}
		return types.NewMap(key, elem)
	case *types.Chan:
		elem := t.instantiateType(ta, typ.Elem())
		if elem == typ.Elem() {
			return typ
		}
		return types.NewChan(typ.Dir(), elem)
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
