// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"strconv"
	"strings"
)

// lookupType returns the types.Type for an AST expression.
// Returns nil if the type is not known.
func (t *translator) lookupType(e ast.Expr) types.Type {
	if typ := t.importer.info.TypeOf(e); typ != nil {
		return typ
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
		if !t.sameType(ot.Type, nt) {
			panic("expression type changed")
		}
		return
	}
	if ot, ok := t.types[e]; ok {
		if !t.sameType(ot, nt) {
			panic("expression type changed")
		}
		return
	}
	t.types[e] = nt
}

// instantiateType instantiates typ using ta.
func (t *translator) instantiateType(ta *typeArgs, typ types.Type) types.Type {
	if t.err != nil {
		return nil
	}

	t.typeDepth++
	defer func() { t.typeDepth-- }()
	if t.typeDepth > 25 {
		t.err = fmt.Errorf("looping while instantiating %v %v", typ, ta.types)
		return nil
	}

	var inProgress *typeInstantiation
	for _, inst := range t.typeInstantiations(typ) {
		if t.sameTypes(ta.types, inst.types) {
			if inst.typ == nil {
				inProgress = inst
				break
			}
			return inst.typ
		}
	}

	ityp := t.doInstantiateType(ta, typ)
	if inProgress != nil {
		if inProgress.typ == nil {
			inProgress.typ = ityp
		} else {
			ityp = inProgress.typ
		}
	} else {
		typinst := &typeInstantiation{
			types: ta.types,
			typ:   ityp,
		}
		t.addTypeInstantiation(typ, typinst)
	}

	return ityp
}

// doInstantiateType does the work of instantiating typ using ta.
// This should only be called from instantiateType.
func (t *translator) doInstantiateType(ta *typeArgs, typ types.Type) types.Type {
	switch typ := typ.(type) {
	case *types.Basic:
		return typ
	case *types.Array:
		elem := typ.Elem()
		instElem := t.instantiateType(ta, elem)
		if elem == instElem {
			return typ
		}
		return types.NewArray(instElem, typ.Len())
	case *types.Slice:
		elem := typ.Elem()
		instElem := t.instantiateType(ta, elem)
		if elem == instElem {
			return typ
		}
		return types.NewSlice(instElem)
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
	case *types.Pointer:
		elem := typ.Elem()
		instElem := t.instantiateType(ta, elem)
		if elem == instElem {
			return typ
		}
		return types.NewPointer(instElem)
	case *types.Tuple:
		return t.instantiateTypeTuple(ta, typ)
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
	case *types.Interface:
		nm := typ.NumExplicitMethods()
		methods := make([]*types.Func, nm)
		changed := false
		for i := 0; i < nm; i++ {
			m := typ.ExplicitMethod(i)
			instSig := t.instantiateType(ta, m.Type()).(*types.Signature)
			if instSig != m.Type() {
				m = types.NewFunc(m.Pos(), m.Pkg(), m.Name(), instSig)
				changed = true
			}
			methods[i] = m
		}
		ne := typ.NumEmbeddeds()
		embeddeds := make([]types.Type, ne)
		for i := 0; i < ne; i++ {
			e := typ.EmbeddedType(i)
			instE := t.instantiateType(ta, e)
			if e != instE {
				changed = true
			}
			embeddeds[i] = instE
		}
		if !changed {
			return typ
		}
		return types.NewInterfaceType(methods, embeddeds)
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
	case *types.Named:
		targs := typ.TArgs()
		targsChanged := false
		if len(targs) > 0 {
			newTargs := make([]types.Type, 0, len(targs))
			for _, targ := range targs {
				newTarg := t.instantiateType(ta, targ)
				if newTarg != targ {
					targsChanged = true
				}
				newTargs = append(newTargs, newTarg)
			}
			targs = newTargs
		}
		if targsChanged {
			return t.updateTArgs(typ, targs)
		}
		return typ
	case *types.TypeParam:
		if instType, ok := ta.typ(typ); ok {
			return instType
		}
		return typ
	default:
		panic(fmt.Sprintf("unimplemented Type %T", typ))
	}
}

// setTargs returns a new named type with updated type arguments.
func (t *translator) updateTArgs(typ *types.Named, targs []types.Type) *types.Named {
	nm := typ.NumMethods()
	methods := make([]*types.Func, 0, nm)
	for i := 0; i < nm; i++ {
		methods = append(methods, typ.Method(i))
	}
	obj := typ.Obj()
	obj = types.NewTypeName(obj.Pos(), obj.Pkg(), obj.Name(), nil)
	nt := types.NewNamed(obj, typ.Underlying(), methods)
	nt.SetTArgs(targs)
	return nt
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

// addTypePackages adds all packages mentioned in typ to t.typePackages.
func (t *translator) addTypePackages(typ types.Type) {
	switch typ := typ.(type) {
	case *types.Basic:
	case *types.Array:
		t.addTypePackages(typ.Elem())
	case *types.Slice:
		t.addTypePackages(typ.Elem())
	case *types.Struct:
		n := typ.NumFields()
		for i := 0; i < n; i++ {
			t.addTypePackages(typ.Field(i).Type())
		}
	case *types.Pointer:
		t.addTypePackages(typ.Elem())
	case *types.Tuple:
		n := typ.Len()
		for i := 0; i < n; i++ {
			t.addTypePackages(typ.At(i).Type())
		}
	case *types.Signature:
		// We'll have seen typ.Recv elsewhere.
		t.addTypePackages(typ.Params())
		t.addTypePackages(typ.Results())
	case *types.Interface:
		nm := typ.NumExplicitMethods()
		for i := 0; i < nm; i++ {
			t.addTypePackages(typ.ExplicitMethod(i).Type())
		}
		ne := typ.NumEmbeddeds()
		for i := 0; i < ne; i++ {
			t.addTypePackages(typ.EmbeddedType(i))
		}
	case *types.Map:
		t.addTypePackages(typ.Key())
		t.addTypePackages(typ.Elem())
	case *types.Chan:
		t.addTypePackages(typ.Elem())
	case *types.Named:
		// This is the point of this whole method.
		if typ.Obj().Pkg() != nil {
			t.typePackages[typ.Obj().Pkg()] = true
		}
	case *types.TypeParam:
	default:
		panic(fmt.Sprintf("unimplemented Type %T", typ))
	}
}

// withoutTags returns a type with no struct tags. If typ has no
// struct tags anyhow, this just returns typ.
func (t *translator) withoutTags(typ types.Type) types.Type {
	switch typ := typ.(type) {
	case *types.Basic:
		return typ
	case *types.Array:
		elem := typ.Elem()
		elemNoTags := t.withoutTags(elem)
		if elem == elemNoTags {
			return typ
		}
		return types.NewArray(elemNoTags, typ.Len())
	case *types.Slice:
		elem := typ.Elem()
		elemNoTags := t.withoutTags(elem)
		if elem == elemNoTags {
			return typ
		}
		return types.NewSlice(elemNoTags)
	case *types.Struct:
		n := typ.NumFields()
		fields := make([]*types.Var, n)
		changed := false
		hasTag := false
		for i := 0; i < n; i++ {
			v := typ.Field(i)
			typeNoTags := t.withoutTags(v.Type())
			if v.Type() != typeNoTags {
				changed = true
			}
			fields[i] = types.NewVar(v.Pos(), v.Pkg(), v.Name(), typeNoTags)
			if typ.Tag(i) != "" {
				hasTag = true
			}
		}
		if !changed && !hasTag {
			return typ
		}
		return types.NewStruct(fields, nil)
	case *types.Pointer:
		elem := typ.Elem()
		elemNoTags := t.withoutTags(elem)
		if elem == elemNoTags {
			return typ
		}
		return types.NewPointer(elemNoTags)
	case *types.Tuple:
		return t.tupleWithoutTags(typ)
	case *types.Signature:
		params := t.tupleWithoutTags(typ.Params())
		results := t.tupleWithoutTags(typ.Results())
		if params == typ.Params() && results == typ.Results() {
			return typ
		}
		r := types.NewSignature(typ.Recv(), params, results, typ.Variadic())
		if tparams := typ.TParams(); tparams != nil {
			r.SetTParams(tparams)
		}
		return r
	case *types.Interface:
		nm := typ.NumExplicitMethods()
		methods := make([]*types.Func, nm)
		changed := false
		for i := 0; i < nm; i++ {
			m := typ.ExplicitMethod(i)
			sigNoTags := t.withoutTags(m.Type()).(*types.Signature)
			if sigNoTags != m.Type() {
				m = types.NewFunc(m.Pos(), m.Pkg(), m.Name(), sigNoTags)
				changed = true
			}
			methods[i] = m
		}
		ne := typ.NumEmbeddeds()
		embeddeds := make([]types.Type, ne)
		for i := 0; i < ne; i++ {
			e := typ.EmbeddedType(i)
			eNoTags := t.withoutTags(e)
			if e != eNoTags {
				changed = true
			}
			embeddeds[i] = eNoTags
		}
		if !changed {
			return typ
		}
		return types.NewInterfaceType(methods, embeddeds)
	case *types.Map:
		key := t.withoutTags(typ.Key())
		elem := t.withoutTags(typ.Elem())
		if key == typ.Key() && elem == typ.Elem() {
			return typ
		}
		return types.NewMap(key, elem)
	case *types.Chan:
		elem := t.withoutTags(typ.Elem())
		if elem == typ.Elem() {
			return typ
		}
		return types.NewChan(typ.Dir(), elem)
	case *types.Named:
		targs := typ.TArgs()
		targsChanged := false
		if len(targs) > 0 {
			newTargs := make([]types.Type, 0, len(targs))
			for _, targ := range targs {
				newTarg := t.withoutTags(targ)
				if newTarg != targ {
					targsChanged = true
				}
				newTargs = append(newTargs, newTarg)
			}
			targs = newTargs
		}
		if targsChanged {
			return t.updateTArgs(typ, targs)
		}
		return typ
	case *types.TypeParam:
		return typ
	default:
		panic(fmt.Sprintf("unimplemented Type %T", typ))
	}
}

// tupleWithoutTags returns a tuple with all struct tag removed.
func (t *translator) tupleWithoutTags(tuple *types.Tuple) *types.Tuple {
	if tuple == nil {
		return nil
	}
	l := tuple.Len()
	typesNoTags := make([]types.Type, l)
	changed := false
	for i := 0; i < l; i++ {
		typ := tuple.At(i).Type()
		typNoTags := t.withoutTags(typ)
		if typ != typNoTags {
			changed = true
		}
		typesNoTags[i] = typNoTags
	}
	if !changed {
		return tuple
	}
	vars := make([]*types.Var, l)
	for i := 0; i < l; i++ {
		v := tuple.At(i)
		vars[i] = types.NewVar(v.Pos(), v.Pkg(), v.Name(), typesNoTags[i])
	}
	return types.NewTuple(vars...)
}

// typeToAST converts a types.Type to an ast.Expr.
func (t *translator) typeToAST(typ types.Type) ast.Expr {
	var r ast.Expr
	switch typ := typ.(type) {
	case *types.Basic:
		r = ast.NewIdent(typ.Name())
	case *types.Array:
		r = &ast.ArrayType{
			Len: &ast.BasicLit{
				Kind:  token.INT,
				Value: strconv.FormatInt(typ.Len(), 10),
			},
			Elt: t.typeToAST(typ.Elem()),
		}
	case *types.Slice:
		r = &ast.ArrayType{
			Elt: t.typeToAST(typ.Elem()),
		}
	case *types.Struct:
		var fields []*ast.Field
		n := typ.NumFields()
		for i := 0; i < n; i++ {
			tf := typ.Field(i)
			var names []*ast.Ident
			if !tf.Embedded() {
				names = []*ast.Ident{
					ast.NewIdent(tf.Name()),
				}
			}
			var atag *ast.BasicLit
			if tag := typ.Tag(i); tag != "" {
				atag = &ast.BasicLit{
					Kind:  token.STRING,
					Value: strconv.Quote(tag),
				}
			}
			af := &ast.Field{
				Names: names,
				Type:  t.typeToAST(tf.Type()),
				Tag:   atag,
			}
			fields = append(fields, af)
		}
		r = &ast.StructType{
			Fields: &ast.FieldList{
				List: fields,
			},
		}
	case *types.Pointer:
		r = &ast.StarExpr{
			X: t.typeToAST(typ.Elem()),
		}
	case *types.Tuple:
		// We should only see this in a types.Signature,
		// where we handle it specially, since there is
		// no ast.Expr that can represent this.
		panic("unexpected types.Tuple")
	case *types.Signature:
		if len(typ.TParams()) > 0 {
			// We should only see type parameters for
			// a package scope function declaration.
			panic("unexpected type parameters")
		}
		r = &ast.FuncType{
			Params:  t.tupleToFieldList(typ.Params()),
			Results: t.tupleToFieldList(typ.Params()),
		}
	case *types.Interface:
		var methods []*ast.Field
		nm := typ.NumExplicitMethods()
		for i := 0; i < nm; i++ {
			m := typ.ExplicitMethod(i)
			f := &ast.Field{
				Names: []*ast.Ident{
					ast.NewIdent(m.Name()),
				},
				Type: t.typeToAST(m.Type()),
			}
			methods = append(methods, f)
		}
		ne := typ.NumEmbeddeds()
		for i := 0; i < ne; i++ {
			e := typ.EmbeddedType(i)
			f := &ast.Field{
				Type: t.typeToAST(e),
			}
			methods = append(methods, f)
		}
		r = &ast.InterfaceType{
			Methods: &ast.FieldList{
				List: methods,
			},
		}
	case *types.Map:
		r = &ast.MapType{
			Key:   t.typeToAST(typ.Key()),
			Value: t.typeToAST(typ.Elem()),
		}
	case *types.Chan:
		var dir ast.ChanDir
		switch typ.Dir() {
		case types.SendRecv:
			dir = ast.SEND | ast.RECV
		case types.SendOnly:
			dir = ast.SEND
		case types.RecvOnly:
			dir = ast.RECV
		default:
			panic("unsupported channel direction")
		}
		r = &ast.ChanType{
			Dir:   dir,
			Value: t.typeToAST(typ.Elem()),
		}
	case *types.Named:
		if len(typ.TArgs()) > 0 {
			_, id := t.lookupInstantiatedType(typ)
			r = id
		} else {
			var sb strings.Builder
			tn := typ.Obj()
			if tn.Pkg() != nil && tn.Pkg() != t.tpkg {
				sb.WriteString(tn.Pkg().Name())
				sb.WriteByte('.')
			}
			sb.WriteString(tn.Name())
			r = ast.NewIdent(sb.String())
		}
	case *types.TypeParam:
		// This should have been instantiated already.
		panic("unexpected type parameter")
	default:
		panic(fmt.Sprintf("unimplemented Type %T", typ))
	}

	t.setType(r, typ)
	return r
}

// tupleToFieldList converts a tupes.Tuple to a ast.FieldList.
func (t *translator) tupleToFieldList(tuple *types.Tuple) *ast.FieldList {
	var fields []*ast.Field
	n := tuple.Len()
	for i := 0; i < n; i++ {
		v := tuple.At(i)
		var names []*ast.Ident
		if v.Name() != "" {
			names = []*ast.Ident{
				ast.NewIdent(v.Name()),
			}
		}
		f := &ast.Field{
			Names: names,
			Type:  t.typeToAST(v.Type()),
		}
		fields = append(fields, f)
	}
	return &ast.FieldList{
		List: fields,
	}
}
