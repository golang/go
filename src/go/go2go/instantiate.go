// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
)

// typeArgs holds type arguments for the function that we are instantiating.
// We can look them up either with a types.Object associated with an ast.Ident,
// or with a types.TypeParam.
type typeArgs struct {
	types []types.Type // type arguments in order
	toAST map[types.Object]ast.Expr
	toTyp map[*types.TypeParam]types.Type
}

// newTypeArgs returns a new typeArgs value.
func newTypeArgs(typeTypes []types.Type) *typeArgs {
	return &typeArgs{
		types: typeTypes,
		toAST: make(map[types.Object]ast.Expr),
		toTyp: make(map[*types.TypeParam]types.Type),
	}
}

// typeArgsFromTParams builds mappings from a list of type parameters
// expressed as ast.Field values.
func typeArgsFromFields(t *translator, astTypes []ast.Expr, typeTypes []types.Type, tparams []*ast.Field) *typeArgs {
	ta := newTypeArgs(typeTypes)
	for i, tf := range tparams {
		for _, tn := range tf.Names {
			obj, ok := t.info.Defs[tn]
			if !ok {
				panic(fmt.Sprintf("no object for type parameter %q", tn))
			}
			objType := obj.Type()
			objParam, ok := objType.(*types.TypeParam)
			if !ok {
				panic(fmt.Sprintf("%v is not a TypeParam", objParam))
			}
			ta.add(obj, objParam, astTypes[i], typeTypes[i])
		}
	}
	return ta
}

// typeArgsFromTParams builds mappings from a list of type parameters
// expressed as ast.Expr values.
func typeArgsFromExprs(t *translator, astTypes []ast.Expr, typeTypes []types.Type, tparams []ast.Expr) *typeArgs {
	ta := newTypeArgs(typeTypes)
	for i, ti := range tparams {
		obj, ok := t.info.Defs[ti.(*ast.Ident)]
		if !ok {
			panic(fmt.Sprintf("no object for type parameter %q", ti))
		}
		objType := obj.Type()
		objParam, ok := objType.(*types.TypeParam)
		if !ok {
			panic(fmt.Sprintf("%v is not a TypeParam", objParam))
		}
		ta.add(obj, objParam, astTypes[i], typeTypes[i])
	}
	return ta
}

// add adds mappings for obj to ast and typ.
func (ta *typeArgs) add(obj types.Object, objParam *types.TypeParam, ast ast.Expr, typ types.Type) {
	ta.toAST[obj] = ast
	ta.toTyp[objParam] = typ
}

// ast returns the AST for obj, and reports whether it exists.
func (ta *typeArgs) ast(obj types.Object) (ast.Expr, bool) {
	e, ok := ta.toAST[obj]
	return e, ok
}

// typ returns the Type for param, and reports whether it exists.
func (ta *typeArgs) typ(param *types.TypeParam) (types.Type, bool) {
	t, ok := ta.toTyp[param]
	return t, ok
}

// instantiateFunction creates a new instantiation of a function.
func (t *translator) instantiateFunction(fnident *ast.Ident, astTypes []ast.Expr, typeTypes []types.Type) (*ast.Ident, error) {
	name, err := t.instantiatedName(fnident, typeTypes)
	if err != nil {
		return nil, err
	}

	decl, err := t.findFuncDecl(fnident)
	if err != nil {
		return nil, err
	}

	ta := typeArgsFromFields(t, astTypes, typeTypes, decl.Type.TParams.List)

	instIdent := ast.NewIdent(name)

	newDecl := &ast.FuncDecl{
		Doc:  decl.Doc,
		Recv: t.instantiateFieldList(ta, decl.Recv),
		Name: instIdent,
		Type: t.instantiateExpr(ta, decl.Type).(*ast.FuncType),
		Body: t.instantiateBlockStmt(ta, decl.Body),
	}
	t.newDecls = append(t.newDecls, newDecl)

	return instIdent, nil
}

// findFuncDecl looks for the FuncDecl for id.
// FIXME: Handle imported packages.
func (t *translator) findFuncDecl(id *ast.Ident) (*ast.FuncDecl, error) {
	obj, ok := t.info.Uses[id]
	if !ok {
		return nil, fmt.Errorf("could not find Object for %q", id.Name)
	}
	decl, ok := t.idToFunc[obj]
	if !ok {
		return nil, fmt.Errorf("could not find function body for %q", id.Name)
	}
	return decl, nil
}

// instantiateType creates a new instantiation of a type.
func (t *translator) instantiateTypeDecl(tident *ast.Ident, typ *types.Named, astTypes []ast.Expr, typeTypes []types.Type) (*ast.Ident, types.Type, error) {
	name, err := t.instantiatedName(tident, typeTypes)
	if err != nil {
		return nil, nil, err
	}

	spec, err := t.findTypeSpec(tident)
	if err != nil {
		return nil, nil, err
	}

	ta := typeArgsFromFields(t, astTypes, typeTypes, spec.TParams.List)

	instIdent := ast.NewIdent(name)

	newSpec := &ast.TypeSpec{
		Doc:     spec.Doc,
		Name:    instIdent,
		Assign:  spec.Assign,
		Type:    t.instantiateExpr(ta, spec.Type),
		Comment: spec.Comment,
	}
	newDecl := &ast.GenDecl{
		Tok:   token.TYPE,
		Specs: []ast.Spec{newSpec},
	}
	t.newDecls = append(t.newDecls, newDecl)

	instType := t.instantiateType(ta, typ.Underlying())

	nm := typ.NumMethods()
	for i := 0; i < nm; i++ {
		method := typ.Method(i)
		mast := t.idToFunc[method]
		if mast == nil {
			panic(fmt.Sprintf("no AST for method %v", method))
		}
		rtyp := mast.Recv.List[0].Type
		newRtype := ast.Expr(ast.NewIdent(name))
		if p, ok := rtyp.(*ast.StarExpr); ok {
			rtyp = p.X
			newRtype = &ast.StarExpr{
				X: newRtype,
			}
		}
		tparams := rtyp.(*ast.CallExpr).Args
		ta := typeArgsFromExprs(t, astTypes, typeTypes, tparams)
		newDecl := &ast.FuncDecl{
			Doc:  mast.Doc,
			Recv: &ast.FieldList{
				Opening: mast.Recv.Opening,
				List: []*ast.Field{
					{
						Doc:     mast.Recv.List[0].Doc,
						Names:   []*ast.Ident{
							mast.Recv.List[0].Names[0],
						},
						Type:    newRtype,
						Comment: mast.Recv.List[0].Comment,
					},
				},
				Closing: mast.Recv.Closing,
			},
			Name: mast.Name,
			Type: t.instantiateExpr(ta, mast.Type).(*ast.FuncType),
			Body: t.instantiateBlockStmt(ta, mast.Body),
		}
		t.newDecls = append(t.newDecls, newDecl)
	}

	return instIdent, instType, nil
}

// findTypeSpec looks for the TypeSpec for id.
// FIXME: Handle imported packages.
func (t *translator) findTypeSpec(id *ast.Ident) (*ast.TypeSpec, error) {
	obj, ok := t.info.Uses[id]
	if !ok {
		return nil, fmt.Errorf("could not find Object for %q", id.Name)
	}
	spec, ok := t.idToTypeSpec[obj]
	if !ok {
		return nil, fmt.Errorf("could not find type spec for %q", id.Name)
	}
	return spec, nil
}

// instantiateBlockStmt instantiates a BlockStmt.
func (t *translator) instantiateBlockStmt(ta *typeArgs, pbs *ast.BlockStmt) *ast.BlockStmt {
	changed := false
	stmts := make([]ast.Stmt, len(pbs.List))
	for i, s := range pbs.List {
		is := t.instantiateStmt(ta, s)
		stmts[i] = is
		if is != s {
			changed = true
		}
	}
	if !changed {
		return pbs
	}
	return &ast.BlockStmt{
		Lbrace: pbs.Lbrace,
		List:   stmts,
		Rbrace: pbs.Rbrace,
	}
}

// instantiateStmt instantiates a statement.
func (t *translator) instantiateStmt(ta *typeArgs, s ast.Stmt) ast.Stmt {
	switch s := s.(type) {
	case *ast.BlockStmt:
		return t.instantiateBlockStmt(ta, s)
	case *ast.ExprStmt:
		x := t.instantiateExpr(ta, s.X)
		if x == s.X {
			return s
		}
		return &ast.ExprStmt{
			X: x,
		}
	case *ast.AssignStmt:
		lhs, lchanged := t.instantiateExprList(ta, s.Lhs)
		rhs, rchanged := t.instantiateExprList(ta, s.Rhs)
		if !lchanged && !rchanged {
			return s
		}
		return &ast.AssignStmt{
			Lhs:    lhs,
			TokPos: s.TokPos,
			Tok:    s.Tok,
			Rhs:    rhs,
		}
	case *ast.RangeStmt:
		key := t.instantiateExpr(ta, s.Key)
		value := t.instantiateExpr(ta, s.Value)
		x := t.instantiateExpr(ta, s.X)
		body := t.instantiateBlockStmt(ta, s.Body)
		if key == s.Key && value == s.Value && x == s.X && body == s.Body {
			return s
		}
		return &ast.RangeStmt{
			For:    s.For,
			Key:    key,
			Value:  value,
			TokPos: s.TokPos,
			Tok:    s.Tok,
			X:      x,
			Body:   body,
		}
	case *ast.ReturnStmt:
		results, changed := t.instantiateExprList(ta, s.Results)
		if !changed {
			return s
		}
		return &ast.ReturnStmt{
			Return:  s.Return,
			Results: results,
		}
	default:
		panic(fmt.Sprintf("unimplemented Stmt %T", s))
	}
}

// instantiateFieldList instantiates a field list.
func (t *translator) instantiateFieldList(ta *typeArgs, fl *ast.FieldList) *ast.FieldList {
	if fl == nil {
		return nil
	}
	nfl := make([]*ast.Field, len(fl.List))
	changed := false
	for i, f := range fl.List {
		nf := t.instantiateField(ta, f)
		if nf != f {
			changed = true
		}
		nfl[i] = nf
	}
	if !changed {
		return fl
	}
	return &ast.FieldList{
		Opening: fl.Opening,
		List:    nfl,
		Closing: fl.Closing,
	}
}

// instantiateField instantiates a field.
func (t *translator) instantiateField(ta *typeArgs, f *ast.Field) *ast.Field {
	typ := t.instantiateExpr(ta, f.Type)
	if typ == f.Type {
		return f
	}
	return &ast.Field{
		Doc:     f.Doc,
		Names:   f.Names,
		Type:    typ,
		Tag:     f.Tag,
		Comment: f.Comment,
	}
}

// instantiateExpr instantiates an expression.
func (t *translator) instantiateExpr(ta *typeArgs, e ast.Expr) ast.Expr {
	if e == nil {
		return nil
	}
	var r ast.Expr
	switch e := e.(type) {
	case *ast.CallExpr:
		fun := t.instantiateExpr(ta, e.Fun)
		args, argsChanged := t.instantiateExprList(ta, e.Args)
		if fun == e.Fun && !argsChanged {
			return e
		}
		r = &ast.CallExpr{
			Fun:      fun,
			Lparen:   e.Lparen,
			Args:     args,
			Ellipsis: e.Ellipsis,
			Rparen:   e.Rparen,
		}
	case *ast.Ident:
		obj := t.info.ObjectOf(e)
		if obj != nil {
			if typ, ok := ta.ast(obj); ok {
				return typ
			}
		}
		return e
	case *ast.SelectorExpr:
		x := t.instantiateExpr(ta, e.X)
		if x == e.X {
			return e
		}
		r = &ast.SelectorExpr{
			X:   x,
			Sel: e.Sel,
		}
	case *ast.StarExpr:
		x := t.instantiateExpr(ta, e.X)
		if x == e.X {
			return e
		}
		r = &ast.StarExpr{
			Star: e.Star,
			X:    x,
		}
	case *ast.FuncType:
		params := t.instantiateFieldList(ta, e.Params)
		results := t.instantiateFieldList(ta, e.Results)
		if e.TParams == nil && params == e.Params && results == e.Results {
			return e
		}
		return &ast.FuncType{
			Func:    e.Func,
			TParams: nil,
			Params:  params,
			Results: results,
		}
	case *ast.ArrayType:
		ln := t.instantiateExpr(ta, e.Len)
		elt := t.instantiateExpr(ta, e.Elt)
		if ln == e.Len && elt == e.Elt {
			return e
		}
		return &ast.ArrayType{
			Lbrack: e.Lbrack,
			Len:    ln,
			Elt:    elt,
		}
	case *ast.StructType:
		fields := t.instantiateFieldList(ta, e.Fields)
		if fields == e.Fields {
			return e
		}
		return &ast.StructType{
			Struct:     e.Struct,
			Fields:     fields,
			Incomplete: e.Incomplete,
		}
	default:
		panic(fmt.Sprintf("unimplemented Expr %T", e))
	}

	// We fall down to here for expressions that are not types.
	t.setType(r, t.instantiateType(ta, t.lookupType(e)))

	return r
}

// instantiateExprList instantiates an expression list.
func (t *translator) instantiateExprList(ta *typeArgs, el []ast.Expr) ([]ast.Expr, bool) {
	nel := make([]ast.Expr, len(el))
	changed := false
	for i, e := range el {
		ne := t.instantiateExpr(ta, e)
		if ne != e {
			changed = true
		}
		nel[i] = ne
	}
	if !changed {
		return el, false
	}
	return nel, true
}
