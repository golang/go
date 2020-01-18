// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/ast"
	"go/types"
)

type typemap map[types.Object]ast.Expr

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

	targs := make(typemap, len(decl.Type.TParams.List))
	for i, tf := range decl.Type.TParams.List {
		for _, tn := range tf.Names {
			obj, ok := t.info.Defs[tn]
			if !ok {
				panic(fmt.Sprintf("no object for type parameter %q", tn))
			}
			targs[obj] = astTypes[i]
		}
	}

	instIdent := ast.NewIdent(name)

	newDecl := &ast.FuncDecl{
		Doc:  decl.Doc,
		Recv: t.instantiateFieldList(targs, decl.Recv),
		Name: instIdent,
		Type: t.instantiateExpr(targs, decl.Type).(*ast.FuncType),
		Body: t.instantiateBlockStmt(targs, decl.Body),
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

// instantiateBlockStmt instantiates a BlockStmt.
func (t *translator) instantiateBlockStmt(targs typemap, pbs *ast.BlockStmt) *ast.BlockStmt {
	changed := false
	stmts := make([]ast.Stmt, len(pbs.List))
	for i, s := range pbs.List {
		is := t.instantiateStmt(targs, s)
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
func (t *translator) instantiateStmt(targs typemap, s ast.Stmt) ast.Stmt {
	switch s := s.(type) {
	case *ast.BlockStmt:
		return t.instantiateBlockStmt(targs, s)
	case *ast.ExprStmt:
		x := t.instantiateExpr(targs, s.X)
		if x == s.X {
			return s
		}
		return &ast.ExprStmt{
			X: x,
		}
	case *ast.RangeStmt:
		key := t.instantiateExpr(targs, s.Key)
		value := t.instantiateExpr(targs, s.Value)
		x := t.instantiateExpr(targs, s.X)
		body := t.instantiateBlockStmt(targs, s.Body)
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
	default:
		panic(fmt.Sprintf("unimplemented Stmt %T", s))
	}
}

// instantiateFieldList instantiates a field list.
func (t *translator) instantiateFieldList(targs typemap, fl *ast.FieldList) *ast.FieldList {
	if fl == nil {
		return nil
	}
	nfl := make([]*ast.Field, len(fl.List))
	changed := false
	for i, f := range fl.List {
		nf := t.instantiateField(targs, f)
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
func (t *translator) instantiateField(targs typemap, f *ast.Field) *ast.Field {
	typ := t.instantiateExpr(targs, f.Type)
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
func (t *translator) instantiateExpr(targs typemap, e ast.Expr) ast.Expr {
	if e == nil {
		return nil
	}
	switch e := e.(type) {
	case *ast.CallExpr:
		fun := t.instantiateExpr(targs, e.Fun)
		args, argsChanged := t.instantiateExprList(targs, e.Args)
		if fun == e.Fun && !argsChanged {
			return e
		}
		return &ast.CallExpr{
			Fun:      fun,
			Lparen:   e.Lparen,
			Args:     args,
			Ellipsis: e.Ellipsis,
			Rparen:   e.Rparen,
		}
	case *ast.Ident:
		obj, ok := t.info.Uses[e]
		if ok {
			typ, ok := targs[obj]
			if ok {
				return typ
			}
		}
		return e
	case *ast.SelectorExpr:
		x := t.instantiateExpr(targs, e.X)
		if x == e.X {
			return e
		}
		return &ast.SelectorExpr{
			X:   x,
			Sel: e.Sel,
		}
	case *ast.FuncType:
		params := t.instantiateFieldList(targs, e.Params)
		results := t.instantiateFieldList(targs, e.Results)
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
		ln := t.instantiateExpr(targs, e.Len)
		elt := t.instantiateExpr(targs, e.Elt)
		if ln == e.Len && elt == e.Elt {
			return e
		}
		return &ast.ArrayType{
			Lbrack: e.Lbrack,
			Len:    ln,
			Elt:    elt,
		}
	default:
		panic(fmt.Sprintf("unimplemented Expr %T", e))
	}
}

// instantiateExprList instantiates an expression list.
func (t *translator) instantiateExprList(targs typemap, el []ast.Expr) ([]ast.Expr, bool) {
	nel := make([]ast.Expr, len(el))
	changed := false
	for i, e := range el {
		ne := t.instantiateExpr(targs, e)
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
