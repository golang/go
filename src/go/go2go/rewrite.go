// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"bufio"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"strings"
)

var config = printer.Config{
	Mode:     printer.UseSpaces | printer.TabIndent | printer.SourcePos,
	Tabwidth: 8,
}

// addFuncIDS finds IDs for generic functions and adds them to a map.
func addFuncIDs(info *types.Info, f *ast.File, m map[types.Object]*ast.FuncDecl) {
	for _, decl := range f.Decls {
		if fd, ok := decl.(*ast.FuncDecl); ok && isParameterizedFuncDecl(fd) {
			obj, ok := info.Defs[fd.Name]
			if !ok {
				panic(fmt.Sprintf("no types.Object for %q", fd.Name.Name))
			}
			m[obj] = fd
		}
	}
}

// isParameterizedFuncDecl reports whether fd is a parameterized function.
func isParameterizedFuncDecl(fd *ast.FuncDecl) bool {
	return fd.Type.TParams != nil
}

// A translator is used to translate a file from Go with contracts to Go 1.
type translator struct {
	info               *types.Info
	types              map[ast.Expr]types.Type
	idToFunc           map[types.Object]*ast.FuncDecl
	instantiations     map[*ast.Ident][]*instantiation
	newDecls           []ast.Decl
	typeInstantiations map[types.Type][]*typeInstantiation

	// err is set if we have seen an error during this translation.
	// This is used by the rewrite methods.
	err error
}

// An instantiation is a single instantiation of a function.
type instantiation struct {
	types []types.Type
	decl  *ast.Ident
}

// A typeInstantiation is a single instantiation of a type.
type typeInstantiation struct {
	types []types.Type
	typ   types.Type
}

// rewrite rewrites the contents of one file.
func rewriteFile(dir string, fset *token.FileSet, info *types.Info, idToFunc map[types.Object]*ast.FuncDecl, filename string, file *ast.File) (err error) {
	if err := rewriteAST(info, idToFunc, file); err != nil {
		return err
	}

	filename = filepath.Base(filename)
	goFile := strings.TrimSuffix(filename, filepath.Ext(filename)) + ".go"
	o, err := os.Create(filepath.Join(dir, goFile))
	if err != nil {
		return err
	}
	defer func() {
		if closeErr := o.Close(); err == nil {
			err = closeErr
		}
	}()

	w := bufio.NewWriter(o)
	defer func() {
		if flushErr := w.Flush(); err == nil {
			err = flushErr
		}
	}()
	fmt.Fprintln(w, rewritePrefix)

	return config.Fprint(w, fset, file)
}

// rewriteAST rewrites the AST for a file.
func rewriteAST(info *types.Info, idToFunc map[types.Object]*ast.FuncDecl, file *ast.File) (err error) {
	t := translator{
		info:               info,
		types:              make(map[ast.Expr]types.Type),
		idToFunc:           idToFunc,
		instantiations:     make(map[*ast.Ident][]*instantiation),
		typeInstantiations: make(map[types.Type][]*typeInstantiation),
	}
	t.translate(file)
	return t.err
}

// translate translates the AST for a file from Go with contracts to Go 1.
func (t *translator) translate(file *ast.File) {
	declsToDo := file.Decls
	file.Decls = nil
	for len(declsToDo) > 0 {
		newDecls := make([]ast.Decl, 0, len(declsToDo))
		for i, decl := range declsToDo {
			switch decl := decl.(type) {
			case (*ast.FuncDecl):
				if !isParameterizedFuncDecl(decl) {
					t.translateFuncDecl(&declsToDo[i])
					newDecls = append(newDecls, decl)
				}
			case (*ast.GenDecl):
				switch decl.Tok {
				case token.TYPE:
					for j := range decl.Specs {
						t.translateTypeSpec(&decl.Specs[j])
					}
				case token.VAR, token.CONST:
					for j := range decl.Specs {
						t.translateValueSpec(&decl.Specs[j])
					}
				}
				newDecls = append(newDecls, decl)
			default:
				newDecls = append(newDecls, decl)
			}
		}
		file.Decls = append(file.Decls, newDecls...)
		declsToDo = t.newDecls
		t.newDecls = nil
	}
}

// translateTypeSpec translates a type from Go with contracts to Go 1.
func (t *translator) translateTypeSpec(ps *ast.Spec) {
	ts := (*ps).(*ast.TypeSpec)
	if ts.TParams == nil {
		t.translateExpr(&ts.Type)
		return
	}
	panic("parameterized type")
}

// translateValueSpec translates a variable or constant from Go with
// contracts to Go 1.
func (t *translator) translateValueSpec(ps *ast.Spec) {
	vs := (*ps).(*ast.ValueSpec)
	t.translateExpr(&vs.Type)
	for i := range vs.Values {
		t.translateExpr(&vs.Values[i])
	}
}

// translateFuncDecl translates a function from Go with contracts to Go 1.
func (t *translator) translateFuncDecl(pd *ast.Decl) {
	if t.err != nil {
		return
	}
	fd := (*pd).(*ast.FuncDecl)
	if fd.Type.TParams != nil {
		panic("parameterized function")
	}
	if fd.Recv != nil {
		t.translateFieldList(fd.Recv)
	}
	t.translateFieldList(fd.Type.Params)
	t.translateFieldList(fd.Type.Results)
	t.translateBlockStmt(fd.Body)
}

// translateBlockStmt translates a block statement from Go with
// contracts to Go 1.
func (t *translator) translateBlockStmt(pbs *ast.BlockStmt) {
	for i := range pbs.List {
		t.translateStmt(&pbs.List[i])
	}
}

// translateStmt translates a statement from Go with contracts to Go 1.
func (t *translator) translateStmt(ps *ast.Stmt) {
	if t.err != nil {
		return
	}
	switch s := (*ps).(type) {
	case *ast.BlockStmt:
		t.translateBlockStmt(s)
	case *ast.ExprStmt:
		t.translateExpr(&s.X)
	case *ast.RangeStmt:
		t.translateExpr(&s.Key)
		t.translateExpr(&s.Value)
		t.translateExpr(&s.X)
		t.translateBlockStmt(s.Body)
	default:
		panic(fmt.Sprintf("unimplemented Stmt %T", s))
	}
}

// translateExpr translates an expression from Go with contracts to Go 1.
func (t *translator) translateExpr(pe *ast.Expr) {
	if t.err != nil {
		return
	}
	if *pe == nil {
		return
	}
	switch e := (*pe).(type) {
	case *ast.Ident:
		return
	case *ast.CallExpr:
		t.translateExprList(e.Args)
		ftyp := t.lookupType(e.Fun).(*types.Signature)
		if ftyp.TParams() != nil {
			t.translateFunctionInstantiation(pe)
		}
		t.translateExpr(&e.Fun)
	case *ast.StarExpr:
		t.translateExpr(&e.X)
	case *ast.SelectorExpr:
		t.translateExpr(&e.X)
	case *ast.ArrayType:
		t.translateExpr(&e.Elt)
	case *ast.BasicLit:
		return
	case *ast.CompositeLit:
		t.translateExpr(&e.Type)
		t.translateExprList(e.Elts)
	default:
		panic(fmt.Sprintf("unimplemented Expr %T", e))
	}
}

// translateExprList translate an expression list from Go with
// contracts to Go 1.
func (t *translator) translateExprList(el []ast.Expr) {
	for i := range el {
		t.translateExpr(&el[i])
	}
}

// translateFieldList translates a field list from Go with contracts to Go 1.
func (t *translator) translateFieldList(fl *ast.FieldList) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		t.translateField(f)
	}
}

// translateField translates a field from Go with contracts to Go 1.
func (t *translator) translateField(f *ast.Field) {
	t.translateExpr(&f.Type)
}

// translateFunctionInstantiation translates an instantiated function
// to Go 1.
func (t *translator) translateFunctionInstantiation(pe *ast.Expr) {
	call := (*pe).(*ast.CallExpr)
	fnident, ok := call.Fun.(*ast.Ident)
	if !ok {
		panic("instantiated function non-ident")
	}

	types := make([]types.Type, 0, len(call.Args))
	for _, arg := range call.Args {
		types = append(types, t.lookupType(arg))
	}

	instantiations := t.instantiations[fnident]
	for _, inst := range instantiations {
		if t.sameTypes(types, inst.types) {
			*pe = inst.decl
			return
		}
	}

	instIdent, err := t.instantiateFunction(fnident, call.Args, types)
	if err != nil {
		t.err = err
		return
	}

	n := &instantiation{
		types: types,
		decl:  instIdent,
	}
	t.instantiations[fnident] = append(instantiations, n)

	*pe = instIdent
}

// sameTypes reports whether two type slices are the same.
func (t *translator) sameTypes(a, b []types.Type) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		if x != b[i] {
			return false
		}
	}
	return true
}
