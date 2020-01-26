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

// addIDs finds IDs for generic functions and types and adds them to a map.
func addIDs(info *types.Info, f *ast.File, mf map[types.Object]*ast.FuncDecl, mt map[types.Object]*ast.TypeSpec) {
	for _, decl := range f.Decls {
		switch decl := decl.(type) {
		case *ast.FuncDecl:
			if isParameterizedFuncDecl(decl, info) {
				obj, ok := info.Defs[decl.Name]
				if !ok {
					panic(fmt.Sprintf("no types.Object for %q", decl.Name.Name))
				}
				mf[obj] = decl
			}
		case *ast.GenDecl:
			if decl.Tok == token.TYPE {
				for _, s := range decl.Specs {
					ts := s.(*ast.TypeSpec)
					obj, ok := info.Defs[ts.Name]
					if !ok {
						panic(fmt.Sprintf("no types.Object for %q", ts.Name.Name))
					}
					mt[obj] = ts
				}
			}
		}
	}
}

// isParameterizedFuncDecl reports whether fd is a parameterized function.
func isParameterizedFuncDecl(fd *ast.FuncDecl, info *types.Info) bool {
	if fd.Type.TParams != nil {
		return true
	}
	if fd.Recv != nil {
		rtyp := info.TypeOf(fd.Recv.List[0].Type)
		if rtyp == nil {
			// Already instantiated.
			return false
		}
		if p, ok := rtyp.(*types.Pointer); ok {
			rtyp = p.Elem()
		}
		if named, ok := rtyp.(*types.Named); ok {
			if named.TParams() != nil {
				return true
			}
		}
	}
	return false
}

// isParameterizedTypeDecl reports whether s is a parameterized type.
func isParameterizedTypeDecl(s ast.Spec) bool {
	ts := s.(*ast.TypeSpec)
	return ts.TParams != nil
}

// A translator is used to translate a file from Go with contracts to Go 1.
type translator struct {
	fset               *token.FileSet
	info               *types.Info
	types              map[ast.Expr]types.Type
	idToFunc           map[types.Object]*ast.FuncDecl
	idToTypeSpec       map[types.Object]*ast.TypeSpec
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
	decl  *ast.Ident
	typ   types.Type
}

// rewrite rewrites the contents of one file.
func rewriteFile(dir string, fset *token.FileSet, info *types.Info, idToFunc map[types.Object]*ast.FuncDecl, idToTypeSpec map[types.Object]*ast.TypeSpec, filename string, file *ast.File) (err error) {
	if err := rewriteAST(fset, info, idToFunc, idToTypeSpec, file); err != nil {
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
func rewriteAST(fset *token.FileSet, info *types.Info, idToFunc map[types.Object]*ast.FuncDecl, idToTypeSpec map[types.Object]*ast.TypeSpec, file *ast.File) (err error) {
	t := translator{
		fset:               fset,
		info:               info,
		types:              make(map[ast.Expr]types.Type),
		idToFunc:           idToFunc,
		idToTypeSpec:       idToTypeSpec,
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
			case *ast.FuncDecl:
				if !isParameterizedFuncDecl(decl, t.info) {
					t.translateFuncDecl(&declsToDo[i])
					newDecls = append(newDecls, decl)
				}
			case *ast.GenDecl:
				switch decl.Tok {
				case token.TYPE:
					newSpecs := make([]ast.Spec, 0, len(decl.Specs))
					for j := range decl.Specs {
						if !isParameterizedTypeDecl(decl.Specs[j]) {
							t.translateTypeSpec(&decl.Specs[j])
							newSpecs = append(newSpecs, decl.Specs[j])
						}
					}
					if len(newSpecs) == 0 {
						decl = nil
					} else {
						decl.Specs = newSpecs
					}
				case token.VAR, token.CONST:
					for j := range decl.Specs {
						t.translateValueSpec(&decl.Specs[j])
					}
				}
				if decl != nil {
					newDecls = append(newDecls, decl)
				}
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
	if ts.TParams != nil {
		panic("parameterized type")
	}
	t.translateExpr(&ts.Type)
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
	if *ps == nil {
		return
	}
	switch s := (*ps).(type) {
	case *ast.BlockStmt:
		t.translateBlockStmt(s)
	case *ast.ExprStmt:
		t.translateExpr(&s.X)
	case *ast.AssignStmt:
		t.translateExprList(s.Lhs)
		t.translateExprList(s.Rhs)
	case *ast.IfStmt:
		t.translateStmt(&s.Init)
		t.translateExpr(&s.Cond)
		t.translateBlockStmt(s.Body)
		t.translateStmt(&s.Else)
	case *ast.RangeStmt:
		t.translateExpr(&s.Key)
		t.translateExpr(&s.Value)
		t.translateExpr(&s.X)
		t.translateBlockStmt(s.Body)
	case *ast.DeclStmt:
		d := s.Decl.(*ast.GenDecl)
		switch d.Tok {
		case token.TYPE:
			for i := range d.Specs {
				t.translateTypeSpec(&d.Specs[i])
			}
		case token.CONST, token.VAR:
			for i := range d.Specs {
				t.translateValueSpec(&d.Specs[i])
			}
		default:
			panic(fmt.Sprintf("unknown decl type %v", d.Tok))
		}
	case *ast.ReturnStmt:
		t.translateExprList(s.Results)
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
	case *ast.ParenExpr:
		t.translateExpr(&e.X)
	case *ast.BinaryExpr:
		t.translateExpr(&e.X)
		t.translateExpr(&e.Y)
	case *ast.UnaryExpr:
		t.translateExpr(&e.X)
	case *ast.CallExpr:
		t.translateExprList(e.Args)
		if ftyp, ok := t.lookupType(e.Fun).(*types.Signature); ok && ftyp.TParams() != nil {
			t.translateFunctionInstantiation(pe)
		} else if ntyp, ok := t.lookupType(e.Fun).(*types.Named); ok && ntyp.TParams() != nil {
			t.translateTypeInstantiation(pe)
		}
		t.translateExpr(&e.Fun)
	case *ast.StarExpr:
		t.translateExpr(&e.X)
	case *ast.SelectorExpr:
		t.translateExpr(&e.X)
	case *ast.ArrayType:
		t.translateExpr(&e.Elt)
	case *ast.StructType:
		t.translateFieldList(e.Fields)
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

// translateTypeInstantiation translates an instantiated type to Go 1.
func (t *translator) translateTypeInstantiation(pe *ast.Expr) {
	call := (*pe).(*ast.CallExpr)
	tident, ok := call.Fun.(*ast.Ident)
	if !ok {
		panic("instantiated type non-ident")
	}

	typ := t.lookupType(call.Fun).(*types.Named)

	types := make([]types.Type, 0, len(call.Args))
	for _, arg := range call.Args {
		types = append(types, t.lookupType(arg))
	}

	instantiations := t.typeInstantiations[typ]
	for _, inst := range instantiations {
		if t.sameTypes(types, inst.types) {
			*pe = inst.decl
			return
		}
	}

	instIdent, instType, err := t.instantiateTypeDecl(tident, typ, call.Args, types)
	if err != nil {
		t.err = err
		return
	}

	n := &typeInstantiation{
		types: types,
		decl:  instIdent,
		typ:   instType,
	}
	t.typeInstantiations[typ] = append(instantiations, n)

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
