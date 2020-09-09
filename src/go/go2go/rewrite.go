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
	"sort"
	"strconv"
	"strings"
)

var config = printer.Config{
	Mode:     printer.UseSpaces | printer.TabIndent | printer.SourcePos,
	Tabwidth: 8,
}

// isParameterizedFuncDecl reports whether fd is a parameterized function.
func isParameterizedFuncDecl(fd *ast.FuncDecl, info *types.Info) bool {
	if fd.Type.TParams != nil && len(fd.Type.TParams.List) > 0 {
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

// isTranslatableType reports whether a type spec can be translated to Go1.
// This is false if the type spec relies on any features that use generics.
func isTranslatableType(s ast.Spec, info *types.Info) bool {
	if isParameterizedTypeDecl(s, info) {
		return false
	}
	if isTypeBound(s, info) {
		return false
	}
	if embedsComparable(s, info) {
		return false
	}
	return true
}

// isParameterizedTypeDecl reports whether s is a parameterized type.
func isParameterizedTypeDecl(s ast.Spec, info *types.Info) bool {
	ts := s.(*ast.TypeSpec)
	if ts.TParams != nil && len(ts.TParams.List) > 0 {
		return true
	}
	if ts.Assign == token.NoPos {
		return false
	}

	// This is a type alias. Try to resolve it.
	typ := info.TypeOf(ts.Type)
	if typ == nil {
		return false
	}
	named, ok := typ.(*types.Named)
	if !ok {
		return false
	}
	return len(named.TParams()) > 0 && len(named.TArgs()) == 0
}

// isTypeBound reports whether s is an interface type that includes a
// type bound or that embeds an interface that must be a type bound.
func isTypeBound(s ast.Spec, info *types.Info) bool {
	typ := info.TypeOf(s.(*ast.TypeSpec).Type)
	if typ == nil {
		return false
	}
	if iface, ok := typ.Underlying().(*types.Interface); ok {
		if iface.HasTypeList() {
			return true
		}
	}
	return false
}

// embedsComparable reports whether s is an interface type that embeds
// the predeclared type "comparable", directly or indirectly.
func embedsComparable(s ast.Spec, info *types.Info) bool {
	typ := info.TypeOf(s.(*ast.TypeSpec).Type)
	return typeEmbedsComparable(typ)
}

// typeEmbedsComparable reports whether typ is an interface type
// that embeds the predeclared type "comparable", directly or indirectly.
// This is like embedsComparable, but for a types.Type.
func typeEmbedsComparable(typ types.Type) bool {
	if typ == nil {
		return false
	}
	iface, ok := typ.Underlying().(*types.Interface)
	if !ok {
		return false
	}
	n := iface.NumEmbeddeds()
	if n == 0 {
		return false
	}
	comparable := types.Universe.Lookup("comparable")
	for i := 0; i < n; i++ {
		et := iface.EmbeddedType(i)
		if named, ok := et.(*types.Named); ok && named.Obj() == comparable {
			return true
		}
		if typeEmbedsComparable(et) {
			return true
		}
	}
	return false
}

// A translator is used to translate a file from generic Go to Go 1.
type translator struct {
	fset         *token.FileSet
	importer     *Importer
	tpkg         *types.Package
	types        map[ast.Expr]types.Type
	newDecls     []ast.Decl
	typePackages map[*types.Package]bool

	// typeDepth tracks recursive type instantiations.
	typeDepth int

	// err is set if we have seen an error during this translation.
	// This is used by the rewrite methods.
	err error
}

// instantiations tracks all function and type instantiations for a package.
type instantiations struct {
	funcInstantiations map[string][]*funcInstantiation
	typeInstantiations map[types.Type][]*typeInstantiation
}

// A funcInstantiation is a single instantiation of a function.
type funcInstantiation struct {
	types []types.Type
	decl  *ast.Ident
}

// A typeInstantiation is a single instantiation of a type.
type typeInstantiation struct {
	types      []types.Type
	decl       *ast.Ident
	typ        types.Type
	inProgress bool
}

// funcInstantiations fetches the function instantiations defined in
// the current package, given a generic function name.
func (t *translator) funcInstantiations(key string) []*funcInstantiation {
	insts := t.importer.instantiations[t.tpkg]
	if insts == nil {
		return nil
	}
	return insts.funcInstantiations[key]
}

// addFuncInstantiation adds a new function instantiation.
func (t *translator) addFuncInstantiation(key string, inst *funcInstantiation) {
	insts := t.pkgInstantiations()
	insts.funcInstantiations[key] = append(insts.funcInstantiations[key], inst)
}

// typeInstantiations fetches the type instantiations defined in
// the current package, given a generic type.
func (t *translator) typeInstantiations(typ types.Type) []*typeInstantiation {
	insts := t.importer.instantiations[t.tpkg]
	if insts == nil {
		return nil
	}
	return insts.typeInstantiations[typ]
}

// addTypeInstantiations adds a new type instantiation.
func (t *translator) addTypeInstantiation(typ types.Type, inst *typeInstantiation) {
	insts := t.pkgInstantiations()
	insts.typeInstantiations[typ] = append(insts.typeInstantiations[typ], inst)
}

// pkgInstantiations returns the instantiations structure for the current
// package, creating it if necessary.
func (t *translator) pkgInstantiations() *instantiations {
	insts := t.importer.instantiations[t.tpkg]
	if insts == nil {
		insts = &instantiations{
			funcInstantiations: make(map[string][]*funcInstantiation),
			typeInstantiations: make(map[types.Type][]*typeInstantiation),
		}
		t.importer.instantiations[t.tpkg] = insts
	}
	return insts
}

// rewrite rewrites the contents of one file.
func rewriteFile(dir string, fset *token.FileSet, importer *Importer, importPath string, tpkg *types.Package, filename string, file *ast.File, addImportableName bool) (err error) {
	if err := rewriteAST(fset, importer, importPath, tpkg, file, addImportableName); err != nil {
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
func rewriteAST(fset *token.FileSet, importer *Importer, importPath string, tpkg *types.Package, file *ast.File, addImportableName bool) (err error) {
	t := translator{
		fset:         fset,
		importer:     importer,
		tpkg:         tpkg,
		types:        make(map[ast.Expr]types.Type),
		typePackages: make(map[*types.Package]bool),
	}
	t.translate(file)

	// Add all the transitive imports. This is more than we need,
	// but we're not trying to be elegant here.
	imps := make(map[string]bool)

	for _, p := range importer.transitiveImports(importPath) {
		imps[p] = true
	}
	for pkg := range t.typePackages {
		if pkg != t.tpkg {
			imps[pkg.Path()] = true
		}
	}

	decls := make([]ast.Decl, 0, len(file.Decls))
	var specs []ast.Spec
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT {
			decls = append(decls, decl)
			continue
		}
		for _, spec := range gen.Specs {
			imp := spec.(*ast.ImportSpec)
			if imp.Name != nil {
				specs = append(specs, imp)
			}
			// We picked up Go 2 imports above, but we still
			// need to pick up Go 1 imports here.
			path, err := strconv.Unquote(imp.Path.Value)
			if err != nil || imps[path] {
				continue
			}
			if imp.Name == nil {
				// If Name != nil we are keeping the spec.
				// Don't add the import again here,
				// only if it is needed elsewhere.
				imps[path] = true
			}
			for _, p := range importer.transitiveImports(path) {
				imps[p] = true
			}
		}
	}
	file.Decls = decls

	// If we have a ./ import, let it override a standard import
	// we may have added due to t.typePackages.
	for path := range imps {
		if strings.HasPrefix(path, "./") {
			delete(imps, strings.TrimPrefix(path, "./"))
		}
	}

	paths := make([]string, 0, len(imps))
	for p := range imps {
		paths = append(paths, p)
	}
	sort.Strings(paths)

	for _, p := range paths {
		specs = append(specs, ast.Spec(&ast.ImportSpec{
			Path: &ast.BasicLit{
				Kind:  token.STRING,
				Value: strconv.Quote(p),
			},
		}))
	}
	if len(specs) > 0 {
		first := &ast.GenDecl{
			Tok:   token.IMPORT,
			Specs: specs,
		}
		file.Decls = append([]ast.Decl{first}, file.Decls...)
	}

	// Add a name that other packages can reference to avoid an error
	// about an unused package.
	if addImportableName {
		file.Decls = append(file.Decls,
			&ast.GenDecl{
				Tok: token.TYPE,
				Specs: []ast.Spec{
					&ast.TypeSpec{
						Name: ast.NewIdent(t.importableName()),
						Type: ast.NewIdent("int"),
					},
				},
			})
	}

	// Add a reference for each imported package to avoid an error
	// about an unused package.
	for _, decl := range file.Decls {
		gen, ok := decl.(*ast.GenDecl)
		if !ok || gen.Tok != token.IMPORT {
			continue
		}
		for _, spec := range gen.Specs {
			imp := spec.(*ast.ImportSpec)
			if imp.Name != nil && imp.Name.Name == "_" {
				continue
			}
			path := strings.TrimPrefix(strings.TrimSuffix(imp.Path.Value, `"`), `"`)
			var pname string

			var tok token.Token
			var importableName string
			if pkg, ok := importer.lookupPackage(path); ok {
				tok = token.TYPE
				importableName = t.importableName()
				pname = pkg.Name()
			} else {
				fileDir := filepath.Dir(fset.Position(file.Name.Pos()).Filename)
				pkg, err := importer.ImportFrom(path, fileDir, 0)
				if err != nil {
					return err
				}
				scope := pkg.Scope()
				pname = pkg.Name()
				names := scope.Names()
			nameLoop:
				for _, name := range names {
					if !token.IsExported(name) {
						continue
					}
					obj := scope.Lookup(name)
					switch obj.(type) {
					case *types.TypeName:
						tok = token.TYPE
						importableName = name
						break nameLoop
					case *types.Var, *types.Func:
						tok = token.VAR
						importableName = name
						break nameLoop
					case *types.Const:
						tok = token.CONST
						importableName = name
						break nameLoop
					}
				}
				if importableName == "" {
					return fmt.Errorf("can't find any importable name in package %q", path)
				}
			}

			var name string
			if imp.Name != nil {
				name = imp.Name.Name
			} else {
				name = pname
			}
			var spec ast.Spec
			switch tok {
			case token.CONST, token.VAR:
				spec = &ast.ValueSpec{
					Names: []*ast.Ident{
						ast.NewIdent("_"),
					},
					Values: []ast.Expr{
						&ast.SelectorExpr{
							X:   ast.NewIdent(name),
							Sel: ast.NewIdent(importableName),
						},
					},
				}
			case token.TYPE:
				spec = &ast.TypeSpec{
					Name: ast.NewIdent("_"),
					Type: &ast.SelectorExpr{
						X:   ast.NewIdent(name),
						Sel: ast.NewIdent(importableName),
					},
				}
			default:
				panic("can't happen")
			}
			file.Decls = append(file.Decls,
				&ast.GenDecl{
					Tok:   tok,
					Specs: []ast.Spec{spec},
				})
		}
	}

	return t.err
}

// translate translates the AST for a file from generic Go to Go 1.
func (t *translator) translate(file *ast.File) {
	declsToDo := file.Decls
	file.Decls = nil
	c := 0
	for len(declsToDo) > 0 {
		if c > 50 {
			var sb strings.Builder
			printer.Fprint(&sb, t.fset, declsToDo[0])
			t.err = fmt.Errorf("looping while expanding %v", &sb)
			return
		}
		c++

		newDecls := make([]ast.Decl, 0, len(declsToDo))
		for i, decl := range declsToDo {
			switch decl := decl.(type) {
			case *ast.FuncDecl:
				if !isParameterizedFuncDecl(decl, t.importer.info) {
					t.translateFuncDecl(&declsToDo[i])
					newDecls = append(newDecls, decl)
				}
			case *ast.GenDecl:
				switch decl.Tok {
				case token.TYPE:
					newSpecs := make([]ast.Spec, 0, len(decl.Specs))
					for j := range decl.Specs {
						if isTranslatableType(decl.Specs[j], t.importer.info) {
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

// translateTypeSpec translates a type from generic Go to Go 1.
func (t *translator) translateTypeSpec(ps *ast.Spec) {
	ts := (*ps).(*ast.TypeSpec)
	if ts.TParams != nil && len(ts.TParams.List) > 0 {
		t.err = fmt.Errorf("%s: go2go tool does not support parameterized type here", t.fset.Position((*ps).Pos()))
		return
	}
	t.translateExpr(&ts.Type)
}

// translateValueSpec translates a variable or constant from generic Go to Go 1.
func (t *translator) translateValueSpec(ps *ast.Spec) {
	vs := (*ps).(*ast.ValueSpec)
	t.translateExpr(&vs.Type)
	for i := range vs.Values {
		t.translateExpr(&vs.Values[i])
	}
}

// translateFuncDecl translates a function from generic Go to Go 1.
func (t *translator) translateFuncDecl(pd *ast.Decl) {
	if t.err != nil {
		return
	}
	fd := (*pd).(*ast.FuncDecl)
	if fd.Type.TParams != nil {
		if len(fd.Type.TParams.List) > 0 {
			panic("parameterized function")
		}
		fd.Type.TParams = nil
	}
	if fd.Recv != nil {
		t.translateFieldList(fd.Recv)
	}
	t.translateFieldList(fd.Type.Params)
	t.translateFieldList(fd.Type.Results)
	t.translateBlockStmt(fd.Body)
}

// translateBlockStmt translates a block statement from generic Go to Go 1.
func (t *translator) translateBlockStmt(pbs *ast.BlockStmt) {
	if pbs == nil {
		return
	}
	for i := range pbs.List {
		t.translateStmt(&pbs.List[i])
	}
}

// translateStmt translates a statement from generic Go to Go 1.
func (t *translator) translateStmt(ps *ast.Stmt) {
	if t.err != nil {
		return
	}
	if *ps == nil {
		return
	}
	switch s := (*ps).(type) {
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
	case *ast.EmptyStmt:
	case *ast.LabeledStmt:
		t.translateStmt(&s.Stmt)
	case *ast.ExprStmt:
		t.translateExpr(&s.X)
	case *ast.SendStmt:
		t.translateExpr(&s.Chan)
		t.translateExpr(&s.Value)
	case *ast.IncDecStmt:
		t.translateExpr(&s.X)
	case *ast.AssignStmt:
		t.translateExprList(s.Lhs)
		t.translateExprList(s.Rhs)
	case *ast.GoStmt:
		e := ast.Expr(s.Call)
		t.translateExpr(&e)
		s.Call = e.(*ast.CallExpr)
	case *ast.DeferStmt:
		e := ast.Expr(s.Call)
		t.translateExpr(&e)
		s.Call = e.(*ast.CallExpr)
	case *ast.ReturnStmt:
		t.translateExprList(s.Results)
	case *ast.BranchStmt:
	case *ast.BlockStmt:
		t.translateBlockStmt(s)
	case *ast.IfStmt:
		t.translateStmt(&s.Init)
		t.translateExpr(&s.Cond)
		t.translateBlockStmt(s.Body)
		t.translateStmt(&s.Else)
	case *ast.CaseClause:
		t.translateExprList(s.List)
		t.translateStmtList(s.Body)
	case *ast.SwitchStmt:
		t.translateStmt(&s.Init)
		t.translateExpr(&s.Tag)
		t.translateBlockStmt(s.Body)
	case *ast.TypeSwitchStmt:
		t.translateStmt(&s.Init)
		t.translateStmt(&s.Assign)
		t.translateBlockStmt(s.Body)
	case *ast.CommClause:
		t.translateStmt(&s.Comm)
		t.translateStmtList(s.Body)
	case *ast.SelectStmt:
		t.translateBlockStmt(s.Body)
	case *ast.ForStmt:
		t.translateStmt(&s.Init)
		t.translateExpr(&s.Cond)
		t.translateStmt(&s.Post)
		t.translateBlockStmt(s.Body)
	case *ast.RangeStmt:
		t.translateExpr(&s.Key)
		t.translateExpr(&s.Value)
		t.translateExpr(&s.X)
		t.translateBlockStmt(s.Body)
	default:
		panic(fmt.Sprintf("unimplemented Stmt %T", s))
	}
}

// translateStmtList translates a list of statements generic Go to Go 1.
func (t *translator) translateStmtList(sl []ast.Stmt) {
	for i := range sl {
		t.translateStmt(&sl[i])
	}
}

// translateExpr translates an expression from generic Go to Go 1.
func (t *translator) translateExpr(pe *ast.Expr) {
	if t.err != nil {
		return
	}
	if *pe == nil {
		return
	}
	switch e := (*pe).(type) {
	case *ast.Ident:
		t.translateIdent(pe)
	case *ast.Ellipsis:
		t.translateExpr(&e.Elt)
	case *ast.BasicLit:
	case *ast.FuncLit:
		t.translateFieldList(e.Type.TParams)
		t.translateFieldList(e.Type.Params)
		t.translateFieldList(e.Type.Results)
		t.translateBlockStmt(e.Body)
	case *ast.CompositeLit:
		t.translateExpr(&e.Type)
		t.translateExprList(e.Elts)
	case *ast.ParenExpr:
		t.translateExpr(&e.X)
	case *ast.SelectorExpr:
		t.translateSelectorExpr(pe)
	case *ast.IndexExpr:
		if ftyp, ok := t.lookupType(e.X).(*types.Signature); ok && len(ftyp.TParams()) > 0 {
			t.translateFunctionInstantiation(pe)
		} else if ntyp, ok := t.lookupType(e.X).(*types.Named); ok && len(ntyp.TParams()) > 0 && len(ntyp.TArgs()) == 0 {
			t.translateTypeInstantiation(pe)
		}
		t.translateExpr(&e.X)
		t.translateExpr(&e.Index)
	case *ast.SliceExpr:
		t.translateExpr(&e.X)
		t.translateExpr(&e.Low)
		t.translateExpr(&e.High)
		t.translateExpr(&e.Max)
	case *ast.TypeAssertExpr:
		t.translateExpr(&e.X)
		t.translateExpr(&e.Type)
	case *ast.CallExpr:
		if ftyp, ok := t.lookupType(e.Fun).(*types.Signature); ok && len(ftyp.TParams()) > 0 {
			t.translateFunctionInstantiation(pe)
		} else if ntyp, ok := t.lookupType(e.Fun).(*types.Named); ok && len(ntyp.TParams()) > 0 && len(ntyp.TArgs()) == 0 {
			t.translateTypeInstantiation(pe)
		}
		t.translateExprList(e.Args)
		t.translateExpr(&e.Fun)
	case *ast.StarExpr:
		t.translateExpr(&e.X)
	case *ast.UnaryExpr:
		t.translateExpr(&e.X)
	case *ast.BinaryExpr:
		t.translateExpr(&e.X)
		t.translateExpr(&e.Y)
	case *ast.KeyValueExpr:
		t.translateExpr(&e.Key)
		t.translateExpr(&e.Value)
	case *ast.ArrayType:
		t.translateExpr(&e.Len)
		t.translateExpr(&e.Elt)
	case *ast.StructType:
		t.translateFieldList(e.Fields)
	case *ast.FuncType:
		t.translateFieldList(e.TParams)
		t.translateFieldList(e.Params)
		t.translateFieldList(e.Results)
	case *ast.InterfaceType:
		methods, types := splitFieldList(e.Methods)
		t.translateFieldList(methods)
		t.translateExprList(types)
	case *ast.MapType:
		t.translateExpr(&e.Key)
		t.translateExpr(&e.Value)
	case *ast.ChanType:
		t.translateExpr(&e.Value)
	default:
		panic(fmt.Sprintf("unimplemented Expr %T", e))
	}
}

// translateIdent translates a simple identifier generic Go to Go 1.
// These are usually fine as is, but a reference
// to a non-generic name in another package may need a package qualifier.
func (t *translator) translateIdent(pe *ast.Expr) {
	e := (*pe).(*ast.Ident)
	obj := t.importer.info.ObjectOf(e)
	if obj == nil {
		return
	}
	if named, ok := obj.Type().(*types.Named); ok && len(named.TParams()) > 0 {
		// A generic function that will be instantiated locally.
		return
	}
	ipkg := obj.Pkg()
	if ipkg == nil || ipkg == t.tpkg {
		// We don't need a package qualifier if it's defined
		// in the current package.
		return
	}
	if obj.Parent() != ipkg.Scope() {
		// We only need a package qualifier if it's defined in
		// package scope.
		return
	}

	// Add package qualifier.
	*pe = &ast.SelectorExpr{
		X:   ast.NewIdent(ipkg.Name()),
		Sel: e,
	}
}

// translateSelectorExpr translates a selector expression
// from generic Go to Go 1.
func (t *translator) translateSelectorExpr(pe *ast.Expr) {
	e := (*pe).(*ast.SelectorExpr)

	t.translateExpr(&e.X)

	obj := t.importer.info.ObjectOf(e.Sel)
	if obj == nil {
		return
	}

	// Handle references to promoted fields and methods,
	// if they go through an embedded instantiated field.
	// We have to add a reference to the field we inserted.
	if xType := t.lookupType(e.X); xType != nil {
		if ptr := xType.Pointer(); ptr != nil {
			xType = ptr.Elem()
		}
		fobj, indexes, _ := types.LookupFieldOrMethod(xType, true, obj.Pkg(), obj.Name())
		if fobj != nil && len(indexes) > 1 {
			for _, index := range indexes[:len(indexes)-1] {
				xf := xType.Struct().Field(index)
				// This must be an embedded type.
				// If the field name is the one we expect,
				// don't mention it explicitly,
				// because it might not be exported.
				if xf.Name() == types.TypeString(xf.Type(), relativeTo(xf.Pkg())) {
					continue
				}
				e.X = &ast.SelectorExpr{
					X:   e.X,
					Sel: ast.NewIdent(xf.Name()),
				}
				xType = xf.Type()
				if ptr := xType.Pointer(); ptr != nil {
					xType = ptr.Elem()
				}
			}
		}
	}

	// Handle references to instantiated embedded fields.
	// We have to rewrite the name to the name used in
	// the translated struct definition.
	if f, ok := obj.(*types.Var); ok && f.Embedded() {
		typ := t.lookupType(e)
		if typ == nil {
			typ = f.Type()
		}
		if pt := typ.Pointer(); pt != nil {
			typ = pt.Elem()
		}
		named, ok := typ.(*types.Named)
		if !ok || len(named.TArgs()) == 0 {
			return
		}
		if obj.Name() != named.Obj().Name() {
			return
		}
		_, id := t.lookupInstantiatedType(named)
		*pe = &ast.SelectorExpr{
			X:   e.X,
			Sel: id,
		}
	}
}

// TODO(iant) refactor code and get rid of this?
func splitFieldList(fl *ast.FieldList) (methods *ast.FieldList, types []ast.Expr) {
	if fl == nil {
		return
	}
	var mfields []*ast.Field
	for _, f := range fl.List {
		if len(f.Names) > 0 && f.Names[0].Name == "type" {
			// type list type
			types = append(types, f.Type)
		} else {
			mfields = append(mfields, f)
		}
	}
	copy := *fl
	copy.List = mfields
	methods = &copy
	return
}

// TODO(iant) refactor code and get rid of this?
func mergeFieldList(methods *ast.FieldList, types []ast.Expr) (fl *ast.FieldList) {
	fl = methods
	if len(types) == 0 {
		return
	}
	if fl == nil {
		fl = new(ast.FieldList)
	}
	name := []*ast.Ident{ast.NewIdent("type")}
	for _, typ := range types {
		fl.List = append(fl.List, &ast.Field{Names: name, Type: typ})
	}
	return
}

// translateExprList translate an expression list generic Go to Go 1.
func (t *translator) translateExprList(el []ast.Expr) {
	for i := range el {
		t.translateExpr(&el[i])
	}
}

// translateFieldList translates a field list generic Go to Go 1.
func (t *translator) translateFieldList(fl *ast.FieldList) {
	if fl == nil {
		return
	}
	for _, f := range fl.List {
		t.translateField(f)
	}
}

// translateField translates a field generic Go to Go 1.
func (t *translator) translateField(f *ast.Field) {
	t.translateExpr(&f.Type)
}

// translateFunctionInstantiation translates an instantiated function
// to Go 1.
func (t *translator) translateFunctionInstantiation(pe *ast.Expr) {
	expr := *pe
	qid := t.instantiatedIdent(expr)
	argList, typeList, typeArgs := t.instantiationTypes(expr)
	if t.err != nil {
		return
	}

	var instIdent *ast.Ident
	key := qid.String()
	insts := t.funcInstantiations(key)
	for _, inst := range insts {
		if t.sameTypes(typeList, inst.types) {
			instIdent = inst.decl
			break
		}
	}

	if instIdent == nil {
		var err error
		instIdent, err = t.instantiateFunction(qid, argList, typeList)
		if err != nil {
			t.err = err
			return
		}

		n := &funcInstantiation{
			types: typeList,
			decl:  instIdent,
		}
		t.addFuncInstantiation(key, n)
	}

	if typeArgs {
		*pe = instIdent
	} else {
		switch e := expr.(type) {
		case *ast.CallExpr:
			newCall := *e
			newCall.Fun = instIdent
			*pe = &newCall
		case *ast.IndexExpr:
			*pe = instIdent
		default:
			panic("unexpected AST type")
		}
	}
}

// translateTypeInstantiation translates an instantiated type to Go 1.
func (t *translator) translateTypeInstantiation(pe *ast.Expr) {
	expr := *pe
	qid := t.instantiatedIdent(expr)
	typ := t.lookupType(qid.ident).(*types.Named)
	argList, typeList, typeArgs := t.instantiationTypes(expr)
	if t.err != nil {
		return
	}
	if !typeArgs {
		panic("no type arguments for type")
	}

	var seen *typeInstantiation
	key := t.typeWithoutArgs(typ)
	for _, inst := range t.typeInstantiations(key) {
		if t.sameTypes(typeList, inst.types) {
			if inst.inProgress {
				panic(fmt.Sprintf("%s: circular type instantiation", t.fset.Position((*pe).Pos())))
			}
			if inst.decl == nil {
				// This can happen if we've instantiated
				// the type in instantiateType.
				seen = inst
				break
			}
			*pe = inst.decl
			return
		}
	}

	name, err := t.instantiatedName(qid, typeList)
	if err != nil {
		t.err = err
		return
	}
	instIdent := ast.NewIdent(name)

	if seen != nil {
		seen.decl = instIdent
		seen.inProgress = true
	} else {
		seen = &typeInstantiation{
			types:      typeList,
			decl:       instIdent,
			typ:        nil,
			inProgress: true,
		}
		t.addTypeInstantiation(key, seen)
	}

	defer func() {
		seen.inProgress = false
	}()

	instType, err := t.instantiateTypeDecl(qid, typ, argList, typeList, instIdent)
	if err != nil {
		t.err = err
		return
	}

	if seen.typ == nil {
		seen.typ = instType
	}

	*pe = instIdent
}

// instantiatedIdent returns the qualified identifer that is being
// instantiated.
func (t *translator) instantiatedIdent(x ast.Expr) qualifiedIdent {
	var fun ast.Expr
	switch x := x.(type) {
	case *ast.CallExpr:
		fun = x.Fun
	case *ast.IndexExpr:
		fun = x.X
	default:
		panic(fmt.Sprintf("unexpected AST %T", x))
	}

	switch fun := fun.(type) {
	case *ast.Ident:
		if obj := t.importer.info.ObjectOf(fun); obj != nil && obj.Pkg() != t.tpkg {
			return qualifiedIdent{pkg: obj.Pkg(), ident: fun}
		}
		return qualifiedIdent{ident: fun}
	case *ast.SelectorExpr:
		pkgname, ok := fun.X.(*ast.Ident)
		if !ok {
			break
		}
		pkgobj, ok := t.importer.info.Uses[pkgname]
		if !ok {
			break
		}
		pn, ok := pkgobj.(*types.PkgName)
		if !ok {
			break
		}
		return qualifiedIdent{pkg: pn.Imported(), ident: fun.Sel}
	}
	panic(fmt.Sprintf("instantiated object %T %v is not an identifier", fun, fun))
}

// instantiationTypes returns the type arguments of an instantiation.
// It also returns the AST arguments if they are present.
// The typeArgs result reports whether the AST arguments are types.
func (t *translator) instantiationTypes(x ast.Expr) (argList []ast.Expr, typeList []types.Type, typeArgs bool) {
	var args []ast.Expr
	switch x := x.(type) {
	case *ast.CallExpr:
		args = x.Args
	case *ast.IndexExpr:
		args = []ast.Expr{x.Index}
	default:
		panic("unexpected AST type")
	}
	inferred, haveInferred := t.importer.info.Inferred[x]

	if !haveInferred {
		argList = args
		typeList = make([]types.Type, 0, len(argList))
		for _, arg := range argList {
			if id, ok := arg.(*ast.Ident); ok && id.Name == "_" {
				t.err = fmt.Errorf("%s: go2go tool does not support using _ here", t.fset.Position(arg.Pos()))
				return
			}
			if at := t.lookupType(arg); at == nil {
				panic(fmt.Sprintf("%s: no type found for %T %v", t.fset.Position(arg.Pos()), arg, arg))
			} else {
				typeList = append(typeList, at)
			}
		}
		typeArgs = true
	} else {
		typeList, argList = t.typeListToASTList(inferred.Targs)
	}

	// Instantiating with a locally defined type won't work.
	// Check that here.
	for _, typ := range typeList {
		if named, ok := typ.(*types.Named); ok && named.Obj().Pkg() != nil {
			if scope := named.Obj().Parent(); scope != nil && scope != named.Obj().Pkg().Scope() {
				t.err = fmt.Errorf("%s: go2go tool does not support using locally defined type as type argument", t.fset.Position(x.Pos()))
				return
			}
		}
	}

	return
}

// lookupInstantiatedType looks for an existing instantiation of an
// instantiated type.
func (t *translator) lookupInstantiatedType(typ *types.Named) (types.Type, *ast.Ident) {
	copyType := func(typ *types.Named, newName string) types.Type {
		nm := typ.NumMethods()
		methods := make([]*types.Func, 0, nm)
		for i := 0; i < nm; i++ {
			methods = append(methods, typ.Method(i))
		}
		obj := typ.Obj()
		obj = types.NewTypeName(obj.Pos(), obj.Pkg(), newName, nil)
		nt := types.NewNamed(obj, typ.Underlying(), methods)
		nt.SetTArgs(typ.TArgs())
		return nt
	}

	targs := typ.TArgs()
	key := t.typeWithoutArgs(typ)
	var seen *typeInstantiation
	for _, inst := range t.typeInstantiations(key) {
		if t.sameTypes(targs, inst.types) {
			if inst.inProgress {
				panic(fmt.Sprintf("instantiation for %v in progress", typ))
			}
			if inst.decl == nil {
				// This can happen if we've instantiated
				// the type in instantiateType.
				seen = inst
				break
			}
			if inst.typ == nil {
				panic(fmt.Sprintf("no type for instantiation entry for %v", typ))
			}
			if instNamed, ok := inst.typ.(*types.Named); ok {
				return copyType(instNamed, inst.decl.Name), inst.decl
			}
			return inst.typ, inst.decl
		}
	}

	typeList, argList := t.typeListToASTList(targs)

	qid := qualifiedIdent{ident: ast.NewIdent(typ.Obj().Name())}
	if typPkg := typ.Obj().Pkg(); typPkg != t.tpkg {
		qid.pkg = typPkg
	}

	name, err := t.instantiatedName(qid, typeList)
	if err != nil {
		t.err = err
		return nil, nil
	}
	instIdent := ast.NewIdent(name)

	if seen != nil {
		seen.decl = instIdent
		seen.inProgress = true
	} else {
		seen = &typeInstantiation{
			types:      targs,
			decl:       instIdent,
			typ:        nil,
			inProgress: true,
		}
		t.addTypeInstantiation(key, seen)
	}

	defer func() {
		seen.inProgress = false
	}()

	instType, err := t.instantiateTypeDecl(qid, typ, argList, typeList, instIdent)
	if err != nil {
		t.err = err
		return nil, nil
	}

	if seen.typ == nil {
		seen.typ = instType
	} else {
		instType = seen.typ
	}

	if instNamed, ok := instType.(*types.Named); ok {
		return copyType(instNamed, instIdent.Name), instIdent
	}
	return instType, instIdent
}

// typeWithoutArgs takes a named type with arguments and returns the
// same type without arguments.
func (t *translator) typeWithoutArgs(typ *types.Named) *types.Named {
	if len(typ.TArgs()) == 0 {
		return typ
	}
	name := typ.Obj().Name()
	fields := strings.Split(name, ".")
	if len(fields) > 2 {
		panic(fmt.Sprintf("unparseable instantiated name %q", name))
	}
	if len(fields) > 1 {
		name = fields[1]
	}

	tpkg := typ.Obj().Pkg()
	if tpkg == nil {
		panic(fmt.Sprintf("can't find package for %s", name))
	}
	nobj := tpkg.Scope().Lookup(name)
	if nobj == nil {
		panic(fmt.Sprintf("can't find %q in scope of package %q", name, tpkg.Name()))
	}
	return nobj.Type().(*types.Named)
}

// typeListToASTList returns an AST list for a type list,
// as well as an updated type list.
func (t *translator) typeListToASTList(typeList []types.Type) ([]types.Type, []ast.Expr) {
	argList := make([]ast.Expr, 0, len(typeList))
	for _, typ := range typeList {
		argList = append(argList, t.typeToAST(typ))

		// This inferred type may introduce a reference to
		// packages that we don't otherwise import, and that
		// package name may wind up in arg. Record all packages
		// seen in inferred types so that we add import
		// statements for them, just in case.
		t.addTypePackages(typ)
	}
	return typeList, argList
}

// relativeTo is like types.RelativeTo, but returns just the package name,
// not the package path.
func relativeTo(pkg *types.Package) types.Qualifier {
	return func(other *types.Package) string {
		if pkg == other {
			return "" // same package; unqualified
		}
		return other.Name()
	}
}

// sameTypes reports whether two type slices are the same.
func (t *translator) sameTypes(a, b []types.Type) bool {
	if len(a) != len(b) {
		return false
	}
	for i, x := range a {
		if !t.sameType(x, b[i]) {
			return false
		}
	}
	return true
}

// sameType reports whether two types are the same.
// We have to check type arguments ourselves.
func (t *translator) sameType(a, b types.Type) bool {
	if types.IdenticalIgnoreTags(a, b) {
		return true
	}
	if ap, bp := a.Pointer(), b.Pointer(); ap != nil && bp != nil {
		a = ap.Elem()
		b = bp.Elem()
	}
	an, ok := a.(*types.Named)
	if !ok {
		return false
	}
	bn, ok := b.(*types.Named)
	if !ok {
		return false
	}
	if an.Obj().Name() != bn.Obj().Name() {
		return false
	}
	if len(an.TArgs()) == 0 || len(an.TArgs()) != len(bn.TArgs()) {
		return false
	}
	for i, typ := range an.TArgs() {
		if !t.sameType(typ, bn.TArgs()[i]) {
			return false
		}
	}
	return true
}

// qualifiedIdent is an identifier possibly qualified with a package.
type qualifiedIdent struct {
	pkg   *types.Package // identifier's package; nil for current package
	ident *ast.Ident
}

// String returns a printable name for qid.
func (qid qualifiedIdent) String() string {
	if qid.pkg == nil {
		return qid.ident.Name
	}
	return qid.pkg.Path() + "." + qid.ident.Name
}
