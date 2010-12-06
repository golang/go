// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// INCOMPLETE PACKAGE.
// This package implements typechecking of a Go AST.
// The result of the typecheck is an augmented AST
// with object and type information for each identifier.
//
package typechecker

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/scanner"
	"os"
)


// TODO(gri) don't report errors for objects/types that are marked as bad.


const debug = true // set for debugging output


// An importer takes an import path and returns the data describing the
// respective package's exported interface. The data format is TBD.
//
type Importer func(path string) ([]byte, os.Error)


// CheckPackage typechecks a package and augments the AST by setting
// *ast.Object, *ast.Type, and *ast.Scope fields accordingly. If an
// importer is provided, it is used to handle imports, otherwise they
// are ignored (likely leading to typechecking errors).
//
// If errors are reported, the AST may be incompletely augmented (fields
// may be nil) or contain incomplete object, type, or scope information.
//
func CheckPackage(fset *token.FileSet, pkg *ast.Package, importer Importer) os.Error {
	var tc typechecker
	tc.fset = fset
	tc.importer = importer
	tc.checkPackage(pkg)
	return tc.GetError(scanner.Sorted)
}


// CheckFile typechecks a single file, but otherwise behaves like
// CheckPackage. If the complete package consists of more than just
// one file, the file may not typecheck without errors.
//
func CheckFile(fset *token.FileSet, file *ast.File, importer Importer) os.Error {
	// create a single-file dummy package
	pkg := &ast.Package{file.Name.Name, nil, map[string]*ast.File{fset.Position(file.Name.NamePos).Filename: file}}
	return CheckPackage(fset, pkg, importer)
}


// ----------------------------------------------------------------------------
// Typechecker state

type typechecker struct {
	fset *token.FileSet
	scanner.ErrorVector
	importer Importer
	topScope *ast.Scope           // current top-most scope
	cyclemap map[*ast.Object]bool // for cycle detection
	iota     int                  // current value of iota
}


func (tc *typechecker) Errorf(pos token.Pos, format string, args ...interface{}) {
	tc.Error(tc.fset.Position(pos), fmt.Sprintf(format, args...))
}


func assert(pred bool) {
	if !pred {
		panic("internal error")
	}
}


/*
Typechecking is done in several phases:

phase 1: declare all global objects; also collect all function and method declarations
	- all objects have kind, name, decl fields; the decl field permits
	  quick lookup of an object's declaration
	- constant objects have an iota value
	- type objects have unresolved types with empty scopes, all others have nil types
	- report global double declarations

phase 2: bind methods to their receiver base types
	- received base types must be declared in the package, thus for
	  each method a corresponding (unresolved) type must exist
	- report method double declarations and errors with base types

phase 3: resolve all global objects
	- sequentially iterate through all objects in the global scope
	- resolve types for all unresolved types and assign types to
	  all attached methods
	- assign types to all other objects, possibly by evaluating
	  constant and initializer expressions
	- resolution may recurse; a cyclemap is used to detect cycles
	- report global typing errors

phase 4: sequentially typecheck function and method bodies
	- all global objects are declared and have types and values;
	  all methods have types
	- sequentially process statements in each body; any object
	  referred to must be fully defined at this point
	- report local typing errors
*/

func (tc *typechecker) checkPackage(pkg *ast.Package) {
	// setup package scope
	tc.topScope = Universe
	tc.openScope()
	defer tc.closeScope()

	// TODO(gri) there's no file scope at the moment since we ignore imports

	// phase 1: declare all global objects; also collect all function and method declarations
	var funcs []*ast.FuncDecl
	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			tc.declGlobal(decl)
			if f, isFunc := decl.(*ast.FuncDecl); isFunc {
				funcs = append(funcs, f)
			}
		}
	}

	// phase 2: bind methods to their receiver base types
	for _, m := range funcs {
		if m.Recv != nil {
			tc.bindMethod(m)
		}
	}

	// phase 3: resolve all global objects
	// (note that objects with _ name are also in the scope)
	tc.cyclemap = make(map[*ast.Object]bool)
	for _, obj := range tc.topScope.Objects {
		tc.resolve(obj)
	}
	assert(len(tc.cyclemap) == 0)

	// 4: sequentially typecheck function and method bodies
	for _, f := range funcs {
		tc.checkBlock(f.Body.List, f.Name.Obj.Type)
	}

	pkg.Scope = tc.topScope
}


func (tc *typechecker) declGlobal(global ast.Decl) {
	switch d := global.(type) {
	case *ast.BadDecl:
		// ignore

	case *ast.GenDecl:
		iota := 0
		var prev *ast.ValueSpec
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ImportSpec:
				// TODO(gri) imports go into file scope
			case *ast.ValueSpec:
				switch d.Tok {
				case token.CONST:
					if s.Values == nil {
						// create a new spec with type and values from the previous one
						if prev != nil {
							s = &ast.ValueSpec{s.Doc, s.Names, prev.Type, prev.Values, s.Comment}
						} else {
							// TODO(gri) this should probably go into the const decl code
							tc.Errorf(s.Pos(), "missing initializer for const %s", s.Names[0].Name)
						}
					}
					for _, name := range s.Names {
						tc.decl(ast.Con, name, s, iota)
					}
				case token.VAR:
					for _, name := range s.Names {
						tc.decl(ast.Var, name, s, 0)
					}
				default:
					panic("unreachable")
				}
				prev = s
				iota++
			case *ast.TypeSpec:
				obj := tc.decl(ast.Typ, s.Name, s, 0)
				// give all type objects an unresolved type so
				// that we can collect methods in the type scope
				typ := ast.NewType(ast.Unresolved)
				obj.Type = typ
				typ.Obj = obj
			default:
				panic("unreachable")
			}
		}

	case *ast.FuncDecl:
		if d.Recv == nil {
			tc.decl(ast.Fun, d.Name, d, 0)
		}

	default:
		panic("unreachable")
	}
}


// If x is of the form *T, deref returns T, otherwise it returns x.
func deref(x ast.Expr) ast.Expr {
	if p, isPtr := x.(*ast.StarExpr); isPtr {
		x = p.X
	}
	return x
}


func (tc *typechecker) bindMethod(method *ast.FuncDecl) {
	// a method is declared in the receiver base type's scope
	var scope *ast.Scope
	base := deref(method.Recv.List[0].Type)
	if name, isIdent := base.(*ast.Ident); isIdent {
		// if base is not an *ast.Ident, we had a syntax
		// error and the parser reported an error already
		obj := tc.topScope.Lookup(name.Name)
		if obj == nil {
			tc.Errorf(name.Pos(), "invalid receiver: %s is not declared in this package", name.Name)
		} else if obj.Kind != ast.Typ {
			tc.Errorf(name.Pos(), "invalid receiver: %s is not a type", name.Name)
		} else {
			typ := obj.Type
			assert(typ.Form == ast.Unresolved)
			scope = typ.Scope
		}
	}
	if scope == nil {
		// no receiver type found; use a dummy scope
		// (we still want to type-check the method
		// body, so make sure there is a name object
		// and type)
		// TODO(gri) should we record the scope so
		// that we don't lose the receiver for type-
		// checking of the method body?
		scope = ast.NewScope(nil)
	}
	tc.declInScope(scope, ast.Fun, method.Name, method, 0)
}


func (tc *typechecker) resolve(obj *ast.Object) {
	// check for declaration cycles
	if tc.cyclemap[obj] {
		tc.Errorf(objPos(obj), "illegal cycle in declaration of %s", obj.Name)
		obj.Kind = ast.Bad
		return
	}
	tc.cyclemap[obj] = true
	defer func() {
		tc.cyclemap[obj] = false, false
	}()

	// resolve non-type objects
	typ := obj.Type
	if typ == nil {
		switch obj.Kind {
		case ast.Bad:
			// ignore

		case ast.Con:
			tc.declConst(obj)

		case ast.Var:
			tc.declVar(obj)
			//obj.Type = tc.typeFor(nil, obj.Decl.(*ast.ValueSpec).Type, false)

		case ast.Fun:
			obj.Type = ast.NewType(ast.Function)
			t := obj.Decl.(*ast.FuncDecl).Type
			tc.declSignature(obj.Type, nil, t.Params, t.Results)

		default:
			// type objects have non-nil types when resolve is called
			if debug {
				fmt.Printf("kind = %s\n", obj.Kind)
			}
			panic("unreachable")
		}
		return
	}

	// resolve type objects
	if typ.Form == ast.Unresolved {
		tc.typeFor(typ, typ.Obj.Decl.(*ast.TypeSpec).Type, false)

		// provide types for all methods
		for _, obj := range typ.Scope.Objects {
			if obj.Kind == ast.Fun {
				assert(obj.Type == nil)
				obj.Type = ast.NewType(ast.Method)
				f := obj.Decl.(*ast.FuncDecl)
				t := f.Type
				tc.declSignature(obj.Type, f.Recv, t.Params, t.Results)
			}
		}
	}
}


func (tc *typechecker) checkBlock(body []ast.Stmt, ftype *ast.Type) {
	tc.openScope()
	defer tc.closeScope()

	// inject function/method parameters into block scope, if any
	if ftype != nil {
		for _, par := range ftype.Params.Objects {
			obj := tc.topScope.Insert(par)
			assert(obj == par) // ftype has no double declarations
		}
	}

	for _, stmt := range body {
		tc.checkStmt(stmt)
	}
}


// ----------------------------------------------------------------------------
// Types

// unparen removes parentheses around x, if any.
func unparen(x ast.Expr) ast.Expr {
	if ux, hasParens := x.(*ast.ParenExpr); hasParens {
		return unparen(ux.X)
	}
	return x
}


func (tc *typechecker) declFields(scope *ast.Scope, fields *ast.FieldList, ref bool) (n uint) {
	if fields != nil {
		for _, f := range fields.List {
			typ := tc.typeFor(nil, f.Type, ref)
			for _, name := range f.Names {
				fld := tc.declInScope(scope, ast.Var, name, f, 0)
				fld.Type = typ
				n++
			}
		}
	}
	return n
}


func (tc *typechecker) declSignature(typ *ast.Type, recv, params, results *ast.FieldList) {
	assert((typ.Form == ast.Method) == (recv != nil))
	typ.Params = ast.NewScope(nil)
	tc.declFields(typ.Params, recv, true)
	tc.declFields(typ.Params, params, true)
	typ.N = tc.declFields(typ.Params, results, true)
}


func (tc *typechecker) typeFor(def *ast.Type, x ast.Expr, ref bool) (typ *ast.Type) {
	x = unparen(x)

	// type name
	if t, isIdent := x.(*ast.Ident); isIdent {
		obj := tc.find(t)

		if obj.Kind != ast.Typ {
			tc.Errorf(t.Pos(), "%s is not a type", t.Name)
			if def == nil {
				typ = ast.NewType(ast.BadType)
			} else {
				typ = def
				typ.Form = ast.BadType
			}
			typ.Expr = x
			return
		}

		if !ref {
			tc.resolve(obj) // check for cycles even if type resolved
		}
		typ = obj.Type

		if def != nil {
			// new type declaration: copy type structure
			def.Form = typ.Form
			def.N = typ.N
			def.Key, def.Elt = typ.Key, typ.Elt
			def.Params = typ.Params
			def.Expr = x
			typ = def
		}
		return
	}

	// type literal
	typ = def
	if typ == nil {
		typ = ast.NewType(ast.BadType)
	}
	typ.Expr = x

	switch t := x.(type) {
	case *ast.SelectorExpr:
		if debug {
			fmt.Println("qualified identifier unimplemented")
		}
		typ.Form = ast.BadType

	case *ast.StarExpr:
		typ.Form = ast.Pointer
		typ.Elt = tc.typeFor(nil, t.X, true)

	case *ast.ArrayType:
		if t.Len != nil {
			typ.Form = ast.Array
			// TODO(gri) compute the real length
			// (this may call resolve recursively)
			(*typ).N = 42
		} else {
			typ.Form = ast.Slice
		}
		typ.Elt = tc.typeFor(nil, t.Elt, t.Len == nil)

	case *ast.StructType:
		typ.Form = ast.Struct
		tc.declFields(typ.Scope, t.Fields, false)

	case *ast.FuncType:
		typ.Form = ast.Function
		tc.declSignature(typ, nil, t.Params, t.Results)

	case *ast.InterfaceType:
		typ.Form = ast.Interface
		tc.declFields(typ.Scope, t.Methods, true)

	case *ast.MapType:
		typ.Form = ast.Map
		typ.Key = tc.typeFor(nil, t.Key, true)
		typ.Elt = tc.typeFor(nil, t.Value, true)

	case *ast.ChanType:
		typ.Form = ast.Channel
		typ.N = uint(t.Dir)
		typ.Elt = tc.typeFor(nil, t.Value, true)

	default:
		if debug {
			fmt.Printf("x is %T\n", x)
		}
		panic("unreachable")
	}

	return
}


// ----------------------------------------------------------------------------
// TODO(gri) implement these place holders

func (tc *typechecker) declConst(*ast.Object) {
}


func (tc *typechecker) declVar(*ast.Object) {
}


func (tc *typechecker) checkStmt(ast.Stmt) {
}
