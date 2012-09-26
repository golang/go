// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which typechecks a package.

package types

import (
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"sort"
)

type checker struct {
	fset   *token.FileSet
	pkg    *ast.Package
	errors scanner.ErrorList
	types  map[ast.Expr]Type
}

// declare declares an object of the given kind and name (ident) in scope;
// decl is the corresponding declaration in the AST. An error is reported
// if the object was declared before.
//
// TODO(gri) This is very similar to the declare function in go/parser; it
// is only used to associate methods with their respective receiver base types.
// In a future version, it might be simpler and cleaner do to all the resolution
// in the type-checking phase. It would simplify the parser, AST, and also
// reduce some amount of code duplication.
//
func (check *checker) declare(scope *ast.Scope, kind ast.ObjKind, ident *ast.Ident, decl ast.Decl) {
	assert(ident.Obj == nil) // identifier already declared or resolved
	obj := ast.NewObj(kind, ident.Name)
	obj.Decl = decl
	ident.Obj = obj
	if ident.Name != "_" {
		if alt := scope.Insert(obj); alt != nil {
			prevDecl := ""
			if pos := alt.Pos(); pos.IsValid() {
				prevDecl = fmt.Sprintf("\n\tprevious declaration at %s", check.fset.Position(pos))
			}
			check.errorf(ident.Pos(), fmt.Sprintf("%s redeclared in this block%s", ident.Name, prevDecl))
		}
	}
}

func (check *checker) decl(pos token.Pos, obj *ast.Object, lhs []*ast.Ident, typ ast.Expr, rhs []ast.Expr, iota int) {
	if len(lhs) == 0 {
		check.invalidAST(pos, "missing lhs in declaration")
		return
	}

	var t Type
	if typ != nil {
		t = check.typ(typ, false)
	}

	// len(lhs) >= 1
	if len(lhs) == len(rhs) {
		// check only corresponding lhs and rhs
		var l, r ast.Expr
		for i, ident := range lhs {
			if ident.Obj == obj {
				l = lhs[i]
				r = rhs[i]
				break
			}
		}
		assert(l != nil)
		obj.Type = t
		// check rhs
		var x operand
		check.expr(&x, r, t, iota)
		// assign to lhs
		check.assignment(l, &x, true)
		return
	}

	if t != nil {
		for _, name := range lhs {
			name.Obj.Type = t
		}
	}

	// check initial values, if any
	if len(rhs) > 0 {
		// TODO(gri) should try to avoid this conversion
		lhx := make([]ast.Expr, len(lhs))
		for i, e := range lhs {
			lhx[i] = e
		}
		check.assignNtoM(lhx, rhs, true, iota)
	}
}

// specValues returns the list of initialization expressions
// for the given part (spec) of a constant declaration.
// TODO(gri) Make this more efficient by caching results
// (using a map in checker).
func (check *checker) specValues(spec *ast.ValueSpec) []ast.Expr {
	if len(spec.Values) > 0 {
		return spec.Values
	}

	// find the corresponding values
	for _, file := range check.pkg.Files {
		for _, d := range file.Decls {
			if d, ok := d.(*ast.GenDecl); ok && d.Tok == token.CONST {
				var values []ast.Expr
				for _, s := range d.Specs {
					if s, ok := s.(*ast.ValueSpec); ok {
						if len(s.Values) > 0 {
							values = s.Values
						}
						if s == spec {
							return values
						}
					}
				}
			}
		}
	}

	check.invalidAST(spec.Pos(), "no initialization values provided")
	return nil
}

// obj type checks an object.
func (check *checker) obj(obj *ast.Object, cycleOk bool) {
	if trace {
		fmt.Printf("obj(%s)\n", obj.Name)
	}

	if obj.Type != nil {
		// object has already been type checked
		return
	}

	switch obj.Kind {
	case ast.Bad, ast.Pkg:
		// nothing to do

	case ast.Con:
		if obj.Data == nil {
			check.errorf(obj.Pos(), "illegal cycle in initialization of %s", obj.Name)
			return
		}
		spec, ok := obj.Decl.(*ast.ValueSpec)
		assert(ok)
		// The Data stored with the constant is the value of iota for that
		// ast.ValueSpec. Use it for the evaluation of the initialization
		// expressions.
		iota := obj.Data.(int)
		obj.Data = nil
		check.decl(spec.Pos(), obj, spec.Names, spec.Type, check.specValues(spec), iota)

	case ast.Var:
		// TODO(gri) missing cycle detection
		spec, ok := obj.Decl.(*ast.ValueSpec)
		if !ok {
			// TODO(gri) the assertion fails for "x, y := 1, 2, 3" it seems
			fmt.Printf("var = %s\n", obj.Name)
		}
		assert(ok)
		check.decl(spec.Pos(), obj, spec.Names, spec.Type, spec.Values, 0)

	case ast.Typ:
		typ := &NamedType{Obj: obj}
		obj.Type = typ // "mark" object so recursion terminates
		typ.Underlying = underlying(check.typ(obj.Decl.(*ast.TypeSpec).Type, cycleOk))
		// collect associated methods, if any
		if obj.Data != nil {
			scope := obj.Data.(*ast.Scope)
			// struct fields must not conflict with methods
			if t, ok := typ.Underlying.(*Struct); ok {
				for _, f := range t.Fields {
					if m := scope.Lookup(f.Name); m != nil {
						check.errorf(m.Pos(), "type %s has both field and method named %s", obj.Name, f.Name)
					}
				}
			}
			// collect methods
			methods := make(ObjList, len(scope.Objects))
			i := 0
			for _, m := range scope.Objects {
				methods[i] = m
				i++
			}
			methods.Sort()
			typ.Methods = methods
			// methods cannot be associated with an interface type
			// (do this check after sorting for reproducible error positions - needed for testing)
			if _, ok := typ.Underlying.(*Interface); ok {
				for _, m := range methods {
					recv := m.Decl.(*ast.FuncDecl).Recv.List[0].Type
					check.errorf(recv.Pos(), "invalid receiver type %s (%s is an interface type)", obj.Name, obj.Name)
				}
			}
		}

	case ast.Fun:
		fdecl := obj.Decl.(*ast.FuncDecl)
		ftyp := check.typ(fdecl.Type, cycleOk).(*Signature)
		obj.Type = ftyp
		if fdecl.Recv != nil {
			// TODO(gri) handle method receiver
		}
		check.stmt(fdecl.Body)

	default:
		panic("unreachable")
	}
}

func check(fset *token.FileSet, pkg *ast.Package, types map[ast.Expr]Type) error {
	var check checker
	check.fset = fset
	check.pkg = pkg
	check.types = types

	// Compute sorted list of file names so that
	// package file iterations are reproducible (needed for testing).
	filenames := make([]string, len(pkg.Files))
	{
		i := 0
		for filename := range pkg.Files {
			filenames[i] = filename
			i++
		}
		sort.Strings(filenames)
	}

	// Associate methods with types
	// TODO(gri) All other objects are resolved by the parser.
	//           Consider doing this in the parser (and provide the info
	//           in the AST. In the long-term (might require Go 1 API
	//           changes) it's probably easier to do all the resolution
	//           in one place in the type checker. See also comment
	//           with checker.declare.
	for _, filename := range filenames {
		file := pkg.Files[filename]
		for _, decl := range file.Decls {
			if meth, ok := decl.(*ast.FuncDecl); ok && meth.Recv != nil {
				// The receiver type is one of the following (enforced by parser):
				// - *ast.Ident
				// - *ast.StarExpr{*ast.Ident}
				// - *ast.BadExpr (parser error)
				typ := meth.Recv.List[0].Type
				if ptr, ok := typ.(*ast.StarExpr); ok {
					typ = ptr.X
				}
				// determine receiver base type object (or nil if error)
				var obj *ast.Object
				if ident, ok := typ.(*ast.Ident); ok && ident.Obj != nil {
					obj = ident.Obj
					if obj.Kind != ast.Typ {
						check.errorf(ident.Pos(), "%s is not a type", ident.Name)
						obj = nil
					}
					// TODO(gri) determine if obj was defined in this package
					/*
						if check.notLocal(obj) {
							check.errorf(ident.Pos(), "cannot define methods on non-local type %s", ident.Name)
							obj = nil
						}
					*/
				} else {
					// If it's not an identifier or the identifier wasn't declared/resolved,
					// the parser/resolver already reported an error. Nothing to do here.
				}
				// determine base type scope (or nil if error)
				var scope *ast.Scope
				if obj != nil {
					if obj.Data != nil {
						scope = obj.Data.(*ast.Scope)
					} else {
						scope = ast.NewScope(nil)
						obj.Data = scope
					}
				} else {
					// use a dummy scope so that meth can be declared in
					// presence of an error and get an associated object
					// (always use a new scope so that we don't get double
					// declaration errors)
					scope = ast.NewScope(nil)
				}
				check.declare(scope, ast.Fun, meth.Name, meth)
			}
		}
	}

	// Sort objects so that we get reproducible error
	// positions (this is only needed for testing).
	// TODO(gri): Consider ast.Scope implementation that
	// provides both a list and a map for fast lookup.
	// Would permit the use of scopes instead of ObjMaps
	// elsewhere.
	list := make(ObjList, len(pkg.Scope.Objects))
	{
		i := 0
		for _, obj := range pkg.Scope.Objects {
			list[i] = obj
			i++
		}
		list.Sort()
	}

	// Check global objects.
	for _, obj := range list {
		check.obj(obj, false)
	}

	// TODO(gri) Missing pieces:
	// - blank (_) objects and init functions are not in scopes but should be type-checked

	// do not remove multiple errors per line - depending on
	// order or error reporting this may hide the real error
	return check.errors.Err()
}
