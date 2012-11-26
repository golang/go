// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which typechecks a package.

package types

import (
	"fmt"
	"go/ast"
	"go/token"
	"sort"
)

// enable for debugging
const trace = false

type checker struct {
	fset *token.FileSet
	pkg  *ast.Package
	errh func(token.Pos, string)
	mapf func(ast.Expr, Type)

	// lazily initialized
	firsterr  error
	filenames []string                      // sorted list of package file names for reproducible iteration order
	initexprs map[*ast.ValueSpec][]ast.Expr // "inherited" initialization expressions for constant declarations
	functypes []*Signature                  // stack of function signatures; actively typechecked function on top
	pos       []token.Pos                   // stack of expr positions; debugging support, used if trace is set
}

// declare declares an object of the given kind and name (ident) in scope;
// decl is the corresponding declaration in the AST. An error is reported
// if the object was declared before.
//
// TODO(gri) This is very similar to the declare function in go/parser; it
// is only used to associate methods with their respective receiver base types.
// In a future version, it might be simpler and cleaner to do all the resolution
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

func (check *checker) valueSpec(pos token.Pos, obj *ast.Object, lhs []*ast.Ident, typ ast.Expr, rhs []ast.Expr, iota int) {
	if len(lhs) == 0 {
		check.invalidAST(pos, "missing lhs in declaration")
		return
	}

	// determine type for all of lhs, if any
	// (but only set it for the object we typecheck!)
	var t Type
	if typ != nil {
		t = check.typ(typ, false)
	}

	// len(lhs) > 0
	if len(lhs) == len(rhs) {
		// check only lhs and rhs corresponding to obj
		var l, r ast.Expr
		for i, name := range lhs {
			if name.Obj == obj {
				l = lhs[i]
				r = rhs[i]
				break
			}
		}
		assert(l != nil)
		obj.Type = t
		check.assign1to1(l, r, nil, true, iota)
		return
	}

	// there must be a type or initialization expressions
	if t == nil && len(rhs) == 0 {
		check.invalidAST(pos, "missing type or initialization expression")
		t = Typ[Invalid]
	}

	// if we have a type, mark all of lhs
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

func (check *checker) function(typ *Signature, body *ast.BlockStmt) {
	check.functypes = append(check.functypes, typ)
	check.stmt(body)
	check.functypes = check.functypes[0 : len(check.functypes)-1]
}

// object typechecks an object by assigning it a type; obj.Type must be nil.
// Callers must check obj.Type before calling object; this eliminates a call
// for each identifier that has been typechecked already, a common scenario.
//
func (check *checker) object(obj *ast.Object, cycleOk bool) {
	assert(obj.Type == nil)

	switch obj.Kind {
	case ast.Bad, ast.Pkg:
		// nothing to do

	case ast.Con, ast.Var:
		// The obj.Data field for constants and variables is initialized
		// to the respective (hypothetical, for variables) iota value by
		// the parser. The object's fields can be in one of the following
		// states:
		// Type != nil  =>  the constant value is Data
		// Type == nil  =>  the object is not typechecked yet, and Data can be:
		// Data is int  =>  Data is the value of iota for this declaration
		// Data == nil  =>  the object's expression is being evaluated
		if obj.Data == nil {
			check.errorf(obj.Pos(), "illegal cycle in initialization of %s", obj.Name)
			obj.Type = Typ[Invalid]
			return
		}
		spec := obj.Decl.(*ast.ValueSpec)
		iota := obj.Data.(int)
		obj.Data = nil
		// determine initialization expressions
		values := spec.Values
		if len(values) == 0 && obj.Kind == ast.Con {
			values = check.initexprs[spec]
		}
		check.valueSpec(spec.Pos(), obj, spec.Names, spec.Type, values, iota)

	case ast.Typ:
		typ := &NamedType{Obj: obj}
		obj.Type = typ // "mark" object so recursion terminates
		typ.Underlying = underlying(check.typ(obj.Decl.(*ast.TypeSpec).Type, cycleOk))
		// typecheck associated method signatures
		if obj.Data != nil {
			scope := obj.Data.(*ast.Scope)
			switch t := typ.Underlying.(type) {
			case *Struct:
				// struct fields must not conflict with methods
				for _, f := range t.Fields {
					if m := scope.Lookup(f.Name); m != nil {
						check.errorf(m.Pos(), "type %s has both field and method named %s", obj.Name, f.Name)
					}
				}
				// ok to continue
			case *Interface:
				// methods cannot be associated with an interface type
				for _, m := range scope.Objects {
					recv := m.Decl.(*ast.FuncDecl).Recv.List[0].Type
					check.errorf(recv.Pos(), "invalid receiver type %s (%s is an interface type)", obj.Name, obj.Name)
				}
				// ok to continue
			}
			// typecheck method signatures
			for _, m := range scope.Objects {
				mdecl := m.Decl.(*ast.FuncDecl)
				// TODO(gri) At the moment, the receiver is type-checked when checking
				// the method body. Also, we don't properly track if the receiver is
				// a pointer (i.e., currently, method sets are too large). FIX THIS.
				mtyp := check.typ(mdecl.Type, cycleOk).(*Signature)
				m.Type = mtyp
			}
		}

	case ast.Fun:
		fdecl := obj.Decl.(*ast.FuncDecl)
		check.collectParams(fdecl.Recv) // ensure method base is type-checked
		ftyp := check.typ(fdecl.Type, cycleOk).(*Signature)
		obj.Type = ftyp
		check.function(ftyp, fdecl.Body)

	default:
		panic("unreachable")
	}
}

// assocInitvals associates "inherited" initialization expressions
// with the corresponding *ast.ValueSpec in the check.initexprs map
// for constant declarations without explicit initialization expressions.
//
func (check *checker) assocInitvals(decl *ast.GenDecl) {
	var values []ast.Expr
	for _, s := range decl.Specs {
		if s, ok := s.(*ast.ValueSpec); ok {
			if len(s.Values) > 0 {
				values = s.Values
			} else {
				check.initexprs[s] = values
			}
		}
	}
	if len(values) == 0 {
		check.invalidAST(decl.Pos(), "no initialization values provided")
	}
}

// assocMethod associates a method declaration with the respective
// receiver base type. meth.Recv must exist.
//
func (check *checker) assocMethod(meth *ast.FuncDecl) {
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

func (check *checker) assocInitvalsOrMethod(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.GenDecl:
		if d.Tok == token.CONST {
			check.assocInitvals(d)
		}
	case *ast.FuncDecl:
		if d.Recv != nil {
			check.assocMethod(d)
		}
	}
}

func (check *checker) decl(decl ast.Decl) {
	switch d := decl.(type) {
	case *ast.BadDecl:
		// ignore
	case *ast.GenDecl:
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ImportSpec:
				// nothing to do (handled by ast.NewPackage)
			case *ast.ValueSpec:
				for _, name := range s.Names {
					if obj := name.Obj; obj.Type == nil {
						check.object(obj, false)
					}
				}
			case *ast.TypeSpec:
				if obj := s.Name.Obj; obj.Type == nil {
					check.object(obj, false)
				}
			default:
				check.invalidAST(s.Pos(), "unknown ast.Spec node %T", s)
			}
		}
	case *ast.FuncDecl:
		if d.Name.Name == "init" {
			// initialization function
			// TODO(gri) ignore for now (has no object associated with it)
			// (should probably collect in a first phase and properly initialize)
			return
		}
		if obj := d.Name.Obj; obj.Type == nil {
			check.object(obj, false)
		}
	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}

// iterate calls f for each package-level declaration.
func (check *checker) iterate(f func(*checker, ast.Decl)) {
	list := check.filenames

	if list == nil {
		// initialize lazily
		for filename := range check.pkg.Files {
			list = append(list, filename)
		}
		sort.Strings(list)
		check.filenames = list
	}

	for _, filename := range list {
		for _, decl := range check.pkg.Files[filename].Decls {
			f(check, decl)
		}
	}
}

// A bailout panic is raised to indicate early termination.
type bailout struct{}

func check(fset *token.FileSet, pkg *ast.Package, errh func(token.Pos, string), f func(ast.Expr, Type)) (err error) {
	// initialize checker
	var check checker
	check.fset = fset
	check.pkg = pkg
	check.errh = errh
	check.mapf = f
	check.initexprs = make(map[*ast.ValueSpec][]ast.Expr)

	// handle panics
	defer func() {
		switch p := recover().(type) {
		case nil:
			// normal return - nothing to do
		case bailout:
			// early exit
			err = check.firsterr
		default:
			// unexpected panic: don't crash clients
			// panic(p) // enable for debugging
			err = fmt.Errorf("types.check internal error: %v", p)
		}
	}()

	// determine missing constant initialization expressions
	// and associate methods with types
	check.iterate((*checker).assocInitvalsOrMethod)

	// typecheck all declarations
	check.iterate((*checker).decl)

	return
}
