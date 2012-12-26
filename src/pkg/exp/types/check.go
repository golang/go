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

// enable for debugging
const trace = false

type checker struct {
	ctxt  *Context
	fset  *token.FileSet
	files []*ast.File

	// lazily initialized
	firsterr  error
	initexprs map[*ast.ValueSpec][]ast.Expr // "inherited" initialization expressions for constant declarations
	funclist  []function                    // list of functions/methods with correct signatures and non-empty bodies
	funcsig   *Signature                    // signature of currently typechecked function
	pos       []token.Pos                   // stack of expr positions; debugging support, used if trace is set
}

type function struct {
	obj  *ast.Object // for debugging/tracing only
	sig  *Signature
	body *ast.BlockStmt
}

// later adds a function with non-empty body to the list of functions
// that need to be processed after all package-level declarations
// are typechecked.
//
func (check *checker) later(obj *ast.Object, sig *Signature, body *ast.BlockStmt) {
	// functions implemented elsewhere (say in assembly) have no body
	if body != nil {
		check.funclist = append(check.funclist, function{obj, sig, body})
	}
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
						// ok to continue
					}
				}
			case *Interface:
				// methods cannot be associated with an interface type
				for _, m := range scope.Objects {
					recv := m.Decl.(*ast.FuncDecl).Recv.List[0].Type
					check.errorf(recv.Pos(), "invalid receiver type %s (%s is an interface type)", obj.Name, obj.Name)
					// ok to continue
				}
			}
			// typecheck method signatures
			for _, obj := range scope.Objects {
				mdecl := obj.Decl.(*ast.FuncDecl)
				sig := check.typ(mdecl.Type, cycleOk).(*Signature)
				params, _ := check.collectParams(mdecl.Recv, false)
				sig.Recv = params[0] // the parser/assocMethod ensure there is exactly one parameter
				obj.Type = sig
				check.later(obj, sig, mdecl.Body)
			}
		}

	case ast.Fun:
		fdecl := obj.Decl.(*ast.FuncDecl)
		// methods are typechecked when their receivers are typechecked
		if fdecl.Recv == nil {
			sig := check.typ(fdecl.Type, cycleOk).(*Signature)
			if obj.Name == "init" && (len(sig.Params) != 0 || len(sig.Results) != 0) {
				check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
				// ok to continue
			}
			obj.Type = sig
			check.later(obj, sig, fdecl.Body)
		}

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
	// determine receiver base type object
	var obj *ast.Object
	if ident, ok := typ.(*ast.Ident); ok && ident.Obj != nil {
		obj = ident.Obj
		if obj.Kind != ast.Typ {
			check.errorf(ident.Pos(), "%s is not a type", ident.Name)
			return // ignore this method
		}
		// TODO(gri) determine if obj was defined in this package
		/*
			if check.notLocal(obj) {
				check.errorf(ident.Pos(), "cannot define methods on non-local type %s", ident.Name)
				return // ignore this method
			}
		*/
	} else {
		// If it's not an identifier or the identifier wasn't declared/resolved,
		// the parser/resolver already reported an error. Nothing to do here.
		return // ignore this method
	}
	// declare method in receiver base type scope
	var scope *ast.Scope
	if obj.Data != nil {
		scope = obj.Data.(*ast.Scope)
	} else {
		scope = ast.NewScope(nil)
		obj.Data = scope
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
		// methods are checked when their respective base types are checked
		if d.Recv != nil {
			return
		}
		obj := d.Name.Obj
		// Initialization functions don't have an object associated with them
		// since they are not in any scope. Create a dummy object for them.
		if d.Name.Name == "init" {
			assert(obj == nil) // all other functions should have an object
			obj = ast.NewObj(ast.Fun, d.Name.Name)
			obj.Decl = d
			d.Name.Obj = obj
		}
		if obj.Type == nil {
			check.object(obj, false)
		}
	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}

// iterate calls f for each package-level declaration.
func (check *checker) iterate(f func(*checker, ast.Decl)) {
	for _, file := range check.files {
		for _, decl := range file.Decls {
			f(check, decl)
		}
	}
}

// sortedFiles returns the sorted list of package files given a package file map.
func sortedFiles(m map[string]*ast.File) []*ast.File {
	keys := make([]string, len(m))
	i := 0
	for k, _ := range m {
		keys[i] = k
		i++
	}
	sort.Strings(keys)

	files := make([]*ast.File, len(m))
	for i, k := range keys {
		files[i] = m[k]
	}

	return files
}

// A bailout panic is raised to indicate early termination.
type bailout struct{}

func check(ctxt *Context, fset *token.FileSet, files map[string]*ast.File) (pkg *ast.Package, err error) {
	// initialize checker
	check := checker{
		ctxt:      ctxt,
		fset:      fset,
		files:     sortedFiles(files),
		initexprs: make(map[*ast.ValueSpec][]ast.Expr),
	}

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
			// TODO(gri) add a test case for this scenario
			err = fmt.Errorf("types internal error: %v", p)
		}
	}()

	// resolve identifiers
	imp := ctxt.Import
	if imp == nil {
		imp = GcImport
	}
	pkg, err = ast.NewPackage(fset, files, imp, Universe)
	if err != nil {
		if list, _ := err.(scanner.ErrorList); len(list) > 0 {
			for _, err := range list {
				check.err(err)
			}
		} else {
			check.err(err)
		}
	}

	// determine missing constant initialization expressions
	// and associate methods with types
	check.iterate((*checker).assocInitvalsOrMethod)

	// typecheck all declarations
	check.iterate((*checker).decl)

	// typecheck all function/method bodies
	// (funclist may grow when checking statements - do not use range clause!)
	for i := 0; i < len(check.funclist); i++ {
		f := check.funclist[i]
		if trace {
			s := "<function literal>"
			if f.obj != nil {
				s = f.obj.Name
			}
			fmt.Println("---", s)
		}
		check.funcsig = f.sig
		check.stmtList(f.body.List)
	}

	return
}
