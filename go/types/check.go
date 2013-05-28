// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements the Check function, which typechecks a package.

package types

import (
	"fmt"
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// debugging/development support
const (
	debug = true  // leave on during development
	trace = false // turn on for detailed type resolution traces
	// TODO(gri) remove this flag and clean up code under the assumption that resolve == true.
	resolve = true // if set, resolve all identifiers in the type checker (don't use ast.Objects anymore)
)

// exprInfo stores type and constant value for an untyped expression.
type exprInfo struct {
	isLhs bool // expression is lhs operand of a shift with delayed type check
	typ   *Basic
	val   exact.Value // constant value; or nil (if not a constant)
}

// A checker is an instance of the type checker.
type checker struct {
	ctxt *Context
	fset *token.FileSet

	// lazily initialized
	pkg         *Package                          // current package
	firsterr    error                             // first error encountered
	idents      map[*ast.Ident]Object             // maps identifiers to their unique object
	objects     map[*ast.Object]Object            // maps *ast.Objects to their unique object
	initspecs   map[*ast.ValueSpec]*ast.ValueSpec // "inherited" type and initialization expressions for constant declarations
	methods     map[*TypeName]*Scope              // maps type names to associated methods
	conversions map[*ast.CallExpr]bool            // set of type-checked conversions (to distinguish from calls)
	untyped     map[ast.Expr]exprInfo             // map of expressions without final type

	// functions
	funclist []function  // list of functions/methods with correct signatures and non-empty bodies
	funcsig  *Signature  // signature of currently typechecked function
	pos      []token.Pos // stack of expr positions; debugging support, used if trace is set

	// these are only valid if resolve is set
	objMap   map[Object]*decl // if set we are in the package-global declaration phase (otherwise all objects seen must be declared)
	topScope *Scope           // topScope for lookups, non-global declarations
}

func (check *checker) openScope() {
	check.topScope = &Scope{Outer: check.topScope}
}

func (check *checker) closeScope() {
	check.topScope = check.topScope.Outer
}

func (check *checker) register(id *ast.Ident, obj Object) {
	if resolve {
		// TODO(gri) Document how if an identifier can be registered more than once.
		if f := check.ctxt.Ident; f != nil {
			f(id, obj)
		}
		return
	}

	// When an expression is evaluated more than once (happens
	// in rare cases, e.g. for statement expressions, see
	// comment in stmt.go), the object has been registered
	// before. Don't do anything in that case.
	if alt := check.idents[id]; alt != nil {
		assert(alt == obj)
		return
	}
	check.idents[id] = obj
	if f := check.ctxt.Ident; f != nil {
		f(id, obj)
	}
}

// lookup returns the unique Object denoted by the identifier.
// For identifiers without assigned *ast.Object, it uses the
// checker.idents map; for identifiers with an *ast.Object it
// uses the checker.objects map.
//
// TODO(gri) Once identifier resolution is done entirely by
//           the typechecker, only scopes are needed. Need
//           to update the comment above.
//
func (check *checker) lookup(ident *ast.Ident) Object {
	if resolve {
		for scope := check.topScope; scope != nil; scope = scope.Outer {
			if obj := scope.Lookup(ident.Name); obj != nil {
				check.register(ident, obj)
				return obj
			}
		}
		return nil
	}

	// old code
	obj := check.idents[ident]
	astObj := ident.Obj

	if obj != nil {
		assert(astObj == nil || check.objects[astObj] == nil || check.objects[astObj] == obj)
		return obj
	}

	if astObj == nil {
		return nil
	}

	if obj = check.objects[astObj]; obj == nil {
		obj = newObj(check.pkg, astObj)
		check.objects[astObj] = obj
	}
	check.register(ident, obj)

	return obj
}

type function struct {
	file *Scope // only valid if resolve is set
	obj  *Func  // for debugging/tracing only
	sig  *Signature
	body *ast.BlockStmt
}

// later adds a function with non-empty body to the list of functions
// that need to be processed after all package-level declarations
// are typechecked.
//
func (check *checker) later(f *Func, sig *Signature, body *ast.BlockStmt) {
	// functions implemented elsewhere (say in assembly) have no body
	if body != nil {
		check.funclist = append(check.funclist, function{check.topScope, f, sig, body})
	}
}

func (check *checker) declareIdent(scope *Scope, ident *ast.Ident, obj Object) {
	if resolve {
		unreachable()
	}
	assert(check.lookup(ident) == nil) // identifier already declared or resolved
	check.register(ident, obj)
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

func (check *checker) valueSpec(pos token.Pos, obj Object, lhs []*ast.Ident, spec *ast.ValueSpec, iota int) {
	if resolve {
		unreachable()
	}

	if len(lhs) == 0 {
		check.invalidAST(pos, "missing lhs in declaration")
		return
	}

	// determine type for all of lhs, if any
	// (but only set it for the object we typecheck!)
	var typ Type
	if spec.Type != nil {
		typ = check.typ(spec.Type, false)
	}

	// len(lhs) > 0
	rhs := spec.Values
	if len(lhs) == len(rhs) {
		// check only lhs and rhs corresponding to obj
		var l, r ast.Expr
		for i, name := range lhs {
			if check.lookup(name) == obj {
				l = lhs[i]
				r = rhs[i]
				break
			}
		}
		isConst := false
		switch obj := obj.(type) {
		case *Const:
			obj.typ = typ
			isConst = true
		case *Var:
			obj.typ = typ
		default:
			unreachable()
		}
		check.assign1to1(l, r, nil, true, iota, isConst)
		return
	}

	// there must be a type or initialization expressions
	if typ == nil && len(rhs) == 0 {
		check.invalidAST(pos, "missing type or initialization expression")
		typ = Typ[Invalid]
	}

	// if we have a type, mark all of lhs
	if typ != nil {
		for _, name := range lhs {
			switch obj := check.lookup(name).(type) {
			case *Const:
				obj.typ = typ
			case *Var:
				obj.typ = typ
			default:
				unreachable()
			}
		}
	}

	// check initial values, if any
	if len(rhs) > 0 {
		// TODO(gri) should try to avoid this conversion
		lhx := make([]ast.Expr, len(lhs))
		for i, e := range lhs {
			lhx[i] = e
		}
		_, isConst := obj.(*Const)
		check.assignNtoM(lhx, rhs, true, iota, isConst)
	}
}

// object typechecks an object by assigning it a type.
//
func (check *checker) object(obj Object, cycleOk bool) {
	if resolve {
		unreachable()
	}

	if obj.Type() != nil {
		return // already checked
	}

	switch obj := obj.(type) {
	case *Package:
		// nothing to do
		if resolve {
			unreachable()
		}

	case *Const:
		// The obj.Val field for constants is initialized to its respective
		// iota value (type int) by the parser.
		// If the object's type is Typ[Invalid], the object value is ignored.
		// If the object's type is valid, the object value must be a legal
		// constant value; it may be nil to indicate that we don't know the
		// value of the constant (e.g., in: "const x = float32("foo")" we
		// know that x is a constant and has type float32, but we don't
		// have a value due to the error in the conversion).
		if obj.visited {
			check.errorf(obj.Pos(), "illegal cycle in initialization of constant %s", obj.name)
			obj.typ = Typ[Invalid]
			return
		}
		obj.visited = true
		spec := obj.spec
		iota, ok := exact.Int64Val(obj.val)
		assert(ok)
		obj.val = exact.MakeUnknown()
		// determine spec for type and initialization expressions
		init := spec
		if len(init.Values) == 0 {
			init = check.initspecs[spec]
		}
		check.valueSpec(spec.Pos(), obj, spec.Names, init, int(iota))

	case *Var:
		if obj.visited {
			check.errorf(obj.Pos(), "illegal cycle in initialization of variable %s", obj.name)
			obj.typ = Typ[Invalid]
			return
		}
		obj.visited = true
		switch d := obj.decl.(type) {
		case *ast.Field:
			unreachable() // function parameters are always typed when collected
		case *ast.ValueSpec:
			check.valueSpec(d.Pos(), obj, d.Names, d, 0)
		case *ast.AssignStmt:
			unreachable() // assign1to1 sets the type for failing short var decls
		default:
			unreachable() // see also function newObj
		}

	case *TypeName:
		typ := &Named{obj: obj}
		obj.typ = typ // "mark" object so recursion terminates
		typ.underlying = check.typ(obj.spec.Type, cycleOk).Underlying()
		// typecheck associated method signatures
		if scope := check.methods[obj]; scope != nil {
			switch t := typ.underlying.(type) {
			case *Struct:
				// struct fields must not conflict with methods
				for _, f := range t.fields {
					if m := scope.Lookup(f.Name); m != nil {
						check.errorf(m.Pos(), "type %s has both field and method named %s", obj.name, f.Name)
						// ok to continue
					}
				}
			case *Interface:
				// methods cannot be associated with an interface type
				for _, m := range scope.Entries {
					recv := m.(*Func).decl.Recv.List[0].Type
					check.errorf(recv.Pos(), "invalid receiver type %s (%s is an interface type)", obj.name, obj.name)
					// ok to continue
				}
			}
			// typecheck method signatures
			var methods ObjSet
			for _, obj := range scope.Entries {
				m := obj.(*Func)
				sig := check.typ(m.decl.Type, cycleOk).(*Signature)
				params, _ := check.collectParams(sig.scope, m.decl.Recv, false)
				sig.recv = params[0] // the parser/assocMethod ensure there is exactly one parameter
				m.typ = sig
				assert(methods.Insert(obj) == nil)
				check.later(m, sig, m.decl.Body)
			}
			typ.methods = methods
			delete(check.methods, obj) // we don't need this scope anymore
		}

	case *Func:
		fdecl := obj.decl
		// methods are typechecked when their receivers are typechecked
		if fdecl.Recv == nil {
			sig := check.typ(fdecl.Type, cycleOk).(*Signature)
			if obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
				check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
				// ok to continue
			}
			obj.typ = sig
			check.later(obj, sig, fdecl.Body)
		}

	default:
		unreachable()
	}
}

// assocInitvals associates "inherited" initialization expressions
// with the corresponding *ast.ValueSpec in the check.initspecs map
// for constant declarations without explicit initialization expressions.
//
func (check *checker) assocInitvals(decl *ast.GenDecl) {
	if resolve {
		unreachable()
	}

	var last *ast.ValueSpec
	for _, s := range decl.Specs {
		if s, ok := s.(*ast.ValueSpec); ok {
			if len(s.Values) > 0 {
				last = s
			} else {
				check.initspecs[s] = last
			}
		}
	}
	if last == nil {
		check.invalidAST(decl.Pos(), "no initialization values provided")
	}
}

// assocMethod associates a method declaration with the respective
// receiver base type. meth.Recv must exist.
//
func (check *checker) assocMethod(meth *ast.FuncDecl) {
	if resolve {
		unreachable()
	}

	// The receiver type is one of the following (enforced by parser):
	// - *ast.Ident
	// - *ast.StarExpr{*ast.Ident}
	// - *ast.BadExpr (parser error)
	typ := meth.Recv.List[0].Type
	if ptr, ok := typ.(*ast.StarExpr); ok {
		typ = ptr.X
	}
	// determine receiver base type name
	ident, ok := typ.(*ast.Ident)
	if !ok {
		// not an identifier - parser reported error already
		return // ignore this method
	}
	// determine receiver base type object
	var tname *TypeName
	if obj := check.lookup(ident); obj != nil {
		obj, ok := obj.(*TypeName)
		if !ok {
			check.errorf(ident.Pos(), "%s is not a type", ident.Name)
			return // ignore this method
		}
		if obj.spec == nil {
			check.errorf(ident.Pos(), "cannot define method on non-local type %s", ident.Name)
			return // ignore this method
		}
		tname = obj
	} else {
		// identifier not declared/resolved - parser reported error already
		return // ignore this method
	}
	// declare method in receiver base type scope
	scope := check.methods[tname]
	if scope == nil {
		scope = new(Scope)
		check.methods[tname] = scope
	}
	check.declareIdent(scope, meth.Name, &Func{pkg: check.pkg, name: meth.Name.Name, decl: meth})
}

func (check *checker) decl(decl ast.Decl) {
	if resolve {
		unreachable() // during top-level type-checking
	}

	switch d := decl.(type) {
	case *ast.BadDecl:
		// ignore
	case *ast.GenDecl:
		for _, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ImportSpec:
				// nothing to do (handled by check.resolve)
			case *ast.ValueSpec:
				for _, name := range s.Names {
					check.object(check.lookup(name), false)
				}
			case *ast.TypeSpec:
				check.object(check.lookup(s.Name), false)
			default:
				check.invalidAST(s.Pos(), "unknown ast.Spec node %T", s)
			}
		}
	case *ast.FuncDecl:
		// methods are checked when their respective base types are checked
		if d.Recv != nil {
			return
		}
		obj := check.lookup(d.Name)
		// Initialization functions don't have an object associated with them
		// since they are not in any scope. Create a dummy object for them.
		if d.Name.Name == "init" {
			assert(obj == nil) // all other functions should have an object
			obj = &Func{pkg: check.pkg, name: d.Name.Name, decl: d}
			check.register(d.Name, obj)
		}
		check.object(obj, false)
	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}

// A bailout panic is raised to indicate early termination.
type bailout struct{}

func check(ctxt *Context, path string, fset *token.FileSet, files ...*ast.File) (pkg *Package, err error) {
	pkg = &Package{
		path:    path,
		scope:   &Scope{Outer: Universe},
		imports: make(map[string]*Package),
	}

	// initialize checker
	check := checker{
		ctxt:        ctxt,
		fset:        fset,
		pkg:         pkg,
		idents:      make(map[*ast.Ident]Object),
		objects:     make(map[*ast.Object]Object),
		initspecs:   make(map[*ast.ValueSpec]*ast.ValueSpec),
		methods:     make(map[*TypeName]*Scope),
		conversions: make(map[*ast.CallExpr]bool),
		untyped:     make(map[ast.Expr]exprInfo),
	}

	// handle panics
	defer func() {
		switch p := recover().(type) {
		case nil, bailout:
			// normal return or early exit
			err = check.firsterr
		default:
			// unexpected panic: don't crash clients
			if debug {
				check.dump("INTERNAL PANIC: %v", p)
				panic(p)
			}
			// TODO(gri) add a test case for this scenario
			err = fmt.Errorf("types internal error: %v", p)
		}
	}()

	// determine package name and files
	i := 0
	for _, file := range files {
		switch name := file.Name.Name; pkg.name {
		case "":
			pkg.name = name
			fallthrough
		case name:
			files[i] = file
			i++
		default:
			check.errorf(file.Package, "package %s; expected %s", name, pkg.name)
			// ignore this file
		}
	}
	files = files[:i]

	// resolve identifiers
	imp := ctxt.Import
	if imp == nil {
		imp = GcImport
	}

	if resolve {
		check.resolveFiles(files, imp)

	} else {
		methods := check.resolve(imp, files)

		// associate methods with types
		for _, m := range methods {
			check.assocMethod(m)
		}

		// typecheck all declarations
		for _, f := range files {
			for _, d := range f.Decls {
				check.decl(d)
			}
		}
	}

	// typecheck all function/method bodies
	// (funclist may grow when checking statements - do not use range clause!)
	for i := 0; i < len(check.funclist); i++ {
		f := check.funclist[i]
		if trace {
			s := "<function literal>"
			if f.obj != nil {
				s = f.obj.name
			}
			fmt.Println("---", s)
		}
		check.topScope = f.sig.scope // open the function scope
		check.funcsig = f.sig
		check.stmtList(f.body.List)
		if f.sig.results.Len() > 0 && f.body != nil && !check.isTerminating(f.body, "") {
			check.errorf(f.body.Rbrace, "missing return")
		}
	}

	// remaining untyped expressions must indeed be untyped
	if debug {
		for x, info := range check.untyped {
			if !isUntyped(info.typ) {
				check.dump("%s: %s (type %s) is not untyped", x.Pos(), x, info.typ)
				panic(0)
			}
		}
	}

	// notify client of any untyped types left
	// TODO(gri) Consider doing this before and
	// after function body checking for smaller
	// map size and more immediate feedback.
	if ctxt.Expr != nil {
		for x, info := range check.untyped {
			ctxt.Expr(x, info.typ, info.val)
		}
	}

	return
}
