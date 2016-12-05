// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/constant"
	"go/token"
)

func (check *Checker) reportAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "\tother declaration of %s", obj.Name()) // secondary error, \t indented
	}
}

func (check *Checker) declare(scope *Scope, id *ast.Ident, obj Object, pos token.Pos) {
	// spec: "The blank identifier, represented by the underscore
	// character _, may be used in a declaration like any other
	// identifier but the declaration does not introduce a new
	// binding."
	if obj.Name() != "_" {
		if alt := scope.Insert(obj); alt != nil {
			check.errorf(obj.Pos(), "%s redeclared in this block", obj.Name())
			check.reportAltDecl(alt)
			return
		}
		obj.setScopePos(pos)
	}
	if id != nil {
		check.recordDef(id, obj)
	}
}

// objDecl type-checks the declaration of obj in its respective (file) context.
// See check.typ for the details on def and path.
func (check *Checker) objDecl(obj Object, def *Named, path []*TypeName) {
	if obj.Type() != nil {
		return // already checked - nothing to do
	}

	if trace {
		check.trace(obj.Pos(), "-- declaring %s", obj.Name())
		check.indent++
		defer func() {
			check.indent--
			check.trace(obj.Pos(), "=> %s", obj)
		}()
	}

	d := check.objMap[obj]
	if d == nil {
		check.dump("%s: %s should have been declared", obj.Pos(), obj.Name())
		unreachable()
	}

	// save/restore current context and setup object context
	defer func(ctxt context) {
		check.context = ctxt
	}(check.context)
	check.context = context{
		scope: d.file,
	}

	// Const and var declarations must not have initialization
	// cycles. We track them by remembering the current declaration
	// in check.decl. Initialization expressions depending on other
	// consts, vars, or functions, add dependencies to the current
	// check.decl.
	switch obj := obj.(type) {
	case *Const:
		check.decl = d // new package-level const decl
		check.constDecl(obj, d.typ, d.init)
	case *Var:
		check.decl = d // new package-level var decl
		check.varDecl(obj, d.lhs, d.typ, d.init)
	case *TypeName:
		// invalid recursive types are detected via path
		check.typeDecl(obj, d.typ, def, path)
	case *Func:
		// functions may be recursive - no need to track dependencies
		check.funcDecl(obj, d)
	// Alias-related code. Keep for now.
	// case *Alias:
	// 	// aliases cannot be recursive - no need to track dependencies
	// 	check.aliasDecl(obj, d)
	default:
		unreachable()
	}
}

func (check *Checker) constDecl(obj *Const, typ, init ast.Expr) {
	assert(obj.typ == nil)

	if obj.visited {
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// use the correct value of iota
	assert(check.iota == nil)
	check.iota = obj.val
	defer func() { check.iota = nil }()

	// provide valid constant value under all circumstances
	obj.val = constant.MakeUnknown()

	// determine type, if any
	if typ != nil {
		t := check.typ(typ)
		if !isConstType(t) {
			check.errorf(typ.Pos(), "invalid constant type %s", t)
			obj.typ = Typ[Invalid]
			return
		}
		obj.typ = t
	}

	// check initialization
	var x operand
	if init != nil {
		check.expr(&x, init)
	}
	check.initConst(obj, &x)
}

func (check *Checker) varDecl(obj *Var, lhs []*Var, typ, init ast.Expr) {
	assert(obj.typ == nil)

	if obj.visited {
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// var declarations cannot use iota
	assert(check.iota == nil)

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ)
		// We cannot spread the type to all lhs variables if there
		// are more than one since that would mark them as checked
		// (see Checker.objDecl) and the assignment of init exprs,
		// if any, would not be checked.
		//
		// TODO(gri) If we have no init expr, we should distribute
		// a given type otherwise we need to re-evalate the type
		// expr for each lhs variable, leading to duplicate work.
	}

	// check initialization
	if init == nil {
		if typ == nil {
			// error reported before by arityMatch
			obj.typ = Typ[Invalid]
		}
		return
	}

	if lhs == nil || len(lhs) == 1 {
		assert(lhs == nil || lhs[0] == obj)
		var x operand
		check.expr(&x, init)
		check.initVar(obj, &x, "variable declaration")
		return
	}

	if debug {
		// obj must be one of lhs
		found := false
		for _, lhs := range lhs {
			if obj == lhs {
				found = true
				break
			}
		}
		if !found {
			panic("inconsistent lhs")
		}
	}

	// We have multiple variables on the lhs and one init expr.
	// Make sure all variables have been given the same type if
	// one was specified, otherwise they assume the type of the
	// init expression values (was issue #15755).
	if typ != nil {
		for _, lhs := range lhs {
			lhs.typ = obj.typ
		}
	}

	check.initVars(lhs, []ast.Expr{init}, token.NoPos)
}

// underlying returns the underlying type of typ; possibly by following
// forward chains of named types. Such chains only exist while named types
// are incomplete.
func underlying(typ Type) Type {
	for {
		n, _ := typ.(*Named)
		if n == nil {
			break
		}
		typ = n.underlying
	}
	return typ
}

func (n *Named) setUnderlying(typ Type) {
	if n != nil {
		n.underlying = typ
	}
}

func (check *Checker) typeDecl(obj *TypeName, typ ast.Expr, def *Named, path []*TypeName) {
	assert(obj.typ == nil)

	// type declarations cannot use iota
	assert(check.iota == nil)

	named := &Named{obj: obj}
	def.setUnderlying(named)
	obj.typ = named // make sure recursive type declarations terminate

	// determine underlying type of named
	check.typExpr(typ, named, append(path, obj))

	// The underlying type of named may be itself a named type that is
	// incomplete:
	//
	//	type (
	//		A B
	//		B *C
	//		C A
	//	)
	//
	// The type of C is the (named) type of A which is incomplete,
	// and which has as its underlying type the named type B.
	// Determine the (final, unnamed) underlying type by resolving
	// any forward chain (they always end in an unnamed type).
	named.underlying = underlying(named.underlying)

	// check and add associated methods
	// TODO(gri) It's easy to create pathological cases where the
	// current approach is incorrect: In general we need to know
	// and add all methods _before_ type-checking the type.
	// See https://play.golang.org/p/WMpE0q2wK8
	check.addMethodDecls(obj)
}

func (check *Checker) addMethodDecls(obj *TypeName) {
	// get associated methods
	methods := check.methods[obj.name]
	if len(methods) == 0 {
		return // no methods
	}
	delete(check.methods, obj.name)

	// use an objset to check for name conflicts
	var mset objset

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	base := obj.typ.(*Named)
	if t, _ := base.underlying.(*Struct); t != nil {
		for _, fld := range t.fields {
			if fld.name != "_" {
				assert(mset.insert(fld) == nil)
			}
		}
	}

	// Checker.Files may be called multiple times; additional package files
	// may add methods to already type-checked types. Add pre-existing methods
	// so that we can detect redeclarations.
	for _, m := range base.methods {
		assert(m.name != "_")
		assert(mset.insert(m) == nil)
	}

	// type-check methods
	for _, m := range methods {
		// spec: "For a base type, the non-blank names of methods bound
		// to it must be unique."
		if m.name != "_" {
			if alt := mset.insert(m); alt != nil {
				switch alt.(type) {
				case *Var:
					check.errorf(m.pos, "field and method with the same name %s", m.name)
				case *Func:
					check.errorf(m.pos, "method %s already declared for %s", m.name, base)
				default:
					unreachable()
				}
				check.reportAltDecl(alt)
				continue
			}
		}
		check.objDecl(m, nil, nil)
		// methods with blank _ names cannot be found - don't keep them
		if m.name != "_" {
			base.methods = append(base.methods, m)
		}
	}
}

func (check *Checker) funcDecl(obj *Func, decl *declInfo) {
	assert(obj.typ == nil)

	// func declarations cannot use iota
	assert(check.iota == nil)

	sig := new(Signature)
	obj.typ = sig // guard against cycles
	fdecl := decl.fdecl
	check.funcType(sig, fdecl.Recv, fdecl.Type)
	if sig.recv == nil && obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
		check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
		// ok to continue
	}

	// function body must be type-checked after global declarations
	// (functions implemented elsewhere have no body)
	if !check.conf.IgnoreFuncBodies && fdecl.Body != nil {
		check.later(obj.name, decl, sig, fdecl.Body)
	}
}

// original returns the original Object if obj is an Alias;
// otherwise it returns obj. The result is never an Alias,
// but it may be nil.
func original(obj Object) Object {
	// an alias stands for the original object; use that one instead
	if alias, _ := obj.(*disabledAlias); alias != nil {
		obj = alias.orig
		// aliases always refer to non-alias originals
		if _, ok := obj.(*disabledAlias); ok {
			panic("original is an alias")
		}
	}
	return obj
}

func (check *Checker) aliasDecl(obj *disabledAlias, decl *declInfo) {
	assert(obj.typ == nil)

	// alias declarations cannot use iota
	assert(check.iota == nil)

	// assume alias is invalid to start with
	obj.typ = Typ[Invalid]

	// rhs must be package-qualified identifer pkg.sel (see also call.go: checker.selector)
	// TODO(gri) factor this code out and share with checker.selector
	rhs := decl.init
	var pkg *Package
	var sel *ast.Ident
	if sexpr, ok := rhs.(*ast.SelectorExpr); ok {
		if ident, ok := sexpr.X.(*ast.Ident); ok {
			_, obj := check.scope.LookupParent(ident.Name, check.pos)
			if pname, _ := obj.(*PkgName); pname != nil {
				assert(pname.pkg == check.pkg)
				check.recordUse(ident, pname)
				pname.used = true
				pkg = pname.imported
				sel = sexpr.Sel
			}
		}
	}
	if pkg == nil {
		check.errorf(rhs.Pos(), "invalid alias: %v is not a package-qualified identifier", rhs)
		return
	}

	// qualified identifier must denote an exported object
	orig := pkg.scope.Lookup(sel.Name)
	if orig == nil || !orig.Exported() {
		if !pkg.fake {
			check.errorf(rhs.Pos(), "%s is not exported by package %s", sel.Name, pkg.name)
		}
		return
	}
	check.recordUse(sel, orig)
	orig = original(orig)

	// avoid further errors if the imported object is an alias that's broken
	if orig == nil {
		return
	}

	// An alias declaration must not refer to package unsafe.
	if orig.Pkg() == Unsafe {
		check.errorf(rhs.Pos(), "invalid alias: %s refers to package unsafe (%v)", obj.Name(), orig)
		return
	}

	// The original must be of the same kind as the alias declaration.
	var why string
	switch obj.kind {
	case token.CONST:
		if _, ok := orig.(*Const); !ok {
			why = "constant"
		}
	case token.TYPE:
		if _, ok := orig.(*TypeName); !ok {
			why = "type"
		}
	case token.VAR:
		if _, ok := orig.(*Var); !ok {
			why = "variable"
		}
	case token.FUNC:
		if _, ok := orig.(*Func); !ok {
			why = "function"
		}
	default:
		unreachable()
	}
	if why != "" {
		check.errorf(rhs.Pos(), "invalid alias: %v is not a %s", orig, why)
		return
	}

	// alias is valid
	obj.typ = orig.Type()
	obj.orig = orig
}

func (check *Checker) declStmt(decl ast.Decl) {
	pkg := check.pkg

	switch d := decl.(type) {
	case *ast.BadDecl:
		// ignore

	case *ast.GenDecl:
		var last *ast.ValueSpec // last ValueSpec with type or init exprs seen
		for iota, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ValueSpec:
				switch d.Tok {
				case token.CONST:
					// determine which init exprs to use
					switch {
					case s.Type != nil || len(s.Values) > 0:
						last = s
					case last == nil:
						last = new(ast.ValueSpec) // make sure last exists
					}

					// declare all constants
					lhs := make([]*Const, len(s.Names))
					for i, name := range s.Names {
						obj := NewConst(name.Pos(), pkg, name.Name, nil, constant.MakeInt64(int64(iota)))
						lhs[i] = obj

						var init ast.Expr
						if i < len(last.Values) {
							init = last.Values[i]
						}

						check.constDecl(obj, last.Type, init)
					}

					check.arityMatch(s, last)

					// spec: "The scope of a constant or variable identifier declared
					// inside a function begins at the end of the ConstSpec or VarSpec
					// (ShortVarDecl for short variable declarations) and ends at the
					// end of the innermost containing block."
					scopePos := s.End()
					for i, name := range s.Names {
						check.declare(check.scope, name, lhs[i], scopePos)
					}

				case token.VAR:
					lhs0 := make([]*Var, len(s.Names))
					for i, name := range s.Names {
						lhs0[i] = NewVar(name.Pos(), pkg, name.Name, nil)
					}

					// initialize all variables
					for i, obj := range lhs0 {
						var lhs []*Var
						var init ast.Expr
						switch len(s.Values) {
						case len(s.Names):
							// lhs and rhs match
							init = s.Values[i]
						case 1:
							// rhs is expected to be a multi-valued expression
							lhs = lhs0
							init = s.Values[0]
						default:
							if i < len(s.Values) {
								init = s.Values[i]
							}
						}
						check.varDecl(obj, lhs, s.Type, init)
						if len(s.Values) == 1 {
							// If we have a single lhs variable we are done either way.
							// If we have a single rhs expression, it must be a multi-
							// valued expression, in which case handling the first lhs
							// variable will cause all lhs variables to have a type
							// assigned, and we are done as well.
							if debug {
								for _, obj := range lhs0 {
									assert(obj.typ != nil)
								}
							}
							break
						}
					}

					check.arityMatch(s, nil)

					// declare all variables
					// (only at this point are the variable scopes (parents) set)
					scopePos := s.End() // see constant declarations
					for i, name := range s.Names {
						// see constant declarations
						check.declare(check.scope, name, lhs0[i], scopePos)
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				// spec: "The scope of a type identifier declared inside a function
				// begins at the identifier in the TypeSpec and ends at the end of
				// the innermost containing block."
				scopePos := s.Name.Pos()
				check.declare(check.scope, s.Name, obj, scopePos)
				check.typeDecl(obj, s.Type, nil, nil)

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
