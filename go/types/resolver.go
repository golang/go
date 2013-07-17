// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"errors"
	"go/ast"
	"go/token"
	"strconv"

	"code.google.com/p/go.tools/go/exact"
)

func (check *checker) declare(scope *Scope, id *ast.Ident, obj Object) {
	if obj.Name() == "_" {
		// blank identifiers are not declared
		obj.setParent(scope)
	} else if alt := scope.Insert(obj); alt != nil {
		check.errorf(obj.Pos(), "%s redeclared in this block", obj.Name())
		if pos := alt.Pos(); pos.IsValid() {
			check.errorf(pos, "previous declaration of %s", obj.Name())
		}
		obj = nil // for callIdent below
	}
	if id != nil {
		check.callIdent(id, obj)
	}
}

// A decl describes a package-level const, type, var, or func declaration.
type decl struct {
	file *Scope   // scope of file containing this declaration
	typ  ast.Expr // type, or nil
	init ast.Expr // initialization expression, or nil
}

// An mdecl describes a method declaration.
type mdecl struct {
	file *Scope // scope of file containing this declaration
	meth *ast.FuncDecl
}

// A multiExpr describes the lhs variables and a single but
// (expected to be) multi-valued rhs init expr of a variable
// declaration.
type multiExpr struct {
	lhs      []*Var
	rhs      []ast.Expr // len(rhs) == 1
	ast.Expr            // dummy to satisfy ast.Expr interface
}

// arityMatch checks that the lhs and rhs of a const or var decl
// have the appropriate number of names and init exprs. For const
// decls, init is the value spec providing the init exprs; for
// var decls, init is nil (the init exprs are in s in this case).
func (check *checker) arityMatch(s, init *ast.ValueSpec) {
	l := len(s.Names)
	r := len(s.Values)
	if init != nil {
		r = len(init.Values)
	}

	switch {
	case init == nil && r == 0:
		// var decl w/o init expr
		if s.Type == nil {
			check.errorf(s.Pos(), "missing type or init expr")
		}
	case l < r:
		if l < len(s.Values) {
			// init exprs from s
			n := s.Values[l]
			check.errorf(n.Pos(), "extra init expr %s", n)
		} else {
			// init exprs "inherited"
			check.errorf(s.Pos(), "extra init expr at %s", init.Pos())
		}
	case l > r && (init != nil || r != 1):
		n := s.Names[r]
		check.errorf(n.Pos(), "missing init expr for %s", n)
	}
}

func (check *checker) resolveFiles(files []*ast.File, importer Importer) {
	pkg := check.pkg

	// Phase 1: Pre-declare all package scope objects so that they can be found
	//          when type-checking package objects.

	var scopes []*Scope // corresponding file scope per file
	var objList []Object
	var objMap = make(map[Object]*decl)
	var methods []*mdecl
	var fileScope *Scope // current file scope, used by collect

	declare := func(ident *ast.Ident, obj Object, typ, init ast.Expr) {
		assert(ident.Name == obj.Name())

		// spec: "A package-scope or file-scope identifier with name init
		// may only be declared to be a function with this (func()) signature."
		if ident.Name == "init" {
			f, _ := obj.(*Func)
			if f == nil {
				check.callIdent(ident, nil)
				check.errorf(ident.Pos(), "cannot declare init - must be func")
				return
			}
			// don't declare init functions in the package scope - they are invisible
			f.parent = pkg.scope
			check.callIdent(ident, obj)
		} else {
			check.declare(pkg.scope, ident, obj)
		}

		objList = append(objList, obj)
		objMap[obj] = &decl{fileScope, typ, init}
	}

	for _, file := range files {
		// the package identifier denotes the current package, but it is in no scope
		check.callIdent(file.Name, pkg)

		fileScope = NewScope(pkg.scope)
		if retainASTLinks {
			fileScope.node = file
		}
		scopes = append(scopes, fileScope)

		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.BadDecl:
				// ignore

			case *ast.GenDecl:
				var last *ast.ValueSpec // last ValueSpec with type or init exprs seen
				for iota, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.ImportSpec:
						if importer == nil {
							continue
						}
						path, _ := strconv.Unquote(s.Path.Value)
						imp, err := importer(pkg.imports, path)
						if imp == nil && err == nil {
							err = errors.New("Context.Import returned nil")
						}
						if err != nil {
							check.errorf(s.Path.Pos(), "could not import %s (%s)", path, err)
							continue
						}

						// local name overrides imported package name
						name := imp.name
						if s.Name != nil {
							name = s.Name.Name
							if name == "init" {
								check.errorf(s.Name.Pos(), "cannot declare init - must be func")
								continue
							}
						}

						imp2 := NewPackage(s.Pos(), path, name, imp.scope, nil, imp.complete)
						if s.Name != nil {
							check.callIdent(s.Name, imp2)
						} else {
							check.callImplicitObj(s, imp2)
						}

						// add import to file scope
						if name == "." {
							// merge imported scope with file scope
							for _, obj := range imp.scope.entries {
								// gcimported package scopes contain non-exported
								// objects such as types used in partially exported
								// objects - do not accept them
								if obj.IsExported() {
									// Note: This will change each imported object's scope!
									//       May be an issue for types aliases.
									check.declare(fileScope, nil, obj)
									check.callImplicitObj(s, obj)
								}
							}
						} else {
							// declare imported package object in file scope
							check.declare(fileScope, nil, imp2)
						}

					case *ast.ValueSpec:
						switch d.Tok {
						case token.CONST:
							// determine which initialization expressions to use
							switch {
							case s.Type != nil || len(s.Values) > 0:
								last = s
							case last == nil:
								last = new(ast.ValueSpec) // make sure last exists
							}

							// declare all constants
							for i, name := range s.Names {
								obj := NewConst(name.Pos(), pkg, name.Name, nil, exact.MakeInt64(int64(iota)))

								var init ast.Expr
								if i < len(last.Values) {
									init = last.Values[i]
								}

								declare(name, obj, last.Type, init)
							}

							check.arityMatch(s, last)

						case token.VAR:
							// declare all variables
							lhs := make([]*Var, len(s.Names))
							for i, name := range s.Names {
								obj := NewVar(name.Pos(), pkg, name.Name, nil)
								lhs[i] = obj

								var init ast.Expr
								switch len(s.Values) {
								case len(s.Names):
									// lhs and rhs match
									init = s.Values[i]
								case 1:
									// rhs must be a multi-valued expression
									// (lhs may not be fully set up yet, but
									// that's fine because declare simply collects
									// the information for later processing.)
									init = &multiExpr{lhs, s.Values, nil}
								default:
									if i < len(s.Values) {
										init = s.Values[i]
									}
								}

								declare(name, obj, s.Type, init)
							}

							check.arityMatch(s, nil)

						default:
							check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
						}

					case *ast.TypeSpec:
						obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
						declare(s.Name, obj, s.Type, nil)

					default:
						check.invalidAST(s.Pos(), "unknown ast.Spec node %T", s)
					}
				}

			case *ast.FuncDecl:
				if d.Recv != nil {
					// collect method
					methods = append(methods, &mdecl{fileScope, d})
					continue
				}
				obj := NewFunc(d.Name.Pos(), pkg, d.Name.Name, nil)
				obj.decl = d
				declare(d.Name, obj, nil, nil)

			default:
				check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
			}
		}
	}

	// Phase 2: Objects in file scopes and package scopes must have different names.
	for _, scope := range scopes {
		for _, obj := range scope.entries {
			if alt := pkg.scope.Lookup(nil, obj.Name()); alt != nil {
				check.errorf(alt.Pos(), "%s already declared in this file through import of package %s", obj.Name(), obj.Pkg().Name())
			}
		}
	}

	// Phase 3: Associate methods with types.
	//          We do this after all top-level type names have been collected.

	for _, meth := range methods {
		m := meth.meth
		// The receiver type must be one of the following:
		// - *ast.Ident
		// - *ast.StarExpr{*ast.Ident}
		// - *ast.BadExpr (parser error)
		typ := m.Recv.List[0].Type
		if ptr, ok := typ.(*ast.StarExpr); ok {
			typ = ptr.X
		}
		// determine receiver base type name
		// Note: We cannot simply call check.typ because this will require
		//       check.objMap to be usable, which it isn't quite yet.
		ident, ok := typ.(*ast.Ident)
		if !ok {
			// Disabled for now since the parser reports this error.
			// check.errorf(typ.Pos(), "receiver base type must be an (unqualified) identifier")
			continue // ignore this method
		}
		// determine receiver base type object
		var tname *TypeName
		if obj := pkg.scope.LookupParent(ident.Name); obj != nil {
			obj, ok := obj.(*TypeName)
			if !ok {
				check.errorf(ident.Pos(), "%s is not a type", ident.Name)
				continue // ignore this method
			}
			if obj.pkg != pkg {
				check.errorf(ident.Pos(), "cannot define method on non-local type %s", ident.Name)
				continue // ignore this method
			}
			tname = obj
		} else {
			// identifier not declared/resolved
			if ident.Name == "_" {
				check.errorf(ident.Pos(), "cannot use _ as value or type")
			} else {
				check.errorf(ident.Pos(), "undeclared name: %s", ident.Name)
			}
			continue // ignore this method
		}
		// declare method in receiver base type scope
		scope := check.methods[tname] // lazily allocated
		if scope == nil {
			scope = new(Scope)
			check.methods[tname] = scope
		}
		fun := NewFunc(m.Name.Pos(), check.pkg, m.Name.Name, nil)
		fun.decl = m
		check.declare(scope, m.Name, fun)
		// HACK(gri) change method parent scope to file scope containing the declaration
		fun.parent = meth.file // remember the file scope
	}

	// Phase 4) Typecheck all objects in objList but not function bodies.

	check.objMap = objMap // indicate that we are checking global declarations (objects may not have a type yet)
	for _, obj := range objList {
		if obj.Type() == nil {
			check.declareObject(obj, nil, false)
		}
	}
	check.objMap = nil // done with global declarations

	// Phase 5) Typecheck all functions.
	// - done by the caller for now
}

// declareObject completes the declaration of obj in its respective file scope.
// See declareType for the details on def and cycleOk.
func (check *checker) declareObject(obj Object, def *Named, cycleOk bool) {
	d := check.objMap[obj]

	// adjust file scope for current object
	oldScope := check.topScope
	check.topScope = d.file // for lookup

	// save current iota
	oldIota := check.iota
	check.iota = nil

	switch obj := obj.(type) {
	case *Const:
		check.declareConst(obj, d.typ, d.init)
	case *Var:
		check.declareVar(obj, d.typ, d.init)
	case *TypeName:
		check.declareType(obj, d.typ, def, cycleOk)
	case *Func:
		check.declareFunc(obj)
	default:
		unreachable()
	}

	check.iota = oldIota
	check.topScope = oldScope
}

func (check *checker) declareConst(obj *Const, typ, init ast.Expr) {
	if obj.visited {
		check.errorf(obj.Pos(), "illegal cycle in initialization of constant %s", obj.name)
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// use the correct value of iota
	assert(check.iota == nil)
	check.iota = obj.val

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ, nil, false)
	}

	// check initialization
	var x operand
	if init != nil {
		check.expr(&x, init)
	}
	check.initConst(obj, &x)

	check.iota = nil
}

func (check *checker) declareVar(obj *Var, typ, init ast.Expr) {
	if obj.visited {
		check.errorf(obj.Pos(), "illegal cycle in initialization of variable %s", obj.name)
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// var declarations cannot use iota
	assert(check.iota == nil)

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ, nil, false)
	}

	// check initialization
	if init == nil {
		if typ == nil {
			// error reported before by arityMatch
			obj.typ = Typ[Invalid]
		}
		return
	}

	if m, _ := init.(*multiExpr); m != nil {
		check.initVars(m.lhs, m.rhs, true)
		return
	}

	var x operand
	check.expr(&x, init)
	check.initVar(obj, &x)
}

func (check *checker) declareType(obj *TypeName, typ ast.Expr, def *Named, cycleOk bool) {
	assert(obj.Type() == nil)

	// type declarations cannot use iota
	assert(check.iota == nil)

	named := &Named{obj: obj}
	obj.typ = named // make sure recursive type declarations terminate

	// If this type (named) defines the type of another (def) type declaration,
	// set def's underlying type to this type so that we can resolve the true
	// underlying of def later.
	if def != nil {
		def.underlying = named
	}

	// Typecheck typ - it may be a named type that is not yet complete.
	// For instance, consider:
	//
	//	type (
	//		A B
	//		B *C
	//		C A
	//	)
	//
	// When we declare obj = C, typ is the identifier A which is incomplete.
	u := check.typ(typ, named, cycleOk)

	// Determine the unnamed underlying type.
	// In the above example, the underlying type of A was (temporarily) set
	// to B whose underlying type was set to *C. Such "forward chains" always
	// end in an unnamed type (cycles are terminated with an invalid type).
	for {
		n, _ := u.(*Named)
		if n == nil {
			break
		}
		u = n.underlying
	}
	named.underlying = u

	// the underlying type has been determined
	named.complete = true

	// typecheck associated method signatures
	if scope := check.methods[obj]; scope != nil {
		switch t := named.underlying.(type) {
		case *Struct:
			// struct fields must not conflict with methods
			if t.fields == nil {
				break
			}
			for _, f := range t.fields {
				if m := scope.Lookup(nil, f.name); m != nil {
					check.errorf(m.Pos(), "type %s has both field and method named %s", obj.name, f.name)
					// ok to continue
				}
			}
		case *Interface:
			// methods cannot be associated with an interface type
			for _, m := range scope.entries {
				recv := m.(*Func).decl.Recv.List[0].Type
				check.errorf(recv.Pos(), "invalid receiver type %s (%s is an interface type)", obj.name, obj.name)
				// ok to continue
			}
		}
		// typecheck method signatures
		var methods []*Func
		if scope.NumEntries() > 0 {
			for _, obj := range scope.entries {
				m := obj.(*Func)

				// set the correct file scope for checking this method type
				fileScope := m.parent
				assert(fileScope != nil)
				oldScope := check.topScope
				check.topScope = fileScope

				sig := check.typ(m.decl.Type, nil, cycleOk).(*Signature)
				params, _ := check.collectParams(sig.scope, m.decl.Recv, false)

				check.topScope = oldScope // reset topScope

				sig.recv = params[0] // the parser/assocMethod ensure there is exactly one parameter
				m.typ = sig
				methods = append(methods, m)
				check.later(m, sig, m.decl.Body)
			}
		}
		named.methods = methods
		delete(check.methods, obj) // we don't need this scope anymore
	}
}

func (check *checker) declareFunc(obj *Func) {
	// func declarations cannot use iota
	assert(check.iota == nil)

	fdecl := obj.decl
	// methods are typechecked when their receivers are typechecked
	// TODO(gri) there is no reason to make this a special case: receivers are simply parameters
	if fdecl.Recv == nil {
		obj.typ = Typ[Invalid] // guard against cycles
		sig := check.typ(fdecl.Type, nil, false).(*Signature)
		if obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
			check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
			// ok to continue
		}
		obj.typ = sig
		check.later(obj, sig, fdecl.Body)
	}
}

func (check *checker) declStmt(decl ast.Decl) {
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
						obj := NewConst(name.Pos(), pkg, name.Name, nil, exact.MakeInt64(int64(iota)))
						lhs[i] = obj

						var init ast.Expr
						if i < len(last.Values) {
							init = last.Values[i]
						}

						check.declareConst(obj, last.Type, init)
					}

					check.arityMatch(s, last)

					for i, name := range s.Names {
						check.declare(check.topScope, name, lhs[i])
					}

				case token.VAR:
					// For declareVar called with a multiExpr we need the fully
					// initialized lhs. Compute it in a separate pre-pass.
					lhs := make([]*Var, len(s.Names))
					for i, name := range s.Names {
						lhs[i] = NewVar(name.Pos(), pkg, name.Name, nil)
					}

					// declare all variables
					for i, obj := range lhs {
						var init ast.Expr
						switch len(s.Values) {
						case len(s.Names):
							// lhs and rhs match
							init = s.Values[i]
						case 1:
							// rhs is expected to be a multi-valued expression
							init = &multiExpr{lhs, s.Values, nil}
						default:
							if i < len(s.Values) {
								init = s.Values[i]
							}
						}

						check.declareVar(obj, s.Type, init)
					}

					check.arityMatch(s, nil)

					for i, name := range s.Names {
						check.declare(check.topScope, name, lhs[i])
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				check.declare(check.topScope, s.Name, obj)
				check.declareType(obj, s.Type, nil, false)

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
