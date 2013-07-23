// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"errors"
	"fmt"
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
		check.recordObject(id, obj)
	}
}

// A declInfo describes a package-level const, type, var, or func declaration.
type declInfo struct {
	file  *Scope        // scope of file containing this declaration
	typ   ast.Expr      // type, or nil
	init  ast.Expr      // initialization expression, or nil
	fdecl *ast.FuncDecl // function declaration, or nil
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

// TODO(gri) Split resolveFiles into smaller components.

func (check *checker) resolveFiles(files []*ast.File) {
	pkg := check.pkg

	// Phase 1: Pre-declare all package-level objects so that they can be found
	//          independent of source order. Collect methods for a later phase.

	var (
		fileScope *Scope                       // current file scope, used by declare
		scopes    []*Scope                     // corresponding file scope per file
		objList   []Object                     // list of objects to type-check
		objMap    = make(map[Object]*declInfo) // declaration info for each object (incl. methods)
	)

	declare := func(ident *ast.Ident, obj Object, typ, init ast.Expr, fdecl *ast.FuncDecl) {
		assert(ident.Name == obj.Name())

		// spec: "A package-scope or file-scope identifier with name init
		// may only be declared to be a function with this (func()) signature."
		if ident.Name == "init" {
			f, _ := obj.(*Func)
			if f == nil {
				check.recordObject(ident, nil)
				check.errorf(ident.Pos(), "cannot declare init - must be func")
				return
			}
			// don't declare init functions in the package scope - they are invisible
			f.parent = pkg.scope
			check.recordObject(ident, obj)
		} else {
			check.declare(pkg.scope, ident, obj)
		}

		objList = append(objList, obj)
		objMap[obj] = &declInfo{fileScope, typ, init, fdecl}
	}

	importer := check.conf.Import
	if importer == nil {
		importer = GcImport
	}

	for _, file := range files {
		// the package identifier denotes the current package, but it is in no scope
		check.recordObject(file.Name, pkg)

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
						path, _ := strconv.Unquote(s.Path.Value)
						imp, err := importer(pkg.imports, path)
						if imp == nil && err == nil {
							err = errors.New("Config.Import returned nil")
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
							check.recordObject(s.Name, imp2)
						} else {
							check.recordImplicit(s, imp2)
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
									check.recordImplicit(s, obj)
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

								declare(name, obj, last.Type, init, nil)
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

								declare(name, obj, s.Type, init, nil)
							}

							check.arityMatch(s, nil)

						default:
							check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
						}

					case *ast.TypeSpec:
						obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
						declare(s.Name, obj, s.Type, nil, nil)

					default:
						check.invalidAST(s.Pos(), "unknown ast.Spec node %T", s)
					}
				}

			case *ast.FuncDecl:
				obj := NewFunc(d.Name.Pos(), pkg, d.Name.Name, nil)
				if d.Recv == nil {
					// regular function
					declare(d.Name, obj, nil, nil, d)
					continue
				}

				// Methods are associated with the receiver base type
				// which we don't have yet. Instead, collect methods
				// with receiver base type name so that they are known
				// when the receiver base type is type-checked.

				// The receiver type must be one of the following
				// - *ast.Ident
				// - *ast.StarExpr{*ast.Ident}
				// - *ast.BadExpr (parser error)
				typ := d.Recv.List[0].Type
				if ptr, _ := typ.(*ast.StarExpr); ptr != nil {
					typ = ptr.X
				}

				// Associate method with receiver base type name, if possible.
				// Methods with receiver types that are not type names, that
				// are blank _ names, or methods with blank names are ignored;
				// the respective error will be reported when the method signature
				// is type-checked.
				if ident, _ := typ.(*ast.Ident); ident != nil && ident.Name != "_" && obj.name != "_" {
					check.methods[ident.Name] = append(check.methods[ident.Name], obj)
				}

				// Collect methods like other objects.
				objList = append(objList, obj)
				objMap[obj] = &declInfo{fileScope, nil, nil, d}

			default:
				check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
			}
		}
	}

	// Phase 2: Verify that objects in package and file scopes have different names.

	for _, scope := range scopes {
		for _, obj := range scope.entries {
			if alt := pkg.scope.Lookup(nil, obj.Name()); alt != nil {
				check.errorf(alt.Pos(), "%s already declared in this file through import of package %s", obj.Name(), obj.Pkg().Name())
			}
		}
	}

	// Phase 3: Typecheck all objects in objList, but not function bodies.

	check.objMap = objMap // indicate that we are checking global declarations (objects may not have a type yet)
	for _, obj := range objList {
		if obj.Type() == nil {
			check.declareObject(obj, nil, false)
		}
	}
	check.objMap = nil // done with global declarations

	// At this point we may have a non-empty check.methods map; this means that not all
	// entries were deleted at the end of declareType, because the respective receiver
	// base types were not declared. In that case, an error was reported when declaring
	// those methods. We can now safely discard this map.
	check.methods = nil

	// Phase 4: Typecheck all functions bodies.

	// (funclist may grow when checking statements - cannot use range clause)
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
		check.declareFunc(obj, d.fdecl)
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

	// type-check signatures of associated methods
	methods := check.methods[obj.name]
	if len(methods) == 0 {
		return // no methods
	}

	// spec: "For a base type, the non-blank names of methods bound
	// to it must be unique."
	// => use a scope to determine redeclarations
	scope := NewScope(nil)

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	// => pre-populate the scope to find conflicts
	if t, _ := named.underlying.(*Struct); t != nil {
		for _, fld := range t.fields {
			if fld.name != "_" {
				scope.Insert(fld)
			}
		}
	}

	// check each method
	for _, m := range methods {
		assert(m.name != "_") // _ methods were excluded before
		mdecl := check.objMap[m]
		alt := scope.Insert(m)
		m.parent = mdecl.file // correct parent scope (scope.Insert used scope)

		if alt != nil {
			switch alt := alt.(type) {
			case *Var:
				check.errorf(m.pos, "field and method with the same name %s", m.name)
				if pos := alt.pos; pos.IsValid() {
					check.errorf(pos, "previous declaration of %s", m.name)
				}
				m = nil
			case *Func:
				check.errorf(m.pos, "method %s already declared for %s", m.name, named)
				if pos := alt.pos; pos.IsValid() {
					check.errorf(pos, "previous declaration of %s", m.name)
				}
				m = nil
			}
		}

		check.recordObject(mdecl.fdecl.Name, m)

		// If the method is valid, type-check its signature,
		// and collect it with the named base type.
		if m != nil {
			check.declareObject(m, nil, true)
			named.methods = append(named.methods, m)
		}
	}

	delete(check.methods, obj.name) // we don't need them anymore
}

func (check *checker) declareFunc(obj *Func, fdecl *ast.FuncDecl) {
	// func declarations cannot use iota
	assert(check.iota == nil)

	obj.typ = Typ[Invalid] // guard against cycles
	sig := check.funcType(fdecl.Recv, fdecl.Type, nil)
	if sig.recv == nil && obj.name == "init" && (sig.params.Len() > 0 || sig.results.Len() > 0) {
		check.errorf(fdecl.Pos(), "func init must have no arguments and no return values")
		// ok to continue
	}
	obj.typ = sig
	check.later(obj, sig, fdecl.Body)
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
