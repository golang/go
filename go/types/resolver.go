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

func (check *checker) declareShort(scope *Scope, list []Object) {
	n := 0 // number of new objects
	for _, obj := range list {
		if obj.Name() == "_" {
			obj.setParent(scope)
			continue // blank identifiers are not visible
		}
		if scope.Insert(obj) == nil {
			n++ // new declaration
		}
	}
	if n == 0 {
		check.errorf(list[0].Pos(), "no new variables on left side of :=")
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

// A projExpr projects the index'th value of a multi-valued expression.
// projExpr implements ast.Expr.
type projExpr struct {
	lhs      []*Var // all variables on the lhs
	ast.Expr        // rhs
}

func (check *checker) resolveFiles(files []*ast.File, importer Importer) {
	pkg := check.pkg

	// Phase 1: Pre-declare all package scope objects so that they can be found
	//          when type-checking package objects.

	var scopes []*Scope // corresponding file scope per file
	var objList []Object
	var objMap = make(map[Object]*decl)
	var methods []*mdecl
	var fileScope *Scope // current file scope, used by add

	add := func(obj Object, typ, init ast.Expr) {
		objList = append(objList, obj)
		objMap[obj] = &decl{fileScope, typ, init}
		// TODO(gri) move check.declare call here
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
				var last *ast.ValueSpec // last list of const initializers seen
				for iota, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.ImportSpec:
						if importer == nil {
							continue
						}
						path, _ := strconv.Unquote(s.Path.Value)
						imp, err := importer(pkg.imports, path)
						if imp == nil && err == nil {
							err = errors.New("Context.Import returned niil")
						}
						if err != nil {
							check.errorf(s.Path.Pos(), "could not import %s (%s)", path, err)
							continue
						}

						// local name overrides imported package name
						name := imp.name
						if s.Name != nil {
							name = s.Name.Name
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
								if ast.IsExported(obj.Name()) {
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
							if len(s.Values) > 0 {
								last = s
							}

							// declare all constants
							for i, name := range s.Names {
								obj := NewConst(name.Pos(), pkg, name.Name, nil, exact.MakeInt64(int64(iota)))
								check.declare(pkg.scope, name, obj)

								var init ast.Expr
								if i < len(last.Values) {
									init = last.Values[i]
								}
								add(obj, last.Type, init)
							}

							// arity of lhs and rhs must match
							if lhs, rhs := len(s.Names), len(s.Values); rhs > 0 {
								switch {
								case lhs < rhs:
									x := s.Values[lhs]
									check.errorf(x.Pos(), "too many initialization expressions")
								case lhs > rhs && rhs != 1:
									n := s.Names[rhs]
									check.errorf(n.Pos(), "missing initialization expression for %s", n)
								}
							}

						case token.VAR:
							// declare all variables
							lhs := make([]*Var, len(s.Names))
							for i, name := range s.Names {
								obj := NewVar(name.Pos(), pkg, name.Name, nil)
								lhs[i] = obj
								check.declare(pkg.scope, name, obj)

								var init ast.Expr
								switch len(s.Values) {
								case len(s.Names):
									// lhs and rhs match
									init = s.Values[i]
								case 1:
									// rhs must be a multi-valued expression
									init = &projExpr{lhs, s.Values[0]}
								default:
									if i < len(s.Values) {
										init = s.Values[i]
									}
								}
								add(obj, s.Type, init)
							}

							// report if there are too many initialization expressions
							if lhs, rhs := len(s.Names), len(s.Values); rhs > 0 {
								switch {
								case lhs < rhs:
									x := s.Values[lhs]
									check.errorf(x.Pos(), "too many initialization expressions")
								case lhs > rhs && rhs != 1:
									n := s.Names[rhs]
									check.errorf(n.Pos(), "missing initialization expression for %s", n)
								}
							}

						default:
							check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
						}

					case *ast.TypeSpec:
						obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
						check.declare(pkg.scope, s.Name, obj)
						add(obj, s.Type, nil)

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
				if obj.name == "init" {
					// init functions are not visible - don't declare them in package scope
					obj.parent = pkg.scope
					check.callIdent(d.Name, obj)
				} else {
					check.declare(pkg.scope, d.Name, obj)
				}
				add(obj, nil, nil)

			default:
				check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
			}
		}
	}

	// Phase 2: Objects in file scopes and package scopes must have different names.
	for _, scope := range scopes {
		for _, obj := range scope.entries {
			if alt := pkg.scope.Lookup(nil, obj.Name()); alt != nil {
				// TODO(gri) better error message
				check.errorf(alt.Pos(), "%s redeclared in this block by import of package %s", obj.Name(), obj.Pkg().Name())
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

	check.objMap = objMap // indicate we are doing global declarations (objects may not have a type yet)
	check.topScope = pkg.scope
	for _, obj := range objList {
		if obj.Type() == nil {
			check.declareObject(obj, false)
		}
	}
	check.objMap = nil // done with global declarations

	// Phase 5) Typecheck all functions.
	// - done by the caller for now
}

func (check *checker) declareObject(obj Object, cycleOk bool) {
	d := check.objMap[obj]

	// adjust file scope for current object
	oldScope := check.topScope
	check.topScope = d.file // for lookup

	switch obj := obj.(type) {
	case *Const:
		check.declareConst(obj, d.typ, d.init)
	case *Var:
		check.declareVar(obj, d.typ, d.init)
	case *TypeName:
		check.declareType(obj, d.typ, cycleOk)
	case *Func:
		check.declareFunc(obj)
	default:
		unreachable()
	}

	check.topScope = oldScope
}

func (check *checker) declareConst(obj *Const, typ, init ast.Expr) {
	if obj.visited {
		check.errorf(obj.Pos(), "illegal cycle in initialization of constant %s", obj.name)
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true
	iota, ok := exact.Int64Val(obj.val) // set in phase 1
	assert(ok)
	// obj.val = exact.MakeUnknown() //do we need this? should we set val to nil?

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ, false)
	}

	var x operand

	if init == nil {
		goto Error // error reported before
	}

	check.expr(&x, init, nil, int(iota))
	if x.mode == invalid {
		goto Error
	}

	check.assign(obj, &x)
	return

Error:
	if obj.typ == nil {
		obj.typ = Typ[Invalid]
	} else {
		obj.val = exact.MakeUnknown()
	}
}

func (check *checker) declareVar(obj *Var, typ, init ast.Expr) {
	if obj.visited {
		check.errorf(obj.Pos(), "illegal cycle in initialization of variable %s", obj.name)
		obj.typ = Typ[Invalid]
		return
	}
	obj.visited = true

	// determine type, if any
	if typ != nil {
		obj.typ = check.typ(typ, false)
	}

	if init == nil {
		if typ == nil {
			obj.typ = Typ[Invalid]
		}
		return // error reported before
	}

	// unpack projection expression, if any
	proj, multi := init.(*projExpr)
	if multi {
		init = proj.Expr
	}

	var x operand
	check.expr(&x, init, nil, -1)
	if x.mode == invalid {
		goto Error
	}

	if multi {
		if t, ok := x.typ.(*Tuple); ok && len(proj.lhs) == t.Len() {
			// function result
			x.mode = value
			for i, lhs := range proj.lhs {
				x.expr = nil // TODO(gri) should do better here
				x.typ = t.At(i).typ
				check.assign(lhs, &x)
			}
			return
		}

		if x.mode == valueok && len(proj.lhs) == 2 {
			// comma-ok expression
			x.mode = value
			check.assign(proj.lhs[0], &x)

			x.typ = Typ[UntypedBool]
			check.assign(proj.lhs[1], &x)
			return
		}

		// TODO(gri) better error message
		check.errorf(proj.lhs[0].Pos(), "assignment count mismatch")
		goto Error
	}

	check.assign(obj, &x)
	return

Error:
	// mark all involved variables so we can avoid repeated error messages
	if multi {
		for _, obj := range proj.lhs {
			if obj.typ == nil {
				obj.typ = Typ[Invalid]
				obj.visited = true
			}
		}
	} else if obj.typ == nil {
		obj.typ = Typ[Invalid]
	}
	return
}

func (check *checker) declareType(obj *TypeName, typ ast.Expr, cycleOk bool) {
	named := &Named{obj: obj}
	obj.typ = named // mark object so recursion terminates in case of cycles
	named.underlying = check.typ(typ, cycleOk).Underlying()

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
		if !scope.IsEmpty() {
			for _, obj := range scope.entries {
				m := obj.(*Func)

				// set the correct file scope for checking this method type
				fileScope := m.parent
				assert(fileScope != nil)
				oldScope := check.topScope
				check.topScope = fileScope

				sig := check.typ(m.decl.Type, cycleOk).(*Signature)
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
	fdecl := obj.decl
	// methods are typechecked when their receivers are typechecked
	// TODO(gri) there is no reason to make this a special case: receivers are simply parameters
	if fdecl.Recv == nil {
		obj.typ = Typ[Invalid] // guard against cycles
		sig := check.typ(fdecl.Type, false).(*Signature)
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
		var last []ast.Expr // last list of const initializers seen
		for iota, spec := range d.Specs {
			switch s := spec.(type) {
			case *ast.ValueSpec:
				switch d.Tok {
				case token.CONST:
					// determine which initialization expressions to use
					if len(s.Values) > 0 {
						last = s.Values
					}

					// declare all constants
					lhs := make([]*Const, len(s.Names))
					for i, name := range s.Names {
						obj := NewConst(name.Pos(), pkg, name.Name, nil, exact.MakeInt64(int64(iota)))
						check.callIdent(name, obj)
						lhs[i] = obj

						var init ast.Expr
						if i < len(last) {
							init = last[i]
						}

						check.declareConst(obj, s.Type, init)
					}

					// arity of lhs and rhs must match
					switch lhs, rhs := len(s.Names), len(last); {
					case lhs < rhs:
						x := last[lhs]
						check.errorf(x.Pos(), "too many initialization expressions")
					case lhs > rhs:
						n := s.Names[rhs]
						check.errorf(n.Pos(), "missing initialization expression for %s", n)
					}

					for _, obj := range lhs {
						check.declare(check.topScope, nil, obj)
					}

				case token.VAR:
					// declare all variables
					lhs := make([]*Var, len(s.Names))
					for i, name := range s.Names {
						obj := NewVar(name.Pos(), pkg, name.Name, nil)
						check.callIdent(name, obj)
						lhs[i] = obj
					}

					// iterate in 2 phases because declareVar requires fully initialized lhs!
					for i, obj := range lhs {
						var init ast.Expr
						switch len(s.Values) {
						case len(s.Names):
							// lhs and rhs match
							init = s.Values[i]
						case 1:
							// rhs must be a multi-valued expression
							init = &projExpr{lhs, s.Values[0]}
						default:
							if i < len(s.Values) {
								init = s.Values[i]
							}
						}

						check.declareVar(obj, s.Type, init)
					}

					// arity of lhs and rhs must match
					if lhs, rhs := len(s.Names), len(s.Values); rhs > 0 {
						switch {
						case lhs < rhs:
							x := s.Values[lhs]
							check.errorf(x.Pos(), "too many initialization expressions")
						case lhs > rhs && rhs != 1:
							n := s.Names[rhs]
							check.errorf(n.Pos(), "missing initialization expression for %s", n)
						}
					}

					for _, obj := range lhs {
						check.declare(check.topScope, nil, obj)
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				check.declare(check.topScope, s.Name, obj)
				check.declareType(obj, s.Type, false)

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
