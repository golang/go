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

func (check *checker) reportAltDecl(obj Object) {
	if pos := obj.Pos(); pos.IsValid() {
		// We use "other" rather than "previous" here because
		// the first declaration seen may not be textually
		// earlier in the source.
		check.errorf(pos, "other declaration of %s", obj.Name())
	}
}

func (check *checker) declareObj(scope *Scope, id *ast.Ident, obj Object) {
	if alt := scope.Insert(obj); alt != nil {
		check.errorf(obj.Pos(), "%s redeclared in this block", obj.Name())
		check.reportAltDecl(alt)
		return
	}
	if id != nil {
		check.recordObject(id, obj)
	}
}

func (check *checker) declareFld(oset *objset, id *ast.Ident, obj Object) {
	if alt := oset.insert(obj); alt != nil {
		check.errorf(obj.Pos(), "%s redeclared", obj.Name())
		check.reportAltDecl(alt)
		return
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

type funcInfo struct {
	obj  *Func // for debugging/tracing only
	sig  *Signature
	body *ast.BlockStmt
}

// later appends a function with non-empty body to check.funcList.
func (check *checker) later(f *Func, sig *Signature, body *ast.BlockStmt) {
	// functions implemented elsewhere (say in assembly) have no body
	if !check.conf.IgnoreFuncBodies && body != nil {
		check.funcList = append(check.funcList, funcInfo{f, sig, body})
	}
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
	//          independent of source order. Associate methods with receiver
	//          base type names.

	var (
		fileScope *Scope                      // current file scope
		objList   []Object                    // list of package-level objects to type-check
		objMap    = make(map[Object]declInfo) // declaration info for each package-level object
	)

	// declare declares obj in the package scope, records its ident -> obj mapping,
	// and updates objList and objMap. The object must not be a function or method.
	declare := func(ident *ast.Ident, obj Object, typ, init ast.Expr) {
		assert(ident.Name == obj.Name())

		// spec: "A package-scope or file-scope identifier with name init
		// may only be declared to be a function with this (func()) signature."
		if ident.Name == "init" {
			check.errorf(ident.Pos(), "cannot declare init - must be func")
			return
		}

		check.declareObj(pkg.scope, ident, obj)
		objList = append(objList, obj)
		objMap[obj] = declInfo{fileScope, typ, init, nil}
	}

	importer := check.conf.Import
	if importer == nil {
		importer = GcImport
	}

	var (
		seenPkgs   = make(map[*Package]bool) // imported packages that have been seen already
		fileScopes []*Scope                  // file scope for each file
		dotImports []map[*Package]token.Pos  // positions of dot-imports for each file
	)

	for _, file := range files {
		// The package identifier denotes the current package,
		// but there is no corresponding package object.
		check.recordObject(file.Name, nil)

		fileScope = NewScope(pkg.scope)
		check.recordScope(file, fileScope)
		fileScopes = append(fileScopes, fileScope)
		dotImports = append(dotImports, nil) // element (map) is lazily allocated

		for _, decl := range file.Decls {
			switch d := decl.(type) {
			case *ast.BadDecl:
				// ignore

			case *ast.GenDecl:
				var last *ast.ValueSpec // last ValueSpec with type or init exprs seen
				for iota, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.ImportSpec:
						// import package
						var imp *Package
						path, _ := strconv.Unquote(s.Path.Value)
						if path == "C" && check.conf.FakeImportC {
							// TODO(gri) shouldn't create a new one each time
							imp = NewPackage("C", "C", NewScope(nil))
							imp.fake = true
						} else {
							var err error
							imp, err = importer(check.conf.Packages, path)
							if imp == nil && err == nil {
								err = errors.New("Config.Import returned nil but no error")
							}
							if err != nil {
								check.errorf(s.Path.Pos(), "could not import %s (%s)", path, err)
								continue
							}
						}

						// add package to list of explicit imports
						// (this functionality is provided as a convenience
						// for clients; it is not needed for type-checking)
						if !seenPkgs[imp] {
							seenPkgs[imp] = true
							if imp != Unsafe {
								pkg.imports = append(pkg.imports, imp)
							}
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

						obj := NewPkgName(s.Pos(), imp, name)
						if s.Name != nil {
							// in a dot-import, the dot represents the package
							check.recordObject(s.Name, obj)
						} else {
							check.recordImplicit(s, obj)
						}

						// add import to file scope
						if name == "." {
							// merge imported scope with file scope
							for _, obj := range imp.scope.elems {
								// A package scope may contain non-exported objects,
								// do not import them!
								if obj.IsExported() {
									// Note: This will change each imported object's scope!
									//       May be an issue for type aliases.
									check.declareObj(fileScope, nil, obj)
									check.recordImplicit(s, obj)
								}
							}
							// add position to set of dot-import positions for this file
							// (this is only needed for "imported but not used" errors)
							posSet := dotImports[len(dotImports)-1]
							if posSet == nil {
								posSet = make(map[*Package]token.Pos)
								dotImports[len(dotImports)-1] = posSet
							}
							posSet[imp] = s.Pos()
						} else {
							// declare imported package object in file scope
							check.declareObj(fileScope, nil, obj)
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
				name := d.Name.Name
				obj := NewFunc(d.Name.Pos(), pkg, name, nil)
				if d.Recv == nil {
					// regular function
					if name == "init" {
						// don't declare init functions in the package scope - they are invisible
						obj.parent = pkg.scope
						check.recordObject(d.Name, obj)
						// init functions must have a body
						if d.Body == nil {
							check.errorf(obj.pos, "missing function body")
							// ok to continue
						}
					} else {
						check.declareObj(pkg.scope, d.Name, obj)
					}
				} else {
					// Associate method with receiver base type name, if possible.
					// Ignore methods that have an invalid receiver, or a blank _
					// receiver name. They will be type-checked later, with regular
					// functions.
					if list := d.Recv.List; len(list) > 0 {
						typ := list[0].Type
						if ptr, _ := typ.(*ast.StarExpr); ptr != nil {
							typ = ptr.X
						}
						if base, _ := typ.(*ast.Ident); base != nil && base.Name != "_" {
							check.methods[base.Name] = append(check.methods[base.Name], obj)
						}
					}
				}
				objList = append(objList, obj)
				objMap[obj] = declInfo{fileScope, nil, nil, d}

			default:
				check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
			}
		}
	}

	// Phase 2: Verify that objects in package and file scopes have different names.

	for _, scope := range fileScopes {
		for _, obj := range scope.elems {
			if alt := pkg.scope.Lookup(obj.Name()); alt != nil {
				check.errorf(alt.Pos(), "%s already declared in this file through import of package %s", obj.Name(), obj.Pkg().Name())
			}
		}
	}

	// Phase 3: Typecheck all objects in objList, but not function bodies.

	check.objMap = objMap // indicate that we are checking global declarations (objects may not have a type yet)
	for _, obj := range objList {
		if obj.Type() == nil {
			check.objDecl(obj, nil, false)
		}
	}

	// done with package-level declarations
	check.objMap = nil
	objList = nil

	// At this point we may have a non-empty check.methods map; this means that not all
	// entries were deleted at the end of typeDecl because the respective receiver base
	// types were not declared. In that case, an error was reported when declaring those
	// methods. We can now safely discard this map.
	check.methods = nil

	// Phase 4: Typecheck all functions bodies.

	// Note: funcList may grow while iterating through it - cannot use range clause.
	for i := 0; i < len(check.funcList); i++ {
		f := check.funcList[i]
		if trace {
			s := "<function literal>"
			if f.obj != nil {
				s = f.obj.name
			}
			fmt.Println("---", s)
		}

		check.topScope = f.sig.scope // open function scope
		check.funcSig = f.sig
		check.labels = nil // lazily allocated
		check.stmtList(f.body.List, false)

		if f.sig.results.Len() > 0 && !check.isTerminating(f.body, "") {
			check.errorf(f.body.Rbrace, "missing return")
		}

		// spec: "It is illegal to define a label that is never used."
		if check.labels != nil {
			for _, obj := range check.labels.elems {
				if l := obj.(*Label); !l.used {
					check.errorf(l.pos, "%s defined but not used", l.name)
				}
			}
		}
	}

	// Phase 5: Check for declared but not used packages and variables.
	// Note: must happen after checking all functions because closures may affect outer scopes

	// spec: "It is illegal (...) to directly import a package without referring to
	// any of its exported identifiers. To import a package solely for its side-effects
	// (initialization), use the blank identifier as explicit package name."

	for i, scope := range fileScopes {
		var usedDotImports map[*Package]bool // lazily allocated
		for _, obj := range scope.elems {
			switch obj := obj.(type) {
			case *PkgName:
				// Unused "blank imports" are automatically ignored
				// since _ identifiers are not entered into scopes.
				if !obj.used {
					check.errorf(obj.pos, "%q imported but not used", obj.pkg.path)
				}
			default:
				// All other objects in the file scope must be dot-
				// imported. If an object was used, mark its package
				// as used.
				if obj.isUsed() {
					if usedDotImports == nil {
						usedDotImports = make(map[*Package]bool)
					}
					usedDotImports[obj.Pkg()] = true
				}
			}
		}
		// Iterate through all dot-imports for this file and
		// check if the corresponding package was used.
		for pkg, pos := range dotImports[i] {
			if !usedDotImports[pkg] {
				check.errorf(pos, "%q imported but not used", pkg.path)
			}
		}
	}

	// Each set of implicitly declared lhs variables in a type switch acts collectively
	// as a single lhs variable. If any one was 'used', all of them are 'used'. Handle
	// them before the general analysis.
	for _, vars := range check.lhsVarsList {
		// len(vars) > 0
		var used bool
		for _, v := range vars {
			if v.used {
				used = true
			}
			v.used = true // avoid later error
		}
		if !used {
			v := vars[0]
			check.errorf(v.pos, "%s declared but not used", v.name)
		}
	}

	// spec: "Implementation restriction: A compiler may make it illegal to
	// declare a variable inside a function body if the variable is never used."
	for _, f := range check.funcList {
		check.usage(f.sig.scope)
	}
}

func (check *checker) usage(scope *Scope) {
	for _, obj := range scope.elems {
		if v, _ := obj.(*Var); v != nil && !v.used {
			check.errorf(v.pos, "%s declared but not used", v.name)
		}
	}
	for _, scope := range scope.children {
		check.usage(scope)
	}
}

// objDecl type-checks the declaration of obj in its respective file scope.
// See typeDecl for the details on def and cycleOk.
func (check *checker) objDecl(obj Object, def *Named, cycleOk bool) {
	d := check.objMap[obj]

	// adjust file scope for current object
	oldScope := check.topScope
	check.topScope = d.file // for lookup

	// save current iota
	oldIota := check.iota
	check.iota = nil

	switch obj := obj.(type) {
	case *Const:
		check.constDecl(obj, d.typ, d.init)
	case *Var:
		check.varDecl(obj, d.typ, d.init)
	case *TypeName:
		check.typeDecl(obj, d.typ, def, cycleOk)
	case *Func:
		check.funcDecl(obj, d.fdecl)
	default:
		unreachable()
	}

	check.iota = oldIota
	check.topScope = oldScope
}

func (check *checker) constDecl(obj *Const, typ, init ast.Expr) {
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

func (check *checker) varDecl(obj *Var, typ, init ast.Expr) {
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
		check.initVars(m.lhs, m.rhs, token.NoPos)
		return
	}

	var x operand
	check.expr(&x, init)
	check.initVar(obj, &x)
}

func (check *checker) typeDecl(obj *TypeName, typ ast.Expr, def *Named, cycleOk bool) {
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
	// => use an objset to determine redeclarations
	var mset objset

	// spec: "If the base type is a struct type, the non-blank method
	// and field names must be distinct."
	// => pre-populate the objset to find conflicts
	// TODO(gri) consider keeping the objset with the struct instead
	if t, _ := named.underlying.(*Struct); t != nil {
		for _, fld := range t.fields {
			assert(mset.insert(fld) == nil)
		}
	}

	// check each method
	for _, m := range methods {
		if alt := mset.insert(m); alt != nil {
			switch alt.(type) {
			case *Var:
				check.errorf(m.pos, "field and method with the same name %s", m.name)
			case *Func:
				check.errorf(m.pos, "method %s already declared for %s", m.name, named)
			default:
				unreachable()
			}
			check.reportAltDecl(alt)
			continue
		}
		check.recordObject(check.objMap[m].fdecl.Name, m)
		check.objDecl(m, nil, true)
		// Methods with blank _ names cannot be found.
		// Don't add them to the method list.
		if m.name != "_" {
			named.methods = append(named.methods, m)
		}
	}

	delete(check.methods, obj.name) // we don't need them anymore
}

func (check *checker) funcDecl(obj *Func, fdecl *ast.FuncDecl) {
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

						check.constDecl(obj, last.Type, init)
					}

					check.arityMatch(s, last)

					for i, name := range s.Names {
						check.declareObj(check.topScope, name, lhs[i])
					}

				case token.VAR:
					// For varDecl called with a multiExpr we need the fully
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

						check.varDecl(obj, s.Type, init)
					}

					check.arityMatch(s, nil)

					for i, name := range s.Names {
						check.declareObj(check.topScope, name, lhs[i])
					}

				default:
					check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
				}

			case *ast.TypeSpec:
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
				check.declareObj(check.topScope, s.Name, obj)
				check.typeDecl(obj, s.Type, nil, false)

			default:
				check.invalidAST(s.Pos(), "const, type, or var declaration expected")
			}
		}

	default:
		check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
	}
}
