// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"errors"
	"fmt"
	"go/ast"
	"go/token"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"code.google.com/p/go.tools/go/exact"
)

// A declInfo describes a package-level const, type, var, or func declaration.
type declInfo struct {
	file  *Scope        // scope of file containing this declaration
	lhs   []*Var        // lhs of n:1 variable declarations, or nil
	typ   ast.Expr      // type, or nil
	init  ast.Expr      // init expression, or nil
	fdecl *ast.FuncDecl // func declaration, or nil

	deps map[Object]bool // type and init dependencies; lazily allocated
	mark int             // for dependency analysis
}

// hasInitializer reports whether the declared object has an initialization
// expression or function body.
func (d *declInfo) hasInitializer() bool {
	return d.init != nil || d.fdecl != nil && d.fdecl.Body != nil
}

// addDep adds obj as a dependency to d.
func (d *declInfo) addDep(obj Object) {
	m := d.deps
	if m == nil {
		m = make(map[Object]bool)
		d.deps = m
	}
	m[obj] = true
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
			// TODO(gri) avoid declared but not used error here
		} else {
			// init exprs "inherited"
			check.errorf(s.Pos(), "extra init expr at %s", init.Pos())
			// TODO(gri) avoid declared but not used error here
		}
	case l > r && (init != nil || r != 1):
		n := s.Names[r]
		check.errorf(n.Pos(), "missing init expr for %s", n)
	}
}

func validatedImportPath(path string) (string, error) {
	s, err := strconv.Unquote(path)
	if err != nil {
		return "", err
	}
	if s == "" {
		return "", fmt.Errorf("empty string")
	}
	const illegalChars = `!"#$%&'()*,:;<=>?[\]^{|}` + "`\uFFFD"
	for _, r := range s {
		if !unicode.IsGraphic(r) || unicode.IsSpace(r) || strings.ContainsRune(illegalChars, r) {
			return s, fmt.Errorf("invalid character %#U", r)
		}
	}
	return s, nil
}

// declarePkgObj declares obj in the package scope, records its ident -> obj mapping,
// and updates check.objMap. The object must not be a function or method.
func (check *checker) declarePkgObj(ident *ast.Ident, obj Object, d *declInfo) {
	assert(ident.Name == obj.Name())

	// spec: "A package-scope or file-scope identifier with name init
	// may only be declared to be a function with this (func()) signature."
	if ident.Name == "init" {
		check.errorf(ident.Pos(), "cannot declare init - must be func")
		return
	}

	check.declare(check.pkg.scope, ident, obj)
	check.objMap[obj] = d
}

// collectObjects collects all file and package objects and inserts them
// into their respective scopes. It also performs imports and associates
// methods with receiver base type names.
func (check *checker) collectObjects() {
	pkg := check.pkg

	importer := check.conf.Import
	if importer == nil {
		if DefaultImport == nil {
			panic(`no Config.Import or DefaultImport (missing import _ "code.google.com/p/go.tools/go/gcimporter"?)`)
		}
		importer = DefaultImport
	}

	// pkgImports is the set of packages already imported by any package file seen
	// so far. Used to avoid duplicate entries in pkg.imports. Allocate and populate
	// it (pkg.imports may not be empty if we are checking test files incrementally).
	var pkgImports = make(map[*Package]bool)
	for _, imp := range pkg.imports {
		pkgImports[imp] = true
	}

	for fileNo, file := range check.files {
		// The package identifier denotes the current package,
		// but there is no corresponding package object.
		check.recordDef(file.Name, nil)
		fileScope := check.fileScopes[fileNo]

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
						path, err := validatedImportPath(s.Path.Value)
						if err != nil {
							check.errorf(s.Path.Pos(), "invalid import path (%s)", err)
							continue
						}
						if path == "C" && check.conf.FakeImportC {
							// TODO(gri) shouldn't create a new one each time
							imp = NewPackage("C", "C")
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
						if !pkgImports[imp] {
							pkgImports[imp] = true
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
							check.recordDef(s.Name, obj)
						} else {
							check.recordImplicit(s, obj)
						}

						// add import to file scope
						if name == "." {
							// merge imported scope with file scope
							for _, obj := range imp.scope.elems {
								// A package scope may contain non-exported objects,
								// do not import them!
								if obj.Exported() {
									check.declare(fileScope, nil, obj)
									check.recordImplicit(s, obj)
								}
							}
							// add position to set of dot-import positions for this file
							// (this is only needed for "imported but not used" errors)
							posSet := check.dotImports[fileNo]
							if posSet == nil {
								posSet = make(map[*Package]token.Pos)
								check.dotImports[fileNo] = posSet
							}
							posSet[imp] = s.Pos()
						} else {
							// declare imported package object in file scope
							check.declare(fileScope, nil, obj)
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

								d := &declInfo{file: fileScope, typ: last.Type, init: init}
								check.declarePkgObj(name, obj, d)
							}

							check.arityMatch(s, last)

						case token.VAR:
							lhs := make([]*Var, len(s.Names))
							// If there's exactly one rhs initializer, use
							// the same declInfo d1 for all lhs variables
							// so that each lhs variable depends on the same
							// rhs initializer (n:1 var declaration).
							var d1 *declInfo
							if len(s.Values) == 1 {
								// The lhs elements are only set up after the for loop below,
								// but that's ok because declareVar only collects the declInfo
								// for a later phase.
								d1 = &declInfo{file: fileScope, lhs: lhs, typ: s.Type, init: s.Values[0]}
							}

							// declare all variables
							for i, name := range s.Names {
								obj := NewVar(name.Pos(), pkg, name.Name, nil)
								lhs[i] = obj

								d := d1
								if d == nil {
									// individual assignments
									var init ast.Expr
									if i < len(s.Values) {
										init = s.Values[i]
									}
									d = &declInfo{file: fileScope, typ: s.Type, init: init}
								}

								check.declarePkgObj(name, obj, d)
							}

							check.arityMatch(s, nil)

						default:
							check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
						}

					case *ast.TypeSpec:
						obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
						check.declarePkgObj(s.Name, obj, &declInfo{file: fileScope, typ: s.Type})

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
						check.recordDef(d.Name, obj)
						// init functions must have a body
						if d.Body == nil {
							check.softErrorf(obj.pos, "missing function body")
						}
					} else {
						check.declare(pkg.scope, d.Name, obj)
					}
				} else {
					// method
					check.recordDef(d.Name, obj)
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
							check.assocMethod(base.Name, obj)
						}
					}
				}
				info := &declInfo{file: fileScope, fdecl: d}
				check.objMap[obj] = info

			default:
				check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
			}
		}
	}

	// verify that objects in package and file scopes have different names
	for _, scope := range check.fileScopes {
		for _, obj := range scope.elems {
			if alt := pkg.scope.Lookup(obj.Name()); alt != nil {
				check.errorf(alt.Pos(), "%s already declared in this file through import of package %s", obj.Name(), obj.Pkg().Name())
			}
		}
	}
}

// packageObjects typechecks all package objects in check.objMap, but not function bodies.
func (check *checker) packageObjects(objList []Object) {
	// add new methods to already type-checked types (from a prior Checker.Files call)
	for _, obj := range objList {
		if obj, _ := obj.(*TypeName); obj != nil && obj.typ != nil {
			check.addMethodDecls(obj)
		}
	}

	// pre-allocate space for type declaration paths so that the underlying array is reused
	typePath := make([]*TypeName, 0, 8)

	for _, obj := range objList {
		check.objDecl(obj, nil, typePath)
	}

	// At this point we may have a non-empty check.methods map; this means that not all
	// entries were deleted at the end of typeDecl because the respective receiver base
	// types were not found. In that case, an error was reported when declaring those
	// methods. We can now safely discard this map.
	check.methods = nil
}

// functionBodies typechecks all function bodies.
func (check *checker) functionBodies() {
	for _, f := range check.funcs {
		check.funcBody(f.decl, f.name, f.sig, f.body)
	}
}

// initDependencies computes initialization dependencies.
func (check *checker) initDependencies(objList []Object) {
	// pre-allocate space for initialization paths so that the underlying array is reused
	initPath := make([]Object, 0, 8)

	for _, obj := range objList {
		switch obj.(type) {
		case *Const, *Var:
			if check.objMap[obj].hasInitializer() {
				check.dependencies(obj, initPath)
			}
		}
	}
}

// unusedImports checks for unused imports.
func (check *checker) unusedImports() {
	// if function bodies are not checked, packages' uses are likely missing - don't check
	if check.conf.IgnoreFuncBodies {
		return
	}

	// spec: "It is illegal (...) to directly import a package without referring to
	// any of its exported identifiers. To import a package solely for its side-effects
	// (initialization), use the blank identifier as explicit package name."
	for i, scope := range check.fileScopes {
		var usedDotImports map[*Package]bool // lazily allocated
		for _, obj := range scope.elems {
			switch obj := obj.(type) {
			case *PkgName:
				// Unused "blank imports" are automatically ignored
				// since _ identifiers are not entered into scopes.
				if !obj.used {
					check.softErrorf(obj.pos, "%q imported but not used", obj.pkg.path)
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
		for pkg, pos := range check.dotImports[i] {
			if !usedDotImports[pkg] {
				check.softErrorf(pos, "%q imported but not used", pkg.path)
			}
		}
	}
}

func orderedSetObjects(set map[Object]bool) []Object {
	list := make([]Object, len(set))
	i := 0
	for obj := range set {
		// we don't care about the map element value
		list[i] = obj
		i++
	}
	sort.Sort(inSourceOrder(list))
	return list
}

// inSourceOrder implements the sort.Sort interface.
type inSourceOrder []Object

func (a inSourceOrder) Len() int           { return len(a) }
func (a inSourceOrder) Less(i, j int) bool { return a[i].Pos() < a[j].Pos() }
func (a inSourceOrder) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// dependencies recursively traverses the initialization dependency graph in a depth-first
// manner and appends the encountered variables in postorder to the Info.InitOrder list.
// As a result, that list ends up being sorted topologically in the order of dependencies.
//
// Path contains all nodes on the path to the current node obj (excluding obj).
//
// To detect cyles, the nodes are marked as follows: Initially, all nodes are unmarked
// (declInfo.mark == 0). On the way down, a node is appended to the path, and the node
// is marked with a value > 0 ("in progress"). On the way up, a node is marked with a
// value < 0 ("finished"). A cycle is detected if a node is reached that is marked as
// "in progress".
//
// A cycle must contain at least one variable to be invalid (cycles containing only
// functions are permitted). To detect such a cycle, and in order to print it, the
// mark value indicating "in progress" is the path length up to (and including) the
// current node; i.e. the length of the path after appending the node. Naturally,
// that value is > 0 as required for "in progress" marks. In other words, each node's
// "in progress" mark value corresponds to the node's path index plus 1. Accordingly,
// when the first node of a cycle is reached, that node's mark value indicates the
// start of the cycle in the path. The tail of the path (path[mark-1:]) contains all
// nodes of the cycle.
//
func (check *checker) dependencies(obj Object, path []Object) {
	init := check.objMap[obj]
	if init.mark < 0 {
		return // finished
	}

	if init.mark > 0 {
		// cycle detected - find index of first constant or variable in cycle, if any
		first := -1
		cycle := path[init.mark-1:]
	L:
		for i, obj := range cycle {
			switch obj.(type) {
			case *Const, *Var:
				first = i
				break L
			}
		}
		// only report an error if there's at least one constant or variable
		if first >= 0 {
			obj := cycle[first]
			check.errorf(obj.Pos(), "initialization cycle for %s", obj.Name())
			// print cycle
			i := first
			for _ = range cycle {
				check.errorf(obj.Pos(), "\t%s refers to", obj.Name()) // secondary error, \t indented
				i++
				if i >= len(cycle) {
					i = 0
				}
				obj = cycle[i]
			}
			check.errorf(obj.Pos(), "\t%s", obj.Name())

		}
		init.mark = -1 // avoid further errors
		return
	}

	// init.mark == 0

	path = append(path, obj) // len(path) > 0
	init.mark = len(path)    // init.mark > 0
	for _, obj := range orderedSetObjects(init.deps) {
		check.dependencies(obj, path)
	}
	init.mark = -1 // init.mark < 0

	// record the init order for variables only
	if this, _ := obj.(*Var); this != nil {
		initLhs := init.lhs // possibly nil (see declInfo.lhs field comment)
		if initLhs == nil {
			initLhs = []*Var{this}
		}
		check.Info.InitOrder = append(check.Info.InitOrder, &Initializer{initLhs, init.init})
	}
}
