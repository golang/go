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

	deps map[Object]*declInfo // init dependencies; lazily allocated
	mark int                  // see check.dependencies
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

// TODO(gri) Split resolveFiles into smaller components.

func (check *checker) resolveFiles(files []*ast.File) {
	pkg := check.pkg

	// Phase 1: Pre-declare all package-level objects so that they can be found
	//          independent of source order. Associate methods with receiver
	//          base type names.

	var (
		objMap  = make(map[Object]*declInfo) // corresponding declaration info
		initMap = make(map[Object]*declInfo) // declaration info for variables with initializers
	)

	// declare declares obj in the package scope, records its ident -> obj mapping,
	// and updates objMap. The object must not be a function or method.
	declare := func(ident *ast.Ident, obj Object, d *declInfo) {
		assert(ident.Name == obj.Name())

		// spec: "A package-scope or file-scope identifier with name init
		// may only be declared to be a function with this (func()) signature."
		if ident.Name == "init" {
			check.errorf(ident.Pos(), "cannot declare init - must be func")
			return
		}

		check.declare(pkg.scope, ident, obj)
		objMap[obj] = d
	}

	importer := check.conf.Import
	if importer == nil {
		if DefaultImport == nil {
			panic(`no Config.Import or DefaultImport (missing import _ "code.google.com/p/go.tools/go/gcimporter"?)`)
		}
		importer = DefaultImport
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

		fileScope := NewScope(pkg.scope)
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
						path, err := validatedImportPath(s.Path.Value)
						if err != nil {
							check.errorf(s.Path.Pos(), "invalid import path (%s)", err)
							continue
						}
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
								if obj.Exported() {
									// Note: This will change each imported object's scope!
									//       May be an issue for type aliases.
									check.declare(fileScope, nil, obj)
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

								declare(name, obj, &declInfo{file: fileScope, typ: last.Type, init: init})
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
								// The lhs elements are only set up after the foor loop below,
								// but that's ok because declareVar only collects the declInfo
								// for a later phase.
								d1 = &declInfo{file: fileScope, lhs: lhs, typ: s.Type, init: s.Values[0]}
							}

							// declare all variables
							for i, name := range s.Names {
								obj := NewVar(name.Pos(), pkg, name.Name, nil)
								lhs[i] = obj

								d := d1
								var init ast.Expr
								if d == nil {
									// individual assignments
									if i < len(s.Values) {
										init = s.Values[i]
									}
									d = &declInfo{file: fileScope, typ: s.Type, init: init}
								}

								declare(name, obj, d)

								// remember obj if it has an initializer
								if d1 != nil || init != nil {
									initMap[obj] = d
								}
							}

							check.arityMatch(s, nil)

						default:
							check.invalidAST(s.Pos(), "invalid token %s", d.Tok)
						}

					case *ast.TypeSpec:
						obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Name, nil)
						declare(s.Name, obj, &declInfo{file: fileScope, typ: s.Type})

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
						check.declare(pkg.scope, d.Name, obj)
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
				info := &declInfo{file: fileScope, fdecl: d}
				objMap[obj] = info
				// remember obj if it has a body (= initializer)
				if d.Body != nil {
					initMap[obj] = info
				}

			default:
				check.invalidAST(d.Pos(), "unknown ast.Decl node %T", d)
			}
		}
	}
	seenPkgs = nil // not needed anymore

	// Phase 2: Verify that objects in package and file scopes have different names.

	for _, scope := range fileScopes {
		for _, obj := range scope.elems {
			if alt := pkg.scope.Lookup(obj.Name()); alt != nil {
				check.errorf(alt.Pos(), "%s already declared in this file through import of package %s", obj.Name(), obj.Pkg().Name())
			}
		}
	}

	// Phase 3: Typecheck all objects in objMap, but not function bodies.

	check.objMap = objMap // indicate that we are checking package-level declarations (objects may not have a type yet)
	check.initMap = initMap
	for _, obj := range objectsOf(check.objMap) {
		if obj.Type() == nil {
			check.objDecl(obj, nil, false)
		}
	}
	check.objMap = nil // not needed anymore

	// At this point we may have a non-empty check.methods map; this means that not all
	// entries were deleted at the end of typeDecl because the respective receiver base
	// types were not declared. In that case, an error was reported when declaring those
	// methods. We can now safely discard this map.
	check.methods = nil

	// Phase 4: Typecheck all functions bodies.

	for _, f := range check.funcList {
		check.decl = f.info
		check.funcBody(f.name, f.sig, f.body)
	}

	// Phase 5: Check initialization dependencies.
	// Note: must happen after checking all functions because function bodies
	// may introduce dependencies

	for _, obj := range objectsOf(initMap) {
		if obj, _ := obj.(*Var); obj != nil {
			if init := initMap[obj]; init != nil {
				// provide non-nil path for re-use of underlying array
				check.dependencies(obj, init, make([]Object, 0, 8))
			}
		}
	}
	check.initMap = nil // not needed anymore

	// Phase 6: Check for declared but not used packages and function variables.
	// Note: must happen after checking all functions because function bodies
	// may introduce package uses

	// If function bodies are not checked, packages' uses are likely missing,
	// and there are no unused function variables. Nothing left to do.
	if check.conf.IgnoreFuncBodies {
		return
	}

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
}

// TODO(gri) Instead of this support function, introduce type
// that combines an map[Object]*declInfo and []Object so that
// we can encapsulate this and don't have to depend on correct
// Pos() information.

// objectsOf returns the keys of m in source order.
func objectsOf(m map[Object]*declInfo) []Object {
	list := make([]Object, len(m))
	i := 0
	for obj := range m {
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
// The current node is represented by the pair (obj, init); and path contains all nodes
// on the path to the current node (excluding the current node).
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
func (check *checker) dependencies(obj Object, init *declInfo, path []Object) {
	if init.mark < 0 {
		return // finished
	}

	if init.mark > 0 {
		// cycle detected - find index of first variable in cycle, if any
		first := -1
		cycle := path[init.mark-1:]
		for i, obj := range cycle {
			if _, ok := obj.(*Var); ok {
				first = i
				break
			}
		}
		// only report an error if there's at least one variable
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
			check.errorf(obj.Pos(), "\t%s (cycle start)", obj.Name())

		}
		init.mark = -1 // avoid further errors
		return
	}

	// init.mark == 0

	path = append(path, obj) // len(path) > 0
	init.mark = len(path)    // init.mark > 0
	for _, obj := range objectsOf(init.deps) {
		dep := init.deps[obj]
		check.dependencies(obj, dep, path)
	}
	init.mark = -1 // init.mark < 0

	// record the init order for variables
	if this, _ := obj.(*Var); this != nil {
		initLhs := init.lhs // possibly nil (see declInfo.lhs field comment)
		if initLhs == nil {
			initLhs = []*Var{this}
		}
		check.Info.InitOrder = append(check.Info.InitOrder, &Initializer{initLhs, init.init})
	}
}
