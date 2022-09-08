// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	"sort"
	"strconv"
	"strings"
	"unicode"
)

// A declInfo describes a package-level const, type, var, or func declaration.
type declInfo struct {
	file      *Scope           // scope of file containing this declaration
	lhs       []*Var           // lhs of n:1 variable declarations, or nil
	vtyp      syntax.Expr      // type, or nil (for const and var declarations only)
	init      syntax.Expr      // init/orig expression, or nil (for const and var declarations only)
	inherited bool             // if set, the init expression is inherited from a previous constant declaration
	tdecl     *syntax.TypeDecl // type declaration, or nil
	fdecl     *syntax.FuncDecl // func declaration, or nil

	// The deps field tracks initialization expression dependencies.
	deps map[Object]bool // lazily initialized
}

// hasInitializer reports whether the declared object has an initialization
// expression or function body.
func (d *declInfo) hasInitializer() bool {
	return d.init != nil || d.fdecl != nil && d.fdecl.Body != nil
}

// addDep adds obj to the set of objects d's init expression depends on.
func (d *declInfo) addDep(obj Object) {
	m := d.deps
	if m == nil {
		m = make(map[Object]bool)
		d.deps = m
	}
	m[obj] = true
}

// arity checks that the lhs and rhs of a const or var decl
// have a matching number of names and initialization values.
// If inherited is set, the initialization values are from
// another (constant) declaration.
func (check *Checker) arity(pos syntax.Pos, names []*syntax.Name, inits []syntax.Expr, constDecl, inherited bool) {
	l := len(names)
	r := len(inits)

	const code = _WrongAssignCount
	switch {
	case l < r:
		n := inits[l]
		if inherited {
			check.errorf(pos, code, "extra init expr at %s", n.Pos())
		} else {
			check.errorf(n, code, "extra init expr %s", n)
		}
	case l > r && (constDecl || r != 1): // if r == 1 it may be a multi-valued function and we can't say anything yet
		n := names[r]
		check.errorf(n, code, "missing init expr for %s", n.Value)
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
func (check *Checker) declarePkgObj(ident *syntax.Name, obj Object, d *declInfo) {
	assert(ident.Value == obj.Name())

	// spec: "A package-scope or file-scope identifier with name init
	// may only be declared to be a function with this (func()) signature."
	if ident.Value == "init" {
		check.error(ident, _InvalidInitDecl, "cannot declare init - must be func")
		return
	}

	// spec: "The main package must have package name main and declare
	// a function main that takes no arguments and returns no value."
	if ident.Value == "main" && check.pkg.name == "main" {
		check.error(ident, _InvalidMainDecl, "cannot declare main - must be func")
		return
	}

	check.declare(check.pkg.scope, ident, obj, nopos)
	check.objMap[obj] = d
	obj.setOrder(uint32(len(check.objMap)))
}

// filename returns a filename suitable for debugging output.
func (check *Checker) filename(fileNo int) string {
	file := check.files[fileNo]
	if pos := file.Pos(); pos.IsKnown() {
		// return check.fset.File(pos).Name()
		// TODO(gri) do we need the actual file name here?
		return pos.RelFilename()
	}
	return fmt.Sprintf("file[%d]", fileNo)
}

func (check *Checker) importPackage(pos syntax.Pos, path, dir string) *Package {
	// If we already have a package for the given (path, dir)
	// pair, use it instead of doing a full import.
	// Checker.impMap only caches packages that are marked Complete
	// or fake (dummy packages for failed imports). Incomplete but
	// non-fake packages do require an import to complete them.
	key := importKey{path, dir}
	imp := check.impMap[key]
	if imp != nil {
		return imp
	}

	// no package yet => import it
	if path == "C" && (check.conf.FakeImportC || check.conf.go115UsesCgo) {
		imp = NewPackage("C", "C")
		imp.fake = true // package scope is not populated
		imp.cgo = check.conf.go115UsesCgo
	} else {
		// ordinary import
		var err error
		if importer := check.conf.Importer; importer == nil {
			err = fmt.Errorf("Config.Importer not installed")
		} else if importerFrom, ok := importer.(ImporterFrom); ok {
			imp, err = importerFrom.ImportFrom(path, dir, 0)
			if imp == nil && err == nil {
				err = fmt.Errorf("Config.Importer.ImportFrom(%s, %s, 0) returned nil but no error", path, dir)
			}
		} else {
			imp, err = importer.Import(path)
			if imp == nil && err == nil {
				err = fmt.Errorf("Config.Importer.Import(%s) returned nil but no error", path)
			}
		}
		// make sure we have a valid package name
		// (errors here can only happen through manipulation of packages after creation)
		if err == nil && imp != nil && (imp.name == "_" || imp.name == "") {
			err = fmt.Errorf("invalid package name: %q", imp.name)
			imp = nil // create fake package below
		}
		if err != nil {
			check.errorf(pos, _BrokenImport, "could not import %s (%s)", path, err)
			if imp == nil {
				// create a new fake package
				// come up with a sensible package name (heuristic)
				name := path
				if i := len(name); i > 0 && name[i-1] == '/' {
					name = name[:i-1]
				}
				if i := strings.LastIndex(name, "/"); i >= 0 {
					name = name[i+1:]
				}
				imp = NewPackage(path, name)
			}
			// continue to use the package as best as we can
			imp.fake = true // avoid follow-up lookup failures
		}
	}

	// package should be complete or marked fake, but be cautious
	if imp.complete || imp.fake {
		check.impMap[key] = imp
		// Once we've formatted an error message, keep the pkgPathMap
		// up-to-date on subsequent imports. It is used for package
		// qualification in error messages.
		if check.pkgPathMap != nil {
			check.markImports(imp)
		}
		return imp
	}

	// something went wrong (importer may have returned incomplete package without error)
	return nil
}

// collectObjects collects all file and package objects and inserts them
// into their respective scopes. It also performs imports and associates
// methods with receiver base type names.
func (check *Checker) collectObjects() {
	pkg := check.pkg

	// pkgImports is the set of packages already imported by any package file seen
	// so far. Used to avoid duplicate entries in pkg.imports. Allocate and populate
	// it (pkg.imports may not be empty if we are checking test files incrementally).
	// Note that pkgImports is keyed by package (and thus package path), not by an
	// importKey value. Two different importKey values may map to the same package
	// which is why we cannot use the check.impMap here.
	var pkgImports = make(map[*Package]bool)
	for _, imp := range pkg.imports {
		pkgImports[imp] = true
	}

	type methodInfo struct {
		obj  *Func        // method
		ptr  bool         // true if pointer receiver
		recv *syntax.Name // receiver type name
	}
	var methods []methodInfo // collected methods with valid receivers and non-blank _ names
	var fileScopes []*Scope
	for fileNo, file := range check.files {
		// The package identifier denotes the current package,
		// but there is no corresponding package object.
		check.recordDef(file.PkgName, nil)

		fileScope := NewScope(check.pkg.scope, syntax.StartPos(file), syntax.EndPos(file), check.filename(fileNo))
		fileScopes = append(fileScopes, fileScope)
		check.recordScope(file, fileScope)

		// determine file directory, necessary to resolve imports
		// FileName may be "" (typically for tests) in which case
		// we get "." as the directory which is what we would want.
		fileDir := dir(file.PkgName.Pos().RelFilename()) // TODO(gri) should this be filename?

		first := -1                // index of first ConstDecl in the current group, or -1
		var last *syntax.ConstDecl // last ConstDecl with init expressions, or nil
		for index, decl := range file.DeclList {
			if _, ok := decl.(*syntax.ConstDecl); !ok {
				first = -1 // we're not in a constant declaration
			}

			switch s := decl.(type) {
			case *syntax.ImportDecl:
				// import package
				if s.Path == nil || s.Path.Bad {
					continue // error reported during parsing
				}
				path, err := validatedImportPath(s.Path.Value)
				if err != nil {
					check.errorf(s.Path, _BadImportPath, "invalid import path (%s)", err)
					continue
				}

				imp := check.importPackage(s.Path.Pos(), path, fileDir)
				if imp == nil {
					continue
				}

				// local name overrides imported package name
				name := imp.name
				if s.LocalPkgName != nil {
					name = s.LocalPkgName.Value
					if path == "C" {
						// match 1.17 cmd/compile (not prescribed by spec)
						check.error(s.LocalPkgName, _ImportCRenamed, `cannot rename import "C"`)
						continue
					}
				}

				if name == "init" {
					check.error(s, _InvalidInitDecl, "cannot import package as init - init must be a func")
					continue
				}

				// add package to list of explicit imports
				// (this functionality is provided as a convenience
				// for clients; it is not needed for type-checking)
				if !pkgImports[imp] {
					pkgImports[imp] = true
					pkg.imports = append(pkg.imports, imp)
				}

				pkgName := NewPkgName(s.Pos(), pkg, name, imp)
				if s.LocalPkgName != nil {
					// in a dot-import, the dot represents the package
					check.recordDef(s.LocalPkgName, pkgName)
				} else {
					check.recordImplicit(s, pkgName)
				}

				if imp.fake {
					// match 1.17 cmd/compile (not prescribed by spec)
					pkgName.used = true
				}

				// add import to file scope
				check.imports = append(check.imports, pkgName)
				if name == "." {
					// dot-import
					if check.dotImportMap == nil {
						check.dotImportMap = make(map[dotImportKey]*PkgName)
					}
					// merge imported scope with file scope
					for name, obj := range imp.scope.elems {
						// Note: Avoid eager resolve(name, obj) here, so we only
						// resolve dot-imported objects as needed.

						// A package scope may contain non-exported objects,
						// do not import them!
						if isExported(name) {
							// declare dot-imported object
							// (Do not use check.declare because it modifies the object
							// via Object.setScopePos, which leads to a race condition;
							// the object may be imported into more than one file scope
							// concurrently. See issue #32154.)
							if alt := fileScope.Lookup(name); alt != nil {
								var err error_
								err.errorf(s.LocalPkgName, "%s redeclared in this block", alt.Name())
								err.recordAltDecl(alt)
								check.report(&err)
							} else {
								fileScope.insert(name, obj)
								check.dotImportMap[dotImportKey{fileScope, name}] = pkgName
							}
						}
					}
				} else {
					// declare imported package object in file scope
					// (no need to provide s.LocalPkgName since we called check.recordDef earlier)
					check.declare(fileScope, nil, pkgName, nopos)
				}

			case *syntax.ConstDecl:
				// iota is the index of the current constDecl within the group
				if first < 0 || s.Group == nil || file.DeclList[index-1].(*syntax.ConstDecl).Group != s.Group {
					first = index
					last = nil
				}
				iota := constant.MakeInt64(int64(index - first))

				// determine which initialization expressions to use
				inherited := true
				switch {
				case s.Type != nil || s.Values != nil:
					last = s
					inherited = false
				case last == nil:
					last = new(syntax.ConstDecl) // make sure last exists
					inherited = false
				}

				// declare all constants
				values := unpackExpr(last.Values)
				for i, name := range s.NameList {
					obj := NewConst(name.Pos(), pkg, name.Value, nil, iota)

					var init syntax.Expr
					if i < len(values) {
						init = values[i]
					}

					d := &declInfo{file: fileScope, vtyp: last.Type, init: init, inherited: inherited}
					check.declarePkgObj(name, obj, d)
				}

				// Constants must always have init values.
				check.arity(s.Pos(), s.NameList, values, true, inherited)

			case *syntax.VarDecl:
				lhs := make([]*Var, len(s.NameList))
				// If there's exactly one rhs initializer, use
				// the same declInfo d1 for all lhs variables
				// so that each lhs variable depends on the same
				// rhs initializer (n:1 var declaration).
				var d1 *declInfo
				if _, ok := s.Values.(*syntax.ListExpr); !ok {
					// The lhs elements are only set up after the for loop below,
					// but that's ok because declarePkgObj only collects the declInfo
					// for a later phase.
					d1 = &declInfo{file: fileScope, lhs: lhs, vtyp: s.Type, init: s.Values}
				}

				// declare all variables
				values := unpackExpr(s.Values)
				for i, name := range s.NameList {
					obj := NewVar(name.Pos(), pkg, name.Value, nil)
					lhs[i] = obj

					d := d1
					if d == nil {
						// individual assignments
						var init syntax.Expr
						if i < len(values) {
							init = values[i]
						}
						d = &declInfo{file: fileScope, vtyp: s.Type, init: init}
					}

					check.declarePkgObj(name, obj, d)
				}

				// If we have no type, we must have values.
				if s.Type == nil || values != nil {
					check.arity(s.Pos(), s.NameList, values, false, false)
				}

			case *syntax.TypeDecl:
				if len(s.TParamList) != 0 && !check.allowVersion(pkg, 1, 18) {
					check.versionErrorf(s.TParamList[0], "go1.18", "type parameter")
				}
				obj := NewTypeName(s.Name.Pos(), pkg, s.Name.Value, nil)
				check.declarePkgObj(s.Name, obj, &declInfo{file: fileScope, tdecl: s})

			case *syntax.FuncDecl:
				name := s.Name.Value
				obj := NewFunc(s.Name.Pos(), pkg, name, nil)
				hasTParamError := false // avoid duplicate type parameter errors
				if s.Recv == nil {
					// regular function
					if name == "init" || name == "main" && pkg.name == "main" {
						code := _InvalidInitDecl
						if name == "main" {
							code = _InvalidMainDecl
						}
						if len(s.TParamList) != 0 {
							check.softErrorf(s.TParamList[0], code, "func %s must have no type parameters", name)
							hasTParamError = true
						}
						if t := s.Type; len(t.ParamList) != 0 || len(t.ResultList) != 0 {
							check.softErrorf(s.Name, code, "func %s must have no arguments and no return values", name)
						}
					}
					// don't declare init functions in the package scope - they are invisible
					if name == "init" {
						obj.parent = pkg.scope
						check.recordDef(s.Name, obj)
						// init functions must have a body
						if s.Body == nil {
							// TODO(gri) make this error message consistent with the others above
							check.softErrorf(obj.pos, _MissingInitBody, "missing function body")
						}
					} else {
						check.declare(pkg.scope, s.Name, obj, nopos)
					}
				} else {
					// method
					// d.Recv != nil
					ptr, recv, _ := check.unpackRecv(s.Recv.Type, false)
					// Methods with invalid receiver cannot be associated to a type, and
					// methods with blank _ names are never found; no need to collect any
					// of them. They will still be type-checked with all the other functions.
					if recv != nil && name != "_" {
						methods = append(methods, methodInfo{obj, ptr, recv})
					}
					check.recordDef(s.Name, obj)
				}
				if len(s.TParamList) != 0 && !check.allowVersion(pkg, 1, 18) && !hasTParamError {
					check.versionErrorf(s.TParamList[0], "go1.18", "type parameter")
				}
				info := &declInfo{file: fileScope, fdecl: s}
				// Methods are not package-level objects but we still track them in the
				// object map so that we can handle them like regular functions (if the
				// receiver is invalid); also we need their fdecl info when associating
				// them with their receiver base type, below.
				check.objMap[obj] = info
				obj.setOrder(uint32(len(check.objMap)))

			default:
				check.errorf(s, 0, invalidAST+"unknown syntax.Decl node %T", s)
			}
		}
	}

	// verify that objects in package and file scopes have different names
	for _, scope := range fileScopes {
		for name, obj := range scope.elems {
			if alt := pkg.scope.Lookup(name); alt != nil {
				obj = resolve(name, obj)
				var err error_
				err.code = _DuplicateDecl
				if pkg, ok := obj.(*PkgName); ok {
					err.errorf(alt, "%s already declared through import of %s", alt.Name(), pkg.Imported())
					err.recordAltDecl(pkg)
				} else {
					err.errorf(alt, "%s already declared through dot-import of %s", alt.Name(), obj.Pkg())
					// TODO(gri) dot-imported objects don't have a position; recordAltDecl won't print anything
					err.recordAltDecl(obj)
				}
				check.report(&err)
			}
		}
	}

	// Now that we have all package scope objects and all methods,
	// associate methods with receiver base type name where possible.
	// Ignore methods that have an invalid receiver. They will be
	// type-checked later, with regular functions.
	if methods != nil {
		check.methods = make(map[*TypeName][]*Func)
		for i := range methods {
			m := &methods[i]
			// Determine the receiver base type and associate m with it.
			ptr, base := check.resolveBaseTypeName(m.ptr, m.recv)
			if base != nil {
				m.obj.hasPtrRecv_ = ptr
				check.methods[base] = append(check.methods[base], m.obj)
			}
		}
	}
}

// unpackRecv unpacks a receiver type and returns its components: ptr indicates whether
// rtyp is a pointer receiver, rname is the receiver type name, and tparams are its
// type parameters, if any. The type parameters are only unpacked if unpackParams is
// set. If rname is nil, the receiver is unusable (i.e., the source has a bug which we
// cannot easily work around).
func (check *Checker) unpackRecv(rtyp syntax.Expr, unpackParams bool) (ptr bool, rname *syntax.Name, tparams []*syntax.Name) {
L: // unpack receiver type
	// This accepts invalid receivers such as ***T and does not
	// work for other invalid receivers, but we don't care. The
	// validity of receiver expressions is checked elsewhere.
	for {
		switch t := rtyp.(type) {
		case *syntax.ParenExpr:
			rtyp = t.X
		// case *ast.StarExpr:
		//      ptr = true
		// 	rtyp = t.X
		case *syntax.Operation:
			if t.Op != syntax.Mul || t.Y != nil {
				break
			}
			ptr = true
			rtyp = t.X
		default:
			break L
		}
	}

	// unpack type parameters, if any
	if ptyp, _ := rtyp.(*syntax.IndexExpr); ptyp != nil {
		rtyp = ptyp.X
		if unpackParams {
			for _, arg := range unpackExpr(ptyp.Index) {
				var par *syntax.Name
				switch arg := arg.(type) {
				case *syntax.Name:
					par = arg
				case *syntax.BadExpr:
					// ignore - error already reported by parser
				case nil:
					check.error(ptyp, 0, invalidAST+"parameterized receiver contains nil parameters")
				default:
					check.errorf(arg, _BadDecl, "receiver type parameter %s must be an identifier", arg)
				}
				if par == nil {
					par = syntax.NewName(arg.Pos(), "_")
				}
				tparams = append(tparams, par)
			}

		}
	}

	// unpack receiver name
	if name, _ := rtyp.(*syntax.Name); name != nil {
		rname = name
	}

	return
}

// resolveBaseTypeName returns the non-alias base type name for typ, and whether
// there was a pointer indirection to get to it. The base type name must be declared
// in package scope, and there can be at most one pointer indirection. If no such type
// name exists, the returned base is nil.
func (check *Checker) resolveBaseTypeName(seenPtr bool, typ syntax.Expr) (ptr bool, base *TypeName) {
	// Algorithm: Starting from a type expression, which may be a name,
	// we follow that type through alias declarations until we reach a
	// non-alias type name. If we encounter anything but pointer types or
	// parentheses we're done. If we encounter more than one pointer type
	// we're done.
	ptr = seenPtr
	var seen map[*TypeName]bool
	for {
		typ = unparen(typ)

		// check if we have a pointer type
		// if pexpr, _ := typ.(*ast.StarExpr); pexpr != nil {
		if pexpr, _ := typ.(*syntax.Operation); pexpr != nil && pexpr.Op == syntax.Mul && pexpr.Y == nil {
			// if we've already seen a pointer, we're done
			if ptr {
				return false, nil
			}
			ptr = true
			typ = unparen(pexpr.X) // continue with pointer base type
		}

		// typ must be a name
		name, _ := typ.(*syntax.Name)
		if name == nil {
			return false, nil
		}

		// name must denote an object found in the current package scope
		// (note that dot-imported objects are not in the package scope!)
		obj := check.pkg.scope.Lookup(name.Value)
		if obj == nil {
			return false, nil
		}

		// the object must be a type name...
		tname, _ := obj.(*TypeName)
		if tname == nil {
			return false, nil
		}

		// ... which we have not seen before
		if seen[tname] {
			return false, nil
		}

		// we're done if tdecl defined tname as a new type
		// (rather than an alias)
		tdecl := check.objMap[tname].tdecl // must exist for objects in package scope
		if !tdecl.Alias {
			return ptr, tname
		}

		// otherwise, continue resolving
		typ = tdecl.Type
		if seen == nil {
			seen = make(map[*TypeName]bool)
		}
		seen[tname] = true
	}
}

// packageObjects typechecks all package objects, but not function bodies.
func (check *Checker) packageObjects() {
	// process package objects in source order for reproducible results
	objList := make([]Object, len(check.objMap))
	i := 0
	for obj := range check.objMap {
		objList[i] = obj
		i++
	}
	sort.Sort(inSourceOrder(objList))

	// add new methods to already type-checked types (from a prior Checker.Files call)
	for _, obj := range objList {
		if obj, _ := obj.(*TypeName); obj != nil && obj.typ != nil {
			check.collectMethods(obj)
		}
	}

	// We process non-alias type declarations first, followed by alias declarations,
	// and then everything else. This appears to avoid most situations where the type
	// of an alias is needed before it is available.
	// There may still be cases where this is not good enough (see also issue #25838).
	// In those cases Checker.ident will report an error ("invalid use of type alias").
	var aliasList []*TypeName
	var othersList []Object // everything that's not a type
	// phase 1: non-alias type declarations
	for _, obj := range objList {
		if tname, _ := obj.(*TypeName); tname != nil {
			if check.objMap[tname].tdecl.Alias {
				aliasList = append(aliasList, tname)
			} else {
				check.objDecl(obj, nil)
			}
		} else {
			othersList = append(othersList, obj)
		}
	}
	// phase 2: alias type declarations
	for _, obj := range aliasList {
		check.objDecl(obj, nil)
	}
	// phase 3: all other declarations
	for _, obj := range othersList {
		check.objDecl(obj, nil)
	}

	// At this point we may have a non-empty check.methods map; this means that not all
	// entries were deleted at the end of typeDecl because the respective receiver base
	// types were not found. In that case, an error was reported when declaring those
	// methods. We can now safely discard this map.
	check.methods = nil
}

// inSourceOrder implements the sort.Sort interface.
type inSourceOrder []Object

func (a inSourceOrder) Len() int           { return len(a) }
func (a inSourceOrder) Less(i, j int) bool { return a[i].order() < a[j].order() }
func (a inSourceOrder) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }

// unusedImports checks for unused imports.
func (check *Checker) unusedImports() {
	// If function bodies are not checked, packages' uses are likely missing - don't check.
	if check.conf.IgnoreFuncBodies {
		return
	}

	// spec: "It is illegal (...) to directly import a package without referring to
	// any of its exported identifiers. To import a package solely for its side-effects
	// (initialization), use the blank identifier as explicit package name."

	for _, obj := range check.imports {
		if !obj.used && obj.name != "_" {
			check.errorUnusedPkg(obj)
		}
	}
}

func (check *Checker) errorUnusedPkg(obj *PkgName) {
	// If the package was imported with a name other than the final
	// import path element, show it explicitly in the error message.
	// Note that this handles both renamed imports and imports of
	// packages containing unconventional package declarations.
	// Note that this uses / always, even on Windows, because Go import
	// paths always use forward slashes.
	path := obj.imported.path
	elem := path
	if i := strings.LastIndex(elem, "/"); i >= 0 {
		elem = elem[i+1:]
	}
	if obj.name == "" || obj.name == "." || obj.name == elem {
		if check.conf.CompilerErrorMessages {
			check.softErrorf(obj, _UnusedImport, "imported and not used: %q", path)
		} else {
			check.softErrorf(obj, _UnusedImport, "%q imported but not used", path)
		}
	} else {
		if check.conf.CompilerErrorMessages {
			check.softErrorf(obj, _UnusedImport, "imported and not used: %q as %s", path, obj.name)
		} else {
			check.softErrorf(obj, _UnusedImport, "%q imported but not used as %s", path, obj.name)
		}
	}
}

// dir makes a good-faith attempt to return the directory
// portion of path. If path is empty, the result is ".".
// (Per the go/build package dependency tests, we cannot import
// path/filepath and simply use filepath.Dir.)
func dir(path string) string {
	if i := strings.LastIndexAny(path, `/\`); i > 0 {
		return path[:i]
	}
	// i <= 0
	return "."
}
