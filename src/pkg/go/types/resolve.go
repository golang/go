// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/ast"
	"strconv"
)

func (check *checker) declareObj(scope, altScope *ast.Scope, obj *ast.Object) {
	alt := scope.Insert(obj)
	if alt == nil && altScope != nil {
		// see if there is a conflicting declaration in altScope
		alt = altScope.Lookup(obj.Name)
	}
	if alt != nil {
		prevDecl := ""
		if pos := alt.Pos(); pos.IsValid() {
			prevDecl = fmt.Sprintf("\n\tprevious declaration at %s", check.fset.Position(pos))
		}
		check.errorf(obj.Pos(), fmt.Sprintf("%s redeclared in this block%s", obj.Name, prevDecl))
	}
}

func resolve(scope *ast.Scope, ident *ast.Ident) bool {
	for ; scope != nil; scope = scope.Outer {
		if obj := scope.Lookup(ident.Name); obj != nil {
			ident.Obj = obj
			return true
		}
	}
	// handle universe scope lookups
	return false
}

// TODO(gri) eventually resolve should only return *Package.
func (check *checker) resolve(importer Importer) (*ast.Package, *Package) {
	// complete package scope
	pkgName := ""
	pkgScope := ast.NewScope(Universe)

	i := 0
	for _, file := range check.files {
		// package names must match
		switch name := file.Name.Name; {
		case pkgName == "":
			pkgName = name
		case name != pkgName:
			check.errorf(file.Package, "package %s; expected %s", name, pkgName)
			continue // ignore this file
		}

		// keep this file
		check.files[i] = file
		i++

		// collect top-level file objects in package scope
		for _, obj := range file.Scope.Objects {
			check.declareObj(pkgScope, nil, obj)
		}
	}
	check.files = check.files[0:i]

	// package global mapping of imported package ids to package objects
	imports := make(map[string]*Package)

	// complete file scopes with imports and resolve identifiers
	for _, file := range check.files {
		// build file scope by processing all imports
		importErrors := false
		fileScope := ast.NewScope(pkgScope)
		for _, spec := range file.Imports {
			if importer == nil {
				importErrors = true
				continue
			}
			path, _ := strconv.Unquote(spec.Path.Value)
			pkg, err := importer(imports, path)
			if err != nil {
				check.errorf(spec.Path.Pos(), "could not import %s (%s)", path, err)
				importErrors = true
				continue
			}
			// TODO(gri) If a local package name != "." is provided,
			// global identifier resolution could proceed even if the
			// import failed. Consider adjusting the logic here a bit.

			// local name overrides imported package name
			name := pkg.Name
			if spec.Name != nil {
				name = spec.Name.Name
			}

			// add import to file scope
			if name == "." {
				// merge imported scope with file scope
				// TODO(gri) Imported packages use Objects but the current
				//           package scope is based on ast.Scope and ast.Objects
				//           at the moment. Don't try to convert the imported
				//           objects for now. Once we get rid of ast.Object
				//           dependency, this loop can be enabled again.
				panic("cannot handle dot-import")
				/*
					for _, obj := range pkg.Scope.Elems {
						check.declareObj(fileScope, pkgScope, obj)
					}
				*/
			} else if name != "_" {
				// declare imported package object in file scope
				// (do not re-use pkg in the file scope but create
				// a new object instead; the Decl field is different
				// for different files)
				obj := ast.NewObj(ast.Pkg, name)
				obj.Decl = spec
				obj.Data = pkg
				check.declareObj(fileScope, pkgScope, obj)
			}
		}

		// resolve identifiers
		if importErrors {
			// don't use the universe scope without correct imports
			// (objects in the universe may be shadowed by imports;
			// with missing imports, identifiers might get resolved
			// incorrectly to universe objects)
			pkgScope.Outer = nil
		}
		i := 0
		for _, ident := range file.Unresolved {
			if !resolve(fileScope, ident) {
				check.errorf(ident.Pos(), "undeclared name: %s", ident.Name)
				file.Unresolved[i] = ident
				i++
			}

		}
		file.Unresolved = file.Unresolved[0:i]
		pkgScope.Outer = Universe // reset outer scope
	}

	// TODO(gri) Once we have a pkgScope of type *Scope, only return *Package.
	return &ast.Package{Name: pkgName, Scope: pkgScope}, &Package{Name: pkgName, Imports: imports}
}
