// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements NewPackage.

package ast

import (
	"fmt"
	"go/scanner"
	"go/token"
	"os"
)


type pkgBuilder struct {
	scanner.ErrorVector
	fset *token.FileSet
}


func (p *pkgBuilder) error(pos token.Pos, msg string) {
	p.Error(p.fset.Position(pos), msg)
}


func (p *pkgBuilder) errorf(pos token.Pos, format string, args ...interface{}) {
	p.error(pos, fmt.Sprintf(format, args...))
}


func (p *pkgBuilder) declare(scope, altScope *Scope, obj *Object) {
	alt := scope.Insert(obj)
	if alt == nil && altScope != nil {
		// see if there is a conflicting declaration in altScope
		alt = altScope.Lookup(obj.Name)
	}
	if alt != nil {
		prevDecl := ""
		if pos := alt.Pos(); pos.IsValid() {
			prevDecl = fmt.Sprintf("\n\tprevious declaration at %s", p.fset.Position(pos))
		}
		p.error(obj.Pos(), fmt.Sprintf("%s redeclared in this block%s", obj.Name, prevDecl))
	}
}


func resolve(scope *Scope, ident *Ident) bool {
	for ; scope != nil; scope = scope.Outer {
		if obj := scope.Lookup(ident.Name); obj != nil {
			ident.Obj = obj
			return true
		}
	}
	return false
}


// NewPackage uses an Importer to resolve imports. Given an importPath,
// an importer returns the imported package's name, its scope of exported
// objects, and an error, if any.
//
type Importer func(path string) (name string, scope *Scope, err os.Error)


// NewPackage creates a new Package node from a set of File nodes. It resolves
// unresolved identifiers across files and updates each file's Unresolved list
// accordingly. If a non-nil importer and universe scope are provided, they are
// used to resolve identifiers not declared in any of the package files. Any
// remaining unresolved identifiers are reported as undeclared. If the files
// belong to different packages, one package name is selected and files with
// different package name are reported and then ignored.
// The result is a package node and a scanner.ErrorList if there were errors.
//
func NewPackage(fset *token.FileSet, files map[string]*File, importer Importer, universe *Scope) (*Package, os.Error) {
	var p pkgBuilder
	p.fset = fset

	// complete package scope
	pkgName := ""
	pkgScope := NewScope(universe)
	for _, file := range files {
		// package names must match
		switch name := file.Name.Name; {
		case pkgName == "":
			pkgName = name
		case name != pkgName:
			p.errorf(file.Package, "package %s; expected %s", name, pkgName)
			continue // ignore this file
		}

		// collect top-level file objects in package scope
		for _, obj := range file.Scope.Objects {
			p.declare(pkgScope, nil, obj)
		}
	}

	// imports maps import paths to package names and scopes
	// TODO(gri): Eventually we like to get to the import scope from
	//            a package object. Then we can have a map path -> Obj.
	type importedPkg struct {
		name  string
		scope *Scope
	}
	imports := make(map[string]*importedPkg)

	// complete file scopes with imports and resolve identifiers
	for _, file := range files {
		// ignore file if it belongs to a different package
		// (error has already been reported)
		if file.Name.Name != pkgName {
			continue
		}

		// build file scope by processing all imports
		importErrors := false
		fileScope := NewScope(pkgScope)
		for _, spec := range file.Imports {
			// add import to global map of imports
			path := string(spec.Path.Value)
			path = path[1 : len(path)-1] // strip ""'s
			pkg := imports[path]
			if pkg == nil {
				if importer == nil {
					importErrors = true
					continue
				}
				name, scope, err := importer(path)
				if err != nil {
					p.errorf(spec.Path.Pos(), "could not import %s (%s)", path, err)
					importErrors = true
					continue
				}
				pkg = &importedPkg{name, scope}
				imports[path] = pkg
				// TODO(gri) If a local package name != "." is provided,
				// global identifier resolution could proceed even if the
				// import failed. Consider adjusting the logic here a bit.
			}
			// local name overrides imported package name
			name := pkg.name
			if spec.Name != nil {
				name = spec.Name.Name
			}
			// add import to file scope
			if name == "." {
				// merge imported scope with file scope
				for _, obj := range pkg.scope.Objects {
					p.declare(fileScope, pkgScope, obj)
				}
			} else {
				// declare imported package object in file scope
				obj := NewObj(Pkg, name)
				obj.Decl = spec
				p.declare(fileScope, pkgScope, obj)
			}
		}

		// resolve identifiers
		if importErrors {
			// don't use the universe scope without correct imports
			// (objects in the universe may be shadowed by imports;
			// with missing imports identifiers might get resolved
			// wrongly)
			pkgScope.Outer = nil
		}
		i := 0
		for _, ident := range file.Unresolved {
			if !resolve(fileScope, ident) {
				p.errorf(ident.Pos(), "undeclared name: %s", ident.Name)
				file.Unresolved[i] = ident
				i++
			}

		}
		file.Unresolved = file.Unresolved[0:i]
		pkgScope.Outer = universe // reset universe scope
	}

	// collect all import paths and respective package scopes
	importedScopes := make(map[string]*Scope)
	for path, pkg := range imports {
		importedScopes[path] = pkg.scope
	}

	return &Package{pkgName, pkgScope, importedScopes, files}, p.GetError(scanner.Sorted)
}
