// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements NewPackage.

package ast

import (
	"fmt"
	"go/scanner"
	"go/token"
	"strconv"
)

type pkgBuilder struct {
	fset   *token.FileSet
	errors scanner.ErrorList
}

func (p *pkgBuilder) error(pos token.Pos, msg string) {
	p.errors.Add(p.fset.Position(pos), msg)
}

func (p *pkgBuilder) errorf(pos token.Pos, format string, args ...any) {
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

// An Importer resolves import paths to package Objects.
// The imports map records the packages already imported,
// indexed by package id (canonical import path).
// An Importer must determine the canonical import path and
// check the map to see if it is already present in the imports map.
// If so, the Importer can return the map entry. Otherwise, the
// Importer should load the package data for the given path into
// a new *[Object] (pkg), record pkg in the imports map, and then
// return pkg.
//
// Deprecated: use the type checker [go/types] instead; see [Object].
type Importer func(imports map[string]*Object, path string) (pkg *Object, err error)

// NewPackage creates a new [Package] node from a set of [File] nodes. It resolves
// unresolved identifiers across files and updates each file's Unresolved list
// accordingly. If a non-nil importer and universe scope are provided, they are
// used to resolve identifiers not declared in any of the package files. Any
// remaining unresolved identifiers are reported as undeclared. If the files
// belong to different packages, one package name is selected and files with
// different package names are reported and then ignored.
// The result is a package node and a [scanner.ErrorList] if there were errors.
//
// Deprecated: use the type checker [go/types] instead; see [Object].
func NewPackage(fset *token.FileSet, files map[string]*File, importer Importer, universe *Scope) (*Package, error) {
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

	// package global mapping of imported package ids to package objects
	imports := make(map[string]*Object)

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
			if importer == nil {
				importErrors = true
				continue
			}
			path, _ := strconv.Unquote(spec.Path.Value)
			pkg, err := importer(imports, path)
			if err != nil {
				p.errorf(spec.Path.Pos(), "could not import %s (%s)", path, err)
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
				for _, obj := range pkg.Data.(*Scope).Objects {
					p.declare(fileScope, pkgScope, obj)
				}
			} else if name != "_" {
				// declare imported package object in file scope
				// (do not re-use pkg in the file scope but create
				// a new object instead; the Decl field is different
				// for different files)
				obj := NewObj(Pkg, name)
				obj.Decl = spec
				obj.Data = pkg.Data
				p.declare(fileScope, pkgScope, obj)
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
				p.errorf(ident.Pos(), "undeclared name: %s", ident.Name)
				file.Unresolved[i] = ident
				i++
			}

		}
		file.Unresolved = file.Unresolved[0:i]
		pkgScope.Outer = universe // reset universe scope
	}

	p.errors.Sort()
	return &Package{pkgName, pkgScope, imports, files}, p.errors.Err()
}
