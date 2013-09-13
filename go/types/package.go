// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

// A Package describes a Go package.
type Package struct {
	path     string
	name     string
	scope    *Scope
	complete bool
	imports  []*Package
	fake     bool // scope lookup errors are silently dropped if package is fake (internal use only)
}

// NewPackage returns a new Package for the given package path,
// name, and scope. The package is not complete and contains no
// explicit imports.
func NewPackage(path, name string, scope *Scope) *Package {
	return &Package{path: path, name: name, scope: scope}
}

// Path returns the package path.
func (pkg *Package) Path() string { return pkg.path }

// Name returns the package name.
func (pkg *Package) Name() string { return pkg.name }

// Scope returns the (complete or incomplete) package scope
// holding the objects declared at package level (TypeNames,
// Consts, Vars, and Funcs).
func (pkg *Package) Scope() *Scope { return pkg.scope }

// A package is complete if its scope contains (at least) all
// exported objects; otherwise it is incomplete.
func (pkg *Package) Complete() bool { return pkg.complete }

// Imports returns the list of packages explicitly imported by
// pkg; the list is in source order. Package unsafe is excluded.
func (pkg *Package) Imports() []*Package { return pkg.imports }
