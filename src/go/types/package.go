// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import "fmt"

// A Package describes a Go package.
type Package struct {
	path     string
	name     string
	scope    *Scope
	complete bool
	imports  []*Package
	fake     bool // scope lookup errors are silently dropped if package is fake (internal use only)
}

// NewPackage returns a new Package for the given package path and name;
// the name must not be the blank identifier.
// The package is not complete and contains no explicit imports.
func NewPackage(path, name string) *Package {
	if name == "_" {
		panic("invalid package name _")
	}
	scope := NewScope(Universe, fmt.Sprintf("package %q", path))
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

// MarkComplete marks a package as complete.
func (pkg *Package) MarkComplete() { pkg.complete = true }

// Imports returns the list of packages explicitly imported by
// pkg; the list is in source order. Package unsafe is excluded.
func (pkg *Package) Imports() []*Package { return pkg.imports }

// SetImports sets the list of explicitly imported packages to list.
// It is the caller's responsibility to make sure list elements are unique.
func (pkg *Package) SetImports(list []*Package) { pkg.imports = list }

func (pkg *Package) String() string {
	return fmt.Sprintf("package %s (%q)", pkg.name, pkg.path)
}
