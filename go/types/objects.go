// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// TODO(gri) provide a complete set of factory functions!
// TODO(gri) clean up the Pos functions.
// TODO(gri) clean up the internal links to decls/specs, etc.

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	Pkg() *Package // nil for objects in the Universe scope
	Outer() *Scope // the scope in which this object is declared
	Name() string
	Type() Type
	Pos() token.Pos // position of object identifier in declaration
	// TODO(gri) provide String method!

	setOuter(*Scope)
}

// A Package represents the contents (objects) of a Go package.
type Package struct {
	pos      token.Pos // position of package import path or local package identifier, if present
	name     string
	path     string // import path, "" for current (non-imported) package
	outer    *Scope
	scope    *Scope              // package-level scope
	imports  map[string]*Package // map of import paths to imported packages
	complete bool                // if set, this package was imported completely

	spec *ast.ImportSpec
}

func NewPackage(path, name string) *Package {
	return &Package{name: name, path: path, complete: true}
}

func (obj *Package) Pkg() *Package { return obj }
func (obj *Package) Outer() *Scope { return obj.outer }
func (obj *Package) Scope() *Scope { return obj.scope }
func (obj *Package) Name() string  { return obj.name }
func (obj *Package) Type() Type    { return Typ[Invalid] }
func (obj *Package) Pos() token.Pos {
	if obj.pos.IsValid() {
		return obj.pos
	}
	if obj.spec != nil {
		return obj.spec.Pos()
	}
	return token.NoPos
}
func (obj *Package) Path() string                 { return obj.path }
func (obj *Package) Imports() map[string]*Package { return obj.imports }
func (obj *Package) Complete() bool               { return obj.complete }
func (obj *Package) setOuter(*Scope) {            /* don't do anything - this is the package's scope */
}

// A Const represents a declared constant.
type Const struct {
	pos   token.Pos // position of identifier in constant declaration
	pkg   *Package
	outer *Scope
	name  string
	typ   Type
	val   exact.Value

	visited bool // for initialization cycle detection
	spec    *ast.ValueSpec
}

func (obj *Const) Pkg() *Package { return obj.pkg }
func (obj *Const) Outer() *Scope { return obj.outer }
func (obj *Const) Name() string  { return obj.name }
func (obj *Const) Type() Type    { return obj.typ }
func (obj *Const) Pos() token.Pos {
	if obj.pos.IsValid() {
		return obj.pos
	}
	if obj.spec == nil {
		return token.NoPos
	}
	for _, n := range obj.spec.Names {
		if n.Name == obj.name {
			return n.Pos()
		}
	}
	return token.NoPos
}
func (obj *Const) Val() exact.Value  { return obj.val }
func (obj *Const) setOuter(s *Scope) { obj.outer = s }

// A TypeName represents a declared type.
type TypeName struct {
	pos   token.Pos // position of identifier in type declaration
	pkg   *Package
	outer *Scope
	name  string
	typ   Type // *Named or *Basic

	spec *ast.TypeSpec
}

func NewTypeName(pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{token.NoPos, pkg, nil, name, typ, nil}
}

func (obj *TypeName) Pkg() *Package { return obj.pkg }
func (obj *TypeName) Outer() *Scope { return obj.outer }
func (obj *TypeName) Name() string  { return obj.name }
func (obj *TypeName) Type() Type    { return obj.typ }
func (obj *TypeName) Pos() token.Pos {
	if obj.pos.IsValid() {
		return obj.pos
	}
	if obj.spec == nil {
		return token.NoPos
	}
	return obj.spec.Pos()
}
func (obj *TypeName) setOuter(s *Scope) { obj.outer = s }

// A Variable represents a declared variable (including function parameters and results).
type Var struct {
	pos   token.Pos // position of identifier in variable declaration
	pkg   *Package  // nil for parameters
	outer *Scope
	name  string
	typ   Type

	visited bool // for initialization cycle detection
	decl    interface{}
}

func NewVar(pkg *Package, name string, typ Type) *Var {
	return &Var{token.NoPos, pkg, nil, name, typ, false, nil}
}

func (obj *Var) Pkg() *Package { return obj.pkg }
func (obj *Var) Outer() *Scope { return obj.outer }
func (obj *Var) Name() string  { return obj.name }
func (obj *Var) Type() Type    { return obj.typ }
func (obj *Var) Pos() token.Pos {
	if obj.pos.IsValid() {
		return obj.pos
	}
	switch d := obj.decl.(type) {
	case *ast.Field:
		for _, n := range d.Names {
			if n.Name == obj.name {
				return n.Pos()
			}
		}
	case *ast.ValueSpec:
		for _, n := range d.Names {
			if n.Name == obj.name {
				return n.Pos()
			}
		}
	case *ast.AssignStmt:
		for _, x := range d.Lhs {
			if ident, isIdent := x.(*ast.Ident); isIdent && ident.Name == obj.name {
				return ident.Pos()
			}
		}
	}
	return token.NoPos
}
func (obj *Var) setOuter(s *Scope) { obj.outer = s }

// A Func represents a declared function.
type Func struct {
	pos   token.Pos
	pkg   *Package
	outer *Scope
	name  string
	typ   Type // *Signature or *Builtin

	decl *ast.FuncDecl
}

func (obj *Func) Pkg() *Package { return obj.pkg }
func (obj *Func) Outer() *Scope { return obj.outer }
func (obj *Func) Name() string  { return obj.name }
func (obj *Func) Type() Type    { return obj.typ }
func (obj *Func) Pos() token.Pos {
	if obj.pos.IsValid() {
		return obj.pos
	}
	if obj.decl != nil && obj.decl.Name != nil {
		return obj.decl.Name.Pos()
	}
	return token.NoPos
}
func (obj *Func) setOuter(s *Scope) { obj.outer = s }
