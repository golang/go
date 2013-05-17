// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	Pkg() *Package // nil for objects in the Universe scope
	Scope() *Scope
	Name() string
	Type() Type
	Pos() token.Pos
	// TODO(gri) provide String method!
}

// A Package represents the contents (objects) of a Go package.
type Package struct {
	name     string
	path     string              // import path, "" for current (non-imported) package
	scope    *Scope              // package-level scope
	imports  map[string]*Package // map of import paths to imported packages
	complete bool                // if set, this package was imported completely

	spec *ast.ImportSpec
}

func NewPackage(path, name string) *Package {
	return &Package{name: name, path: path, complete: true}
}

func (obj *Package) Pkg() *Package { return obj }
func (obj *Package) Scope() *Scope { return obj.scope }
func (obj *Package) Name() string  { return obj.name }
func (obj *Package) Type() Type    { return Typ[Invalid] }
func (obj *Package) Pos() token.Pos {
	if obj.spec == nil {
		return token.NoPos
	}
	return obj.spec.Pos()
}
func (obj *Package) Path() string                 { return obj.path }
func (obj *Package) Imports() map[string]*Package { return obj.imports }
func (obj *Package) Complete() bool               { return obj.complete }

// A Const represents a declared constant.
type Const struct {
	pkg  *Package
	name string
	typ  Type
	val  exact.Value

	visited bool // for initialization cycle detection
	spec    *ast.ValueSpec
}

func (obj *Const) Pkg() *Package { return obj.pkg }
func (obj *Const) Scope() *Scope { panic("unimplemented") }
func (obj *Const) Name() string  { return obj.name }
func (obj *Const) Type() Type    { return obj.typ }
func (obj *Const) Pos() token.Pos {
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
func (obj *Const) Val() exact.Value { return obj.val }

// A TypeName represents a declared type.
type TypeName struct {
	pkg  *Package
	name string
	typ  Type // *Named or *Basic

	spec *ast.TypeSpec
}

func NewTypeName(pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{pkg, name, typ, nil}
}

func (obj *TypeName) Pkg() *Package { return obj.pkg }
func (obj *TypeName) Scope() *Scope { panic("unimplemented") }
func (obj *TypeName) Name() string  { return obj.name }
func (obj *TypeName) Type() Type    { return obj.typ }
func (obj *TypeName) Pos() token.Pos {
	if obj.spec == nil {
		return token.NoPos
	}
	return obj.spec.Pos()
}

// A Variable represents a declared variable (including function parameters and results).
type Var struct {
	pkg  *Package // nil for parameters
	name string
	typ  Type

	visited bool // for initialization cycle detection
	decl    interface{}
}

func NewVar(pkg *Package, name string, typ Type) *Var {
	return &Var{pkg, name, typ, false, nil}
}

func (obj *Var) Pkg() *Package { return obj.pkg }
func (obj *Var) Scope() *Scope { panic("unimplemented") }
func (obj *Var) Name() string  { return obj.name }
func (obj *Var) Type() Type    { return obj.typ }
func (obj *Var) Pos() token.Pos {
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

// A Func represents a declared function.
type Func struct {
	pkg  *Package
	name string
	typ  Type // *Signature or *Builtin

	decl *ast.FuncDecl
}

func (obj *Func) Pkg() *Package { return obj.pkg }
func (obj *Func) Scope() *Scope { panic("unimplemented") }
func (obj *Func) Name() string  { return obj.name }
func (obj *Func) Type() Type    { return obj.typ }
func (obj *Func) Pos() token.Pos {
	if obj.decl != nil && obj.decl.Name != nil {
		return obj.decl.Name.Pos()
	}
	return token.NoPos
}

// newObj returns a new Object for a given *ast.Object.
// It does not canonicalize them (it always returns a new one).
// For canonicalization, see check.lookup.
//
// TODO(gri) Once we do identifier resolution completely in
//           the typechecker, this functionality can go.
//
func newObj(pkg *Package, astObj *ast.Object) Object {
	assert(pkg != nil)
	name := astObj.Name
	typ, _ := astObj.Type.(Type)
	switch astObj.Kind {
	case ast.Bad:
		// ignore
	case ast.Pkg:
		unreachable()
	case ast.Con:
		iota := astObj.Data.(int)
		return &Const{pkg: pkg, name: name, typ: typ, val: exact.MakeInt64(int64(iota)), spec: astObj.Decl.(*ast.ValueSpec)}
	case ast.Typ:
		return &TypeName{pkg: pkg, name: name, typ: typ, spec: astObj.Decl.(*ast.TypeSpec)}
	case ast.Var:
		switch astObj.Decl.(type) {
		case *ast.Field: // function parameters
		case *ast.ValueSpec: // proper variable declarations
		case *ast.AssignStmt: // short variable declarations
		default:
			unreachable() // everything else is not ok
		}
		return &Var{pkg: pkg, name: name, typ: typ, decl: astObj.Decl}
	case ast.Fun:
		return &Func{pkg: pkg, name: name, typ: typ, decl: astObj.Decl.(*ast.FuncDecl)}
	case ast.Lbl:
		unreachable() // for now
	}
	unreachable()
	return nil
}
