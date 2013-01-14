// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"
)

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	GetName() string
	GetType() Type
	GetPos() token.Pos

	anObject()
}

// A Package represents the contents (objects) of a Go package.
type Package struct {
	Name     string
	Path     string              // import path, "" for current (non-imported) package
	Scope    *Scope              // package-level scope
	Imports  map[string]*Package // map of import paths to imported packages
	Complete bool                // if set, this package was imported completely

	spec *ast.ImportSpec
}

// A Const represents a declared constant.
type Const struct {
	Name string
	Type Type
	Val  interface{}

	spec *ast.ValueSpec
}

// A TypeName represents a declared type.
type TypeName struct {
	Name string
	Type Type // *NamedType or *Basic

	spec *ast.TypeSpec
}

// A Variable represents a declared variable (including function parameters and results).
type Var struct {
	Name string
	Type Type

	visited bool // for initialization cycle detection
	decl    interface{}
}

// A Func represents a declared function.
type Func struct {
	Name string
	Type Type // *Signature or *Builtin

	decl *ast.FuncDecl
}

func (obj *Package) GetName() string  { return obj.Name }
func (obj *Const) GetName() string    { return obj.Name }
func (obj *TypeName) GetName() string { return obj.Name }
func (obj *Var) GetName() string      { return obj.Name }
func (obj *Func) GetName() string     { return obj.Name }

func (obj *Package) GetType() Type  { return Typ[Invalid] }
func (obj *Const) GetType() Type    { return obj.Type }
func (obj *TypeName) GetType() Type { return obj.Type }
func (obj *Var) GetType() Type      { return obj.Type }
func (obj *Func) GetType() Type     { return obj.Type }

func (obj *Package) GetPos() token.Pos { return obj.spec.Pos() }
func (obj *Const) GetPos() token.Pos {
	for _, n := range obj.spec.Names {
		if n.Name == obj.Name {
			return n.Pos()
		}
	}
	return token.NoPos
}
func (obj *TypeName) GetPos() token.Pos { return obj.spec.Pos() }
func (obj *Var) GetPos() token.Pos {
	switch d := obj.decl.(type) {
	case *ast.Field:
		for _, n := range d.Names {
			if n.Name == obj.Name {
				return n.Pos()
			}
		}
	case *ast.ValueSpec:
		for _, n := range d.Names {
			if n.Name == obj.Name {
				return n.Pos()
			}
		}
	case *ast.AssignStmt:
		for _, x := range d.Lhs {
			if ident, isIdent := x.(*ast.Ident); isIdent && ident.Name == obj.Name {
				return ident.Pos()
			}
		}
	}
	return token.NoPos
}
func (obj *Func) GetPos() token.Pos { return obj.decl.Name.Pos() }

func (*Package) anObject()  {}
func (*Const) anObject()    {}
func (*TypeName) anObject() {}
func (*Var) anObject()      {}
func (*Func) anObject()     {}

// newObj returns a new Object for a given *ast.Object.
// It does not canonicalize them (it always returns a new one).
// For canonicalization, see check.lookup.
//
// TODO(gri) Once we do identifier resolution completely in
//           in the typechecker, this functionality can go.
//
func newObj(astObj *ast.Object) Object {
	name := astObj.Name
	typ, _ := astObj.Type.(Type)
	switch astObj.Kind {
	case ast.Bad:
		// ignore
	case ast.Pkg:
		unreachable()
	case ast.Con:
		return &Const{Name: name, Type: typ, Val: astObj.Data, spec: astObj.Decl.(*ast.ValueSpec)}
	case ast.Typ:
		return &TypeName{Name: name, Type: typ, spec: astObj.Decl.(*ast.TypeSpec)}
	case ast.Var:
		switch astObj.Decl.(type) {
		case *ast.Field, *ast.ValueSpec, *ast.AssignStmt: // these are ok
		default:
			unreachable()
		}
		return &Var{Name: name, Type: typ, decl: astObj.Decl}
	case ast.Fun:
		return &Func{Name: name, Type: typ, decl: astObj.Decl.(*ast.FuncDecl)}
	case ast.Lbl:
		unreachable() // for now
	}
	unreachable()
	return nil
}
