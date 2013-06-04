// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"go/token"

	"code.google.com/p/go.tools/go/exact"
)

// TODO(gri) All objects looks very similar now. Maybe just have single Object struct with an object kind?
// TODO(gri) Document factory, accessor methods, and fields.

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	Parent() *Scope // the scope in which this object is declared
	Pos() token.Pos // position of object identifier in declaration
	Pkg() *Package  // nil for objects in the Universe scope and labels
	Name() string   // the package local object name
	Type() Type     // the object type

	setParent(*Scope)

	// TODO(gri) provide String method!
}

// An object implements the common parts of an Object.
type object struct {
	parent *Scope
	pos    token.Pos
	pkg    *Package
	name   string
	typ    Type
}

func (obj *object) Parent() *Scope { return obj.parent }
func (obj *object) Pos() token.Pos { return obj.pos }
func (obj *object) Pkg() *Package  { return obj.pkg }
func (obj *object) Name() string   { return obj.name }
func (obj *object) Type() Type     { return obj.typ }

func (obj *object) setParent(parent *Scope) { obj.parent = parent }

// A Package represents the contents (objects) of a Go package.
type Package struct {
	object
	path     string              // import path, "" for current (non-imported) package
	scope    *Scope              // imported objects
	imports  map[string]*Package // map of import paths to imported packages
	complete bool                // if set, this package was imported completely
}

func NewPackage(pos token.Pos, path, name string, scope *Scope, imports map[string]*Package, complete bool) *Package {
	obj := &Package{object{nil, pos, nil, name, Typ[Invalid]}, path, scope, imports, complete}
	obj.pkg = obj
	return obj
}

func (obj *Package) Path() string                 { return obj.path }
func (obj *Package) Scope() *Scope                { return obj.scope }
func (obj *Package) Imports() map[string]*Package { return obj.imports }
func (obj *Package) Complete() bool               { return obj.complete }

// A Const represents a declared constant.
type Const struct {
	object
	val exact.Value

	visited bool // for initialization cycle detection
}

func NewConst(pos token.Pos, pkg *Package, name string, typ Type, val exact.Value) *Const {
	return &Const{object{nil, pos, pkg, name, typ}, val, false}
}

func (obj *Const) Val() exact.Value { return obj.val }

// A TypeName represents a declared type.
type TypeName struct {
	object
}

func NewTypeName(pos token.Pos, pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{object{nil, pos, pkg, name, typ}}
}

// A Variable represents a declared variable (including function parameters and results).
type Var struct {
	object

	visited bool // for initialization cycle detection
}

func NewVar(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object{nil, pos, pkg, name, typ}, false}
}

// A Field represents a struct field.
type Field struct {
	object
	anonymous bool
}

func NewField(pos token.Pos, pkg *Package, name string, typ Type, anonymous bool) *Field {
	return &Field{object{nil, pos, pkg, name, typ}, anonymous}
}

func (obj *Field) Anonymous() bool { return obj.anonymous }

func (f *Field) isMatch(pkg *Package, name string) bool {
	// spec:
	// "Two identifiers are different if they are spelled differently,
	// or if they appear in different packages and are not exported.
	// Otherwise, they are the same."
	if name != f.name {
		return false
	}
	// f.Name == name
	return ast.IsExported(name) || pkg.path == f.pkg.path
}

// A Func represents a declared function.
type Func struct {
	object

	decl *ast.FuncDecl // TODO(gri) can we get rid of this field?
}

func NewFunc(pos token.Pos, pkg *Package, name string, typ Type) *Func {
	return &Func{object{nil, pos, pkg, name, typ}, nil}
}

// A Label represents a declared label.
type Label struct {
	object
}

func NewLabel(pos token.Pos, name string) *Label {
	return &Label{object{nil, pos, nil, name, nil}}
}
