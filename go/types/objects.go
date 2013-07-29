// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
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
	Parent() *Scope   // scope in which this object is declared
	Pos() token.Pos   // position of object identifier in declaration
	Pkg() *Package    // nil for objects in the Universe scope and labels
	Name() string     // package local object name
	Type() Type       // object type
	IsExported() bool // reports whether the name starts with a capital letter
	Id() string       // object id (see Id below)

	// String returns a human-readable string of the object.
	String() string

	// setParent sets the parent scope of the object.
	setParent(*Scope)

	// sameId reports whether obj.Id() and Id(pkg, name) are the same.
	sameId(pkg *Package, name string) bool
}

// Id returns name if it is exported, otherwise it
// returns the name qualified with the package path.
func Id(pkg *Package, name string) string {
	if ast.IsExported(name) {
		return name
	}
	// unexported names need the package path for differentiation
	path := ""
	// TODO(gri): shouldn't !ast.IsExported(name) => pkg != nil be an precondition?
	// if pkg == nil {
	// 	panic("nil package in lookup of unexported name")
	// }
	if pkg != nil {
		path = pkg.path
		if path == "" {
			path = "?"
		}
	}
	return path + "." + name
}

// An object implements the common parts of an Object.
type object struct {
	parent *Scope
	pos    token.Pos
	pkg    *Package
	name   string
	typ    Type
}

func (obj *object) Parent() *Scope   { return obj.parent }
func (obj *object) Pos() token.Pos   { return obj.pos }
func (obj *object) Pkg() *Package    { return obj.pkg }
func (obj *object) Name() string     { return obj.name }
func (obj *object) Type() Type       { return obj.typ }
func (obj *object) IsExported() bool { return ast.IsExported(obj.name) }
func (obj *object) Id() string       { return Id(obj.pkg, obj.name) }
func (obj *object) String() string   { panic("abstract") }

func (obj *object) toString(kind string, typ Type) string {
	var buf bytes.Buffer

	buf.WriteString(kind)
	buf.WriteByte(' ')
	// For package-level objects, package-qualify the name.
	if obj.pkg != nil && obj.pkg.scope.Lookup(obj.name) == obj {
		buf.WriteString(obj.pkg.name)
		buf.WriteByte('.')
	}
	buf.WriteString(obj.name)
	if typ != nil {
		buf.WriteByte(' ')
		writeType(&buf, typ)
	}

	return buf.String()
}

func (obj *object) setParent(parent *Scope) { obj.parent = parent }

func (obj *object) sameId(pkg *Package, name string) bool {
	// spec:
	// "Two identifiers are different if they are spelled differently,
	// or if they appear in different packages and are not exported.
	// Otherwise, they are the same."
	if name != obj.name {
		return false
	}
	// obj.Name == name
	if obj.IsExported() {
		return true
	}
	// not exported, so packages must be the same (pkg == nil for
	// fields in Universe scope; this can only happen for types
	// introduced via Eval)
	if pkg == nil || obj.pkg == nil {
		return pkg == obj.pkg
	}
	// pkg != nil && obj.pkg != nil
	return pkg.path == obj.pkg.path
}

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

func (obj *Package) String() string               { return fmt.Sprintf("package %s", obj.Path()) }
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

func (obj *Const) String() string   { return obj.toString("const", obj.typ) }
func (obj *Const) Val() exact.Value { return obj.val }

// A TypeName represents a declared type.
type TypeName struct {
	object
}

func NewTypeName(pos token.Pos, pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{object{nil, pos, pkg, name, typ}}
}

func (obj *TypeName) String() string { return obj.toString("type", obj.typ.Underlying()) }

// A Variable represents a declared variable (including function parameters and results, and struct fields).
type Var struct {
	object

	anonymous bool // if set, this variable is an anonymous struct field, and name is the type name
	visited   bool // for initialization cycle detection
}

func NewVar(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object{nil, pos, pkg, name, typ}, false, false}
}

func NewFieldVar(pos token.Pos, pkg *Package, name string, typ Type, anonymous bool) *Var {
	return &Var{object{nil, pos, pkg, name, typ}, anonymous, false}
}

func (obj *Var) Anonymous() bool { return obj.anonymous }
func (obj *Var) String() string  { return obj.toString("var", obj.typ) }

// A Func represents a declared function, concrete method, or abstract
// (interface) method.  Its Type() is a *Signature.
// An abstract method may belong to many interfaces due to embedding.
type Func struct {
	object
}

func NewFunc(pos token.Pos, pkg *Package, name string, typ Type) *Func {
	return &Func{object{nil, pos, pkg, name, typ}}
}

// FullName returns the package- or receiver-type-qualified name of
// function or method obj.
func (obj *Func) FullName() string {
	var buf bytes.Buffer
	obj.fullname(&buf)
	return buf.String()
}

func (obj *Func) fullname(buf *bytes.Buffer) {
	if sig, _ := obj.typ.(*Signature); sig != nil { // may be a *Builtin
		if recv := sig.Recv(); recv != nil {
			buf.WriteByte('(')
			if _, ok := recv.Type().(*Interface); ok {
				// gcimporter creates abstract methods of
				// named interfaces using the interface type
				// (not the named type) as the receiver.
				// Don't print it in full.
				buf.WriteString("interface")
			} else {
				writeType(buf, recv.Type())
			}
			buf.WriteByte(')')
			buf.WriteByte('.')
		} else if obj.pkg != nil {
			buf.WriteString(obj.pkg.name)
			buf.WriteByte('.')
		}
	}
	buf.WriteString(obj.name)
}

func (obj *Func) String() string {
	var buf bytes.Buffer
	buf.WriteString("func ")
	obj.fullname(&buf)
	if sig, _ := obj.typ.(*Signature); sig != nil {
		writeSignature(&buf, sig)
	}
	return buf.String()
}

// A Label represents a declared label.
type Label struct {
	object
}

func NewLabel(pos token.Pos, name string) *Label {
	return &Label{object{nil, pos, nil, name, nil}}
}

func (obj *Label) String() string { return fmt.Sprintf("label %s", obj.Name()) }
