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

// TODO(gri) Document factory, accessor methods, and fields. General clean-up.

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

	// isUsed reports whether the object was marked as 'used'.
	isUsed() bool

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
	used   bool
}

func (obj *object) Parent() *Scope   { return obj.parent }
func (obj *object) Pos() token.Pos   { return obj.pos }
func (obj *object) Pkg() *Package    { return obj.pkg }
func (obj *object) Name() string     { return obj.name }
func (obj *object) Type() Type       { return obj.typ }
func (obj *object) IsExported() bool { return ast.IsExported(obj.name) }
func (obj *object) Id() string       { return Id(obj.pkg, obj.name) }
func (obj *object) String() string   { panic("abstract") }

func (obj *object) isUsed() bool { return obj.used }

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

// A PkgName represents an imported Go package.
type PkgName struct {
	object
}

func NewPkgName(pos token.Pos, pkg *Package, name string) *PkgName {
	return &PkgName{object{nil, pos, pkg, name, Typ[Invalid], false}}
}

func (obj *PkgName) String() string { return obj.toString("package", nil) }

// A Const represents a declared constant.
type Const struct {
	object
	val exact.Value

	visited bool // for initialization cycle detection
}

func NewConst(pos token.Pos, pkg *Package, name string, typ Type, val exact.Value) *Const {
	return &Const{object: object{nil, pos, pkg, name, typ, false}, val: val}
}

func (obj *Const) String() string   { return obj.toString("const", obj.typ) }
func (obj *Const) Val() exact.Value { return obj.val }

// A TypeName represents a declared type.
type TypeName struct {
	object
}

func NewTypeName(pos token.Pos, pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{object{nil, pos, pkg, name, typ, false}}
}

func (obj *TypeName) String() string { return obj.toString("type", obj.typ.Underlying()) }

// A Variable represents a declared variable (including function parameters and results, and struct fields).
type Var struct {
	object

	anonymous bool // if set, the variable is an anonymous struct field, and name is the type name
	visited   bool // for initialization cycle detection
}

func NewVar(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, false}}
}

func NewParam(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, true}} // parameters are always 'used'
}

func NewField(pos token.Pos, pkg *Package, name string, typ Type, anonymous bool) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, false}, anonymous: anonymous}
}

func (obj *Var) Anonymous() bool { return obj.anonymous }
func (obj *Var) String() string  { return obj.toString("var", obj.typ) }

// A Func represents a declared function, concrete method, or abstract
// (interface) method.  Its Type() is always a *Signature.
// An abstract method may belong to many interfaces due to embedding.
type Func struct {
	object
}

func NewFunc(pos token.Pos, pkg *Package, name string, sig *Signature) *Func {
	// don't store a nil signature
	var typ Type
	if sig != nil {
		typ = sig
	}
	return &Func{object{nil, pos, pkg, name, typ, false}}
}

// FullName returns the package- or receiver-type-qualified name of
// function or method obj.
func (obj *Func) FullName() string {
	var buf bytes.Buffer
	obj.fullname(&buf)
	return buf.String()
}

func (obj *Func) fullname(buf *bytes.Buffer) {
	if obj.typ != nil {
		sig := obj.typ.(*Signature)
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
	if obj.typ != nil {
		writeSignature(&buf, obj.typ.(*Signature))
	}
	return buf.String()
}

// A Label represents a declared label.
type Label struct {
	object
}

func NewLabel(pos token.Pos, name string) *Label {
	return &Label{object{pos: pos, name: name}}
}

func (obj *Label) String() string { return fmt.Sprintf("label %s", obj.Name()) }

// A Builtin represents a built-in function.
// Builtins don't have a valid type.
type Builtin struct {
	object

	id builtinId
}

func newBuiltin(id builtinId) *Builtin {
	return &Builtin{object{name: predeclaredFuncs[id].name, typ: Typ[Invalid]}, id}
}
