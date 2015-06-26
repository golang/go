// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/token"

	"golang.org/x/tools/go/exact"
)

// TODO(gri) Document factory, accessor methods, and fields. General clean-up.

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	Parent() *Scope // scope in which this object is declared
	Pos() token.Pos // position of object identifier in declaration
	Pkg() *Package  // nil for objects in the Universe scope and labels
	Name() string   // package local object name
	Type() Type     // object type
	Exported() bool // reports whether the name starts with a capital letter
	Id() string     // object id (see Id below)

	// String returns a human-readable string of the object.
	String() string

	// order reflects a package-level object's source order: if object
	// a is before object b in the source, then a.order() < b.order().
	// order returns a value > 0 for package-level objects; it returns
	// 0 for all other objects (including objects in file scopes).
	order() uint32

	// setOrder sets the order number of the object. It must be > 0.
	setOrder(uint32)

	// setParent sets the parent scope of the object.
	setParent(*Scope)

	// sameId reports whether obj.Id() and Id(pkg, name) are the same.
	sameId(pkg *Package, name string) bool

	// scopePos returns the start position of the scope of this Object
	scopePos() token.Pos

	// setScopePos sets the start position of the scope for this Object.
	setScopePos(pos token.Pos)
}

// Id returns name if it is exported, otherwise it
// returns the name qualified with the package path.
func Id(pkg *Package, name string) string {
	if ast.IsExported(name) {
		return name
	}
	// unexported names need the package path for differentiation
	// (if there's no package, make sure we don't start with '.'
	// as that may change the order of methods between a setup
	// inside a package and outside a package - which breaks some
	// tests)
	path := "_"
	// TODO(gri): shouldn't !ast.IsExported(name) => pkg != nil be an precondition?
	// if pkg == nil {
	// 	panic("nil package in lookup of unexported name")
	// }
	if pkg != nil {
		path = pkg.path
		if path == "" {
			path = "_"
		}
	}
	return path + "." + name
}

// An object implements the common parts of an Object.
type object struct {
	parent    *Scope
	pos       token.Pos
	pkg       *Package
	name      string
	typ       Type
	order_    uint32
	scopePos_ token.Pos
}

func (obj *object) Parent() *Scope      { return obj.parent }
func (obj *object) Pos() token.Pos      { return obj.pos }
func (obj *object) Pkg() *Package       { return obj.pkg }
func (obj *object) Name() string        { return obj.name }
func (obj *object) Type() Type          { return obj.typ }
func (obj *object) Exported() bool      { return ast.IsExported(obj.name) }
func (obj *object) Id() string          { return Id(obj.pkg, obj.name) }
func (obj *object) String() string      { panic("abstract") }
func (obj *object) order() uint32       { return obj.order_ }
func (obj *object) scopePos() token.Pos { return obj.scopePos_ }

func (obj *object) setParent(parent *Scope)   { obj.parent = parent }
func (obj *object) setOrder(order uint32)     { assert(order > 0); obj.order_ = order }
func (obj *object) setScopePos(pos token.Pos) { obj.scopePos_ = pos }

func (obj *object) sameId(pkg *Package, name string) bool {
	// spec:
	// "Two identifiers are different if they are spelled differently,
	// or if they appear in different packages and are not exported.
	// Otherwise, they are the same."
	if name != obj.name {
		return false
	}
	// obj.Name == name
	if obj.Exported() {
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
	imported *Package
	used     bool // set if the package was used
}

func NewPkgName(pos token.Pos, pkg *Package, name string, imported *Package) *PkgName {
	return &PkgName{object{nil, pos, pkg, name, Typ[Invalid], 0, token.NoPos}, imported, false}
}

// Imported returns the package that was imported.
// It is distinct from Pkg(), which is the package containing the import statement.
func (obj *PkgName) Imported() *Package { return obj.imported }

// A Const represents a declared constant.
type Const struct {
	object
	val     exact.Value
	visited bool // for initialization cycle detection
}

func NewConst(pos token.Pos, pkg *Package, name string, typ Type, val exact.Value) *Const {
	return &Const{object{nil, pos, pkg, name, typ, 0, token.NoPos}, val, false}
}

func (obj *Const) Val() exact.Value { return obj.val }

// A TypeName represents a declared type.
type TypeName struct {
	object
}

func NewTypeName(pos token.Pos, pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{object{nil, pos, pkg, name, typ, 0, token.NoPos}}
}

// A Variable represents a declared variable (including function parameters and results, and struct fields).
type Var struct {
	object
	anonymous bool // if set, the variable is an anonymous struct field, and name is the type name
	visited   bool // for initialization cycle detection
	isField   bool // var is struct field
	used      bool // set if the variable was used
}

func NewVar(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, token.NoPos}}
}

func NewParam(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, token.NoPos}, used: true} // parameters are always 'used'
}

func NewField(pos token.Pos, pkg *Package, name string, typ Type, anonymous bool) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, token.NoPos}, anonymous: anonymous, isField: true}
}

func (obj *Var) Anonymous() bool { return obj.anonymous }

func (obj *Var) IsField() bool { return obj.isField }

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
	return &Func{object{nil, pos, pkg, name, typ, 0, token.NoPos}}
}

// FullName returns the package- or receiver-type-qualified name of
// function or method obj.
func (obj *Func) FullName() string {
	var buf bytes.Buffer
	writeFuncName(&buf, obj, nil)
	return buf.String()
}

func (obj *Func) Scope() *Scope {
	return obj.typ.(*Signature).scope
}

// A Label represents a declared label.
type Label struct {
	object
	used bool // set if the label was used
}

func NewLabel(pos token.Pos, pkg *Package, name string) *Label {
	return &Label{object{pos: pos, pkg: pkg, name: name, typ: Typ[Invalid]}, false}
}

// A Builtin represents a built-in function.
// Builtins don't have a valid type.
type Builtin struct {
	object
	id builtinId
}

func newBuiltin(id builtinId) *Builtin {
	return &Builtin{object{name: predeclaredFuncs[id].name, typ: Typ[Invalid]}, id}
}

// Nil represents the predeclared value nil.
type Nil struct {
	object
}

func writeObject(buf *bytes.Buffer, obj Object, qf Qualifier) {
	typ := obj.Type()
	switch obj := obj.(type) {
	case *PkgName:
		fmt.Fprintf(buf, "package %s", obj.Name())
		if path := obj.imported.path; path != "" && path != obj.name {
			fmt.Fprintf(buf, " (%q)", path)
		}
		return

	case *Const:
		buf.WriteString("const")

	case *TypeName:
		buf.WriteString("type")
		typ = typ.Underlying()

	case *Var:
		if obj.isField {
			buf.WriteString("field")
		} else {
			buf.WriteString("var")
		}

	case *Func:
		buf.WriteString("func ")
		writeFuncName(buf, obj, qf)
		if typ != nil {
			WriteSignature(buf, typ.(*Signature), qf)
		}
		return

	case *Label:
		buf.WriteString("label")
		typ = nil

	case *Builtin:
		buf.WriteString("builtin")
		typ = nil

	case *Nil:
		buf.WriteString("nil")
		return

	default:
		panic(fmt.Sprintf("writeObject(%T)", obj))
	}

	buf.WriteByte(' ')

	// For package-level objects, qualify the name.
	if obj.Pkg() != nil && obj.Pkg().scope.Lookup(obj.Name()) == obj {
		writePackage(buf, obj.Pkg(), qf)
	}
	buf.WriteString(obj.Name())
	if typ != nil {
		buf.WriteByte(' ')
		WriteType(buf, typ, qf)
	}
}

func writePackage(buf *bytes.Buffer, pkg *Package, qf Qualifier) {
	if pkg == nil {
		return
	}
	var s string
	if qf != nil {
		s = qf(pkg)
	} else {
		s = pkg.Path()
	}
	if s != "" {
		buf.WriteString(s)
		buf.WriteByte('.')
	}
}

// ObjectString returns the string form of obj.
// The Qualifier controls the printing of
// package-level objects, and may be nil.
func ObjectString(obj Object, qf Qualifier) string {
	var buf bytes.Buffer
	writeObject(&buf, obj, qf)
	return buf.String()
}

func (obj *PkgName) String() string  { return ObjectString(obj, nil) }
func (obj *Const) String() string    { return ObjectString(obj, nil) }
func (obj *TypeName) String() string { return ObjectString(obj, nil) }
func (obj *Var) String() string      { return ObjectString(obj, nil) }
func (obj *Func) String() string     { return ObjectString(obj, nil) }
func (obj *Label) String() string    { return ObjectString(obj, nil) }
func (obj *Builtin) String() string  { return ObjectString(obj, nil) }
func (obj *Nil) String() string      { return ObjectString(obj, nil) }

func writeFuncName(buf *bytes.Buffer, f *Func, qf Qualifier) {
	if f.typ != nil {
		sig := f.typ.(*Signature)
		if recv := sig.Recv(); recv != nil {
			buf.WriteByte('(')
			if _, ok := recv.Type().(*Interface); ok {
				// gcimporter creates abstract methods of
				// named interfaces using the interface type
				// (not the named type) as the receiver.
				// Don't print it in full.
				buf.WriteString("interface")
			} else {
				WriteType(buf, recv.Type(), qf)
			}
			buf.WriteByte(')')
			buf.WriteByte('.')
		} else if f.pkg != nil {
			writePackage(buf, f.pkg, qf)
		}
	}
	buf.WriteString(f.name)
}
