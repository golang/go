// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
)

// An Object describes a named language entity such as a package,
// constant, type, variable, function (incl. methods), or label.
// All objects implement the Object interface.
//
type Object interface {
	Parent() *Scope // scope in which this object is declared; nil for methods and struct fields
	Pos() token.Pos // position of object identifier in declaration
	Pkg() *Package  // package to which this object belongs; nil for labels and objects in the Universe scope
	Name() string   // package local object name
	Type() Type     // object type
	Exported() bool // reports whether the name starts with a capital letter
	Id() string     // object name if exported, qualified name if not exported (see func Id)

	// String returns a human-readable string of the object.
	String() string

	// order reflects a package-level object's source order: if object
	// a is before object b in the source, then a.order() < b.order().
	// order returns a value > 0 for package-level objects; it returns
	// 0 for all other objects (including objects in file scopes).
	order() uint32

	// color returns the object's color.
	color() color

	// setOrder sets the order number of the object. It must be > 0.
	setOrder(uint32)

	// setColor sets the object's color. It must not be white.
	setColor(color color)

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
	// pkg is nil for objects in Universe scope and possibly types
	// introduced via Eval (see also comment in object.sameId)
	if pkg != nil && pkg.path != "" {
		path = pkg.path
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
	color_    color
	scopePos_ token.Pos
}

// color encodes the color of an object (see Checker.objDecl for details).
type color uint32

// An object may be painted in one of three colors.
// Color values other than white or black are considered grey.
const (
	white color = iota
	black
	grey // must be > white and black
)

func (c color) String() string {
	switch c {
	case white:
		return "white"
	case black:
		return "black"
	default:
		return "grey"
	}
}

// colorFor returns the (initial) color for an object depending on
// whether its type t is known or not.
func colorFor(t Type) color {
	if t != nil {
		return black
	}
	return white
}

// Parent returns the scope in which the object is declared.
// The result is nil for methods and struct fields.
func (obj *object) Parent() *Scope { return obj.parent }

// Pos returns the declaration position of the object's identifier.
func (obj *object) Pos() token.Pos { return obj.pos }

// Pkg returns the package to which the object belongs.
// The result is nil for labels and objects in the Universe scope.
func (obj *object) Pkg() *Package { return obj.pkg }

// Name returns the object's (package-local, unqualified) name.
func (obj *object) Name() string { return obj.name }

// Type returns the object's type.
func (obj *object) Type() Type { return obj.typ }

// Exported reports whether the object is exported (starts with a capital letter).
// It doesn't take into account whether the object is in a local (function) scope
// or not.
func (obj *object) Exported() bool { return ast.IsExported(obj.name) }

// Id is a wrapper for Id(obj.Pkg(), obj.Name()).
func (obj *object) Id() string { return Id(obj.pkg, obj.name) }

func (obj *object) String() string      { panic("abstract") }
func (obj *object) order() uint32       { return obj.order_ }
func (obj *object) color() color        { return obj.color_ }
func (obj *object) scopePos() token.Pos { return obj.scopePos_ }

func (obj *object) setParent(parent *Scope)   { obj.parent = parent }
func (obj *object) setOrder(order uint32)     { assert(order > 0); obj.order_ = order }
func (obj *object) setColor(color color)      { assert(color != white); obj.color_ = color }
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
// PkgNames don't have a type.
type PkgName struct {
	object
	imported *Package
	used     bool // set if the package was used
}

// NewPkgName returns a new PkgName object representing an imported package.
// The remaining arguments set the attributes found with all Objects.
func NewPkgName(pos token.Pos, pkg *Package, name string, imported *Package) *PkgName {
	return &PkgName{object{nil, pos, pkg, name, Typ[Invalid], 0, black, token.NoPos}, imported, false}
}

// Imported returns the package that was imported.
// It is distinct from Pkg(), which is the package containing the import statement.
func (obj *PkgName) Imported() *Package { return obj.imported }

// A Const represents a declared constant.
type Const struct {
	object
	val constant.Value
}

// NewConst returns a new constant with value val.
// The remaining arguments set the attributes found with all Objects.
func NewConst(pos token.Pos, pkg *Package, name string, typ Type, val constant.Value) *Const {
	return &Const{object{nil, pos, pkg, name, typ, 0, colorFor(typ), token.NoPos}, val}
}

// Val returns the constant's value.
func (obj *Const) Val() constant.Value { return obj.val }

func (*Const) isDependency() {} // a constant may be a dependency of an initialization expression

// A TypeName represents a name for a (defined or alias) type.
type TypeName struct {
	object
}

// NewTypeName returns a new type name denoting the given typ.
// The remaining arguments set the attributes found with all Objects.
//
// The typ argument may be a defined (Named) type or an alias type.
// It may also be nil such that the returned TypeName can be used as
// argument for NewNamed, which will set the TypeName's type as a side-
// effect.
func NewTypeName(pos token.Pos, pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{object{nil, pos, pkg, name, typ, 0, colorFor(typ), token.NoPos}}
}

// IsAlias reports whether obj is an alias name for a type.
func (obj *TypeName) IsAlias() bool {
	switch t := obj.typ.(type) {
	case nil:
		return false
	case *Basic:
		// unsafe.Pointer is not an alias.
		if obj.pkg == Unsafe {
			return false
		}
		// Any user-defined type name for a basic type is an alias for a
		// basic type (because basic types are pre-declared in the Universe
		// scope, outside any package scope), and so is any type name with
		// a different name than the name of the basic type it refers to.
		// Additionally, we need to look for "byte" and "rune" because they
		// are aliases but have the same names (for better error messages).
		return obj.pkg != nil || t.name != obj.name || t == universeByte || t == universeRune
	case *Named:
		return obj != t.obj
	default:
		return true
	}
}

// A Variable represents a declared variable (including function parameters and results, and struct fields).
type Var struct {
	object
	embedded bool // if set, the variable is an embedded struct field, and name is the type name
	isField  bool // var is struct field
	used     bool // set if the variable was used
}

// NewVar returns a new variable.
// The arguments set the attributes found with all Objects.
func NewVar(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, colorFor(typ), token.NoPos}}
}

// NewParam returns a new variable representing a function parameter.
func NewParam(pos token.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, colorFor(typ), token.NoPos}, used: true} // parameters are always 'used'
}

// NewField returns a new variable representing a struct field.
// For embedded fields, the name is the unqualified type name
/// under which the field is accessible.
func NewField(pos token.Pos, pkg *Package, name string, typ Type, embedded bool) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, colorFor(typ), token.NoPos}, embedded: embedded, isField: true}
}

// Anonymous reports whether the variable is an embedded field.
// Same as Embedded; only present for backward-compatibility.
func (obj *Var) Anonymous() bool { return obj.embedded }

// Embedded reports whether the variable is an embedded field.
func (obj *Var) Embedded() bool { return obj.embedded }

// IsField reports whether the variable is a struct field.
func (obj *Var) IsField() bool { return obj.isField }

func (*Var) isDependency() {} // a variable may be a dependency of an initialization expression

// A Func represents a declared function, concrete method, or abstract
// (interface) method. Its Type() is always a *Signature.
// An abstract method may belong to many interfaces due to embedding.
type Func struct {
	object
}

// NewFunc returns a new function with the given signature, representing
// the function's type.
func NewFunc(pos token.Pos, pkg *Package, name string, sig *Signature) *Func {
	// don't store a nil signature
	var typ Type
	if sig != nil {
		typ = sig
	}
	return &Func{object{nil, pos, pkg, name, typ, 0, colorFor(typ), token.NoPos}}
}

// FullName returns the package- or receiver-type-qualified name of
// function or method obj.
func (obj *Func) FullName() string {
	var buf bytes.Buffer
	writeFuncName(&buf, obj, nil)
	return buf.String()
}

// Scope returns the scope of the function's body block.
func (obj *Func) Scope() *Scope { return obj.typ.(*Signature).scope }

func (*Func) isDependency() {} // a function may be a dependency of an initialization expression

// A Label represents a declared label.
// Labels don't have a type.
type Label struct {
	object
	used bool // set if the label was used
}

// NewLabel returns a new label.
func NewLabel(pos token.Pos, pkg *Package, name string) *Label {
	return &Label{object{pos: pos, pkg: pkg, name: name, typ: Typ[Invalid], color_: black}, false}
}

// A Builtin represents a built-in function.
// Builtins don't have a valid type.
type Builtin struct {
	object
	id builtinId
}

func newBuiltin(id builtinId) *Builtin {
	return &Builtin{object{name: predeclaredFuncs[id].name, typ: Typ[Invalid], color_: black}, id}
}

// Nil represents the predeclared value nil.
type Nil struct {
	object
}

func writeObject(buf *bytes.Buffer, obj Object, qf Qualifier) {
	var tname *TypeName
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
		tname = obj
		buf.WriteString("type")

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

	if typ == nil {
		return
	}

	if tname != nil {
		// We have a type object: Don't print anything more for
		// basic types since there's no more information (names
		// are the same; see also comment in TypeName.IsAlias).
		if _, ok := typ.(*Basic); ok {
			return
		}
		if tname.IsAlias() {
			buf.WriteString(" =")
		} else {
			typ = typ.Underlying()
		}
	}

	buf.WriteByte(' ')
	WriteType(buf, typ, qf)
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
