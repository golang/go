// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"bytes"
	"cmd/compile/internal/syntax"
	"fmt"
	"go/constant"
	"strings"
	"unicode"
	"unicode/utf8"
)

// An Object is a named language entity.
// An Object may be a constant ([Const]), type name ([TypeName]),
// variable or struct field ([Var]), function or method ([Func]),
// imported package ([PkgName]), label ([Label]),
// built-in function ([Builtin]),
// or the predeclared identifier 'nil' ([Nil]).
//
// The environment, which is structured as a tree of Scopes,
// maps each name to the unique Object that it denotes.
type Object interface {
	Parent() *Scope  // scope in which this object is declared; nil for methods and struct fields
	Pos() syntax.Pos // position of object identifier in declaration
	Pkg() *Package   // package to which this object belongs; nil for labels and objects in the Universe scope
	Name() string    // package local object name
	Type() Type      // object type
	Exported() bool  // reports whether the name starts with a capital letter
	Id() string      // object name if exported, qualified name if not exported (see func Id)

	// String returns a human-readable string of the object.
	// Use [ObjectString] to control how package names are formatted in the string.
	String() string

	// order reflects a package-level object's source order: if object
	// a is before object b in the source, then a.order() < b.order().
	// order returns a value > 0 for package-level objects; it returns
	// 0 for all other objects (including objects in file scopes).
	order() uint32

	// color returns the object's color.
	color() color

	// setType sets the type of the object.
	setType(Type)

	// setOrder sets the order number of the object. It must be > 0.
	setOrder(uint32)

	// setColor sets the object's color. It must not be white.
	setColor(color color)

	// setParent sets the parent scope of the object.
	setParent(*Scope)

	// sameId reports whether obj.Id() and Id(pkg, name) are the same.
	// If foldCase is true, names are considered equal if they are equal with case folding
	// and their packages are ignored (e.g., pkg1.m, pkg1.M, pkg2.m, and pkg2.M are all equal).
	sameId(pkg *Package, name string, foldCase bool) bool

	// scopePos returns the start position of the scope of this Object
	scopePos() syntax.Pos

	// setScopePos sets the start position of the scope for this Object.
	setScopePos(pos syntax.Pos)
}

func isExported(name string) bool {
	ch, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(ch)
}

// Id returns name if it is exported, otherwise it
// returns the name qualified with the package path.
func Id(pkg *Package, name string) string {
	if isExported(name) {
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
	pos       syntax.Pos
	pkg       *Package
	name      string
	typ       Type
	order_    uint32
	color_    color
	scopePos_ syntax.Pos
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
func (obj *object) Pos() syntax.Pos { return obj.pos }

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
func (obj *object) Exported() bool { return isExported(obj.name) }

// Id is a wrapper for Id(obj.Pkg(), obj.Name()).
func (obj *object) Id() string { return Id(obj.pkg, obj.name) }

func (obj *object) String() string       { panic("abstract") }
func (obj *object) order() uint32        { return obj.order_ }
func (obj *object) color() color         { return obj.color_ }
func (obj *object) scopePos() syntax.Pos { return obj.scopePos_ }

func (obj *object) setParent(parent *Scope)    { obj.parent = parent }
func (obj *object) setType(typ Type)           { obj.typ = typ }
func (obj *object) setOrder(order uint32)      { assert(order > 0); obj.order_ = order }
func (obj *object) setColor(color color)       { assert(color != white); obj.color_ = color }
func (obj *object) setScopePos(pos syntax.Pos) { obj.scopePos_ = pos }

func (obj *object) sameId(pkg *Package, name string, foldCase bool) bool {
	// If we don't care about capitalization, we also ignore packages.
	if foldCase && strings.EqualFold(obj.name, name) {
		return true
	}
	// spec:
	// "Two identifiers are different if they are spelled differently,
	// or if they appear in different packages and are not exported.
	// Otherwise, they are the same."
	if obj.name != name {
		return false
	}
	// obj.Name == name
	if obj.Exported() {
		return true
	}
	// not exported, so packages must be the same
	return samePkg(obj.pkg, pkg)
}

// cmp reports whether object a is ordered before object b.
// cmp returns:
//
//	-1 if a is before b
//	 0 if a is equivalent to b
//	+1 if a is behind b
//
// Objects are ordered nil before non-nil, exported before
// non-exported, then by name, and finally (for non-exported
// functions) by package path.
func (a *object) cmp(b *object) int {
	if a == b {
		return 0
	}

	// Nil before non-nil.
	if a == nil {
		return -1
	}
	if b == nil {
		return +1
	}

	// Exported functions before non-exported.
	ea := isExported(a.name)
	eb := isExported(b.name)
	if ea != eb {
		if ea {
			return -1
		}
		return +1
	}

	// Order by name and then (for non-exported names) by package.
	if a.name != b.name {
		return strings.Compare(a.name, b.name)
	}
	if !ea {
		return strings.Compare(a.pkg.path, b.pkg.path)
	}

	return 0
}

// A PkgName represents an imported Go package.
// PkgNames don't have a type.
type PkgName struct {
	object
	imported *Package
}

// NewPkgName returns a new PkgName object representing an imported package.
// The remaining arguments set the attributes found with all Objects.
func NewPkgName(pos syntax.Pos, pkg *Package, name string, imported *Package) *PkgName {
	return &PkgName{object{nil, pos, pkg, name, Typ[Invalid], 0, black, nopos}, imported}
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
func NewConst(pos syntax.Pos, pkg *Package, name string, typ Type, val constant.Value) *Const {
	return &Const{object{nil, pos, pkg, name, typ, 0, colorFor(typ), nopos}, val}
}

// Val returns the constant's value.
func (obj *Const) Val() constant.Value { return obj.val }

func (*Const) isDependency() {} // a constant may be a dependency of an initialization expression

// A TypeName is an [Object] that represents a type with a name:
// a defined type ([Named]),
// an alias type ([Alias]),
// a type parameter ([TypeParam]),
// or a predeclared type such as int or error.
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
func NewTypeName(pos syntax.Pos, pkg *Package, name string, typ Type) *TypeName {
	return &TypeName{object{nil, pos, pkg, name, typ, 0, colorFor(typ), nopos}}
}

// NewTypeNameLazy returns a new defined type like NewTypeName, but it
// lazily calls resolve to finish constructing the Named object.
func NewTypeNameLazy(pos syntax.Pos, pkg *Package, name string, load func(named *Named) (tparams []*TypeParam, underlying Type, methods []*Func)) *TypeName {
	obj := NewTypeName(pos, pkg, name, nil)
	NewNamed(obj, nil, nil).loader = load
	return obj
}

// IsAlias reports whether obj is an alias name for a type.
func (obj *TypeName) IsAlias() bool {
	switch t := obj.typ.(type) {
	case nil:
		return false
	// case *Alias:
	//	handled by default case
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
	case *TypeParam:
		return obj != t.obj
	default:
		return true
	}
}

// A Variable represents a declared variable (including function parameters and results, and struct fields).
type Var struct {
	object
	origin   *Var // if non-nil, the Var from which this one was instantiated
	embedded bool // if set, the variable is an embedded struct field, and name is the type name
	isField  bool // var is struct field
	isParam  bool // var is a param, for backport of 'used' check to go1.24 (go.dev/issue/72826)
}

// NewVar returns a new variable.
// The arguments set the attributes found with all Objects.
func NewVar(pos syntax.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, colorFor(typ), nopos}}
}

// NewParam returns a new variable representing a function parameter.
func NewParam(pos syntax.Pos, pkg *Package, name string, typ Type) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, colorFor(typ), nopos}, isParam: true}
}

// NewField returns a new variable representing a struct field.
// For embedded fields, the name is the unqualified type name
// under which the field is accessible.
func NewField(pos syntax.Pos, pkg *Package, name string, typ Type, embedded bool) *Var {
	return &Var{object: object{nil, pos, pkg, name, typ, 0, colorFor(typ), nopos}, embedded: embedded, isField: true}
}

// Anonymous reports whether the variable is an embedded field.
// Same as Embedded; only present for backward-compatibility.
func (obj *Var) Anonymous() bool { return obj.embedded }

// Embedded reports whether the variable is an embedded field.
func (obj *Var) Embedded() bool { return obj.embedded }

// IsField reports whether the variable is a struct field.
func (obj *Var) IsField() bool { return obj.isField }

// Origin returns the canonical Var for its receiver, i.e. the Var object
// recorded in Info.Defs.
//
// For synthetic Vars created during instantiation (such as struct fields or
// function parameters that depend on type arguments), this will be the
// corresponding Var on the generic (uninstantiated) type. For all other Vars
// Origin returns the receiver.
func (obj *Var) Origin() *Var {
	if obj.origin != nil {
		return obj.origin
	}
	return obj
}

func (*Var) isDependency() {} // a variable may be a dependency of an initialization expression

// A Func represents a declared function, concrete method, or abstract
// (interface) method. Its Type() is always a *Signature.
// An abstract method may belong to many interfaces due to embedding.
type Func struct {
	object
	hasPtrRecv_ bool  // only valid for methods that don't have a type yet; use hasPtrRecv() to read
	origin      *Func // if non-nil, the Func from which this one was instantiated
}

// NewFunc returns a new function with the given signature, representing
// the function's type.
func NewFunc(pos syntax.Pos, pkg *Package, name string, sig *Signature) *Func {
	var typ Type
	if sig != nil {
		typ = sig
	} else {
		// Don't store a (typed) nil *Signature.
		// We can't simply replace it with new(Signature) either,
		// as this would violate object.{Type,color} invariants.
		// TODO(adonovan): propose to disallow NewFunc with nil *Signature.
	}
	return &Func{object{nil, pos, pkg, name, typ, 0, colorFor(typ), nopos}, false, nil}
}

// Signature returns the signature (type) of the function or method.
func (obj *Func) Signature() *Signature {
	if obj.typ != nil {
		return obj.typ.(*Signature) // normal case
	}
	// No signature: Signature was called either:
	// - within go/types, before a FuncDecl's initially
	//   nil Func.Type was lazily populated, indicating
	//   a types bug; or
	// - by a client after NewFunc(..., nil),
	//   which is arguably a client bug, but we need a
	//   proposal to tighten NewFunc's precondition.
	// For now, return a trivial signature.
	return new(Signature)
}

// FullName returns the package- or receiver-type-qualified name of
// function or method obj.
func (obj *Func) FullName() string {
	var buf bytes.Buffer
	writeFuncName(&buf, obj, nil)
	return buf.String()
}

// Scope returns the scope of the function's body block.
// The result is nil for imported or instantiated functions and methods
// (but there is also no mechanism to get to an instantiated function).
func (obj *Func) Scope() *Scope { return obj.typ.(*Signature).scope }

// Origin returns the canonical Func for its receiver, i.e. the Func object
// recorded in Info.Defs.
//
// For synthetic functions created during instantiation (such as methods on an
// instantiated Named type or interface methods that depend on type arguments),
// this will be the corresponding Func on the generic (uninstantiated) type.
// For all other Funcs Origin returns the receiver.
func (obj *Func) Origin() *Func {
	if obj.origin != nil {
		return obj.origin
	}
	return obj
}

// Pkg returns the package to which the function belongs.
//
// The result is nil for methods of types in the Universe scope,
// like method Error of the error built-in interface type.
func (obj *Func) Pkg() *Package { return obj.object.Pkg() }

// hasPtrRecv reports whether the receiver is of the form *T for the given method obj.
func (obj *Func) hasPtrRecv() bool {
	// If a method's receiver type is set, use that as the source of truth for the receiver.
	// Caution: Checker.funcDecl (decl.go) marks a function by setting its type to an empty
	// signature. We may reach here before the signature is fully set up: we must explicitly
	// check if the receiver is set (we cannot just look for non-nil obj.typ).
	if sig, _ := obj.typ.(*Signature); sig != nil && sig.recv != nil {
		_, isPtr := deref(sig.recv.typ)
		return isPtr
	}

	// If a method's type is not set it may be a method/function that is:
	// 1) client-supplied (via NewFunc with no signature), or
	// 2) internally created but not yet type-checked.
	// For case 1) we can't do anything; the client must know what they are doing.
	// For case 2) we can use the information gathered by the resolver.
	return obj.hasPtrRecv_
}

func (*Func) isDependency() {} // a function may be a dependency of an initialization expression

// A Label represents a declared label.
// Labels don't have a type.
type Label struct {
	object
	used bool // set if the label was used
}

// NewLabel returns a new label.
func NewLabel(pos syntax.Pos, pkg *Package, name string) *Label {
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
		if isTypeParam(typ) {
			buf.WriteString(" parameter")
		}

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
		buf.WriteString(packagePrefix(obj.Pkg(), qf))
	}
	buf.WriteString(obj.Name())

	if typ == nil {
		return
	}

	if tname != nil {
		switch t := typ.(type) {
		case *Basic:
			// Don't print anything more for basic types since there's
			// no more information.
			return
		case genericType:
			if t.TypeParams().Len() > 0 {
				newTypeWriter(buf, qf).tParamList(t.TypeParams().list())
			}
		}
		if tname.IsAlias() {
			buf.WriteString(" =")
			if alias, ok := typ.(*Alias); ok { // materialized? (gotypesalias=1)
				typ = alias.fromRHS
			}
		} else if t, _ := typ.(*TypeParam); t != nil {
			typ = t.bound
		} else {
			// TODO(gri) should this be fromRHS for *Named?
			// (See discussion in #66559.)
			typ = under(typ)
		}
	}

	// Special handling for any: because WriteType will format 'any' as 'any',
	// resulting in the object string `type any = any` rather than `type any =
	// interface{}`. To avoid this, swap in a different empty interface.
	if obj.Name() == "any" && obj.Parent() == Universe {
		assert(Identical(typ, &emptyInterface))
		typ = &emptyInterface
	}

	buf.WriteByte(' ')
	WriteType(buf, typ, qf)
}

func packagePrefix(pkg *Package, qf Qualifier) string {
	if pkg == nil {
		return ""
	}
	var s string
	if qf != nil {
		s = qf(pkg)
	} else {
		s = pkg.Path()
	}
	if s != "" {
		s += "."
	}
	return s
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
			buf.WriteString(packagePrefix(f.pkg, qf))
		}
	}
	buf.WriteString(f.name)
}
