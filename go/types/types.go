// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"go/ast"
	"sort"
)

// TODO(gri) Revisit factory functions - make sure they have all relevant parameters.

// A Type represents a type of Go.
// All types implement the Type interface.
type Type interface {
	// Underlying returns the underlying type of a type.
	Underlying() Type

	// MethodSet returns the method set of a type.
	MethodSet() *MethodSet

	// String returns a string representation of a type.
	String() string
}

// BasicKind describes the kind of basic type.
type BasicKind int

const (
	Invalid BasicKind = iota // type is invalid

	// predeclared types
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	String
	UnsafePointer

	// types for untyped values
	UntypedBool
	UntypedInt
	UntypedRune
	UntypedFloat
	UntypedComplex
	UntypedString
	UntypedNil

	// aliases
	Byte = Uint8
	Rune = Int32
)

// BasicInfo is a set of flags describing properties of a basic type.
type BasicInfo int

// Properties of basic types.
const (
	IsBoolean BasicInfo = 1 << iota
	IsInteger
	IsUnsigned
	IsFloat
	IsComplex
	IsString
	IsUntyped

	IsOrdered   = IsInteger | IsFloat | IsString
	IsNumeric   = IsInteger | IsFloat | IsComplex
	IsConstType = IsBoolean | IsNumeric | IsString
)

// A Basic represents a basic type.
type Basic struct {
	kind BasicKind
	info BasicInfo
	size int64 // use DefaultSizeof to get size
	name string
}

// Kind returns the kind of basic type b.
func (b *Basic) Kind() BasicKind { return b.kind }

// Info returns information about properties of basic type b.
func (b *Basic) Info() BasicInfo { return b.info }

// Name returns the name of basic type b.
func (b *Basic) Name() string { return b.name }

// An Array represents an array type.
type Array struct {
	len  int64
	elem Type
}

// NewArray returns a new array type for the given element type and length.
func NewArray(elem Type, len int64) *Array { return &Array{len, elem} }

// Len returns the length of array a.
func (a *Array) Len() int64 { return a.len }

// Elem returns element type of array a.
func (a *Array) Elem() Type { return a.elem }

// A Slice represents a slice type.
type Slice struct {
	elem Type
}

// NewSlice returns a new slice type for the given element type.
func NewSlice(elem Type) *Slice { return &Slice{elem} }

// Elem returns the element type of slice s.
func (s *Slice) Elem() Type { return s.elem }

// A Struct represents a struct type.
type Struct struct {
	fields []*Var
	tags   []string // field tags; nil if there are no tags
	// TODO(gri) access to offsets is not threadsafe - fix this
	offsets []int64         // field offsets in bytes, lazily initialized
	mset    cachedMethodSet // method set, lazily initialized
}

// NewStruct returns a new struct with the given fields and corresponding field tags.
// If a field with index i has a tag, tags[i] must be that tag, but len(tags) may be
// only as long as required to hold the tag with the largest index i. Consequently,
// if no field has a tag, tags may be nil.
func NewStruct(fields []*Var, tags []string) *Struct {
	var fset objset
	for _, f := range fields {
		if f.name != "_" && fset.insert(f) != nil {
			panic("multiple fields with the same name")
		}
	}
	if len(tags) > len(fields) {
		panic("more tags than fields")
	}
	return &Struct{fields: fields, tags: tags}
}

// NumFields returns the number of fields in the struct (including blank and anonymous fields).
func (s *Struct) NumFields() int { return len(s.fields) }

// Field returns the i'th field for 0 <= i < NumFields().
func (s *Struct) Field(i int) *Var { return s.fields[i] }

// Tag returns the i'th field tag for 0 <= i < NumFields().
func (s *Struct) Tag(i int) string {
	if i < len(s.tags) {
		return s.tags[i]
	}
	return ""
}

// A Pointer represents a pointer type.
type Pointer struct {
	base Type            // element type
	mset cachedMethodSet // method set, lazily initialized
}

// NewPointer returns a new pointer type for the given element (base) type.
func NewPointer(elem Type) *Pointer { return &Pointer{base: elem} }

// Elem returns the element type for the given pointer p.
func (p *Pointer) Elem() Type { return p.base }

// A Tuple represents an ordered list of variables; a nil *Tuple is a valid (empty) tuple.
// Tuples are used as components of signatures and to represent the type of multiple
// assignments; they are not first class types of Go.
type Tuple struct {
	vars []*Var
}

// NewTuple returns a new tuple for the given variables.
func NewTuple(x ...*Var) *Tuple {
	if len(x) > 0 {
		return &Tuple{x}
	}
	return nil
}

// Len returns the number variables of tuple t.
func (t *Tuple) Len() int {
	if t != nil {
		return len(t.vars)
	}
	return 0
}

// At returns the i'th variable of tuple t.
func (t *Tuple) At(i int) *Var { return t.vars[i] }

// A Signature represents a (non-builtin) function or method type.
type Signature struct {
	scope      *Scope // function scope, always present
	recv       *Var   // nil if not a method
	params     *Tuple // (incoming) parameters from left to right; or nil
	results    *Tuple // (outgoing) results from left to right; or nil
	isVariadic bool   // true if the last parameter's type is of the form ...T
}

// NewSignature returns a new function type for the given receiver, parameters,
// and results, either of which may be nil. If isVariadic is set, the function
// is variadic, it must have at least one parameter, and the last parameter
// must be of unnamed slice type.
func NewSignature(scope *Scope, recv *Var, params, results *Tuple, isVariadic bool) *Signature {
	// TODO(gri) Should we rely on the correct (non-nil) incoming scope
	//           or should this function allocate and populate a scope?
	if isVariadic {
		n := params.Len()
		if n == 0 {
			panic("types.NewSignature: variadic function must have at least one parameter")
		}
		if _, ok := params.At(n - 1).typ.(*Slice); !ok {
			panic("types.NewSignature: variadic parameter must be of unnamed slice type")
		}
	}
	return &Signature{scope, recv, params, results, isVariadic}
}

// Recv returns the receiver of signature s (if a method), or nil if a
// function.
//
// For an abstract method, Recv returns the enclosing interface either
// as a *Named or an *Interface.  Due to embedding, an interface may
// contain methods whose receiver type is a different interface.
func (s *Signature) Recv() *Var { return s.recv }

// Params returns the parameters of signature s, or nil.
func (s *Signature) Params() *Tuple { return s.params }

// Results returns the results of signature s, or nil.
func (s *Signature) Results() *Tuple { return s.results }

// IsVariadic reports whether the signature s is variadic.
func (s *Signature) IsVariadic() bool { return s.isVariadic }

// An Interface represents an interface type.
type Interface struct {
	methods []*Func  // explicitly declared methods
	types   []*Named // explicitly embedded types

	allMethods []*Func         // ordered list of methods declared with or embedded in this interface (TODO(gri): replace with mset)
	mset       cachedMethodSet // method set for interface, lazily initialized
}

// NewInterface returns a new interface for the given methods.
func NewInterface(methods []*Func, types []*Named) *Interface {
	typ := new(Interface)

	var mset objset
	for _, m := range methods {
		if mset.insert(m) != nil {
			panic("multiple methods with the same name")
		}
		// set receiver
		// TODO(gri) Ideally, we should use a named type here instead of
		// typ, for less verbose printing of interface method signatures.
		m.typ.(*Signature).recv = NewVar(m.pos, m.pkg, "", typ)
	}
	sort.Sort(byUniqueMethodName(methods))

	var allMethods []*Func
	if types == nil {
		allMethods = methods
	} else {
		allMethods = append(allMethods, methods...)
		for _, t := range types {
			it := t.Underlying().(*Interface)
			for _, tm := range it.allMethods {
				// Make a copy of the method and adjust its receiver type.
				newm := *tm
				newmtyp := *tm.typ.(*Signature)
				newm.typ = &newmtyp
				newmtyp.recv = NewVar(newm.pos, newm.pkg, "", typ)
				allMethods = append(allMethods, &newm)
			}
		}
		sort.Sort(byUniqueMethodName(allMethods))
	}

	typ.methods = methods
	typ.types = types
	typ.allMethods = allMethods
	return typ
}

// NumMethods returns the number of methods of interface t.
func (t *Interface) NumMethods() int { return len(t.allMethods) }

// Method returns the i'th method of interface t for 0 <= i < t.NumMethods().
func (t *Interface) Method(i int) *Func { return t.allMethods[i] }

// A Map represents a map type.
type Map struct {
	key, elem Type
}

// NewMap returns a new map for the given key and element types.
func NewMap(key, elem Type) *Map {
	return &Map{key, elem}
}

// Key returns the key type of map m.
func (m *Map) Key() Type { return m.key }

// Elem returns the element type of map m.
func (m *Map) Elem() Type { return m.elem }

// A Chan represents a channel type.
type Chan struct {
	dir  ast.ChanDir
	elem Type
}

// NewChan returns a new channel type for the given direction and element type.
func NewChan(dir ast.ChanDir, elem Type) *Chan {
	return &Chan{dir, elem}
}

// Dir returns the direction of channel c.
func (c *Chan) Dir() ast.ChanDir { return c.dir }

// Elem returns the element type of channel c.
func (c *Chan) Elem() Type { return c.elem }

// A Named represents a named type.
type Named struct {
	obj         *TypeName       // corresponding declared object
	underlying  Type            // possibly a *Named if !complete; never a *Named if complete
	complete    bool            // if set, the underlying type has been determined
	methods     []*Func         // methods declared for this type (not the method set of this type)
	mset, pmset cachedMethodSet // method set for T, *T, lazily initialized
}

// NewNamed returns a new named type for the given type name, underlying type, and associated methods.
// The underlying type must not be a *Named.
func NewNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	if _, ok := underlying.(*Named); ok {
		panic("types.NewNamed: underlying type must not be *Named")
	}
	typ := &Named{obj: obj, underlying: underlying, complete: underlying != nil, methods: methods}
	if obj.typ == nil {
		obj.typ = typ
	}
	return typ
}

// TypeName returns the type name for the named type t.
func (t *Named) Obj() *TypeName { return t.obj }

// NumMethods returns the number of explicit methods whose receiver is named type t.
func (t *Named) NumMethods() int { return len(t.methods) }

// Method returns the i'th method of named type t for 0 <= i < t.NumMethods().
func (t *Named) Method(i int) *Func { return t.methods[i] }

// SetUnderlying sets the underlying type and marks t as complete.
// TODO(gri) determine if there's a better solution rather than providing this function
func (t *Named) SetUnderlying(underlying Type) {
	if underlying == nil {
		panic("types.Named.SetUnderlying: underlying type must not be nil")
	}
	if _, ok := underlying.(*Named); ok {
		panic("types.Named.SetUnderlying: underlying type must not be *Named")
	}
	t.underlying = underlying
	t.complete = true
}

// AddMethod adds method m unless it is already in the method list.
// TODO(gri) find a better solution instead of providing this function
func (t *Named) AddMethod(m *Func) {
	if i, _ := lookupMethod(t.methods, m.pkg, m.name); i < 0 {
		t.methods = append(t.methods, m)
	}
}

// Implementations for Type methods.

func (t *Basic) Underlying() Type     { return t }
func (t *Array) Underlying() Type     { return t }
func (t *Slice) Underlying() Type     { return t }
func (t *Struct) Underlying() Type    { return t }
func (t *Pointer) Underlying() Type   { return t }
func (t *Tuple) Underlying() Type     { return t }
func (t *Signature) Underlying() Type { return t }
func (t *Interface) Underlying() Type { return t }
func (t *Map) Underlying() Type       { return t }
func (t *Chan) Underlying() Type      { return t }
func (t *Named) Underlying() Type     { return t.underlying }

func (t *Basic) MethodSet() *MethodSet  { return &emptyMethodSet }
func (t *Array) MethodSet() *MethodSet  { return &emptyMethodSet }
func (t *Slice) MethodSet() *MethodSet  { return &emptyMethodSet }
func (t *Struct) MethodSet() *MethodSet { return t.mset.of(t) }
func (t *Pointer) MethodSet() *MethodSet {
	if named, _ := t.base.(*Named); named != nil {
		// Avoid recomputing mset(*T) for each distinct Pointer
		// instance whose underlying type is a named type.
		return named.pmset.of(t)
	}
	return t.mset.of(t)
}
func (t *Tuple) MethodSet() *MethodSet     { return &emptyMethodSet }
func (t *Signature) MethodSet() *MethodSet { return &emptyMethodSet }
func (t *Interface) MethodSet() *MethodSet { return t.mset.of(t) }
func (t *Map) MethodSet() *MethodSet       { return &emptyMethodSet }
func (t *Chan) MethodSet() *MethodSet      { return &emptyMethodSet }
func (t *Named) MethodSet() *MethodSet     { return t.mset.of(t) }

func (t *Basic) String() string     { return TypeString(nil, t) }
func (t *Array) String() string     { return TypeString(nil, t) }
func (t *Slice) String() string     { return TypeString(nil, t) }
func (t *Struct) String() string    { return TypeString(nil, t) }
func (t *Pointer) String() string   { return TypeString(nil, t) }
func (t *Tuple) String() string     { return TypeString(nil, t) }
func (t *Signature) String() string { return TypeString(nil, t) }
func (t *Interface) String() string { return TypeString(nil, t) }
func (t *Map) String() string       { return TypeString(nil, t) }
func (t *Chan) String() string      { return TypeString(nil, t) }
func (t *Named) String() string     { return TypeString(nil, t) }
