// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"fmt"
	"go/token"
	"sort"
)

// A Type represents a type of Go.
// All types implement the Type interface.
type Type interface {
	// Underlying returns the underlying type of a type
	// w/o following forwarding chains. Only used by
	// client packages (here for backward-compatibility).
	Underlying() Type

	// Under returns the true expanded underlying type.
	// If it doesn't exist, the result is Typ[Invalid].
	// Under must only be called when a type is known
	// to be fully set up.
	Under() Type

	// String returns a string representation of a type.
	String() string

	// Converters
	// A converter must only be called when a type is
	// known to be fully set up. A converter returns
	// a type's operational type (see comment for optype)
	// or nil if the type is receiver is not of the
	// respective type.
	Basic() *Basic
	Array() *Array
	Slice() *Slice
	Struct() *Struct
	Pointer() *Pointer
	Tuple() *Tuple
	Signature() *Signature
	Sum() *Sum
	Interface() *Interface
	Map() *Map
	Chan() *Chan

	// If the receiver for Named and TypeParam is of
	// the respective type (possibly after unpacking
	// an instance type), these methods return that
	// type. Otherwise the result is nil.
	Named() *Named
	TypeParam() *TypeParam
}

// aType implements default type behavior
type aType struct{}

// These methods must be implemented by each type.
func (aType) Underlying() Type { panic("unreachable") }
func (aType) Under() Type      { panic("unreachable") }
func (aType) String() string   { panic("unreachable") }

// Each type is implementing its version of these methods
// (Basic must implement Basic, etc.), the other methods
// are inherited.
func (aType) Basic() *Basic         { return nil }
func (aType) Array() *Array         { return nil }
func (aType) Slice() *Slice         { return nil }
func (aType) Struct() *Struct       { return nil }
func (aType) Pointer() *Pointer     { return nil }
func (aType) Tuple() *Tuple         { return nil }
func (aType) Signature() *Signature { return nil }
func (aType) Sum() *Sum             { return nil }
func (aType) Interface() *Interface { return nil }
func (aType) Map() *Map             { return nil }
func (aType) Chan() *Chan           { return nil }

func (aType) Named() *Named         { return nil }
func (aType) TypeParam() *TypeParam { return nil }

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
	name string
	aType
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
	aType
}

// NewArray returns a new array type for the given element type and length.
// A negative length indicates an unknown length.
func NewArray(elem Type, len int64) *Array { return &Array{len: len, elem: elem} }

// Len returns the length of array a.
// A negative result indicates an unknown length.
func (a *Array) Len() int64 { return a.len }

// Elem returns element type of array a.
func (a *Array) Elem() Type { return a.elem }

// A Slice represents a slice type.
type Slice struct {
	elem Type
	aType
}

// NewSlice returns a new slice type for the given element type.
func NewSlice(elem Type) *Slice { return &Slice{elem: elem} }

// Elem returns the element type of slice s.
func (s *Slice) Elem() Type { return s.elem }

// A Struct represents a struct type.
type Struct struct {
	fields []*Var
	tags   []string // field tags; nil if there are no tags
	aType
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

// NumFields returns the number of fields in the struct (including blank and embedded fields).
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
	base Type // element type
	aType
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
	aType
}

// NewTuple returns a new tuple for the given variables.
func NewTuple(x ...*Var) *Tuple {
	if len(x) > 0 {
		return &Tuple{vars: x}
	}
	return nil
}

// We cannot rely on the embedded X() *X methods because (*Tuple)(nil)
// is a valid *Tuple value but (*Tuple)(nil).X() would panic without
// these implementations. At the moment we only need X = Basic, Named,
// but add all because missing one leads to very confusing bugs.
// TODO(gri) Don't represent empty tuples with a (*Tuple)(nil) pointer;
//           it's too subtle and causes problems.
func (*Tuple) Basic() *Basic     { return nil }
func (*Tuple) Array() *Array     { return nil }
func (*Tuple) Slice() *Slice     { return nil }
func (*Tuple) Struct() *Struct   { return nil }
func (*Tuple) Pointer() *Pointer { return nil }

// func (*Tuple) Tuple() *Tuple      // implemented below
func (*Tuple) Signature() *Signature { return nil }
func (*Tuple) Sum() *Sum             { return nil }
func (*Tuple) Interface() *Interface { return nil }
func (*Tuple) Map() *Map             { return nil }
func (*Tuple) Chan() *Chan           { return nil }

func (*Tuple) Named() *Named         { return nil }
func (*Tuple) TypeParam() *TypeParam { return nil }

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
// The receiver is ignored when comparing signatures for identity.
type Signature struct {
	// We need to keep the scope in Signature (rather than passing it around
	// and store it in the Func Object) because when type-checking a function
	// literal we call the general type checker which returns a general Type.
	// We then unpack the *Signature and use the scope for the literal body.
	rparams  []*TypeName // reveiver type parameters from left to right; or nil
	tparams  []*TypeName // type parameters from left to right; or nil
	scope    *Scope      // function scope, present for package-local signatures
	recv     *Var        // nil if not a method
	params   *Tuple      // (incoming) parameters from left to right; or nil
	results  *Tuple      // (outgoing) results from left to right; or nil
	variadic bool        // true if the last parameter's type is of the form ...T (or string, for append built-in only)
	aType
}

// NewSignature returns a new function type for the given receiver, parameters,
// and results, either of which may be nil. If variadic is set, the function
// is variadic, it must have at least one parameter, and the last parameter
// must be of unnamed slice type.
func NewSignature(recv *Var, params, results *Tuple, variadic bool) *Signature {
	if variadic {
		n := params.Len()
		if n == 0 {
			panic("types.NewSignature: variadic function must have at least one parameter")
		}
		if _, ok := params.At(n - 1).typ.(*Slice); !ok {
			panic("types.NewSignature: variadic parameter must be of unnamed slice type")
		}
	}
	return &Signature{recv: recv, params: params, results: results, variadic: variadic}
}

// Recv returns the receiver of signature s (if a method), or nil if a
// function. It is ignored when comparing signatures for identity.
//
// For an abstract method, Recv returns the enclosing interface either
// as a *Named or an *Interface. Due to embedding, an interface may
// contain methods whose receiver type is a different interface.
func (s *Signature) Recv() *Var { return s.recv }

// TParams returns the type parameters of signature s, or nil.
func (s *Signature) TParams() []*TypeName { return s.tparams }

// SetTParams sets the type parameters of signature s.
func (s *Signature) SetTParams(tparams []*TypeName) { s.tparams = tparams }

// Params returns the parameters of signature s, or nil.
func (s *Signature) Params() *Tuple { return s.params }

// Results returns the results of signature s, or nil.
func (s *Signature) Results() *Tuple { return s.results }

// Variadic reports whether the signature s is variadic.
func (s *Signature) Variadic() bool { return s.variadic }

// A Sum represents a set of possible types.
// Sums are currently used to represent type lists of interfaces
// and thus the underlying types of type parameters; they are not
// first class types of Go.
type Sum struct {
	types []Type // types are unique
	aType
}

// NewSum returns a new Sum type consisting of the provided
// types if there are more than one. If there is exactly one
// type, it returns that type. If the list of types is empty
// the result is nil.
func NewSum(types []Type) Type {
	if len(types) == 0 {
		return nil
	}

	// What should happen if types contains a sum type?
	// Do we flatten the types list? For now we check
	// and panic. This should not be possible for the
	// current use case of type lists.
	// TODO(gri) Come up with the rules for sum types.
	for _, t := range types {
		if _, ok := t.(*Sum); ok {
			panic("sum type contains sum type - unimplemented")
		}
	}

	if len(types) == 1 {
		return types[0]
	}
	return &Sum{types: types}
}

// is reports whether all types in t satisfy pred.
func (s *Sum) is(pred func(Type) bool) bool {
	if s == nil {
		return false
	}
	for _, t := range s.types {
		if !pred(t) {
			return false
		}
	}
	return true
}

// An Interface represents an interface type.
type Interface struct {
	methods   []*Func // ordered list of explicitly declared methods
	types     Type    // (possibly a Sum) type declared with a type list (TODO(gri) need better field name)
	embeddeds []Type  // ordered list of explicitly embedded types

	allMethods []*Func // ordered list of methods declared with or embedded in this interface (TODO(gri): replace with mset)
	allTypes   Type    // intersection of all embedded and locally declared types  (TODO(gri) need better field name)

	obj Object // type declaration defining this interface; or nil (for better error messages)

	aType
}

// unpack unpacks a type into a list of types.
// TODO(gri) Try to eliminate the need for this function.
func unpack(typ Type) []Type {
	if typ == nil {
		return nil
	}
	if sum := typ.Sum(); sum != nil {
		return sum.types
	}
	return []Type{typ}
}

// is reports whether interface t represents types that all satisfy pred.
func (t *Interface) is(pred func(Type) bool) bool {
	if t.allTypes == nil {
		return false // we must have at least one type! (was bug)
	}
	for _, t := range unpack(t.allTypes) {
		if !pred(t) {
			return false
		}
	}
	return true
}

// emptyInterface represents the empty (completed) interface
var emptyInterface = Interface{allMethods: markComplete}

// markComplete is used to mark an empty interface as completely
// set up by setting the allMethods field to a non-nil empty slice.
var markComplete = make([]*Func, 0)

// NewInterface returns a new (incomplete) interface for the given methods and embedded types.
// Each embedded type must have an underlying type of interface type.
// NewInterface takes ownership of the provided methods and may modify their types by setting
// missing receivers. To compute the method set of the interface, Complete must be called.
//
// Deprecated: Use NewInterfaceType instead which allows any (even non-defined) interface types
// to be embedded. This is necessary for interfaces that embed alias type names referring to
// non-defined (literal) interface types.
func NewInterface(methods []*Func, embeddeds []*Named) *Interface {
	tnames := make([]Type, len(embeddeds))
	for i, t := range embeddeds {
		tnames[i] = t
	}
	return NewInterfaceType(methods, tnames)
}

// NewInterfaceType returns a new (incomplete) interface for the given methods and embedded types.
// Each embedded type must have an underlying type of interface type (this property is not
// verified for defined types, which may be in the process of being set up and which don't
// have a valid underlying type yet).
// NewInterfaceType takes ownership of the provided methods and may modify their types by setting
// missing receivers. To compute the method set of the interface, Complete must be called.
func NewInterfaceType(methods []*Func, embeddeds []Type) *Interface {
	if len(methods) == 0 && len(embeddeds) == 0 {
		return &emptyInterface
	}

	// set method receivers if necessary
	typ := new(Interface)
	for _, m := range methods {
		if sig := m.typ.(*Signature); sig.recv == nil {
			sig.recv = NewVar(m.pos, m.pkg, "", typ)
		}
	}

	// All embedded types should be interfaces; however, defined types
	// may not yet be fully resolved. Only verify that non-defined types
	// are interfaces. This matches the behavior of the code before the
	// fix for #25301 (issue #25596).
	for _, t := range embeddeds {
		if _, ok := t.(*Named); !ok && !IsInterface(t) {
			panic("embedded type is not an interface")
		}
	}

	// sort for API stability
	sort.Sort(byUniqueMethodName(methods))
	sort.Stable(byUniqueTypeName(embeddeds))

	typ.methods = methods
	typ.embeddeds = embeddeds
	return typ
}

// NumExplicitMethods returns the number of explicitly declared methods of interface t.
func (t *Interface) NumExplicitMethods() int { return len(t.methods) }

// ExplicitMethod returns the i'th explicitly declared method of interface t for 0 <= i < t.NumExplicitMethods().
// The methods are ordered by their unique Id.
func (t *Interface) ExplicitMethod(i int) *Func { return t.methods[i] }

// NumEmbeddeds returns the number of embedded types in interface t.
func (t *Interface) NumEmbeddeds() int { return len(t.embeddeds) }

// Embedded returns the i'th embedded defined (*Named) type of interface t for 0 <= i < t.NumEmbeddeds().
// The result is nil if the i'th embedded type is not a defined type.
//
// Deprecated: Use EmbeddedType which is not restricted to defined (*Named) types.
func (t *Interface) Embedded(i int) *Named { tname, _ := t.embeddeds[i].(*Named); return tname }

// EmbeddedType returns the i'th embedded type of interface t for 0 <= i < t.NumEmbeddeds().
func (t *Interface) EmbeddedType(i int) Type { return t.embeddeds[i] }

// NumMethods returns the total number of methods of interface t.
// The interface must have been completed.
func (t *Interface) NumMethods() int { t.assertCompleteness(); return len(t.allMethods) }

func (t *Interface) assertCompleteness() {
	if t.allMethods == nil {
		panic("interface is incomplete")
	}
}

// Method returns the i'th method of interface t for 0 <= i < t.NumMethods().
// The methods are ordered by their unique Id.
// The interface must have been completed.
func (t *Interface) Method(i int) *Func { t.assertCompleteness(); return t.allMethods[i] }

// Empty reports whether t is the empty interface.
func (t *Interface) Empty() bool {
	if t.allMethods != nil {
		// interface is complete - quick test
		// A non-nil allTypes may still be empty and represents the bottom type.
		return len(t.allMethods) == 0 && t.allTypes == nil
	}
	return !t.iterate(func(t *Interface) bool {
		if len(t.methods) > 0 || t.types != nil {
			return true
		}
		return false
	}, nil)
}

// HasTypeList reports whether interface t has a type list, possibly from an embedded type.
func (t *Interface) HasTypeList() bool {
	if t.allMethods != nil {
		// interface is complete - quick test
		return t.allTypes != nil
	}

	return t.iterate(func(t *Interface) bool {
		if t.types != nil {
			return true
		}
		return false
	}, nil)
}

// IsComparable reports whether interface t is or embeds the predeclared interface "comparable".
func (t *Interface) IsComparable() bool {
	if t.allMethods != nil {
		// interface is complete - quick test
		_, m := lookupMethod(t.allMethods, nil, "==")
		return m != nil
	}

	return t.iterate(func(t *Interface) bool {
		_, m := lookupMethod(t.methods, nil, "==")
		return m != nil
	}, nil)
}

// IsConstraint reports t.HasTypeList() || t.IsComparable().
func (t *Interface) IsConstraint() bool {
	if t.allMethods != nil {
		// interface is complete - quick test
		if t.allTypes != nil {
			return true
		}
		_, m := lookupMethod(t.allMethods, nil, "==")
		return m != nil
	}

	return t.iterate(func(t *Interface) bool {
		if t.types != nil {
			return true
		}
		_, m := lookupMethod(t.methods, nil, "==")
		return m != nil
	}, nil)
}

// iterate calls f with t and then with any embedded interface of t, recursively, until f returns true.
// iterate reports whether any call to f returned true.
func (t *Interface) iterate(f func(*Interface) bool, seen map[*Interface]bool) bool {
	if f(t) {
		return true
	}
	for _, e := range t.embeddeds {
		// e should be an interface but be careful (it may be invalid)
		if e := e.Interface(); e != nil {
			// Cyclic interfaces such as "type E interface { E }" are not permitted
			// but they are still constructed and we need to detect such cycles.
			if seen[e] {
				continue
			}
			if seen == nil {
				seen = make(map[*Interface]bool)
			}
			seen[e] = true
			if e.iterate(f, seen) {
				return true
			}
		}
	}
	return false
}

// includes reports whether the interface t includes the type typ
// by checking typ against the _underlying_ type of each if the
// types in its typelist.
// Note: Even though the type list is constructed to only contain
// underlying types, it may also contain type parameters (whose
// underlying types are themselves). After instantiation of the
// interface, those type parameters may be replaced with defined
// types, but we still want the underlying types of those (was bug).
// Alternatively, we could recompute the underlying types once,
// after instantiation.
// TODO(gri) investigate the best approach.
func (t *Interface) includes(typ Type) bool {
	if t.allTypes != nil {
		for _, t := range unpack(t.allTypes) {
			if Identical(t.Under(), typ) {
				return true
			}
		}
	}
	return false
}

// Complete computes the interface's method set. It must be called by users of
// NewInterfaceType and NewInterface after the interface's embedded types are
// fully defined and before using the interface type in any way other than to
// form other types. The interface must not contain duplicate methods or a
// panic occurs. Complete returns the receiver.
func (t *Interface) Complete() *Interface {
	// TODO(gri) consolidate this method with Checker.completeInterface
	if t.allMethods != nil {
		return t
	}

	t.allMethods = markComplete // avoid infinite recursion

	var todo []*Func
	var methods []*Func
	var seen objset
	addMethod := func(m *Func, explicit bool) {
		switch other := seen.insert(m); {
		case other == nil:
			methods = append(methods, m)
		case explicit:
			panic("duplicate method " + m.name)
		default:
			// check method signatures after all locally embedded interfaces are computed
			todo = append(todo, m, other.(*Func))
		}
	}

	for _, m := range t.methods {
		addMethod(m, true)
	}

	allTypes := t.types

	for _, typ := range t.embeddeds {
		utyp := typ.Under()
		etyp := utyp.Interface()
		if etyp == nil {
			if utyp != Typ[Invalid] {
				panic(fmt.Sprintf("%s is not an interface", typ))
			}
			continue
		}
		etyp.Complete()
		for _, m := range etyp.allMethods {
			addMethod(m, false)
		}
		allTypes = intersect(allTypes, etyp.allTypes)
	}

	for i := 0; i < len(todo); i += 2 {
		m := todo[i]
		other := todo[i+1]
		if !Identical(m.typ, other.typ) {
			panic("duplicate method " + m.name)
		}
	}

	if methods != nil {
		sort.Sort(byUniqueMethodName(methods))
		t.allMethods = methods
	}
	t.allTypes = allTypes

	return t
}

// A Map represents a map type.
type Map struct {
	key, elem Type
	aType
}

// NewMap returns a new map for the given key and element types.
func NewMap(key, elem Type) *Map {
	return &Map{key: key, elem: elem}
}

// Key returns the key type of map m.
func (m *Map) Key() Type { return m.key }

// Elem returns the element type of map m.
func (m *Map) Elem() Type { return m.elem }

// A Chan represents a channel type.
type Chan struct {
	dir  ChanDir
	elem Type
	aType
}

// A ChanDir value indicates a channel direction.
type ChanDir int

// The direction of a channel is indicated by one of these constants.
const (
	SendRecv ChanDir = iota
	SendOnly
	RecvOnly
)

// NewChan returns a new channel type for the given direction and element type.
func NewChan(dir ChanDir, elem Type) *Chan {
	return &Chan{dir: dir, elem: elem}
}

// Dir returns the direction of channel c.
func (c *Chan) Dir() ChanDir { return c.dir }

// Elem returns the element type of channel c.
func (c *Chan) Elem() Type { return c.elem }

// A Named represents a named (defined) type.
type Named struct {
	check      *Checker    // for Named.Under implementation
	info       typeInfo    // for cycle detection
	obj        *TypeName   // corresponding declared object
	orig       Type        // type (on RHS of declaration) this *Named type is derived of (for cycle reporting)
	underlying Type        // possibly a *Named during setup; never a *Named once set up completely
	tparams    []*TypeName // type parameters, or nil
	targs      []Type      // type arguments (after instantiation), or nil
	methods    []*Func     // methods declared for this type (not the method set of this type); signatures are type-checked lazily
	aType
}

// NewNamed returns a new named type for the given type name, underlying type, and associated methods.
// If the given type name obj doesn't have a type yet, its type is set to the returned named type.
// The underlying type must not be a *Named.
func NewNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	if _, ok := underlying.(*Named); ok {
		panic("types.NewNamed: underlying type must not be *Named")
	}
	typ := &Named{obj: obj, orig: underlying, underlying: underlying, methods: methods}
	if obj.typ == nil {
		obj.typ = typ
	}
	return typ
}

func (check *Checker) NewNamed(obj *TypeName, underlying Type, methods []*Func) *Named {
	typ := &Named{check: check, obj: obj, orig: underlying, underlying: underlying, methods: methods}
	if obj.typ == nil {
		obj.typ = typ
	}
	return typ
}

// Obj returns the type name for the named type t.
func (t *Named) Obj() *TypeName { return t.obj }

// Converter methods
func (t *Named) Basic() *Basic         { return t.Under().Basic() }
func (t *Named) Array() *Array         { return t.Under().Array() }
func (t *Named) Slice() *Slice         { return t.Under().Slice() }
func (t *Named) Struct() *Struct       { return t.Under().Struct() }
func (t *Named) Pointer() *Pointer     { return t.Under().Pointer() }
func (t *Named) Tuple() *Tuple         { return t.Under().Tuple() }
func (t *Named) Signature() *Signature { return t.Under().Signature() }
func (t *Named) Interface() *Interface { return t.Under().Interface() }
func (t *Named) Map() *Map             { return t.Under().Map() }
func (t *Named) Chan() *Chan           { return t.Under().Chan() }

// func (t *Named) Named() *Named      // declared below
func (t *Named) TypeParam() *TypeParam { return t.Under().TypeParam() }

// TODO(gri) Come up with a better representation and API to distinguish
//           between parameterized instantiated and non-instantiated types.

// TParams returns the type parameters of the named type t, or nil.
// The result is non-nil for an (originally) parameterized type even if it is instantiated.
func (t *Named) TParams() []*TypeName { return t.tparams }

// TArgs returns the type arguments after instantiation of the named type t, or nil if not instantiated.
func (t *Named) TArgs() []Type { return t.targs }

// SetTArgs sets the type arguments of Named.
func (t *Named) SetTArgs(args []Type) { t.targs = args }

// NumMethods returns the number of explicit methods whose receiver is named type t.
func (t *Named) NumMethods() int { return len(t.methods) }

// Method returns the i'th method of named type t for 0 <= i < t.NumMethods().
func (t *Named) Method(i int) *Func { return t.methods[i] }

// SetUnderlying sets the underlying type and marks t as complete.
func (t *Named) SetUnderlying(underlying Type) {
	if underlying == nil {
		panic("types.Named.SetUnderlying: underlying type must not be nil")
	}
	if _, ok := underlying.(*Named); ok {
		panic("types.Named.SetUnderlying: underlying type must not be *Named")
	}
	t.underlying = underlying
}

// AddMethod adds method m unless it is already in the method list.
func (t *Named) AddMethod(m *Func) {
	if i, _ := lookupMethod(t.methods, m.pkg, m.name); i < 0 {
		t.methods = append(t.methods, m)
	}
}

// A TypeParam represents a type parameter type.
type TypeParam struct {
	check *Checker  // for lazy type bound completion
	id    uint64    // unique id
	ptr   bool      // pointer designation
	obj   *TypeName // corresponding type name
	index int       // parameter index
	bound Type      // *Named or *Interface; underlying type is always *Interface
	aType
}

// NewTypeParam returns a new TypeParam.
func (check *Checker) NewTypeParam(ptr bool, obj *TypeName, index int, bound Type) *TypeParam {
	assert(bound != nil)
	typ := &TypeParam{check: check, id: check.nextId, ptr: ptr, obj: obj, index: index, bound: bound}
	check.nextId++
	if obj.typ == nil {
		obj.typ = typ
	}
	return typ
}

func (t *TypeParam) Bound() *Interface {
	iface := t.bound.Interface()
	// use the type bound position if we have one
	pos := token.NoPos
	if n, _ := t.bound.(*Named); n != nil {
		pos = n.obj.pos
	}
	t.check.completeInterface(pos, iface)
	return iface
}

// optype returns a type's operational type. Except for
// type parameters, the operational type is the same
// as the underlying type (as returned by Under). For
// Type parameters, the operational type is determined
// by the corresponding type bound's type list. The
// result may be the bottom or top type, but it is never
// the incoming type parameter.
func optype(typ Type) Type {
	if t := typ.TypeParam(); t != nil {
		// If the optype is typ, return the top type as we have
		// no information. It also prevents infinite recursion
		// via the TypeParam converter methods. This can happen
		// for a type parameter list of the form:
		// (type T interface { type T }).
		// See also issue #39680.
		if u := t.Bound().allTypes; u != nil && u != typ {
			// u != typ and u is a type parameter => u.Under() != typ, so this is ok
			return u.Under()
		}
		return theTop
	}
	return typ
}

// Converter methods
func (t *TypeParam) Basic() *Basic         { return optype(t).Basic() }
func (t *TypeParam) Array() *Array         { return optype(t).Array() }
func (t *TypeParam) Slice() *Slice         { return optype(t).Slice() }
func (t *TypeParam) Struct() *Struct       { return optype(t).Struct() }
func (t *TypeParam) Pointer() *Pointer     { return optype(t).Pointer() }
func (t *TypeParam) Tuple() *Tuple         { return optype(t).Tuple() }
func (t *TypeParam) Signature() *Signature { return optype(t).Signature() }
func (t *TypeParam) Sum() *Sum             { return optype(t).Sum() }
func (t *TypeParam) Interface() *Interface { return optype(t).Interface() }
func (t *TypeParam) Map() *Map             { return optype(t).Map() }
func (t *TypeParam) Chan() *Chan           { return optype(t).Chan() }

// func (t *TypeParam) Named() *Named         // Named does not unpack type parameters
// func (t *TypeParam) TypeParam() *TypeParam // declared below

// An instance represents an instantiated generic type syntactically
// (without expanding the instantiation). Type instances appear only
// during type-checking and are replaced by their fully instantiated
// (expanded) types before the end of type-checking.
type instance struct {
	check   *Checker    // for lazy instantiation
	pos     token.Pos   // position of type instantiation; for error reporting only
	base    *Named      // parameterized type to be instantiated
	targs   []Type      // type arguments
	poslist []token.Pos // position of each targ; for error reporting only
	value   Type        // base(targs...) after instantiation or Typ[Invalid]; nil if not yet set
	aType
}

// Converter methods
func (t *instance) Basic() *Basic         { return t.Under().Basic() }
func (t *instance) Array() *Array         { return t.Under().Array() }
func (t *instance) Slice() *Slice         { return t.Under().Slice() }
func (t *instance) Struct() *Struct       { return t.Under().Struct() }
func (t *instance) Pointer() *Pointer     { return t.Under().Pointer() }
func (t *instance) Tuple() *Tuple         { return t.Under().Tuple() }
func (t *instance) Signature() *Signature { return t.Under().Signature() }
func (t *instance) Sum() *Sum             { return t.Under().Sum() }
func (t *instance) Interface() *Interface { return t.Under().Interface() }
func (t *instance) Map() *Map             { return t.Under().Map() }
func (t *instance) Chan() *Chan           { return t.Under().Chan() }

func (t *instance) Named() *Named         { return t.expand().Named() }
func (t *instance) TypeParam() *TypeParam { return t.expand().TypeParam() }

// expand returns the instantiated (= expanded) type of t.
// The result is either an instantiated *Named type, or
// Typ[Invalid] if there was an error.
func (t *instance) expand() Type {
	v := t.value
	if v == nil {
		v = t.check.instantiate(t.pos, t.base, t.targs, t.poslist)
		if v == nil {
			v = Typ[Invalid]
		}
		t.value = v
	}
	// After instantiation we must have an invalid or a *Named type.
	if debug && v != Typ[Invalid] {
		_ = v.(*Named)
	}
	return v
}

// expand expands a type instance into its instantiated
// type and leaves all other types alone. expand does
// not recurse.
func expand(typ Type) Type {
	if t, _ := typ.(*instance); t != nil {
		return t.expand()
	}
	return typ
}

// expandf is set to expand.
// Call expandf when calling expand causes compile-time cycle error.
var expandf func(Type) Type

func init() { expandf = expand }

// bottom represents the bottom of the type lattice.
// It is the underlying type of a type parameter that
// cannot be satisfied by any type, usually because
// the intersection of type constraints left nothing).
type bottom struct {
	aType
}

// theBottom is the singleton bottom type.
var theBottom = &bottom{}

// top represents the top of the type lattice.
// It is the underlying type of a type parameter that
// can be satisfied by any type (ignoring methods),
// usually because the type constraint has no type
// list.
type top struct {
	aType
}

// theTop is the singleton top type.
var theTop = &top{}

// Type-specific implementations of type converters.
func (t *Basic) Basic() *Basic             { return t }
func (t *Array) Array() *Array             { return t }
func (t *Slice) Slice() *Slice             { return t }
func (t *Struct) Struct() *Struct          { return t }
func (t *Pointer) Pointer() *Pointer       { return t }
func (t *Tuple) Tuple() *Tuple             { return t }
func (t *Signature) Signature() *Signature { return t }
func (t *Sum) Sum() *Sum                   { return t }
func (t *Interface) Interface() *Interface { return t }
func (t *Map) Map() *Map                   { return t }
func (t *Chan) Chan() *Chan                { return t }

func (t *Named) Named() *Named             { return t }
func (t *TypeParam) TypeParam() *TypeParam { return t }

// Type-specific implementations of Underlying.
func (t *Basic) Underlying() Type     { return t }
func (t *Array) Underlying() Type     { return t }
func (t *Slice) Underlying() Type     { return t }
func (t *Struct) Underlying() Type    { return t }
func (t *Pointer) Underlying() Type   { return t }
func (t *Tuple) Underlying() Type     { return t }
func (t *Signature) Underlying() Type { return t }
func (t *Sum) Underlying() Type       { return t }
func (t *Interface) Underlying() Type { return t }
func (t *Map) Underlying() Type       { return t }
func (t *Chan) Underlying() Type      { return t }
func (t *Named) Underlying() Type     { return t.underlying }
func (t *TypeParam) Underlying() Type { return t }
func (t *instance) Underlying() Type  { return t }
func (t *bottom) Underlying() Type    { return t }
func (t *top) Underlying() Type       { return t }

// Type-specific implementations of Under.
func (t *Basic) Under() Type     { return t }
func (t *Array) Under() Type     { return t }
func (t *Slice) Under() Type     { return t }
func (t *Struct) Under() Type    { return t }
func (t *Pointer) Under() Type   { return t }
func (t *Tuple) Under() Type     { return t }
func (t *Signature) Under() Type { return t }
func (t *Sum) Under() Type       { return t } // TODO(gri) is this correct?
func (t *Interface) Under() Type { return t }
func (t *Map) Under() Type       { return t }
func (t *Chan) Under() Type      { return t }

// see decl.go for implementation of Named.Under
func (t *TypeParam) Under() Type { return t }
func (t *instance) Under() Type  { return t.expand().Under() }
func (t *bottom) Under() Type    { return t }
func (t *top) Under() Type       { return t }

// Type-specific implementations of String.
func (t *Basic) String() string     { return TypeString(t, nil) }
func (t *Array) String() string     { return TypeString(t, nil) }
func (t *Slice) String() string     { return TypeString(t, nil) }
func (t *Struct) String() string    { return TypeString(t, nil) }
func (t *Pointer) String() string   { return TypeString(t, nil) }
func (t *Tuple) String() string     { return TypeString(t, nil) }
func (t *Signature) String() string { return TypeString(t, nil) }
func (t *Sum) String() string       { return TypeString(t, nil) }
func (t *Interface) String() string { return TypeString(t, nil) }
func (t *Map) String() string       { return TypeString(t, nil) }
func (t *Chan) String() string      { return TypeString(t, nil) }
func (t *Named) String() string     { return TypeString(t, nil) }
func (t *TypeParam) String() string { return TypeString(t, nil) }
func (t *instance) String() string  { return TypeString(t, nil) }
func (t *bottom) String() string    { return TypeString(t, nil) }
func (t *top) String() string       { return TypeString(t, nil) }
