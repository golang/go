// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file provides methods that let us export a Type as an ../ssa:Type.
// We don't export this package's Type directly because it would lead
// to an import cycle with this package and ../ssa.
// TODO: move Type to its own package, then we don't need to dance around import cycles.

package gc

import (
	"cmd/compile/internal/ssa"
	"fmt"
)

// EType describes a kind of type.
type EType uint8

const (
	Txxx = iota

	TINT8
	TUINT8
	TINT16
	TUINT16
	TINT32
	TUINT32
	TINT64
	TUINT64
	TINT
	TUINT
	TUINTPTR

	TCOMPLEX64
	TCOMPLEX128

	TFLOAT32
	TFLOAT64

	TBOOL

	TPTR32
	TPTR64

	TFUNC
	TSLICE
	TARRAY
	TSTRUCT
	TCHAN
	TMAP
	TINTER
	TFORW
	TANY
	TSTRING
	TUNSAFEPTR

	// pseudo-types for literals
	TIDEAL
	TNIL
	TBLANK

	// pseudo-types for frame layout
	TFUNCARGS
	TCHANARGS
	TINTERMETH

	// pseudo-types for import/export
	TDDDFIELD // wrapper: contained type is a ... field

	NTYPE
)

// ChanDir is whether a channel can send, receive, or both.
type ChanDir uint8

func (c ChanDir) CanRecv() bool { return c&Crecv != 0 }
func (c ChanDir) CanSend() bool { return c&Csend != 0 }

const (
	// types of channel
	// must match ../../../../reflect/type.go:/ChanDir
	Crecv ChanDir = 1 << 0
	Csend ChanDir = 1 << 1
	Cboth ChanDir = Crecv | Csend
)

// Types stores pointers to predeclared named types.
//
// It also stores pointers to several special types:
//   - Types[TANY] is the placeholder "any" type recognized by substArgTypes.
//   - Types[TBLANK] represents the blank variable's type.
//   - Types[TIDEAL] represents untyped numeric constants.
//   - Types[TNIL] represents the predeclared "nil" value's type.
//   - Types[TUNSAFEPTR] is package unsafe's Pointer type.
var Types [NTYPE]*Type

var (
	// Predeclared alias types. Kept separate for better error messages.
	bytetype *Type
	runetype *Type

	// Predeclared error interface type.
	errortype *Type

	// Types to represent untyped string and boolean constants.
	idealstring *Type
	idealbool   *Type

	// Types to represent untyped numeric constants.
	// Note: Currently these are only used within the binary export
	// data format. The rest of the compiler only uses Types[TIDEAL].
	idealint     = typ(TIDEAL)
	idealrune    = typ(TIDEAL)
	idealfloat   = typ(TIDEAL)
	idealcomplex = typ(TIDEAL)
)

// A Type represents a Go type.
type Type struct {
	// Extra contains extra etype-specific fields.
	// As an optimization, those etype-specific structs which contain exactly
	// one pointer-shaped field are stored as values rather than pointers when possible.
	//
	// TMAP: *MapType
	// TFORW: *ForwardType
	// TFUNC: *FuncType
	// TINTERMETHOD: InterMethType
	// TSTRUCT: *StructType
	// TINTER: *InterType
	// TDDDFIELD: DDDFieldType
	// TFUNCARGS: FuncArgsType
	// TCHANARGS: ChanArgsType
	// TCHAN: *ChanType
	// TPTR32, TPTR64: PtrType
	// TARRAY: *ArrayType
	// TSLICE: SliceType
	Extra interface{}

	// Width is the width of this Type in bytes.
	Width int64

	methods    Fields
	allMethods Fields

	nod  *Node // canonical OTYPE node
	Orig *Type // original type (type literal or predefined type)

	sliceOf *Type
	ptrTo   *Type

	Sym    *Sym  // symbol containing name, for named types
	Vargen int32 // unique name for OTYPE/ONAME
	Lineno int32 // line at which this type was declared, implicitly or explicitly

	Etype      EType // kind of type
	Noalg      bool  // suppress hash and eq algorithm generation
	Trecur     uint8 // to detect loops
	Local      bool  // created in this file
	Deferwidth bool
	Broke      bool  // broken type definition.
	Align      uint8 // the required alignment of this type, in bytes
	NotInHeap  bool  // type cannot be heap allocated
}

// MapType contains Type fields specific to maps.
type MapType struct {
	Key *Type // Key type
	Val *Type // Val (elem) type

	Bucket *Type // internal struct type representing a hash bucket
	Hmap   *Type // internal struct type representing the Hmap (map header object)
	Hiter  *Type // internal struct type representing hash iterator state
}

// MapType returns t's extra map-specific fields.
func (t *Type) MapType() *MapType {
	t.wantEtype(TMAP)
	return t.Extra.(*MapType)
}

// ForwardType contains Type fields specific to forward types.
type ForwardType struct {
	Copyto      []*Node // where to copy the eventual value to
	Embedlineno int32   // first use of this type as an embedded type
}

// ForwardType returns t's extra forward-type-specific fields.
func (t *Type) ForwardType() *ForwardType {
	t.wantEtype(TFORW)
	return t.Extra.(*ForwardType)
}

// FuncType contains Type fields specific to func types.
type FuncType struct {
	Receiver *Type // function receiver
	Results  *Type // function results
	Params   *Type // function params

	Nname *Node

	// Argwid is the total width of the function receiver, params, and results.
	// It gets calculated via a temporary TFUNCARGS type.
	// Note that TFUNC's Width is Widthptr.
	Argwid int64

	Outnamed bool
}

// FuncType returns t's extra func-specific fields.
func (t *Type) FuncType() *FuncType {
	t.wantEtype(TFUNC)
	return t.Extra.(*FuncType)
}

// InterMethType contains Type fields specific to interface method pseudo-types.
type InterMethType struct {
	Nname *Node
}

// StructType contains Type fields specific to struct types.
type StructType struct {
	fields Fields

	// Maps have three associated internal structs (see struct MapType).
	// Map links such structs back to their map type.
	Map *Type

	Funarg      Funarg // type of function arguments for arg struct
	Haspointers uint8  // 0 unknown, 1 no, 2 yes
}

// Fnstruct records the kind of function argument
type Funarg uint8

const (
	FunargNone    Funarg = iota
	FunargRcvr           // receiver
	FunargParams         // input parameters
	FunargResults        // output results
)

// StructType returns t's extra struct-specific fields.
func (t *Type) StructType() *StructType {
	t.wantEtype(TSTRUCT)
	return t.Extra.(*StructType)
}

// InterType contains Type fields specific to interface types.
type InterType struct {
	fields Fields
}

// PtrType contains Type fields specific to pointer types.
type PtrType struct {
	Elem *Type // element type
}

// DDDFieldType contains Type fields specific to TDDDFIELD types.
type DDDFieldType struct {
	T *Type // reference to a slice type for ... args
}

// ChanArgsType contains Type fields specific to TCHANARGS types.
type ChanArgsType struct {
	T *Type // reference to a chan type whose elements need a width check
}

// // FuncArgsType contains Type fields specific to TFUNCARGS types.
type FuncArgsType struct {
	T *Type // reference to a func type whose elements need a width check
}

// ChanType contains Type fields specific to channel types.
type ChanType struct {
	Elem *Type   // element type
	Dir  ChanDir // channel direction
}

// ChanType returns t's extra channel-specific fields.
func (t *Type) ChanType() *ChanType {
	t.wantEtype(TCHAN)
	return t.Extra.(*ChanType)
}

// ArrayType contains Type fields specific to array types.
type ArrayType struct {
	Elem        *Type // element type
	Bound       int64 // number of elements; <0 if unknown yet
	Haspointers uint8 // 0 unknown, 1 no, 2 yes
}

// SliceType contains Type fields specific to slice types.
type SliceType struct {
	Elem *Type // element type
}

// A Field represents a field in a struct or a method in an interface or
// associated with a named type.
type Field struct {
	Nointerface bool
	Embedded    uint8 // embedded field
	Funarg      Funarg
	Broke       bool // broken field definition
	Isddd       bool // field is ... argument

	Sym   *Sym
	Nname *Node

	Type *Type // field type

	// Offset in bytes of this field or method within its enclosing struct
	// or interface Type.
	Offset int64

	Note string // literal string annotation
}

// End returns the offset of the first byte immediately after this field.
func (f *Field) End() int64 {
	return f.Offset + f.Type.Width
}

// Fields is a pointer to a slice of *Field.
// This saves space in Types that do not have fields or methods
// compared to a simple slice of *Field.
type Fields struct {
	s *[]*Field
}

// Len returns the number of entries in f.
func (f *Fields) Len() int {
	if f.s == nil {
		return 0
	}
	return len(*f.s)
}

// Slice returns the entries in f as a slice.
// Changes to the slice entries will be reflected in f.
func (f *Fields) Slice() []*Field {
	if f.s == nil {
		return nil
	}
	return *f.s
}

// Index returns the i'th element of Fields.
// It panics if f does not have at least i+1 elements.
func (f *Fields) Index(i int) *Field {
	return (*f.s)[i]
}

// Set sets f to a slice.
// This takes ownership of the slice.
func (f *Fields) Set(s []*Field) {
	if len(s) == 0 {
		f.s = nil
	} else {
		// Copy s and take address of t rather than s to avoid
		// allocation in the case where len(s) == 0.
		t := s
		f.s = &t
	}
}

// Append appends entries to f.
func (f *Fields) Append(s ...*Field) {
	if f.s == nil {
		f.s = new([]*Field)
	}
	*f.s = append(*f.s, s...)
}

// typ returns a new Type of the specified kind.
func typ(et EType) *Type {
	t := &Type{
		Etype:  et,
		Width:  BADWIDTH,
		Lineno: lineno,
	}
	t.Orig = t
	// TODO(josharian): lazily initialize some of these?
	switch t.Etype {
	case TMAP:
		t.Extra = new(MapType)
	case TFORW:
		t.Extra = new(ForwardType)
	case TFUNC:
		t.Extra = new(FuncType)
	case TINTERMETH:
		t.Extra = InterMethType{}
	case TSTRUCT:
		t.Extra = new(StructType)
	case TINTER:
		t.Extra = new(InterType)
	case TPTR32, TPTR64:
		t.Extra = PtrType{}
	case TCHANARGS:
		t.Extra = ChanArgsType{}
	case TFUNCARGS:
		t.Extra = FuncArgsType{}
	case TDDDFIELD:
		t.Extra = DDDFieldType{}
	case TCHAN:
		t.Extra = new(ChanType)
	}
	return t
}

// typArray returns a new fixed-length array Type.
func typArray(elem *Type, bound int64) *Type {
	if bound < 0 {
		Fatalf("typArray: invalid bound %v", bound)
	}
	t := typ(TARRAY)
	t.Extra = &ArrayType{Elem: elem, Bound: bound}
	t.NotInHeap = elem.NotInHeap
	return t
}

// typSlice returns the slice Type with element type elem.
func typSlice(elem *Type) *Type {
	if t := elem.sliceOf; t != nil {
		if t.Elem() != elem {
			Fatalf("elem mismatch")
		}
		return t
	}

	t := typ(TSLICE)
	t.Extra = SliceType{Elem: elem}
	elem.sliceOf = t
	return t
}

// typDDDArray returns a new [...]T array Type.
func typDDDArray(elem *Type) *Type {
	t := typ(TARRAY)
	t.Extra = &ArrayType{Elem: elem, Bound: -1}
	t.NotInHeap = elem.NotInHeap
	return t
}

// typChan returns a new chan Type with direction dir.
func typChan(elem *Type, dir ChanDir) *Type {
	t := typ(TCHAN)
	ct := t.ChanType()
	ct.Elem = elem
	ct.Dir = dir
	return t
}

// typMap returns a new map Type with key type k and element (aka value) type v.
func typMap(k, v *Type) *Type {
	t := typ(TMAP)
	mt := t.MapType()
	mt.Key = k
	mt.Val = v
	return t
}

// typPtr returns the pointer type pointing to t.
func typPtr(elem *Type) *Type {
	if t := elem.ptrTo; t != nil {
		if t.Elem() != elem {
			Fatalf("elem mismatch")
		}
		return t
	}

	t := typ(Tptr)
	t.Extra = PtrType{Elem: elem}
	t.Width = int64(Widthptr)
	t.Align = uint8(Widthptr)
	elem.ptrTo = t
	return t
}

// typDDDField returns a new TDDDFIELD type for slice type s.
func typDDDField(s *Type) *Type {
	t := typ(TDDDFIELD)
	t.Extra = DDDFieldType{T: s}
	return t
}

// typChanArgs returns a new TCHANARGS type for channel type c.
func typChanArgs(c *Type) *Type {
	t := typ(TCHANARGS)
	t.Extra = ChanArgsType{T: c}
	return t
}

// typFuncArgs returns a new TFUNCARGS type for func type f.
func typFuncArgs(f *Type) *Type {
	t := typ(TFUNCARGS)
	t.Extra = FuncArgsType{T: f}
	return t
}

func newField() *Field {
	return &Field{
		Offset: BADWIDTH,
	}
}

// substArgTypes substitutes the given list of types for
// successive occurrences of the "any" placeholder in the
// type syntax expression n.Type.
// The result of substArgTypes MUST be assigned back to old, e.g.
// 	n.Left = substArgTypes(n.Left, t1, t2)
func substArgTypes(old *Node, types ...*Type) *Node {
	n := *old // make shallow copy

	for _, t := range types {
		dowidth(t)
	}
	n.Type = substAny(n.Type, &types)
	if len(types) > 0 {
		Fatalf("substArgTypes: too many argument types")
	}
	return &n
}

// substAny walks t, replacing instances of "any" with successive
// elements removed from types.  It returns the substituted type.
func substAny(t *Type, types *[]*Type) *Type {
	if t == nil {
		return nil
	}

	switch t.Etype {
	default:
		// Leave the type unchanged.

	case TANY:
		if len(*types) == 0 {
			Fatalf("substArgTypes: not enough argument types")
		}
		t = (*types)[0]
		*types = (*types)[1:]

	case TPTR32, TPTR64:
		elem := substAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.Copy()
			t.Extra = PtrType{Elem: elem}
		}

	case TARRAY:
		elem := substAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.Copy()
			t.Extra.(*ArrayType).Elem = elem
		}

	case TSLICE:
		elem := substAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.Copy()
			t.Extra = SliceType{Elem: elem}
		}

	case TCHAN:
		elem := substAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.Copy()
			t.Extra.(*ChanType).Elem = elem
		}

	case TMAP:
		key := substAny(t.Key(), types)
		val := substAny(t.Val(), types)
		if key != t.Key() || val != t.Val() {
			t = t.Copy()
			t.Extra.(*MapType).Key = key
			t.Extra.(*MapType).Val = val
		}

	case TFUNC:
		recvs := substAny(t.Recvs(), types)
		params := substAny(t.Params(), types)
		results := substAny(t.Results(), types)
		if recvs != t.Recvs() || params != t.Params() || results != t.Results() {
			t = t.Copy()
			t.FuncType().Receiver = recvs
			t.FuncType().Results = results
			t.FuncType().Params = params
		}

	case TSTRUCT:
		fields := t.FieldSlice()
		var nfs []*Field
		for i, f := range fields {
			nft := substAny(f.Type, types)
			if nft == f.Type {
				continue
			}
			if nfs == nil {
				nfs = append([]*Field(nil), fields...)
			}
			nfs[i] = f.Copy()
			nfs[i].Type = nft
		}
		if nfs != nil {
			t = t.Copy()
			t.SetFields(nfs)
		}
	}

	return t
}

// Copy returns a shallow copy of the Type.
func (t *Type) Copy() *Type {
	if t == nil {
		return nil
	}
	nt := *t
	// copy any *T Extra fields, to avoid aliasing
	switch t.Etype {
	case TMAP:
		x := *t.Extra.(*MapType)
		nt.Extra = &x
	case TFORW:
		x := *t.Extra.(*ForwardType)
		nt.Extra = &x
	case TFUNC:
		x := *t.Extra.(*FuncType)
		nt.Extra = &x
	case TSTRUCT:
		x := *t.Extra.(*StructType)
		nt.Extra = &x
	case TINTER:
		x := *t.Extra.(*InterType)
		nt.Extra = &x
	case TCHAN:
		x := *t.Extra.(*ChanType)
		nt.Extra = &x
	case TARRAY:
		x := *t.Extra.(*ArrayType)
		nt.Extra = &x
	}
	// TODO(mdempsky): Find out why this is necessary and explain.
	if t.Orig == t {
		nt.Orig = &nt
	}
	return &nt
}

func (f *Field) Copy() *Field {
	nf := *f
	return &nf
}

// Iter provides an abstraction for iterating across struct fields and
// interface methods.
type Iter struct {
	s []*Field
}

// iterFields returns the first field or method in struct or interface type t
// and an Iter value to continue iterating across the rest.
func iterFields(t *Type) (*Field, Iter) {
	return t.Fields().Iter()
}

// Iter returns the first field in fs and an Iter value to continue iterating
// across its successor fields.
// Deprecated: New code should use Slice instead.
func (fs *Fields) Iter() (*Field, Iter) {
	i := Iter{s: fs.Slice()}
	f := i.Next()
	return f, i
}

// Next returns the next field or method, if any.
func (i *Iter) Next() *Field {
	if len(i.s) == 0 {
		return nil
	}
	f := i.s[0]
	i.s = i.s[1:]
	return f
}

func (t *Type) wantEtype(et EType) {
	if t.Etype != et {
		Fatalf("want %v, but have %v", et, t)
	}
}

func (t *Type) Recvs() *Type   { return t.FuncType().Receiver }
func (t *Type) Params() *Type  { return t.FuncType().Params }
func (t *Type) Results() *Type { return t.FuncType().Results }

// Recv returns the receiver of function type t, if any.
func (t *Type) Recv() *Field {
	s := t.Recvs()
	if s.NumFields() == 0 {
		return nil
	}
	return s.Field(0)
}

// recvsParamsResults stores the accessor functions for a function Type's
// receiver, parameters, and result parameters, in that order.
// It can be used to iterate over all of a function's parameter lists.
var recvsParamsResults = [3]func(*Type) *Type{
	(*Type).Recvs, (*Type).Params, (*Type).Results,
}

// paramsResults is like recvsParamsResults, but omits receiver parameters.
var paramsResults = [2]func(*Type) *Type{
	(*Type).Params, (*Type).Results,
}

// Key returns the key type of map type t.
func (t *Type) Key() *Type {
	t.wantEtype(TMAP)
	return t.Extra.(*MapType).Key
}

// Val returns the value type of map type t.
func (t *Type) Val() *Type {
	t.wantEtype(TMAP)
	return t.Extra.(*MapType).Val
}

// Elem returns the type of elements of t.
// Usable with pointers, channels, arrays, and slices.
func (t *Type) Elem() *Type {
	switch t.Etype {
	case TPTR32, TPTR64:
		return t.Extra.(PtrType).Elem
	case TARRAY:
		return t.Extra.(*ArrayType).Elem
	case TSLICE:
		return t.Extra.(SliceType).Elem
	case TCHAN:
		return t.Extra.(*ChanType).Elem
	}
	Fatalf("Type.Elem %s", t.Etype)
	return nil
}

// DDDField returns the slice ... type for TDDDFIELD type t.
func (t *Type) DDDField() *Type {
	t.wantEtype(TDDDFIELD)
	return t.Extra.(DDDFieldType).T
}

// ChanArgs returns the channel type for TCHANARGS type t.
func (t *Type) ChanArgs() *Type {
	t.wantEtype(TCHANARGS)
	return t.Extra.(ChanArgsType).T
}

// FuncArgs returns the channel type for TFUNCARGS type t.
func (t *Type) FuncArgs() *Type {
	t.wantEtype(TFUNCARGS)
	return t.Extra.(FuncArgsType).T
}

// Nname returns the associated function's nname.
func (t *Type) Nname() *Node {
	switch t.Etype {
	case TFUNC:
		return t.Extra.(*FuncType).Nname
	case TINTERMETH:
		return t.Extra.(InterMethType).Nname
	}
	Fatalf("Type.Nname %v %v", t.Etype, t)
	return nil
}

// Nname sets the associated function's nname.
func (t *Type) SetNname(n *Node) {
	switch t.Etype {
	case TFUNC:
		t.Extra.(*FuncType).Nname = n
	case TINTERMETH:
		t.Extra = InterMethType{Nname: n}
	default:
		Fatalf("Type.SetNname %v %v", t.Etype, t)
	}
}

// IsFuncArgStruct reports whether t is a struct representing function parameters.
func (t *Type) IsFuncArgStruct() bool {
	return t.Etype == TSTRUCT && t.Extra.(*StructType).Funarg != FunargNone
}

func (t *Type) Methods() *Fields {
	// TODO(mdempsky): Validate t?
	return &t.methods
}

func (t *Type) AllMethods() *Fields {
	// TODO(mdempsky): Validate t?
	return &t.allMethods
}

func (t *Type) Fields() *Fields {
	switch t.Etype {
	case TSTRUCT:
		return &t.Extra.(*StructType).fields
	case TINTER:
		return &t.Extra.(*InterType).fields
	}
	Fatalf("Fields: type %v does not have fields", t)
	return nil
}

// Field returns the i'th field/method of struct/interface type t.
func (t *Type) Field(i int) *Field {
	return t.Fields().Slice()[i]
}

// FieldSlice returns a slice of containing all fields/methods of
// struct/interface type t.
func (t *Type) FieldSlice() []*Field {
	return t.Fields().Slice()
}

// SetFields sets struct/interface type t's fields/methods to fields.
func (t *Type) SetFields(fields []*Field) {
	for _, f := range fields {
		// If type T contains a field F with a go:notinheap
		// type, then T must also be go:notinheap. Otherwise,
		// you could heap allocate T and then get a pointer F,
		// which would be a heap pointer to a go:notinheap
		// type.
		if f.Type != nil && f.Type.NotInHeap {
			t.NotInHeap = true
			break
		}
	}
	t.Fields().Set(fields)
}

func (t *Type) isDDDArray() bool {
	if t.Etype != TARRAY {
		return false
	}
	return t.Extra.(*ArrayType).Bound < 0
}

// ArgWidth returns the total aligned argument size for a function.
// It includes the receiver, parameters, and results.
func (t *Type) ArgWidth() int64 {
	t.wantEtype(TFUNC)
	return t.Extra.(*FuncType).Argwid
}

func (t *Type) Size() int64 {
	dowidth(t)
	return t.Width
}

func (t *Type) Alignment() int64 {
	dowidth(t)
	return int64(t.Align)
}

func (t *Type) SimpleString() string {
	return t.Etype.String()
}

// Compare compares types for purposes of the SSA back
// end, returning an ssa.Cmp (one of CMPlt, CMPeq, CMPgt).
// The answers are correct for an optimizer
// or code generator, but not necessarily typechecking.
// The order chosen is arbitrary, only consistency and division
// into equivalence classes (Types that compare CMPeq) matters.
func (t *Type) Compare(u ssa.Type) ssa.Cmp {
	x, ok := u.(*Type)
	// ssa.CompilerType is smaller than gc.Type
	// bare pointer equality is easy.
	if !ok {
		return ssa.CMPgt
	}
	if x == t {
		return ssa.CMPeq
	}
	return t.cmp(x)
}

func cmpForNe(x bool) ssa.Cmp {
	if x {
		return ssa.CMPlt
	}
	return ssa.CMPgt
}

func (r *Sym) cmpsym(s *Sym) ssa.Cmp {
	if r == s {
		return ssa.CMPeq
	}
	if r == nil {
		return ssa.CMPlt
	}
	if s == nil {
		return ssa.CMPgt
	}
	// Fast sort, not pretty sort
	if len(r.Name) != len(s.Name) {
		return cmpForNe(len(r.Name) < len(s.Name))
	}
	if r.Pkg != s.Pkg {
		if len(r.Pkg.Prefix) != len(s.Pkg.Prefix) {
			return cmpForNe(len(r.Pkg.Prefix) < len(s.Pkg.Prefix))
		}
		if r.Pkg.Prefix != s.Pkg.Prefix {
			return cmpForNe(r.Pkg.Prefix < s.Pkg.Prefix)
		}
	}
	if r.Name != s.Name {
		return cmpForNe(r.Name < s.Name)
	}
	return ssa.CMPeq
}

// cmp compares two *Types t and x, returning ssa.CMPlt,
// ssa.CMPeq, ssa.CMPgt as t<x, t==x, t>x, for an arbitrary
// and optimizer-centric notion of comparison.
func (t *Type) cmp(x *Type) ssa.Cmp {
	// This follows the structure of eqtype in subr.go
	// with two exceptions.
	// 1. Symbols are compared more carefully because a <,=,> result is desired.
	// 2. Maps are treated specially to avoid endless recursion -- maps
	//    contain an internal data type not expressible in Go source code.
	if t == x {
		return ssa.CMPeq
	}
	if t == nil {
		return ssa.CMPlt
	}
	if x == nil {
		return ssa.CMPgt
	}

	if t.Etype != x.Etype {
		return cmpForNe(t.Etype < x.Etype)
	}

	if t.Sym != nil || x.Sym != nil {
		// Special case: we keep byte and uint8 separate
		// for error messages. Treat them as equal.
		switch t.Etype {
		case TUINT8:
			if (t == Types[TUINT8] || t == bytetype) && (x == Types[TUINT8] || x == bytetype) {
				return ssa.CMPeq
			}

		case TINT32:
			if (t == Types[runetype.Etype] || t == runetype) && (x == Types[runetype.Etype] || x == runetype) {
				return ssa.CMPeq
			}
		}
	}

	if c := t.Sym.cmpsym(x.Sym); c != ssa.CMPeq {
		return c
	}

	if x.Sym != nil {
		// Syms non-nil, if vargens match then equal.
		if t.Vargen != x.Vargen {
			return cmpForNe(t.Vargen < x.Vargen)
		}
		return ssa.CMPeq
	}
	// both syms nil, look at structure below.

	switch t.Etype {
	case TBOOL, TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128, TUNSAFEPTR, TUINTPTR,
		TINT8, TINT16, TINT32, TINT64, TINT, TUINT8, TUINT16, TUINT32, TUINT64, TUINT:
		return ssa.CMPeq
	}

	switch t.Etype {
	case TMAP:
		if c := t.Key().cmp(x.Key()); c != ssa.CMPeq {
			return c
		}
		return t.Val().cmp(x.Val())

	case TPTR32, TPTR64, TSLICE:
		// No special cases for these, they are handled
		// by the general code after the switch.

	case TSTRUCT:
		if t.StructType().Map == nil {
			if x.StructType().Map != nil {
				return ssa.CMPlt // nil < non-nil
			}
			// to the fallthrough
		} else if x.StructType().Map == nil {
			return ssa.CMPgt // nil > non-nil
		} else if t.StructType().Map.MapType().Bucket == t {
			// Both have non-nil Map
			// Special case for Maps which include a recursive type where the recursion is not broken with a named type
			if x.StructType().Map.MapType().Bucket != x {
				return ssa.CMPlt // bucket maps are least
			}
			return t.StructType().Map.cmp(x.StructType().Map)
		} else if x.StructType().Map.MapType().Bucket == x {
			return ssa.CMPgt // bucket maps are least
		} // If t != t.Map.Bucket, fall through to general case

		fallthrough
	case TINTER:
		t1, ti := iterFields(t)
		x1, xi := iterFields(x)
		for ; t1 != nil && x1 != nil; t1, x1 = ti.Next(), xi.Next() {
			if t1.Embedded != x1.Embedded {
				return cmpForNe(t1.Embedded < x1.Embedded)
			}
			if t1.Note != x1.Note {
				return cmpForNe(t1.Note < x1.Note)
			}
			if c := t1.Sym.cmpsym(x1.Sym); c != ssa.CMPeq {
				return c
			}
			if c := t1.Type.cmp(x1.Type); c != ssa.CMPeq {
				return c
			}
		}
		if t1 != x1 {
			return cmpForNe(t1 == nil)
		}
		return ssa.CMPeq

	case TFUNC:
		for _, f := range recvsParamsResults {
			// Loop over fields in structs, ignoring argument names.
			ta, ia := iterFields(f(t))
			tb, ib := iterFields(f(x))
			for ; ta != nil && tb != nil; ta, tb = ia.Next(), ib.Next() {
				if ta.Isddd != tb.Isddd {
					return cmpForNe(!ta.Isddd)
				}
				if c := ta.Type.cmp(tb.Type); c != ssa.CMPeq {
					return c
				}
			}
			if ta != tb {
				return cmpForNe(ta == nil)
			}
		}
		return ssa.CMPeq

	case TARRAY:
		if t.NumElem() != x.NumElem() {
			return cmpForNe(t.NumElem() < x.NumElem())
		}

	case TCHAN:
		if t.ChanDir() != x.ChanDir() {
			return cmpForNe(t.ChanDir() < x.ChanDir())
		}

	default:
		e := fmt.Sprintf("Do not know how to compare %v with %v", t, x)
		panic(e)
	}

	// Common element type comparison for TARRAY, TCHAN, TPTR32, TPTR64, and TSLICE.
	return t.Elem().cmp(x.Elem())
}

// IsKind reports whether t is a Type of the specified kind.
func (t *Type) IsKind(et EType) bool {
	return t != nil && t.Etype == et
}

func (t *Type) IsBoolean() bool {
	return t.Etype == TBOOL
}

var unsignedEType = [...]EType{
	TINT8:    TUINT8,
	TUINT8:   TUINT8,
	TINT16:   TUINT16,
	TUINT16:  TUINT16,
	TINT32:   TUINT32,
	TUINT32:  TUINT32,
	TINT64:   TUINT64,
	TUINT64:  TUINT64,
	TINT:     TUINT,
	TUINT:    TUINT,
	TUINTPTR: TUINTPTR,
}

// toUnsigned returns the unsigned equivalent of integer type t.
func (t *Type) toUnsigned() *Type {
	if !t.IsInteger() {
		Fatalf("unsignedType(%v)", t)
	}
	return Types[unsignedEType[t.Etype]]
}

func (t *Type) IsInteger() bool {
	switch t.Etype {
	case TINT8, TUINT8, TINT16, TUINT16, TINT32, TUINT32, TINT64, TUINT64, TINT, TUINT, TUINTPTR:
		return true
	}
	return false
}

func (t *Type) IsSigned() bool {
	switch t.Etype {
	case TINT8, TINT16, TINT32, TINT64, TINT:
		return true
	}
	return false
}

func (t *Type) IsFloat() bool {
	return t.Etype == TFLOAT32 || t.Etype == TFLOAT64
}

func (t *Type) IsComplex() bool {
	return t.Etype == TCOMPLEX64 || t.Etype == TCOMPLEX128
}

// IsPtr reports whether t is a regular Go pointer type.
// This does not include unsafe.Pointer.
func (t *Type) IsPtr() bool {
	return t.Etype == TPTR32 || t.Etype == TPTR64
}

// IsUnsafePtr reports whether t is an unsafe pointer.
func (t *Type) IsUnsafePtr() bool {
	return t.Etype == TUNSAFEPTR
}

// IsPtrShaped reports whether t is represented by a single machine pointer.
// In addition to regular Go pointer types, this includes map, channel, and
// function types and unsafe.Pointer. It does not include array or struct types
// that consist of a single pointer shaped type.
// TODO(mdempsky): Should it? See golang.org/issue/15028.
func (t *Type) IsPtrShaped() bool {
	return t.Etype == TPTR32 || t.Etype == TPTR64 || t.Etype == TUNSAFEPTR ||
		t.Etype == TMAP || t.Etype == TCHAN || t.Etype == TFUNC
}

func (t *Type) IsString() bool {
	return t.Etype == TSTRING
}

func (t *Type) IsMap() bool {
	return t.Etype == TMAP
}

func (t *Type) IsChan() bool {
	return t.Etype == TCHAN
}

func (t *Type) IsSlice() bool {
	return t.Etype == TSLICE
}

func (t *Type) IsArray() bool {
	return t.Etype == TARRAY
}

func (t *Type) IsStruct() bool {
	return t.Etype == TSTRUCT
}

func (t *Type) IsInterface() bool {
	return t.Etype == TINTER
}

// IsEmptyInterface reports whether t is an empty interface type.
func (t *Type) IsEmptyInterface() bool {
	return t.IsInterface() && t.NumFields() == 0
}

func (t *Type) ElemType() ssa.Type {
	// TODO(josharian): If Type ever moves to a shared
	// internal package, remove this silly wrapper.
	return t.Elem()
}
func (t *Type) PtrTo() ssa.Type {
	return ptrto(t)
}

func (t *Type) NumFields() int {
	return t.Fields().Len()
}
func (t *Type) FieldType(i int) ssa.Type {
	return t.Field(i).Type
}
func (t *Type) FieldOff(i int) int64 {
	return t.Field(i).Offset
}
func (t *Type) FieldName(i int) string {
	return t.Field(i).Sym.Name
}

func (t *Type) NumElem() int64 {
	t.wantEtype(TARRAY)
	at := t.Extra.(*ArrayType)
	if at.Bound < 0 {
		Fatalf("NumElem array %v does not have bound yet", t)
	}
	return at.Bound
}

// SetNumElem sets the number of elements in an array type.
// The only allowed use is on array types created with typDDDArray.
// For other uses, create a new array with typArray instead.
func (t *Type) SetNumElem(n int64) {
	t.wantEtype(TARRAY)
	at := t.Extra.(*ArrayType)
	if at.Bound >= 0 {
		Fatalf("SetNumElem array %v already has bound %d", t, at.Bound)
	}
	at.Bound = n
}

// ChanDir returns the direction of a channel type t.
// The direction will be one of Crecv, Csend, or Cboth.
func (t *Type) ChanDir() ChanDir {
	t.wantEtype(TCHAN)
	return t.Extra.(*ChanType).Dir
}

func (t *Type) IsMemory() bool { return false }
func (t *Type) IsFlags() bool  { return false }
func (t *Type) IsVoid() bool   { return false }
func (t *Type) IsTuple() bool  { return false }

// IsUntyped reports whether t is an untyped type.
func (t *Type) IsUntyped() bool {
	if t == nil {
		return false
	}
	if t == idealstring || t == idealbool {
		return true
	}
	switch t.Etype {
	case TNIL, TIDEAL:
		return true
	}
	return false
}
