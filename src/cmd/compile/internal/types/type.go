// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types

import (
	"cmd/internal/obj"
	"cmd/internal/src"
	"fmt"
)

// Dummy Node so we can refer to *Node without actually
// having a gc.Node. Necessary to break import cycles.
// TODO(gri) try to eliminate soon
type Node struct{ _ int }

//go:generate stringer -type EType -trimprefix T

// EType describes a kind of type.
type EType uint8

const (
	Txxx EType = iota

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

	TPTR
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
	TIDEAL // untyped numeric constants
	TNIL
	TBLANK

	// pseudo-types for frame layout
	TFUNCARGS
	TCHANARGS

	// SSA backend types
	TSSA     // internal types used by SSA backend (flags, memory, etc.)
	TTUPLE   // a pair of types, used by SSA backend
	TRESULTS // multiple types; the result of calling a function or method, with a memory at the end.

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
//   - Types[TNIL] represents the predeclared "nil" value's type.
//   - Types[TUNSAFEPTR] is package unsafe's Pointer type.
var Types [NTYPE]*Type

var (
	// Predeclared alias types. Kept separate for better error messages.
	Bytetype *Type
	Runetype *Type

	// Predeclared error interface type.
	Errortype *Type

	// Types to represent untyped string and boolean constants.
	UntypedString *Type
	UntypedBool   *Type

	// Types to represent untyped numeric constants.
	UntypedInt     = New(TIDEAL)
	UntypedRune    = New(TIDEAL)
	UntypedFloat   = New(TIDEAL)
	UntypedComplex = New(TIDEAL)
)

// A Type represents a Go type.
type Type struct {
	// Extra contains extra etype-specific fields.
	// As an optimization, those etype-specific structs which contain exactly
	// one pointer-shaped field are stored as values rather than pointers when possible.
	//
	// TMAP: *Map
	// TFORW: *Forward
	// TFUNC: *Func
	// TSTRUCT: *Struct
	// TINTER: *Interface
	// TFUNCARGS: FuncArgs
	// TCHANARGS: ChanArgs
	// TCHAN: *Chan
	// TPTR: Ptr
	// TARRAY: *Array
	// TSLICE: Slice
	// TSSA: string
	Extra interface{}

	// Width is the width of this Type in bytes.
	Width int64 // valid if Align > 0

	methods    Fields
	allMethods Fields

	Nod  *Node // canonical OTYPE node
	Orig *Type // original type (type literal or predefined type)

	// Cache of composite types, with this type being the element type.
	Cache struct {
		ptr   *Type // *T, or nil
		slice *Type // []T, or nil
	}

	Sym    *Sym  // symbol containing name, for named types
	Vargen int32 // unique name for OTYPE/ONAME

	Etype EType // kind of type
	Align uint8 // the required alignment of this type, in bytes (0 means Width and Align have not yet been computed)

	flags bitset8
}

const (
	typeNotInHeap  = 1 << iota // type cannot be heap allocated
	typeBroke                  // broken type definition
	typeNoalg                  // suppress hash and eq algorithm generation
	typeDeferwidth             // width computation has been deferred and type is on deferredTypeStack
	typeRecur
)

func (t *Type) NotInHeap() bool  { return t.flags&typeNotInHeap != 0 }
func (t *Type) Broke() bool      { return t.flags&typeBroke != 0 }
func (t *Type) Noalg() bool      { return t.flags&typeNoalg != 0 }
func (t *Type) Deferwidth() bool { return t.flags&typeDeferwidth != 0 }
func (t *Type) Recur() bool      { return t.flags&typeRecur != 0 }

func (t *Type) SetNotInHeap(b bool)  { t.flags.set(typeNotInHeap, b) }
func (t *Type) SetBroke(b bool)      { t.flags.set(typeBroke, b) }
func (t *Type) SetNoalg(b bool)      { t.flags.set(typeNoalg, b) }
func (t *Type) SetDeferwidth(b bool) { t.flags.set(typeDeferwidth, b) }
func (t *Type) SetRecur(b bool)      { t.flags.set(typeRecur, b) }

// Pkg returns the package that t appeared in.
//
// Pkg is only defined for function, struct, and interface types
// (i.e., types with named elements). This information isn't used by
// cmd/compile itself, but we need to track it because it's exposed by
// the go/types API.
func (t *Type) Pkg() *Pkg {
	switch t.Etype {
	case TFUNC:
		return t.Extra.(*Func).pkg
	case TSTRUCT:
		return t.Extra.(*Struct).pkg
	case TINTER:
		return t.Extra.(*Interface).pkg
	default:
		Fatalf("Pkg: unexpected kind: %v", t)
		return nil
	}
}

// SetPkg sets the package that t appeared in.
func (t *Type) SetPkg(pkg *Pkg) {
	switch t.Etype {
	case TFUNC:
		t.Extra.(*Func).pkg = pkg
	case TSTRUCT:
		t.Extra.(*Struct).pkg = pkg
	case TINTER:
		t.Extra.(*Interface).pkg = pkg
	default:
		Fatalf("Pkg: unexpected kind: %v", t)
	}
}

// Map contains Type fields specific to maps.
type Map struct {
	Key  *Type // Key type
	Elem *Type // Val (elem) type

	Bucket *Type // internal struct type representing a hash bucket
	Hmap   *Type // internal struct type representing the Hmap (map header object)
	Hiter  *Type // internal struct type representing hash iterator state
}

// MapType returns t's extra map-specific fields.
func (t *Type) MapType() *Map {
	t.wantEtype(TMAP)
	return t.Extra.(*Map)
}

// Forward contains Type fields specific to forward types.
type Forward struct {
	Copyto      []*Type  // where to copy the eventual value to
	Embedlineno src.XPos // first use of this type as an embedded type
}

// ForwardType returns t's extra forward-type-specific fields.
func (t *Type) ForwardType() *Forward {
	t.wantEtype(TFORW)
	return t.Extra.(*Forward)
}

// Func contains Type fields specific to func types.
type Func struct {
	Receiver *Type // function receiver
	Results  *Type // function results
	Params   *Type // function params

	Nname *Node
	pkg   *Pkg

	// Argwid is the total width of the function receiver, params, and results.
	// It gets calculated via a temporary TFUNCARGS type.
	// Note that TFUNC's Width is Widthptr.
	Argwid int64

	Outnamed bool
}

// FuncType returns t's extra func-specific fields.
func (t *Type) FuncType() *Func {
	t.wantEtype(TFUNC)
	return t.Extra.(*Func)
}

// StructType contains Type fields specific to struct types.
type Struct struct {
	fields Fields
	pkg    *Pkg

	// Maps have three associated internal structs (see struct MapType).
	// Map links such structs back to their map type.
	Map *Type

	Funarg Funarg // type of function arguments for arg struct
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
func (t *Type) StructType() *Struct {
	t.wantEtype(TSTRUCT)
	return t.Extra.(*Struct)
}

// Interface contains Type fields specific to interface types.
type Interface struct {
	Fields Fields
	pkg    *Pkg
}

// Ptr contains Type fields specific to pointer types.
type Ptr struct {
	Elem *Type // element type
}

// ChanArgs contains Type fields specific to TCHANARGS types.
type ChanArgs struct {
	T *Type // reference to a chan type whose elements need a width check
}

// // FuncArgs contains Type fields specific to TFUNCARGS types.
type FuncArgs struct {
	T *Type // reference to a func type whose elements need a width check
}

// Chan contains Type fields specific to channel types.
type Chan struct {
	Elem *Type   // element type
	Dir  ChanDir // channel direction
}

// ChanType returns t's extra channel-specific fields.
func (t *Type) ChanType() *Chan {
	t.wantEtype(TCHAN)
	return t.Extra.(*Chan)
}

type Tuple struct {
	first  *Type
	second *Type
	// Any tuple with a memory type must put that memory type second.
}

// Results are the output from calls that will be late-expanded.
type Results struct {
	Types []*Type // Last element is memory output from call.
}

// Array contains Type fields specific to array types.
type Array struct {
	Elem  *Type // element type
	Bound int64 // number of elements; <0 if unknown yet
}

// Slice contains Type fields specific to slice types.
type Slice struct {
	Elem *Type // element type
}

// A Field represents a field in a struct or a method in an interface or
// associated with a named type.
type Field struct {
	flags bitset8

	Embedded uint8 // embedded field

	Pos  src.XPos
	Sym  *Sym
	Type *Type  // field type
	Note string // literal string annotation

	// For fields that represent function parameters, Nname points
	// to the associated ONAME Node.
	Nname *Node

	// Offset in bytes of this field or method within its enclosing struct
	// or interface Type.
	Offset int64
}

const (
	fieldIsDDD = 1 << iota // field is ... argument
	fieldBroke             // broken field definition
	fieldNointerface
)

func (f *Field) IsDDD() bool       { return f.flags&fieldIsDDD != 0 }
func (f *Field) Broke() bool       { return f.flags&fieldBroke != 0 }
func (f *Field) Nointerface() bool { return f.flags&fieldNointerface != 0 }

func (f *Field) SetIsDDD(b bool)       { f.flags.set(fieldIsDDD, b) }
func (f *Field) SetBroke(b bool)       { f.flags.set(fieldBroke, b) }
func (f *Field) SetNointerface(b bool) { f.flags.set(fieldNointerface, b) }

// End returns the offset of the first byte immediately after this field.
func (f *Field) End() int64 {
	return f.Offset + f.Type.Width
}

// IsMethod reports whether f represents a method rather than a struct field.
func (f *Field) IsMethod() bool {
	return f.Type.Etype == TFUNC && f.Type.Recv() != nil
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

// New returns a new Type of the specified kind.
func New(et EType) *Type {
	t := &Type{
		Etype: et,
		Width: BADWIDTH,
	}
	t.Orig = t
	// TODO(josharian): lazily initialize some of these?
	switch t.Etype {
	case TMAP:
		t.Extra = new(Map)
	case TFORW:
		t.Extra = new(Forward)
	case TFUNC:
		t.Extra = new(Func)
	case TSTRUCT:
		t.Extra = new(Struct)
	case TINTER:
		t.Extra = new(Interface)
	case TPTR:
		t.Extra = Ptr{}
	case TCHANARGS:
		t.Extra = ChanArgs{}
	case TFUNCARGS:
		t.Extra = FuncArgs{}
	case TCHAN:
		t.Extra = new(Chan)
	case TTUPLE:
		t.Extra = new(Tuple)
	case TRESULTS:
		t.Extra = new(Results)
	}
	return t
}

// NewArray returns a new fixed-length array Type.
func NewArray(elem *Type, bound int64) *Type {
	if bound < 0 {
		Fatalf("NewArray: invalid bound %v", bound)
	}
	t := New(TARRAY)
	t.Extra = &Array{Elem: elem, Bound: bound}
	t.SetNotInHeap(elem.NotInHeap())
	return t
}

// NewSlice returns the slice Type with element type elem.
func NewSlice(elem *Type) *Type {
	if t := elem.Cache.slice; t != nil {
		if t.Elem() != elem {
			Fatalf("elem mismatch")
		}
		return t
	}

	t := New(TSLICE)
	t.Extra = Slice{Elem: elem}
	elem.Cache.slice = t
	return t
}

// NewChan returns a new chan Type with direction dir.
func NewChan(elem *Type, dir ChanDir) *Type {
	t := New(TCHAN)
	ct := t.ChanType()
	ct.Elem = elem
	ct.Dir = dir
	return t
}

func NewTuple(t1, t2 *Type) *Type {
	t := New(TTUPLE)
	t.Extra.(*Tuple).first = t1
	t.Extra.(*Tuple).second = t2
	return t
}

func NewResults(types []*Type) *Type {
	t := New(TRESULTS)
	t.Extra.(*Results).Types = types
	return t
}

func newSSA(name string) *Type {
	t := New(TSSA)
	t.Extra = name
	return t
}

// NewMap returns a new map Type with key type k and element (aka value) type v.
func NewMap(k, v *Type) *Type {
	t := New(TMAP)
	mt := t.MapType()
	mt.Key = k
	mt.Elem = v
	return t
}

// NewPtrCacheEnabled controls whether *T Types are cached in T.
// Caching is disabled just before starting the backend.
// This allows the backend to run concurrently.
var NewPtrCacheEnabled = true

// NewPtr returns the pointer type pointing to t.
func NewPtr(elem *Type) *Type {
	if elem == nil {
		Fatalf("NewPtr: pointer to elem Type is nil")
	}

	if t := elem.Cache.ptr; t != nil {
		if t.Elem() != elem {
			Fatalf("NewPtr: elem mismatch")
		}
		return t
	}

	t := New(TPTR)
	t.Extra = Ptr{Elem: elem}
	t.Width = int64(Widthptr)
	t.Align = uint8(Widthptr)
	if NewPtrCacheEnabled {
		elem.Cache.ptr = t
	}
	return t
}

// NewChanArgs returns a new TCHANARGS type for channel type c.
func NewChanArgs(c *Type) *Type {
	t := New(TCHANARGS)
	t.Extra = ChanArgs{T: c}
	return t
}

// NewFuncArgs returns a new TFUNCARGS type for func type f.
func NewFuncArgs(f *Type) *Type {
	t := New(TFUNCARGS)
	t.Extra = FuncArgs{T: f}
	return t
}

func NewField() *Field {
	return &Field{
		Offset: BADWIDTH,
	}
}

// SubstAny walks t, replacing instances of "any" with successive
// elements removed from types.  It returns the substituted type.
func SubstAny(t *Type, types *[]*Type) *Type {
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

	case TPTR:
		elem := SubstAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.copy()
			t.Extra = Ptr{Elem: elem}
		}

	case TARRAY:
		elem := SubstAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.copy()
			t.Extra.(*Array).Elem = elem
		}

	case TSLICE:
		elem := SubstAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.copy()
			t.Extra = Slice{Elem: elem}
		}

	case TCHAN:
		elem := SubstAny(t.Elem(), types)
		if elem != t.Elem() {
			t = t.copy()
			t.Extra.(*Chan).Elem = elem
		}

	case TMAP:
		key := SubstAny(t.Key(), types)
		elem := SubstAny(t.Elem(), types)
		if key != t.Key() || elem != t.Elem() {
			t = t.copy()
			t.Extra.(*Map).Key = key
			t.Extra.(*Map).Elem = elem
		}

	case TFUNC:
		recvs := SubstAny(t.Recvs(), types)
		params := SubstAny(t.Params(), types)
		results := SubstAny(t.Results(), types)
		if recvs != t.Recvs() || params != t.Params() || results != t.Results() {
			t = t.copy()
			t.FuncType().Receiver = recvs
			t.FuncType().Results = results
			t.FuncType().Params = params
		}

	case TSTRUCT:
		// Make a copy of all fields, including ones whose type does not change.
		// This prevents aliasing across functions, which can lead to later
		// fields getting their Offset incorrectly overwritten.
		fields := t.FieldSlice()
		nfs := make([]*Field, len(fields))
		for i, f := range fields {
			nft := SubstAny(f.Type, types)
			nfs[i] = f.Copy()
			nfs[i].Type = nft
		}
		t = t.copy()
		t.SetFields(nfs)
	}

	return t
}

// copy returns a shallow copy of the Type.
func (t *Type) copy() *Type {
	if t == nil {
		return nil
	}
	nt := *t
	// copy any *T Extra fields, to avoid aliasing
	switch t.Etype {
	case TMAP:
		x := *t.Extra.(*Map)
		nt.Extra = &x
	case TFORW:
		x := *t.Extra.(*Forward)
		nt.Extra = &x
	case TFUNC:
		x := *t.Extra.(*Func)
		nt.Extra = &x
	case TSTRUCT:
		x := *t.Extra.(*Struct)
		nt.Extra = &x
	case TINTER:
		x := *t.Extra.(*Interface)
		nt.Extra = &x
	case TCHAN:
		x := *t.Extra.(*Chan)
		nt.Extra = &x
	case TARRAY:
		x := *t.Extra.(*Array)
		nt.Extra = &x
	case TTUPLE, TSSA, TRESULTS:
		Fatalf("ssa types cannot be copied")
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

func (t *Type) wantEtype(et EType) {
	if t.Etype != et {
		Fatalf("want %v, but have %v", et, t)
	}
}

func (t *Type) Recvs() *Type   { return t.FuncType().Receiver }
func (t *Type) Params() *Type  { return t.FuncType().Params }
func (t *Type) Results() *Type { return t.FuncType().Results }

func (t *Type) NumRecvs() int   { return t.FuncType().Receiver.NumFields() }
func (t *Type) NumParams() int  { return t.FuncType().Params.NumFields() }
func (t *Type) NumResults() int { return t.FuncType().Results.NumFields() }

// IsVariadic reports whether function type t is variadic.
func (t *Type) IsVariadic() bool {
	n := t.NumParams()
	return n > 0 && t.Params().Field(n-1).IsDDD()
}

// Recv returns the receiver of function type t, if any.
func (t *Type) Recv() *Field {
	s := t.Recvs()
	if s.NumFields() == 0 {
		return nil
	}
	return s.Field(0)
}

// RecvsParamsResults stores the accessor functions for a function Type's
// receiver, parameters, and result parameters, in that order.
// It can be used to iterate over all of a function's parameter lists.
var RecvsParamsResults = [3]func(*Type) *Type{
	(*Type).Recvs, (*Type).Params, (*Type).Results,
}

// RecvsParams is like RecvsParamsResults, but omits result parameters.
var RecvsParams = [2]func(*Type) *Type{
	(*Type).Recvs, (*Type).Params,
}

// ParamsResults is like RecvsParamsResults, but omits receiver parameters.
var ParamsResults = [2]func(*Type) *Type{
	(*Type).Params, (*Type).Results,
}

// Key returns the key type of map type t.
func (t *Type) Key() *Type {
	t.wantEtype(TMAP)
	return t.Extra.(*Map).Key
}

// Elem returns the type of elements of t.
// Usable with pointers, channels, arrays, slices, and maps.
func (t *Type) Elem() *Type {
	switch t.Etype {
	case TPTR:
		return t.Extra.(Ptr).Elem
	case TARRAY:
		return t.Extra.(*Array).Elem
	case TSLICE:
		return t.Extra.(Slice).Elem
	case TCHAN:
		return t.Extra.(*Chan).Elem
	case TMAP:
		return t.Extra.(*Map).Elem
	}
	Fatalf("Type.Elem %s", t.Etype)
	return nil
}

// ChanArgs returns the channel type for TCHANARGS type t.
func (t *Type) ChanArgs() *Type {
	t.wantEtype(TCHANARGS)
	return t.Extra.(ChanArgs).T
}

// FuncArgs returns the func type for TFUNCARGS type t.
func (t *Type) FuncArgs() *Type {
	t.wantEtype(TFUNCARGS)
	return t.Extra.(FuncArgs).T
}

// Nname returns the associated function's nname.
func (t *Type) Nname() *Node {
	switch t.Etype {
	case TFUNC:
		return t.Extra.(*Func).Nname
	}
	Fatalf("Type.Nname %v %v", t.Etype, t)
	return nil
}

// Nname sets the associated function's nname.
func (t *Type) SetNname(n *Node) {
	switch t.Etype {
	case TFUNC:
		t.Extra.(*Func).Nname = n
	default:
		Fatalf("Type.SetNname %v %v", t.Etype, t)
	}
}

// IsFuncArgStruct reports whether t is a struct representing function parameters.
func (t *Type) IsFuncArgStruct() bool {
	return t.Etype == TSTRUCT && t.Extra.(*Struct).Funarg != FunargNone
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
		return &t.Extra.(*Struct).fields
	case TINTER:
		Dowidth(t)
		return &t.Extra.(*Interface).Fields
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
	// If we've calculated the width of t before,
	// then some other type such as a function signature
	// might now have the wrong type.
	// Rather than try to track and invalidate those,
	// enforce that SetFields cannot be called once
	// t's width has been calculated.
	if t.WidthCalculated() {
		Fatalf("SetFields of %v: width previously calculated", t)
	}
	t.wantEtype(TSTRUCT)
	for _, f := range fields {
		// If type T contains a field F with a go:notinheap
		// type, then T must also be go:notinheap. Otherwise,
		// you could heap allocate T and then get a pointer F,
		// which would be a heap pointer to a go:notinheap
		// type.
		if f.Type != nil && f.Type.NotInHeap() {
			t.SetNotInHeap(true)
			break
		}
	}
	t.Fields().Set(fields)
}

func (t *Type) SetInterface(methods []*Field) {
	t.wantEtype(TINTER)
	t.Methods().Set(methods)
}

func (t *Type) WidthCalculated() bool {
	return t.Align > 0
}

// ArgWidth returns the total aligned argument size for a function.
// It includes the receiver, parameters, and results.
func (t *Type) ArgWidth() int64 {
	t.wantEtype(TFUNC)
	return t.Extra.(*Func).Argwid
}

func (t *Type) Size() int64 {
	if t.Etype == TSSA {
		if t == TypeInt128 {
			return 16
		}
		return 0
	}
	Dowidth(t)
	return t.Width
}

func (t *Type) Alignment() int64 {
	Dowidth(t)
	return int64(t.Align)
}

func (t *Type) SimpleString() string {
	return t.Etype.String()
}

// Cmp is a comparison between values a and b.
// -1 if a < b
//  0 if a == b
//  1 if a > b
type Cmp int8

const (
	CMPlt = Cmp(-1)
	CMPeq = Cmp(0)
	CMPgt = Cmp(1)
)

// Compare compares types for purposes of the SSA back
// end, returning a Cmp (one of CMPlt, CMPeq, CMPgt).
// The answers are correct for an optimizer
// or code generator, but not necessarily typechecking.
// The order chosen is arbitrary, only consistency and division
// into equivalence classes (Types that compare CMPeq) matters.
func (t *Type) Compare(x *Type) Cmp {
	if x == t {
		return CMPeq
	}
	return t.cmp(x)
}

func cmpForNe(x bool) Cmp {
	if x {
		return CMPlt
	}
	return CMPgt
}

func (r *Sym) cmpsym(s *Sym) Cmp {
	if r == s {
		return CMPeq
	}
	if r == nil {
		return CMPlt
	}
	if s == nil {
		return CMPgt
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
	return CMPeq
}

// cmp compares two *Types t and x, returning CMPlt,
// CMPeq, CMPgt as t<x, t==x, t>x, for an arbitrary
// and optimizer-centric notion of comparison.
// TODO(josharian): make this safe for recursive interface types
// and use in signatlist sorting. See issue 19869.
func (t *Type) cmp(x *Type) Cmp {
	// This follows the structure of function identical in identity.go
	// with two exceptions.
	// 1. Symbols are compared more carefully because a <,=,> result is desired.
	// 2. Maps are treated specially to avoid endless recursion -- maps
	//    contain an internal data type not expressible in Go source code.
	if t == x {
		return CMPeq
	}
	if t == nil {
		return CMPlt
	}
	if x == nil {
		return CMPgt
	}

	if t.Etype != x.Etype {
		return cmpForNe(t.Etype < x.Etype)
	}

	if t.Sym != nil || x.Sym != nil {
		// Special case: we keep byte and uint8 separate
		// for error messages. Treat them as equal.
		switch t.Etype {
		case TUINT8:
			if (t == Types[TUINT8] || t == Bytetype) && (x == Types[TUINT8] || x == Bytetype) {
				return CMPeq
			}

		case TINT32:
			if (t == Types[Runetype.Etype] || t == Runetype) && (x == Types[Runetype.Etype] || x == Runetype) {
				return CMPeq
			}
		}
	}

	if c := t.Sym.cmpsym(x.Sym); c != CMPeq {
		return c
	}

	if x.Sym != nil {
		// Syms non-nil, if vargens match then equal.
		if t.Vargen != x.Vargen {
			return cmpForNe(t.Vargen < x.Vargen)
		}
		return CMPeq
	}
	// both syms nil, look at structure below.

	switch t.Etype {
	case TBOOL, TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128, TUNSAFEPTR, TUINTPTR,
		TINT8, TINT16, TINT32, TINT64, TINT, TUINT8, TUINT16, TUINT32, TUINT64, TUINT:
		return CMPeq

	case TSSA:
		tname := t.Extra.(string)
		xname := x.Extra.(string)
		// desire fast sorting, not pretty sorting.
		if len(tname) == len(xname) {
			if tname == xname {
				return CMPeq
			}
			if tname < xname {
				return CMPlt
			}
			return CMPgt
		}
		if len(tname) > len(xname) {
			return CMPgt
		}
		return CMPlt

	case TTUPLE:
		xtup := x.Extra.(*Tuple)
		ttup := t.Extra.(*Tuple)
		if c := ttup.first.Compare(xtup.first); c != CMPeq {
			return c
		}
		return ttup.second.Compare(xtup.second)

	case TRESULTS:
		xResults := x.Extra.(*Results)
		tResults := t.Extra.(*Results)
		xl, tl := len(xResults.Types), len(tResults.Types)
		if tl != xl {
			if tl < xl {
				return CMPlt
			}
			return CMPgt
		}
		for i := 0; i < tl; i++ {
			if c := tResults.Types[i].Compare(xResults.Types[i]); c != CMPeq {
				return c
			}
		}
		return CMPeq

	case TMAP:
		if c := t.Key().cmp(x.Key()); c != CMPeq {
			return c
		}
		return t.Elem().cmp(x.Elem())

	case TPTR, TSLICE:
		// No special cases for these, they are handled
		// by the general code after the switch.

	case TSTRUCT:
		if t.StructType().Map == nil {
			if x.StructType().Map != nil {
				return CMPlt // nil < non-nil
			}
			// to the fallthrough
		} else if x.StructType().Map == nil {
			return CMPgt // nil > non-nil
		} else if t.StructType().Map.MapType().Bucket == t {
			// Both have non-nil Map
			// Special case for Maps which include a recursive type where the recursion is not broken with a named type
			if x.StructType().Map.MapType().Bucket != x {
				return CMPlt // bucket maps are least
			}
			return t.StructType().Map.cmp(x.StructType().Map)
		} else if x.StructType().Map.MapType().Bucket == x {
			return CMPgt // bucket maps are least
		} // If t != t.Map.Bucket, fall through to general case

		tfs := t.FieldSlice()
		xfs := x.FieldSlice()
		for i := 0; i < len(tfs) && i < len(xfs); i++ {
			t1, x1 := tfs[i], xfs[i]
			if t1.Embedded != x1.Embedded {
				return cmpForNe(t1.Embedded < x1.Embedded)
			}
			if t1.Note != x1.Note {
				return cmpForNe(t1.Note < x1.Note)
			}
			if c := t1.Sym.cmpsym(x1.Sym); c != CMPeq {
				return c
			}
			if c := t1.Type.cmp(x1.Type); c != CMPeq {
				return c
			}
		}
		if len(tfs) != len(xfs) {
			return cmpForNe(len(tfs) < len(xfs))
		}
		return CMPeq

	case TINTER:
		tfs := t.FieldSlice()
		xfs := x.FieldSlice()
		for i := 0; i < len(tfs) && i < len(xfs); i++ {
			t1, x1 := tfs[i], xfs[i]
			if c := t1.Sym.cmpsym(x1.Sym); c != CMPeq {
				return c
			}
			if c := t1.Type.cmp(x1.Type); c != CMPeq {
				return c
			}
		}
		if len(tfs) != len(xfs) {
			return cmpForNe(len(tfs) < len(xfs))
		}
		return CMPeq

	case TFUNC:
		for _, f := range RecvsParamsResults {
			// Loop over fields in structs, ignoring argument names.
			tfs := f(t).FieldSlice()
			xfs := f(x).FieldSlice()
			for i := 0; i < len(tfs) && i < len(xfs); i++ {
				ta := tfs[i]
				tb := xfs[i]
				if ta.IsDDD() != tb.IsDDD() {
					return cmpForNe(!ta.IsDDD())
				}
				if c := ta.Type.cmp(tb.Type); c != CMPeq {
					return c
				}
			}
			if len(tfs) != len(xfs) {
				return cmpForNe(len(tfs) < len(xfs))
			}
		}
		return CMPeq

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

	// Common element type comparison for TARRAY, TCHAN, TPTR, and TSLICE.
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

// ToUnsigned returns the unsigned equivalent of integer type t.
func (t *Type) ToUnsigned() *Type {
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
	return t.Etype == TPTR
}

// IsPtrElem reports whether t is the element of a pointer (to t).
func (t *Type) IsPtrElem() bool {
	return t.Cache.ptr != nil
}

// IsUnsafePtr reports whether t is an unsafe pointer.
func (t *Type) IsUnsafePtr() bool {
	return t.Etype == TUNSAFEPTR
}

// IsUintptr reports whether t is an uintptr.
func (t *Type) IsUintptr() bool {
	return t.Etype == TUINTPTR
}

// IsPtrShaped reports whether t is represented by a single machine pointer.
// In addition to regular Go pointer types, this includes map, channel, and
// function types and unsafe.Pointer. It does not include array or struct types
// that consist of a single pointer shaped type.
// TODO(mdempsky): Should it? See golang.org/issue/15028.
func (t *Type) IsPtrShaped() bool {
	return t.Etype == TPTR || t.Etype == TUNSAFEPTR ||
		t.Etype == TMAP || t.Etype == TCHAN || t.Etype == TFUNC
}

// HasNil reports whether the set of values determined by t includes nil.
func (t *Type) HasNil() bool {
	switch t.Etype {
	case TCHAN, TFUNC, TINTER, TMAP, TPTR, TSLICE, TUNSAFEPTR:
		return true
	}
	return false
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

func (t *Type) PtrTo() *Type {
	return NewPtr(t)
}

func (t *Type) NumFields() int {
	return t.Fields().Len()
}
func (t *Type) FieldType(i int) *Type {
	if t.Etype == TTUPLE {
		switch i {
		case 0:
			return t.Extra.(*Tuple).first
		case 1:
			return t.Extra.(*Tuple).second
		default:
			panic("bad tuple index")
		}
	}
	if t.Etype == TRESULTS {
		return t.Extra.(*Results).Types[i]
	}
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
	return t.Extra.(*Array).Bound
}

type componentsIncludeBlankFields bool

const (
	IgnoreBlankFields componentsIncludeBlankFields = false
	CountBlankFields  componentsIncludeBlankFields = true
)

// NumComponents returns the number of primitive elements that compose t.
// Struct and array types are flattened for the purpose of counting.
// All other types (including string, slice, and interface types) count as one element.
// If countBlank is IgnoreBlankFields, then blank struct fields
// (and their comprised elements) are excluded from the count.
// struct { x, y [3]int } has six components; [10]struct{ x, y string } has twenty.
func (t *Type) NumComponents(countBlank componentsIncludeBlankFields) int64 {
	switch t.Etype {
	case TSTRUCT:
		if t.IsFuncArgStruct() {
			Fatalf("NumComponents func arg struct")
		}
		var n int64
		for _, f := range t.FieldSlice() {
			if countBlank == IgnoreBlankFields && f.Sym.IsBlank() {
				continue
			}
			n += f.Type.NumComponents(countBlank)
		}
		return n
	case TARRAY:
		return t.NumElem() * t.Elem().NumComponents(countBlank)
	}
	return 1
}

// SoleComponent returns the only primitive component in t,
// if there is exactly one. Otherwise, it returns nil.
// Components are counted as in NumComponents, including blank fields.
func (t *Type) SoleComponent() *Type {
	switch t.Etype {
	case TSTRUCT:
		if t.IsFuncArgStruct() {
			Fatalf("SoleComponent func arg struct")
		}
		if t.NumFields() != 1 {
			return nil
		}
		return t.Field(0).Type.SoleComponent()
	case TARRAY:
		if t.NumElem() != 1 {
			return nil
		}
		return t.Elem().SoleComponent()
	}
	return t
}

// ChanDir returns the direction of a channel type t.
// The direction will be one of Crecv, Csend, or Cboth.
func (t *Type) ChanDir() ChanDir {
	t.wantEtype(TCHAN)
	return t.Extra.(*Chan).Dir
}

func (t *Type) IsMemory() bool {
	if t == TypeMem || t.Etype == TTUPLE && t.Extra.(*Tuple).second == TypeMem {
		return true
	}
	if t.Etype == TRESULTS {
		if types := t.Extra.(*Results).Types; len(types) > 0 && types[len(types)-1] == TypeMem {
			return true
		}
	}
	return false
}
func (t *Type) IsFlags() bool   { return t == TypeFlags }
func (t *Type) IsVoid() bool    { return t == TypeVoid }
func (t *Type) IsTuple() bool   { return t.Etype == TTUPLE }
func (t *Type) IsResults() bool { return t.Etype == TRESULTS }

// IsUntyped reports whether t is an untyped type.
func (t *Type) IsUntyped() bool {
	if t == nil {
		return false
	}
	if t == UntypedString || t == UntypedBool {
		return true
	}
	switch t.Etype {
	case TNIL, TIDEAL:
		return true
	}
	return false
}

// HasPointers reports whether t contains a heap pointer.
// Note that this function ignores pointers to go:notinheap types.
func (t *Type) HasPointers() bool {
	switch t.Etype {
	case TINT, TUINT, TINT8, TUINT8, TINT16, TUINT16, TINT32, TUINT32, TINT64,
		TUINT64, TUINTPTR, TFLOAT32, TFLOAT64, TCOMPLEX64, TCOMPLEX128, TBOOL, TSSA:
		return false

	case TARRAY:
		if t.NumElem() == 0 { // empty array has no pointers
			return false
		}
		return t.Elem().HasPointers()

	case TSTRUCT:
		for _, t1 := range t.Fields().Slice() {
			if t1.Type.HasPointers() {
				return true
			}
		}
		return false

	case TPTR, TSLICE:
		return !t.Elem().NotInHeap()

	case TTUPLE:
		ttup := t.Extra.(*Tuple)
		return ttup.first.HasPointers() || ttup.second.HasPointers()

	case TRESULTS:
		types := t.Extra.(*Results).Types
		for _, et := range types {
			if et.HasPointers() {
				return true
			}
		}
		return false
	}

	return true
}

func (t *Type) Symbol() *obj.LSym {
	return TypeLinkSym(t)
}

// Tie returns 'T' if t is a concrete type,
// 'I' if t is an interface type, and 'E' if t is an empty interface type.
// It is used to build calls to the conv* and assert* runtime routines.
func (t *Type) Tie() byte {
	if t.IsEmptyInterface() {
		return 'E'
	}
	if t.IsInterface() {
		return 'I'
	}
	return 'T'
}

var recvType *Type

// FakeRecvType returns the singleton type used for interface method receivers.
func FakeRecvType() *Type {
	if recvType == nil {
		recvType = NewPtr(New(TSTRUCT))
	}
	return recvType
}

var (
	// TSSA types. HasPointers assumes these are pointer-free.
	TypeInvalid = newSSA("invalid")
	TypeMem     = newSSA("mem")
	TypeFlags   = newSSA("flags")
	TypeVoid    = newSSA("void")
	TypeInt128  = newSSA("int128")
)
