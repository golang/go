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

const (
	sliceBound = -1   // slices have Bound=sliceBound
	dddBound   = -100 // arrays declared as [...]T start life with Bound=dddBound
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
	Etype       EType
	Noalg       bool
	Chan        uint8
	Trecur      uint8 // to detect loops
	Printed     bool
	Funarg      bool // on TSTRUCT and TFIELD
	Local       bool // created in this file
	Deferwidth  bool
	Broke       bool // broken type definition.
	Align       uint8
	Haspointers uint8 // 0 unknown, 1 no, 2 yes
	Outnamed    bool  // on TFUNC

	Nod  *Node // canonical OTYPE node
	Orig *Type // original type (type literal or predefined type)

	methods    Fields
	allMethods Fields

	Sym    *Sym
	Vargen int32 // unique name for OTYPE/ONAME
	Lineno int32

	nname  *Node
	Argwid int64

	// most nodes
	Type  *Type // element type for TARRAY, TCHAN, TMAP, TPTRxx
	Width int64

	// TSTRUCT
	fields Fields

	Down *Type // key type in TMAP; next struct in Funarg TSTRUCT

	// TARRAY
	Bound int64 // negative is slice

	// TMAP
	Bucket *Type // internal type representing a hash bucket
	Hmap   *Type // internal type representing a Hmap (map header object)
	Hiter  *Type // internal type representing hash iterator state
	Map    *Type // link from the above 3 internal types back to the map type.

	Maplineno   int32 // first use of TFORW as map key
	Embedlineno int32 // first use of TFORW as embedded type

	// for TFORW, where to copy the eventual value to
	Copyto []*Node
}

// A Field represents a field in a struct or a method in an interface or
// associated with a named type.
type Field struct {
	Nointerface bool
	Embedded    uint8 // embedded field
	Funarg      bool
	Broke       bool // broken field definition
	Isddd       bool // field is ... argument

	Sym   *Sym
	Nname *Node

	Type *Type // field type

	// Offset in bytes of this field or method within its enclosing struct
	// or interface Type.
	Offset int64

	Note *string // literal string annotation
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
	return t
}

// typArray returns a new fixed-length array Type.
func typArray(elem *Type, bound int64) *Type {
	t := typ(TARRAY)
	t.Type = elem
	t.Bound = bound
	return t
}

// typSlice returns a new slice Type.
func typSlice(elem *Type) *Type {
	t := typ(TARRAY)
	t.Type = elem
	t.Bound = sliceBound
	return t
}

// typDDDArray returns a new [...]T array Type.
func typDDDArray(elem *Type) *Type {
	t := typ(TARRAY)
	t.Type = elem
	t.Bound = dddBound
	return t
}

// typChan returns a new chan Type with direction dir.
func typChan(elem *Type, dir uint8) *Type {
	t := typ(TCHAN)
	t.Type = elem
	t.Chan = dir
	return t
}

// typMap returns a new map Type with key type k and element (aka value) type v.
func typMap(k, v *Type) *Type {
	if k != nil {
		checkMapKeyType(k)
	}

	t := typ(TMAP)
	t.Down = k
	t.Type = v
	return t
}

// typPtr returns a new pointer type pointing to t.
func typPtr(elem *Type) *Type {
	t := typ(Tptr)
	t.Type = elem
	t.Width = int64(Widthptr)
	t.Align = uint8(Widthptr)
	return t
}

// typWrapper returns a new wrapper psuedo-type.
func typWrapper(et EType, wrapped *Type) *Type {
	switch et {
	case TCHANARGS, TFUNCARGS, TDDDFIELD:
	default:
		Fatalf("typWrapper bad etype %s", et)
	}
	t := typ(et)
	t.Type = wrapped
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

	case TPTR32, TPTR64, TCHAN, TARRAY:
		elem := substAny(t.Type, types)
		if elem != t.Type {
			t = t.Copy()
			t.Type = elem
		}

	case TMAP:
		key := substAny(t.Down, types)
		val := substAny(t.Type, types)
		if key != t.Down || val != t.Type {
			t = t.Copy()
			t.Down = key
			t.Type = val
		}

	case TFUNC:
		recvs := substAny(t.Recvs(), types)
		params := substAny(t.Params(), types)
		results := substAny(t.Results(), types)
		if recvs != t.Recvs() || params != t.Params() || results != t.Results() {
			// Note that this code has to be aware of the
			// representation underlying Recvs/Results/Params.
			if recvs == t.Recvs() {
				recvs = recvs.Copy()
			}
			if results == t.Results() {
				results = results.Copy()
			}
			t = t.Copy()
			*t.RecvsP() = recvs
			*t.ResultsP() = results
			*t.ParamsP() = params
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

// IterFields returns the first field or method in struct or interface type t
// and an Iter value to continue iterating across the rest.
func IterFields(t *Type) (*Field, Iter) {
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

func (t *Type) wantEtype2(et1, et2 EType) {
	if t.Etype != et1 && t.Etype != et2 {
		Fatalf("want %v or %v, but have %v", et1, et2, t)
	}
}

func (t *Type) RecvsP() **Type {
	t.wantEtype(TFUNC)
	return &t.Type
}

func (t *Type) ParamsP() **Type {
	t.wantEtype(TFUNC)
	return &t.Type.Down.Down
}

func (t *Type) ResultsP() **Type {
	t.wantEtype(TFUNC)
	return &t.Type.Down
}

func (t *Type) Recvs() *Type   { return *t.RecvsP() }
func (t *Type) Params() *Type  { return *t.ParamsP() }
func (t *Type) Results() *Type { return *t.ResultsP() }

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
	return t.Down
}

// Val returns the value type of map type t.
func (t *Type) Val() *Type {
	t.wantEtype(TMAP)
	return t.Type
}

// Elem returns the type of elements of t.
// Usable with pointers, channels, arrays, and slices.
func (t *Type) Elem() *Type {
	switch t.Etype {
	case TPTR32, TPTR64, TCHAN, TARRAY:
	default:
		Fatalf("Type.Elem %s", t.Etype)
	}
	return t.Type
}

// Wrapped returns the type that pseudo-type t wraps.
func (t *Type) Wrapped() *Type {
	switch t.Etype {
	case TCHANARGS, TFUNCARGS, TDDDFIELD:
	default:
		Fatalf("Type.Wrapped %s", t.Etype)
	}
	return t.Type
}

// Nname returns the associated function's nname.
func (t *Type) Nname() *Node {
	t.wantEtype2(TFUNC, TINTERMETH)
	return t.nname
}

// Nname sets the associated function's nname.
func (t *Type) SetNname(n *Node) {
	t.wantEtype2(TFUNC, TINTERMETH)
	t.nname = n
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
	if t.Etype != TSTRUCT && t.Etype != TINTER {
		Fatalf("Fields: type %v does not have fields", t)
	}
	return &t.fields
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
	t.Fields().Set(fields)
}

func (t *Type) isDDDArray() bool {
	if t.Etype != TARRAY {
		return false
	}
	t.checkBound()
	return t.Bound == dddBound
}

// ArgWidth returns the total aligned argument size for a function.
// It includes the receiver, parameters, and results.
func (t *Type) ArgWidth() int64 {
	t.wantEtype(TFUNC)
	return t.Argwid
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
	return Econv(t.Etype)
}

func (t *Type) Equal(u ssa.Type) bool {
	x, ok := u.(*Type)
	return ok && Eqtype(t, x)
}

// Compare compares types for purposes of the SSA back
// end, returning an ssa.Cmp (one of CMPlt, CMPeq, CMPgt).
// The answers are correct for an optimizer
// or code generator, but not for Go source.
// For example, "type gcDrainFlags int" results in
// two Go-different types that Compare equal.
// The order chosen is also arbitrary, only division into
// equivalence classes (Types that compare CMPeq) matters.
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
	// This follows the structure of Eqtype in subr.go
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

	case TPTR32, TPTR64:
		// No special cases for these two, they are handled
		// by the general code after the switch.

	case TSTRUCT:
		if t.Map == nil {
			if x.Map != nil {
				return ssa.CMPlt // nil < non-nil
			}
			// to the fallthrough
		} else if x.Map == nil {
			return ssa.CMPgt // nil > non-nil
		} else if t.Map.Bucket == t {
			// Both have non-nil Map
			// Special case for Maps which include a recursive type where the recursion is not broken with a named type
			if x.Map.Bucket != x {
				return ssa.CMPlt // bucket maps are least
			}
			return t.Map.cmp(x.Map)
		} // If t != t.Map.Bucket, fall through to general case

		fallthrough
	case TINTER:
		t1, ti := IterFields(t)
		x1, xi := IterFields(x)
		for ; t1 != nil && x1 != nil; t1, x1 = ti.Next(), xi.Next() {
			if t1.Embedded != x1.Embedded {
				return cmpForNe(t1.Embedded < x1.Embedded)
			}
			if t1.Note != x1.Note {
				if t1.Note == nil {
					return ssa.CMPlt
				}
				if x1.Note == nil {
					return ssa.CMPgt
				}
				if *t1.Note != *x1.Note {
					return cmpForNe(*t1.Note < *x1.Note)
				}
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
			ta, ia := IterFields(f(t))
			tb, ib := IterFields(f(x))
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
		e := fmt.Sprintf("Do not know how to compare %s with %s", t, x)
		panic(e)
	}

	// Common element type comparison for TARRAY, TCHAN, TPTR32, and TPTR64.
	return t.Elem().cmp(x.Elem())
}

// IsKind reports whether t is a Type of the specified kind.
func (t *Type) IsKind(et EType) bool {
	return t != nil && t.Etype == et
}

func (t *Type) IsBoolean() bool {
	return t.Etype == TBOOL
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

// checkBound enforces that Bound has an acceptable value.
func (t *Type) checkBound() {
	if t.Bound != sliceBound && t.Bound < 0 && t.Bound != dddBound {
		Fatalf("bad TARRAY bounds %d %s", t.Bound, t)
	}
}

func (t *Type) IsSlice() bool {
	t.checkBound()
	return t.Etype == TARRAY && t.Bound == sliceBound
}

func (t *Type) IsArray() bool {
	t.checkBound()
	return t.Etype == TARRAY && t.Bound >= 0
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
	return Ptrto(t)
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

func (t *Type) NumElem() int64 {
	t.wantEtype(TARRAY)
	t.checkBound()
	return t.Bound
}

// SetNumElem sets the number of elements in an array type.
// It should not be used if at all possible.
// Create a new array/slice/dddArray with typX instead.
// TODO(josharian): figure out how to get rid of this.
func (t *Type) SetNumElem(n int64) {
	t.wantEtype(TARRAY)
	t.Bound = n
}

// ChanDir returns the direction of a channel type t.
// The direction will be one of Crecv, Csend, or Cboth.
func (t *Type) ChanDir() uint8 {
	t.wantEtype(TCHAN)
	return t.Chan
}

func (t *Type) IsMemory() bool { return false }
func (t *Type) IsFlags() bool  { return false }
func (t *Type) IsVoid() bool   { return false }

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
