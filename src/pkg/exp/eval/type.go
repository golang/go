// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"big"
	"go/ast"
	"go/token"
	"log"
	"reflect"
	"sort"
	"unsafe" // For Sizeof
)


// XXX(Spec) The type compatibility section is very confusing because
// it makes it seem like there are three distinct types of
// compatibility: plain compatibility, assignment compatibility, and
// comparison compatibility.  As I understand it, there's really only
// assignment compatibility and comparison and conversion have some
// restrictions and have special meaning in some cases where the types
// are not otherwise assignment compatible.  The comparison
// compatibility section is almost all about the semantics of
// comparison, not the type checking of it, so it would make much more
// sense in the comparison operators section.  The compatibility and
// assignment compatibility sections should be rolled into one.

type Type interface {
	// compat returns whether this type is compatible with another
	// type.  If conv is false, this is normal compatibility,
	// where two named types are compatible only if they are the
	// same named type.  If conv if true, this is conversion
	// compatibility, where two named types are conversion
	// compatible if their definitions are conversion compatible.
	//
	// TODO(austin) Deal with recursive types
	compat(o Type, conv bool) bool
	// lit returns this type's literal.  If this is a named type,
	// this is the unnamed underlying type.  Otherwise, this is an
	// identity operation.
	lit() Type
	// isBoolean returns true if this is a boolean type.
	isBoolean() bool
	// isInteger returns true if this is an integer type.
	isInteger() bool
	// isFloat returns true if this is a floating type.
	isFloat() bool
	// isIdeal returns true if this is an ideal int or float.
	isIdeal() bool
	// Zero returns a new zero value of this type.
	Zero() Value
	// String returns the string representation of this type.
	String() string
	// The position where this type was defined, if any.
	Pos() token.Pos
}

type BoundedType interface {
	Type
	// minVal returns the smallest value of this type.
	minVal() *big.Rat
	// maxVal returns the largest value of this type.
	maxVal() *big.Rat
}

var universePos = token.NoPos

/*
 * Type array maps.  These are used to memoize composite types.
 */

type typeArrayMapEntry struct {
	key  []Type
	v    interface{}
	next *typeArrayMapEntry
}

type typeArrayMap map[uintptr]*typeArrayMapEntry

func hashTypeArray(key []Type) uintptr {
	hash := uintptr(0)
	for _, t := range key {
		hash = hash * 33
		if t == nil {
			continue
		}
		addr := reflect.NewValue(t).(*reflect.PtrValue).Get()
		hash ^= addr
	}
	return hash
}

func newTypeArrayMap() typeArrayMap { return make(map[uintptr]*typeArrayMapEntry) }

func (m typeArrayMap) Get(key []Type) interface{} {
	ent, ok := m[hashTypeArray(key)]
	if !ok {
		return nil
	}

nextEnt:
	for ; ent != nil; ent = ent.next {
		if len(key) != len(ent.key) {
			continue
		}
		for i := 0; i < len(key); i++ {
			if key[i] != ent.key[i] {
				continue nextEnt
			}
		}
		// Found it
		return ent.v
	}

	return nil
}

func (m typeArrayMap) Put(key []Type, v interface{}) interface{} {
	hash := hashTypeArray(key)
	ent := m[hash]

	new := &typeArrayMapEntry{key, v, ent}
	m[hash] = new
	return v
}

/*
 * Common type
 */

type commonType struct{}

func (commonType) isBoolean() bool { return false }

func (commonType) isInteger() bool { return false }

func (commonType) isFloat() bool { return false }

func (commonType) isIdeal() bool { return false }

func (commonType) Pos() token.Pos { return token.NoPos }

/*
 * Bool
 */

type boolType struct {
	commonType
}

var BoolType = universe.DefineType("bool", universePos, &boolType{})

func (t *boolType) compat(o Type, conv bool) bool {
	_, ok := o.lit().(*boolType)
	return ok
}

func (t *boolType) lit() Type { return t }

func (t *boolType) isBoolean() bool { return true }

func (boolType) String() string {
	// Use angle brackets as a convention for printing the
	// underlying, unnamed type.  This should only show up in
	// debug output.
	return "<bool>"
}

func (t *boolType) Zero() Value {
	res := boolV(false)
	return &res
}

/*
 * Uint
 */

type uintType struct {
	commonType

	// 0 for architecture-dependent types
	Bits uint
	// true for uintptr, false for all others
	Ptr  bool
	name string
}

var (
	Uint8Type  = universe.DefineType("uint8", universePos, &uintType{commonType{}, 8, false, "uint8"})
	Uint16Type = universe.DefineType("uint16", universePos, &uintType{commonType{}, 16, false, "uint16"})
	Uint32Type = universe.DefineType("uint32", universePos, &uintType{commonType{}, 32, false, "uint32"})
	Uint64Type = universe.DefineType("uint64", universePos, &uintType{commonType{}, 64, false, "uint64"})

	UintType    = universe.DefineType("uint", universePos, &uintType{commonType{}, 0, false, "uint"})
	UintptrType = universe.DefineType("uintptr", universePos, &uintType{commonType{}, 0, true, "uintptr"})
)

func (t *uintType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*uintType)
	return ok && t == t2
}

func (t *uintType) lit() Type { return t }

func (t *uintType) isInteger() bool { return true }

func (t *uintType) String() string { return "<" + t.name + ">" }

func (t *uintType) Zero() Value {
	switch t.Bits {
	case 0:
		if t.Ptr {
			res := uintptrV(0)
			return &res
		} else {
			res := uintV(0)
			return &res
		}
	case 8:
		res := uint8V(0)
		return &res
	case 16:
		res := uint16V(0)
		return &res
	case 32:
		res := uint32V(0)
		return &res
	case 64:
		res := uint64V(0)
		return &res
	}
	panic("unexpected uint bit count")
}

func (t *uintType) minVal() *big.Rat { return big.NewRat(0, 1) }

func (t *uintType) maxVal() *big.Rat {
	bits := t.Bits
	if bits == 0 {
		if t.Ptr {
			bits = uint(8 * unsafe.Sizeof(uintptr(0)))
		} else {
			bits = uint(8 * unsafe.Sizeof(uint(0)))
		}
	}
	numer := big.NewInt(1)
	numer.Lsh(numer, bits)
	numer.Sub(numer, idealOne)
	return new(big.Rat).SetInt(numer)
}

/*
 * Int
 */

type intType struct {
	commonType

	// XXX(Spec) Numeric types: "There is also a set of
	// architecture-independent basic numeric types whose size
	// depends on the architecture."  Should that be
	// architecture-dependent?

	// 0 for architecture-dependent types
	Bits uint
	name string
}

var (
	Int8Type  = universe.DefineType("int8", universePos, &intType{commonType{}, 8, "int8"})
	Int16Type = universe.DefineType("int16", universePos, &intType{commonType{}, 16, "int16"})
	Int32Type = universe.DefineType("int32", universePos, &intType{commonType{}, 32, "int32"})
	Int64Type = universe.DefineType("int64", universePos, &intType{commonType{}, 64, "int64"})

	IntType = universe.DefineType("int", universePos, &intType{commonType{}, 0, "int"})
)

func (t *intType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*intType)
	return ok && t == t2
}

func (t *intType) lit() Type { return t }

func (t *intType) isInteger() bool { return true }

func (t *intType) String() string { return "<" + t.name + ">" }

func (t *intType) Zero() Value {
	switch t.Bits {
	case 8:
		res := int8V(0)
		return &res
	case 16:
		res := int16V(0)
		return &res
	case 32:
		res := int32V(0)
		return &res
	case 64:
		res := int64V(0)
		return &res

	case 0:
		res := intV(0)
		return &res
	}
	panic("unexpected int bit count")
}

func (t *intType) minVal() *big.Rat {
	bits := t.Bits
	if bits == 0 {
		bits = uint(8 * unsafe.Sizeof(int(0)))
	}
	numer := big.NewInt(-1)
	numer.Lsh(numer, bits-1)
	return new(big.Rat).SetInt(numer)
}

func (t *intType) maxVal() *big.Rat {
	bits := t.Bits
	if bits == 0 {
		bits = uint(8 * unsafe.Sizeof(int(0)))
	}
	numer := big.NewInt(1)
	numer.Lsh(numer, bits-1)
	numer.Sub(numer, idealOne)
	return new(big.Rat).SetInt(numer)
}

/*
 * Ideal int
 */

type idealIntType struct {
	commonType
}

var IdealIntType Type = &idealIntType{}

func (t *idealIntType) compat(o Type, conv bool) bool {
	_, ok := o.lit().(*idealIntType)
	return ok
}

func (t *idealIntType) lit() Type { return t }

func (t *idealIntType) isInteger() bool { return true }

func (t *idealIntType) isIdeal() bool { return true }

func (t *idealIntType) String() string { return "ideal integer" }

func (t *idealIntType) Zero() Value { return &idealIntV{idealZero} }

/*
 * Float
 */

type floatType struct {
	commonType

	// 0 for architecture-dependent type
	Bits uint

	name string
}

var (
	Float32Type = universe.DefineType("float32", universePos, &floatType{commonType{}, 32, "float32"})
	Float64Type = universe.DefineType("float64", universePos, &floatType{commonType{}, 64, "float64"})
)

func (t *floatType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*floatType)
	return ok && t == t2
}

func (t *floatType) lit() Type { return t }

func (t *floatType) isFloat() bool { return true }

func (t *floatType) String() string { return "<" + t.name + ">" }

func (t *floatType) Zero() Value {
	switch t.Bits {
	case 32:
		res := float32V(0)
		return &res
	case 64:
		res := float64V(0)
		return &res
	}
	panic("unexpected float bit count")
}

var maxFloat32Val *big.Rat
var maxFloat64Val *big.Rat
var minFloat32Val *big.Rat
var minFloat64Val *big.Rat

func (t *floatType) minVal() *big.Rat {
	bits := t.Bits
	switch bits {
	case 32:
		return minFloat32Val
	case 64:
		return minFloat64Val
	}
	log.Panicf("unexpected floating point bit count: %d", bits)
	panic("unreachable")
}

func (t *floatType) maxVal() *big.Rat {
	bits := t.Bits
	switch bits {
	case 32:
		return maxFloat32Val
	case 64:
		return maxFloat64Val
	}
	log.Panicf("unexpected floating point bit count: %d", bits)
	panic("unreachable")
}

/*
 * Ideal float
 */

type idealFloatType struct {
	commonType
}

var IdealFloatType Type = &idealFloatType{}

func (t *idealFloatType) compat(o Type, conv bool) bool {
	_, ok := o.lit().(*idealFloatType)
	return ok
}

func (t *idealFloatType) lit() Type { return t }

func (t *idealFloatType) isFloat() bool { return true }

func (t *idealFloatType) isIdeal() bool { return true }

func (t *idealFloatType) String() string { return "ideal float" }

func (t *idealFloatType) Zero() Value { return &idealFloatV{big.NewRat(0, 1)} }

/*
 * String
 */

type stringType struct {
	commonType
}

var StringType = universe.DefineType("string", universePos, &stringType{})

func (t *stringType) compat(o Type, conv bool) bool {
	_, ok := o.lit().(*stringType)
	return ok
}

func (t *stringType) lit() Type { return t }

func (t *stringType) String() string { return "<string>" }

func (t *stringType) Zero() Value {
	res := stringV("")
	return &res
}

/*
 * Array
 */

type ArrayType struct {
	commonType
	Len  int64
	Elem Type
}

var arrayTypes = make(map[int64]map[Type]*ArrayType)

// Two array types are identical if they have identical element types
// and the same array length.

func NewArrayType(len int64, elem Type) *ArrayType {
	ts, ok := arrayTypes[len]
	if !ok {
		ts = make(map[Type]*ArrayType)
		arrayTypes[len] = ts
	}
	t, ok := ts[elem]
	if !ok {
		t = &ArrayType{commonType{}, len, elem}
		ts[elem] = t
	}
	return t
}

func (t *ArrayType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*ArrayType)
	if !ok {
		return false
	}
	return t.Len == t2.Len && t.Elem.compat(t2.Elem, conv)
}

func (t *ArrayType) lit() Type { return t }

func (t *ArrayType) String() string { return "[]" + t.Elem.String() }

func (t *ArrayType) Zero() Value {
	res := arrayV(make([]Value, t.Len))
	// TODO(austin) It's unfortunate that each element is
	// separately heap allocated.  We could add ZeroArray to
	// everything, though that doesn't help with multidimensional
	// arrays.  Or we could do something unsafe.  We'll have this
	// same problem with structs.
	for i := int64(0); i < t.Len; i++ {
		res[i] = t.Elem.Zero()
	}
	return &res
}

/*
 * Struct
 */

type StructField struct {
	Name      string
	Type      Type
	Anonymous bool
}

type StructType struct {
	commonType
	Elems []StructField
}

var structTypes = newTypeArrayMap()

// Two struct types are identical if they have the same sequence of
// fields, and if corresponding fields have the same names and
// identical types. Two anonymous fields are considered to have the
// same name.

func NewStructType(fields []StructField) *StructType {
	// Start by looking up just the types
	fts := make([]Type, len(fields))
	for i, f := range fields {
		fts[i] = f.Type
	}
	tMapI := structTypes.Get(fts)
	if tMapI == nil {
		tMapI = structTypes.Put(fts, make(map[string]*StructType))
	}
	tMap := tMapI.(map[string]*StructType)

	// Construct key for field names
	key := ""
	for _, f := range fields {
		// XXX(Spec) It's not clear if struct { T } and struct
		// { T T } are either identical or compatible.  The
		// "Struct Types" section says that the name of that
		// field is "T", which suggests that they are
		// identical, but it really means that it's the name
		// for the purpose of selector expressions and nothing
		// else.  We decided that they should be neither
		// identical or compatible.
		if f.Anonymous {
			key += "!"
		}
		key += f.Name + " "
	}

	// XXX(Spec) Do the tags also have to be identical for the
	// types to be identical?  I certainly hope so, because
	// otherwise, this is the only case where two distinct type
	// objects can represent identical types.

	t, ok := tMap[key]
	if !ok {
		// Create new struct type
		t = &StructType{commonType{}, fields}
		tMap[key] = t
	}
	return t
}

func (t *StructType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*StructType)
	if !ok {
		return false
	}
	if len(t.Elems) != len(t2.Elems) {
		return false
	}
	for i, e := range t.Elems {
		e2 := t2.Elems[i]
		// XXX(Spec) An anonymous and a non-anonymous field
		// are neither identical nor compatible.
		if e.Anonymous != e2.Anonymous ||
			(!e.Anonymous && e.Name != e2.Name) ||
			!e.Type.compat(e2.Type, conv) {
			return false
		}
	}
	return true
}

func (t *StructType) lit() Type { return t }

func (t *StructType) String() string {
	s := "struct {"
	for i, f := range t.Elems {
		if i > 0 {
			s += "; "
		}
		if !f.Anonymous {
			s += f.Name + " "
		}
		s += f.Type.String()
	}
	return s + "}"
}

func (t *StructType) Zero() Value {
	res := structV(make([]Value, len(t.Elems)))
	for i, f := range t.Elems {
		res[i] = f.Type.Zero()
	}
	return &res
}

/*
 * Pointer
 */

type PtrType struct {
	commonType
	Elem Type
}

var ptrTypes = make(map[Type]*PtrType)

// Two pointer types are identical if they have identical base types.

func NewPtrType(elem Type) *PtrType {
	t, ok := ptrTypes[elem]
	if !ok {
		t = &PtrType{commonType{}, elem}
		ptrTypes[elem] = t
	}
	return t
}

func (t *PtrType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*PtrType)
	if !ok {
		return false
	}
	return t.Elem.compat(t2.Elem, conv)
}

func (t *PtrType) lit() Type { return t }

func (t *PtrType) String() string { return "*" + t.Elem.String() }

func (t *PtrType) Zero() Value { return &ptrV{nil} }

/*
 * Function
 */

type FuncType struct {
	commonType
	// TODO(austin) Separate receiver Type for methods?
	In       []Type
	Variadic bool
	Out      []Type
	builtin  string
}

var funcTypes = newTypeArrayMap()
var variadicFuncTypes = newTypeArrayMap()

// Create singleton function types for magic built-in functions
var (
	capType     = &FuncType{builtin: "cap"}
	closeType   = &FuncType{builtin: "close"}
	closedType  = &FuncType{builtin: "closed"}
	lenType     = &FuncType{builtin: "len"}
	makeType    = &FuncType{builtin: "make"}
	newType     = &FuncType{builtin: "new"}
	panicType   = &FuncType{builtin: "panic"}
	printType   = &FuncType{builtin: "print"}
	printlnType = &FuncType{builtin: "println"}
	copyType    = &FuncType{builtin: "copy"}
)

// Two function types are identical if they have the same number of
// parameters and result values and if corresponding parameter and
// result types are identical. All "..." parameters have identical
// type. Parameter and result names are not required to match.

func NewFuncType(in []Type, variadic bool, out []Type) *FuncType {
	inMap := funcTypes
	if variadic {
		inMap = variadicFuncTypes
	}

	outMapI := inMap.Get(in)
	if outMapI == nil {
		outMapI = inMap.Put(in, newTypeArrayMap())
	}
	outMap := outMapI.(typeArrayMap)

	tI := outMap.Get(out)
	if tI != nil {
		return tI.(*FuncType)
	}

	t := &FuncType{commonType{}, in, variadic, out, ""}
	outMap.Put(out, t)
	return t
}

func (t *FuncType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*FuncType)
	if !ok {
		return false
	}
	if len(t.In) != len(t2.In) || t.Variadic != t2.Variadic || len(t.Out) != len(t2.Out) {
		return false
	}
	for i := range t.In {
		if !t.In[i].compat(t2.In[i], conv) {
			return false
		}
	}
	for i := range t.Out {
		if !t.Out[i].compat(t2.Out[i], conv) {
			return false
		}
	}
	return true
}

func (t *FuncType) lit() Type { return t }

func typeListString(ts []Type, ns []*ast.Ident) string {
	s := ""
	for i, t := range ts {
		if i > 0 {
			s += ", "
		}
		if ns != nil && ns[i] != nil {
			s += ns[i].Name + " "
		}
		if t == nil {
			// Some places use nil types to represent errors
			s += "<none>"
		} else {
			s += t.String()
		}
	}
	return s
}

func (t *FuncType) String() string {
	if t.builtin != "" {
		return "built-in function " + t.builtin
	}
	args := typeListString(t.In, nil)
	if t.Variadic {
		if len(args) > 0 {
			args += ", "
		}
		args += "..."
	}
	s := "func(" + args + ")"
	if len(t.Out) > 0 {
		s += " (" + typeListString(t.Out, nil) + ")"
	}
	return s
}

func (t *FuncType) Zero() Value { return &funcV{nil} }

type FuncDecl struct {
	Type *FuncType
	Name *ast.Ident // nil for function literals
	// InNames will be one longer than Type.In if this function is
	// variadic.
	InNames  []*ast.Ident
	OutNames []*ast.Ident
}

func (t *FuncDecl) String() string {
	s := "func"
	if t.Name != nil {
		s += " " + t.Name.Name
	}
	s += funcTypeString(t.Type, t.InNames, t.OutNames)
	return s
}

func funcTypeString(ft *FuncType, ins []*ast.Ident, outs []*ast.Ident) string {
	s := "("
	s += typeListString(ft.In, ins)
	if ft.Variadic {
		if len(ft.In) > 0 {
			s += ", "
		}
		s += "..."
	}
	s += ")"
	if len(ft.Out) > 0 {
		s += " (" + typeListString(ft.Out, outs) + ")"
	}
	return s
}

/*
 * Interface
 */

// TODO(austin) Interface values, types, and type compilation are
// implemented, but none of the type checking or semantics of
// interfaces are.

type InterfaceType struct {
	commonType
	// TODO(austin) This should be a map from names to
	// *FuncType's.  We only need the sorted list for generating
	// the type map key.  It's detrimental for everything else.
	methods []IMethod
}

type IMethod struct {
	Name string
	Type *FuncType
}

var interfaceTypes = newTypeArrayMap()

func NewInterfaceType(methods []IMethod, embeds []*InterfaceType) *InterfaceType {
	// Count methods of embedded interfaces
	nMethods := len(methods)
	for _, e := range embeds {
		nMethods += len(e.methods)
	}

	// Combine methods
	allMethods := make([]IMethod, nMethods)
	copy(allMethods, methods)
	n := len(methods)
	for _, e := range embeds {
		for _, m := range e.methods {
			allMethods[n] = m
			n++
		}
	}

	// Sort methods
	sort.Sort(iMethodSorter(allMethods))

	mts := make([]Type, len(allMethods))
	for i, m := range methods {
		mts[i] = m.Type
	}
	tMapI := interfaceTypes.Get(mts)
	if tMapI == nil {
		tMapI = interfaceTypes.Put(mts, make(map[string]*InterfaceType))
	}
	tMap := tMapI.(map[string]*InterfaceType)

	key := ""
	for _, m := range allMethods {
		key += m.Name + " "
	}

	t, ok := tMap[key]
	if !ok {
		t = &InterfaceType{commonType{}, allMethods}
		tMap[key] = t
	}
	return t
}

type iMethodSorter []IMethod

func (s iMethodSorter) Less(a, b int) bool { return s[a].Name < s[b].Name }

func (s iMethodSorter) Swap(a, b int) { s[a], s[b] = s[b], s[a] }

func (s iMethodSorter) Len() int { return len(s) }

func (t *InterfaceType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*InterfaceType)
	if !ok {
		return false
	}
	if len(t.methods) != len(t2.methods) {
		return false
	}
	for i, e := range t.methods {
		e2 := t2.methods[i]
		if e.Name != e2.Name || !e.Type.compat(e2.Type, conv) {
			return false
		}
	}
	return true
}

func (t *InterfaceType) lit() Type { return t }

func (t *InterfaceType) String() string {
	// TODO(austin) Instead of showing embedded interfaces, this
	// shows their methods.
	s := "interface {"
	for i, m := range t.methods {
		if i > 0 {
			s += "; "
		}
		s += m.Name + funcTypeString(m.Type, nil, nil)
	}
	return s + "}"
}

// implementedBy tests if o implements t, returning nil, true if it does.
// Otherwise, it returns a method of t that o is missing and false.
func (t *InterfaceType) implementedBy(o Type) (*IMethod, bool) {
	if len(t.methods) == 0 {
		return nil, true
	}

	// The methods of a named interface types are those of the
	// underlying type.
	if it, ok := o.lit().(*InterfaceType); ok {
		o = it
	}

	// XXX(Spec) Interface types: "A type implements any interface
	// comprising any subset of its methods" It's unclear if
	// methods must have identical or compatible types.  6g
	// requires identical types.

	switch o := o.(type) {
	case *NamedType:
		for _, tm := range t.methods {
			sm, ok := o.methods[tm.Name]
			if !ok || sm.decl.Type != tm.Type {
				return &tm, false
			}
		}
		return nil, true

	case *InterfaceType:
		var ti, oi int
		for ti < len(t.methods) && oi < len(o.methods) {
			tm, om := &t.methods[ti], &o.methods[oi]
			switch {
			case tm.Name == om.Name:
				if tm.Type != om.Type {
					return tm, false
				}
				ti++
				oi++
			case tm.Name > om.Name:
				oi++
			default:
				return tm, false
			}
		}
		if ti < len(t.methods) {
			return &t.methods[ti], false
		}
		return nil, true
	}

	return &t.methods[0], false
}

func (t *InterfaceType) Zero() Value { return &interfaceV{} }

/*
 * Slice
 */

type SliceType struct {
	commonType
	Elem Type
}

var sliceTypes = make(map[Type]*SliceType)

// Two slice types are identical if they have identical element types.

func NewSliceType(elem Type) *SliceType {
	t, ok := sliceTypes[elem]
	if !ok {
		t = &SliceType{commonType{}, elem}
		sliceTypes[elem] = t
	}
	return t
}

func (t *SliceType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*SliceType)
	if !ok {
		return false
	}
	return t.Elem.compat(t2.Elem, conv)
}

func (t *SliceType) lit() Type { return t }

func (t *SliceType) String() string { return "[]" + t.Elem.String() }

func (t *SliceType) Zero() Value {
	// The value of an uninitialized slice is nil. The length and
	// capacity of a nil slice are 0.
	return &sliceV{Slice{nil, 0, 0}}
}

/*
 * Map type
 */

type MapType struct {
	commonType
	Key  Type
	Elem Type
}

var mapTypes = make(map[Type]map[Type]*MapType)

func NewMapType(key Type, elem Type) *MapType {
	ts, ok := mapTypes[key]
	if !ok {
		ts = make(map[Type]*MapType)
		mapTypes[key] = ts
	}
	t, ok := ts[elem]
	if !ok {
		t = &MapType{commonType{}, key, elem}
		ts[elem] = t
	}
	return t
}

func (t *MapType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*MapType)
	if !ok {
		return false
	}
	return t.Elem.compat(t2.Elem, conv) && t.Key.compat(t2.Key, conv)
}

func (t *MapType) lit() Type { return t }

func (t *MapType) String() string { return "map[" + t.Key.String() + "] " + t.Elem.String() }

func (t *MapType) Zero() Value {
	// The value of an uninitialized map is nil.
	return &mapV{nil}
}

/*
type ChanType struct {
	// TODO(austin)
}
*/

/*
 * Named types
 */

type Method struct {
	decl *FuncDecl
	fn   Func
}

type NamedType struct {
	NamePos token.Pos
	Name    string
	// Underlying type.  If incomplete is true, this will be nil.
	// If incomplete is false and this is still nil, then this is
	// a placeholder type representing an error.
	Def Type
	// True while this type is being defined.
	incomplete bool
	methods    map[string]Method
}

// TODO(austin) This is temporarily needed by the debugger's remote
// type parser.  This should only be possible with block.DefineType.
func NewNamedType(name string) *NamedType {
	return &NamedType{token.NoPos, name, nil, true, make(map[string]Method)}
}

func (t *NamedType) Pos() token.Pos {
	return t.NamePos
}

func (t *NamedType) Complete(def Type) {
	if !t.incomplete {
		log.Panicf("cannot complete already completed NamedType %+v", *t)
	}
	// We strip the name from def because multiple levels of
	// naming are useless.
	if ndef, ok := def.(*NamedType); ok {
		def = ndef.Def
	}
	t.Def = def
	t.incomplete = false
}

func (t *NamedType) compat(o Type, conv bool) bool {
	t2, ok := o.(*NamedType)
	if ok {
		if conv {
			// Two named types are conversion compatible
			// if their literals are conversion
			// compatible.
			return t.Def.compat(t2.Def, conv)
		} else {
			// Two named types are compatible if their
			// type names originate in the same type
			// declaration.
			return t == t2
		}
	}
	// A named and an unnamed type are compatible if the
	// respective type literals are compatible.
	return o.compat(t.Def, conv)
}

func (t *NamedType) lit() Type { return t.Def.lit() }

func (t *NamedType) isBoolean() bool { return t.Def.isBoolean() }

func (t *NamedType) isInteger() bool { return t.Def.isInteger() }

func (t *NamedType) isFloat() bool { return t.Def.isFloat() }

func (t *NamedType) isIdeal() bool { return false }

func (t *NamedType) String() string { return t.Name }

func (t *NamedType) Zero() Value { return t.Def.Zero() }

/*
 * Multi-valued type
 */

// MultiType is a special type used for multi-valued expressions, akin
// to a tuple type.  It's not generally accessible within the
// language.
type MultiType struct {
	commonType
	Elems []Type
}

var multiTypes = newTypeArrayMap()

func NewMultiType(elems []Type) *MultiType {
	if t := multiTypes.Get(elems); t != nil {
		return t.(*MultiType)
	}

	t := &MultiType{commonType{}, elems}
	multiTypes.Put(elems, t)
	return t
}

func (t *MultiType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*MultiType)
	if !ok {
		return false
	}
	if len(t.Elems) != len(t2.Elems) {
		return false
	}
	for i := range t.Elems {
		if !t.Elems[i].compat(t2.Elems[i], conv) {
			return false
		}
	}
	return true
}

var EmptyType Type = NewMultiType([]Type{})

func (t *MultiType) lit() Type { return t }

func (t *MultiType) String() string {
	if len(t.Elems) == 0 {
		return "<none>"
	}
	return typeListString(t.Elems, nil)
}

func (t *MultiType) Zero() Value {
	res := make([]Value, len(t.Elems))
	for i, t := range t.Elems {
		res[i] = t.Zero()
	}
	return multiV(res)
}

/*
 * Initialize the universe
 */

func init() {
	numer := big.NewInt(0xffffff)
	numer.Lsh(numer, 127-23)
	maxFloat32Val = new(big.Rat).SetInt(numer)
	numer.SetInt64(0x1fffffffffffff)
	numer.Lsh(numer, 1023-52)
	maxFloat64Val = new(big.Rat).SetInt(numer)
	minFloat32Val = new(big.Rat).Neg(maxFloat32Val)
	minFloat64Val = new(big.Rat).Neg(maxFloat64Val)

	// To avoid portability issues all numeric types are distinct
	// except byte, which is an alias for uint8.

	// Make byte an alias for the named type uint8.  Type aliases
	// are otherwise impossible in Go, so just hack it here.
	universe.defs["byte"] = universe.defs["uint8"]

	// Built-in functions
	universe.DefineConst("cap", universePos, capType, nil)
	universe.DefineConst("close", universePos, closeType, nil)
	universe.DefineConst("closed", universePos, closedType, nil)
	universe.DefineConst("copy", universePos, copyType, nil)
	universe.DefineConst("len", universePos, lenType, nil)
	universe.DefineConst("make", universePos, makeType, nil)
	universe.DefineConst("new", universePos, newType, nil)
	universe.DefineConst("panic", universePos, panicType, nil)
	universe.DefineConst("print", universePos, printType, nil)
	universe.DefineConst("println", universePos, printlnType, nil)
}
