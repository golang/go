// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"go/ast";
	"go/token";
	"log";
	"reflect";
	"unsafe";			// For Sizeof
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

var universePos = token.Position{"<universe>", 0, 0, 0};

/*
 * Type array maps.  These are used to memoize composite types.
 */

type typeArrayMapEntry struct {
	key []Type;
	v interface {};
	next *typeArrayMapEntry;
}

type typeArrayMap map[uintptr] *typeArrayMapEntry

func hashTypeArray(key []Type) uintptr {
	hash := uintptr(0);
	for _, t := range key {
		hash = hash * 33;
		if t == nil {
			continue;
		}
		addr := reflect.NewValue(t).(*reflect.PtrValue).Get();
		hash ^= addr;
	}
	return hash;
}

func newTypeArrayMap() typeArrayMap {
	return make(map[uintptr] *typeArrayMapEntry);
}

func (m typeArrayMap) Get(key []Type) (interface{}) {
	ent, ok := m[hashTypeArray(key)];
	if !ok {
		return nil;
	}

nextEnt:
	for ; ent != nil; ent = ent.next {
		if len(key) != len(ent.key) {
			continue;
		}
		for i := 0; i < len(key); i++ {
			if key[i] != ent.key[i] {
				continue nextEnt;
			}
		}
		// Found it
		return ent.v;
	}

	return nil;
}

func (m typeArrayMap) Put(key []Type, v interface{}) interface{} {
	hash := hashTypeArray(key);
	ent, _ := m[hash];

	new := &typeArrayMapEntry{key, v, ent};
	m[hash] = new;
	return v;
}

/*
 * Common type
 */

type commonType struct {
}

func (commonType) isBoolean() bool {
	return false;
}

func (commonType) isInteger() bool {
	return false;
}

func (commonType) isFloat() bool {
	return false;
}

func (commonType) isIdeal() bool {
	return false;
}

func (commonType) Pos() token.Position {
	return token.Position{};
}

/*
 * Bool
 */

type boolType struct {
	commonType;
}

var BoolType = universe.DefineType("bool", universePos, &boolType{});

func (t *boolType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*boolType);
	return ok;
}

func (t *boolType) lit() Type {
	return t;
}

func (t *boolType) isBoolean() bool {
	return true;
}

func (boolType) String() string {
	// Use angle brackets as a convention for printing the
	// underlying, unnamed type.  This should only show up in
	// debug output.
	return "<bool>";
}

func (t *boolType) Zero() Value

/*
 * Uint
 */

type uintType struct {
	commonType;

	// 0 for architecture-dependent types
	Bits uint;
	// true for uintptr, false for all others
	Ptr bool;
	name string;
}

var (
	Uint8Type   = universe.DefineType("uint8",   universePos, &uintType{commonType{}, 8,  false, "uint8"});
	Uint16Type  = universe.DefineType("uint16",  universePos, &uintType{commonType{}, 16, false, "uint16"});
	Uint32Type  = universe.DefineType("uint32",  universePos, &uintType{commonType{}, 32, false, "uint32"});
	Uint64Type  = universe.DefineType("uint64",  universePos, &uintType{commonType{}, 64, false, "uint64"});

	UintType    = universe.DefineType("uint",    universePos, &uintType{commonType{}, 0,  false, "uint"});
	UintptrType = universe.DefineType("uintptr", universePos, &uintType{commonType{}, 0,  true,  "uintptr"});
)

func init() {
	// To avoid portability issues all numeric types are distinct
	// except byte, which is an alias for uint8.

	// Make byte an alias for the named type uint8.  Type aliases
	// are otherwise impossible in Go, so just hack it here.
	universe.defs["byte"] = universe.defs["uint8"];
}

func (t *uintType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*uintType);
	return ok && t == t2;;
}

func (t *uintType) lit() Type {
	return t;
}

func (t *uintType) isInteger() bool {
	return true;
}

func (t *uintType) String() string {
	return "<" + t.name + ">";
}

func (t *uintType) Zero() Value

func (t *uintType) minVal() *bignum.Rational {
	return bignum.Rat(0, 1);
}

func (t *uintType) maxVal() *bignum.Rational {
	bits := t.Bits;
	if bits == 0 {
		if t.Ptr {
			bits = uint(8 * unsafe.Sizeof(uintptr(0)));
		} else {
			bits = uint(8 * unsafe.Sizeof(uint(0)));
		}
	}
	return bignum.MakeRat(bignum.Int(1).Shl(bits).Add(bignum.Int(-1)), bignum.Nat(1));
}

/*
 * Int
 */

type intType struct {
	commonType;

	// XXX(Spec) Numeric types: "There is also a set of
	// architecture-independent basic numeric types whose size
	// depends on the architecture."  Should that be
	// architecture-dependent?

	// 0 for architecture-dependent types
	Bits uint;
	name string;
}

var (
	Int8Type  = universe.DefineType("int8",  universePos, &intType{commonType{}, 8,  "int8"});
	Int16Type = universe.DefineType("int16", universePos, &intType{commonType{}, 16, "int16"});
	Int32Type = universe.DefineType("int32", universePos, &intType{commonType{}, 32, "int32"});
	Int64Type = universe.DefineType("int64", universePos, &intType{commonType{}, 64, "int64"});

	IntType   = universe.DefineType("int",   universePos, &intType{commonType{}, 0,  "int"});
)

func (t *intType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*intType);
	return ok && t == t2;
}

func (t *intType) lit() Type {
	return t;
}

func (t *intType) isInteger() bool {
	return true;
}

func (t *intType) String() string {
	return "<" + t.name + ">";
}

func (t *intType) Zero() Value

func (t *intType) minVal() *bignum.Rational {
	bits := t.Bits;
	if bits == 0 {
		bits = uint(8 * unsafe.Sizeof(int(0)));
	}
	return bignum.MakeRat(bignum.Int(-1).Shl(bits - 1), bignum.Nat(1));
}

func (t *intType) maxVal() *bignum.Rational {
	bits := t.Bits;
	if bits == 0 {
		bits = uint(8 * unsafe.Sizeof(int(0)));
	}
	return bignum.MakeRat(bignum.Int(1).Shl(bits - 1).Add(bignum.Int(-1)), bignum.Nat(1));
}

/*
 * Ideal int
 */

type idealIntType struct {
	commonType;
}

var IdealIntType Type = &idealIntType{}

func (t *idealIntType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*idealIntType);
	return ok;
}

func (t *idealIntType) lit() Type {
	return t;
}

func (t *idealIntType) isInteger() bool {
	return true;
}

func (t *idealIntType) isIdeal() bool {
	return true;
}

func (t *idealIntType) String() string {
	return "ideal integer";
}

func (t *idealIntType) Zero() Value

/*
 * Float
 */

type floatType struct {
	commonType;

	// 0 for architecture-dependent type
	Bits uint;

	name string;
}

var (
	Float32Type = universe.DefineType("float32", universePos, &floatType{commonType{}, 32, "float32"});
	Float64Type = universe.DefineType("float64", universePos, &floatType{commonType{}, 64, "float64"});
	FloatType   = universe.DefineType("float",   universePos, &floatType{commonType{}, 0,  "float"});
)

func (t *floatType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*floatType);
	return ok && t == t2;
}

func (t *floatType) lit() Type {
	return t;
}

func (t *floatType) isFloat() bool {
	return true;
}

func (t *floatType) String() string {
	return "<" + t.name + ">";
}

func (t *floatType) Zero() Value

var maxFloat32Val = bignum.MakeRat(bignum.Int(0xffffff).Shl(127-23), bignum.Nat(1));
var maxFloat64Val = bignum.MakeRat(bignum.Int(0x1fffffffffffff).Shl(1023-52), bignum.Nat(1));
var minFloat32Val = maxFloat32Val.Neg();
var minFloat64Val = maxFloat64Val.Neg();

func (t *floatType) minVal() *bignum.Rational {
	bits := t.Bits;
	if bits == 0 {
		bits = uint(8 * unsafe.Sizeof(float(0)));
	}
	switch bits {
	case 32:
		return minFloat32Val;
	case 64:
		return minFloat64Val;
	}
	log.Crashf("unexpected floating point bit count: %d", bits);
	panic();
}

func (t *floatType) maxVal() *bignum.Rational {
	bits := t.Bits;
	if bits == 0 {
		bits = uint(8 * unsafe.Sizeof(float(0)));
	}
	switch bits {
	case 32:
		return maxFloat32Val;
	case 64:
		return maxFloat64Val;
	}
	log.Crashf("unexpected floating point bit count: %d", bits);
	panic();
}

/*
 * Ideal float
 */

type idealFloatType struct {
	commonType;
}

var IdealFloatType Type = &idealFloatType{};

func (t *idealFloatType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*idealFloatType);
	return ok;
}

func (t *idealFloatType) lit() Type {
	return t;
}

func (t *idealFloatType) isFloat() bool {
	return true;
}

func (t *idealFloatType) isIdeal() bool {
	return true;
}

func (t *idealFloatType) String() string {
	return "ideal float";
}

func (t *idealFloatType) Zero() Value

/*
 * String
 */

type stringType struct {
	commonType;
}

var StringType = universe.DefineType("string", universePos, &stringType{});

func (t *stringType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*stringType);
	return ok;
}

func (t *stringType) lit() Type {
	return t;
}

func (t *stringType) String() string {
	return "<string>";
}

func (t *stringType) Zero() Value

/*
 * Array
 */

type ArrayType struct {
	commonType;
	Len int64;
	Elem Type;
}

var arrayTypes = make(map[int64] map[Type] *ArrayType);

// Two array types are identical if they have identical element types
// and the same array length.

func NewArrayType(len int64, elem Type) *ArrayType {
	ts, ok := arrayTypes[len];
	if !ok {
		ts = make(map[Type] *ArrayType);
		arrayTypes[len] = ts;
	}
	t, ok := ts[elem];
	if !ok {
		t = &ArrayType{commonType{}, len, elem};
		ts[elem] = t;
	}
	return t;
}

func (t *ArrayType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*ArrayType);
	if !ok {
		return false;
	}
	return t.Len == t2.Len && t.Elem.compat(t2.Elem, conv);
}

func (t *ArrayType) lit() Type {
	return t;
}

func (t *ArrayType) String() string {
	return "[]" + t.Elem.String();
}

func (t *ArrayType) Zero() Value

/*
 * Struct
 */

type StructField struct {
	Name string;
	Type Type;
	Anonymous bool;
}

type StructType struct {
	commonType;
	Elems []StructField;
	maxDepth int;
}

var structTypes = newTypeArrayMap()

// Two struct types are identical if they have the same sequence of
// fields, and if corresponding fields have the same names and
// identical types. Two anonymous fields are considered to have the
// same name.

func NewStructType(fields []StructField) *StructType {
	// Start by looking up just the types
	fts := make([]Type, len(fields));
	for i, f := range fields {
		fts[i] = f.Type;
	}
	tMapI := structTypes.Get(fts);
	if tMapI == nil {
		tMapI = structTypes.Put(fts, make(map[string] *StructType));
	}
	tMap := tMapI.(map[string] *StructType);

	// Construct key for field names
	key := "";
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
			key += "!";
		}
		key += f.Name + " ";
	}

	// XXX(Spec) Do the tags also have to be identical for the
	// types to be identical?  I certainly hope so, because
	// otherwise, this is the only case where two distinct type
	// objects can represent identical types.

	t, ok := tMap[key];
	if !ok {
		// Create new struct type

		// Compute max anonymous field depth
		maxDepth := 1;
		for _, f := range fields {
			// TODO(austin) Careful of type T struct { *T }
			if st, ok := f.Type.(*StructType); ok {
				if st.maxDepth + 1 > maxDepth {
					maxDepth = st.maxDepth + 1;
				}
			}
		}

		t = &StructType{commonType{}, fields, maxDepth};
		tMap[key] = t;
	}
	return t;
}

func (t *StructType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*StructType);
	if !ok {
		return false;
	}
	if len(t.Elems) != len(t2.Elems) {
		return false;
	}
	for i, e := range t.Elems {
		e2 := t2.Elems[i];
		// XXX(Spec) An anonymous and a non-anonymous field
		// are neither identical nor compatible.
		if (e.Anonymous != e2.Anonymous ||
		    (!e.Anonymous && e.Name != e2.Name) ||
		    !e.Type.compat(e2.Type, conv)) {
			return false;
		}
	}
	return true;
}

func (t *StructType) lit() Type {
	return t;
}

func (t *StructType) String() string {
	s := "struct {";
	for i, f := range t.Elems {
		if i > 0 {
			s += "; ";
		}
		if !f.Anonymous {
			s += f.Name + " ";
		}
		s += f.Type.String();
	}
	return s + "}";
}

func (t *StructType) Zero() Value

/*
 * Pointer
 */

type PtrType struct {
	commonType;
	Elem Type;
}

var ptrTypes = make(map[Type] *PtrType)

// Two pointer types are identical if they have identical base types.

func NewPtrType(elem Type) *PtrType {
	t, ok := ptrTypes[elem];
	if !ok {
		t = &PtrType{commonType{}, elem};
		ptrTypes[elem] = t;
	}
	return t;
}

func (t *PtrType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*PtrType);
	if !ok {
		return false;
	}
	return t.Elem.compat(t2.Elem, conv);
}

func (t *PtrType) lit() Type {
	return t;
}

func (t *PtrType) String() string {
	return "*" + t.Elem.String();
}

func (t *PtrType) Zero() Value

/*
 * Function
 */

type FuncType struct {
	commonType;
	// TODO(austin) Separate receiver Type for methods?
	In []Type;
	Variadic bool;
	Out []Type;
}

var funcTypes = newTypeArrayMap();
var variadicFuncTypes = newTypeArrayMap();

// Two function types are identical if they have the same number of
// parameters and result values and if corresponding parameter and
// result types are identical. All "..." parameters have identical
// type. Parameter and result names are not required to match.

func NewFuncType(in []Type, variadic bool, out []Type) *FuncType {
	inMap := funcTypes;
	if variadic {
		inMap = variadicFuncTypes;
	}

	outMapI := inMap.Get(in);
	if outMapI == nil {
		outMapI = inMap.Put(in, newTypeArrayMap());
	}
	outMap := outMapI.(typeArrayMap);

	tI := outMap.Get(out);
	if tI != nil {
		return tI.(*FuncType);
	}

	t := &FuncType{commonType{}, in, variadic, out};
	outMap.Put(out, t);
	return t;
}

func (t *FuncType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*FuncType);
	if !ok {
		return false;
	}
	if len(t.In) != len(t2.In) || t.Variadic != t2.Variadic || len(t.Out) != len(t2.Out) {
		return false;
	}
	for i := range t.In {
		if !t.In[i].compat(t2.In[i], conv) {
			return false;
		}
	}
	for i := range t.Out {
		if !t.Out[i].compat(t2.Out[i], conv) {
			return false;
		}
	}
	return true;
}

func (t *FuncType) lit() Type {
	return t;
}

func typeListString(ts []Type, ns []*ast.Ident) string {
	s := "";
	for i, t := range ts {
		if i > 0 {
			s += ", ";
		}
		if ns != nil && ns[i] != nil {
			s += ns[i].Value + " ";
		}
		if t == nil {
			// Some places use nil types to represent errors
			s += "<none>";
		} else {
			s += t.String();
		}
	}
	return s;
}

func (t *FuncType) String() string {
	args := typeListString(t.In, nil);
	if t.Variadic {
		if len(args) > 0 {
			args += ", ";
		}
		args += "...";
	}
	s := "func(" + args + ")";
	if len(t.Out) > 0 {
		s += " (" + typeListString(t.Out, nil) + ")";
	}
	return s;
}

func (t *FuncType) Zero() Value

type FuncDecl struct {
	Type *FuncType;
	Name *ast.Ident;		// nil for function literals
	// InNames will be one longer than Type.In if this function is
	// variadic.
	InNames []*ast.Ident;
	OutNames []*ast.Ident;
}

func (t *FuncDecl) String() string {
	args := typeListString(t.Type.In, t.InNames);
	if t.Type.Variadic {
		if len(args) > 0 {
			args += ", ";
		}
		args += "...";
	}
	s := "func";
	if t.Name != nil {
		s += " " + t.Name.Value;
	}
	s += "(" + args + ")";
	if len(t.Type.Out) > 0 {
		s += " (" + typeListString(t.Type.Out, t.OutNames) + ")";
	}
	return s;
}

/*
type InterfaceType struct {
	// TODO(austin)
}

type SliceType struct {
	// TODO(austin)
}

type MapType struct {
	// TODO(austin)
}

type ChanType struct {
	// TODO(austin)
}
*/

/*
 * Named types
 */

type Method struct {
	decl *FuncDecl;
	fn Func;
}

type NamedType struct {
	token.Position;
	name string;
	// Underlying type.  If incomplete is true, this will be nil.
	// If incomplete is false and this is still nil, then this is
	// a placeholder type representing an error.
	def Type;
	// True while this type is being defined.
	incomplete bool;
	methods map[string] Method;
}

func (t *NamedType) compat(o Type, conv bool) bool {
	t2, ok := o.(*NamedType);
	if ok {
		if conv {
			// Two named types are conversion compatible
			// if their literals are conversion
			// compatible.
			return t.def.compat(t2.def, conv);
		} else {
			// Two named types are compatible if their
			// type names originate in the same type
			// declaration.
			return t == t2;
		}
	}
	// A named and an unnamed type are compatible if the
	// respective type literals are compatible.
	return o.compat(t.def, conv);
}

func (t *NamedType) lit() Type {
	return t.def.lit();
}

func (t *NamedType) isBoolean() bool {
	return t.def.isBoolean();
}

func (t *NamedType) isInteger() bool {
	return t.def.isInteger();
}

func (t *NamedType) isFloat() bool {
	return t.def.isFloat();
}

func (t *NamedType) isIdeal() bool {
	return false;
}

func (t *NamedType) String() string {
	return t.name;
}

func (t *NamedType) Zero() Value {
	return t.def.Zero();
}

/*
 * Multi-valued type
 */

// MultiType is a special type used for multi-valued expressions, akin
// to a tuple type.  It's not generally accessible within the
// language.
type MultiType struct {
	commonType;
	Elems []Type;
}

var multiTypes = newTypeArrayMap()

func NewMultiType(elems []Type) *MultiType {
	if t := multiTypes.Get(elems); t != nil {
		return t.(*MultiType);
	}

	t := &MultiType{commonType{}, elems};
	multiTypes.Put(elems, t);
	return t;
}

func (t *MultiType) compat(o Type, conv bool) bool {
	t2, ok := o.lit().(*MultiType);
	if !ok {
		return false;
	}
	if len(t.Elems) != len(t2.Elems) {
		return false;
	}
	for i := range t.Elems {
		if !t.Elems[i].compat(t2.Elems[i], conv) {
			return false;
		}
	}
	return true;
}

var EmptyType Type = NewMultiType([]Type{});

func (t *MultiType) lit() Type {
	return t;
}

func (t *MultiType) String() string {
	if len(t.Elems) == 0 {
		return "<none>";
	}
	return typeListString(t.Elems, nil);
}

func (t *MultiType) Zero() Value
