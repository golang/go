// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"log";
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

// XXX(Spec) Comparison compatibility: "Values of any type may be
// compared to other values of compatible static type."  That should
// be *identical* type.

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

type boolType struct {
	commonType;
}

var BoolType Type = &boolType{};

func (t *boolType) literal() Type {
	return t;
}

func (t *boolType) compatible(o Type) bool {
	return Type(t) == o;
}

func (t *boolType) isBoolean() bool {
	return true;
}

func (boolType) String() string {
	return "bool";
}

func (t *boolType) Zero() Value

type uintType struct {
	commonType;

	Bits uint;
	// true for uintptr, false for all others
	Ptr bool;

	name string;
}

// TODO(austin) These are all technically *named types*, which matters
// for some things.  Perhaps these should be the underlying unnamed
// types and the named types should be created when they are put in
// the universal scope.
var (
	Uint8Type   Type = &uintType{commonType{}, 8,  false, "uint8"};
	Uint16Type  Type = &uintType{commonType{}, 16, false, "uint16"};
	Uint32Type  Type = &uintType{commonType{}, 32, false, "uint32"};
	Uint64Type  Type = &uintType{commonType{}, 64, false, "uint64"};

	UintType    Type = &uintType{commonType{}, 64, false, "uint"};
	UintptrType Type = &uintType{commonType{}, 64, true,  "uintptr"};
)

func (t *uintType) literal() Type {
	return t;
}

func (t *uintType) compatible(o Type) bool {
	return Type(t) == o;
}

func (t *uintType) isInteger() bool {
	return true;
}

func (t *uintType) String() string {
	return t.name;
}

func (t *uintType) Zero() Value

func (t *uintType) minVal() *bignum.Rational {
	return bignum.Rat(0, 1);
}

func (t *uintType) maxVal() *bignum.Rational {
	return bignum.MakeRat(bignum.Int(1).Shl(t.Bits).Add(bignum.Int(-1)), bignum.Nat(1));
}

type intType struct {
	commonType;

	// XXX(Spec) Numeric types: "There is also a set of
	// architecture-independent basic numeric types whose size
	// depends on the architecture."  Should that be
	// architecture-dependent?

	Bits uint;

	name string;
}

var (
	Int8Type  Type = &intType{commonType{}, 8,  "int8"};
	Int16Type Type = &intType{commonType{}, 16, "int16"};
	Int32Type Type = &intType{commonType{}, 32, "int32"};
	Int64Type Type = &intType{commonType{}, 64, "int64"};

	IntType   Type = &intType{commonType{}, 64, "int"};
)

func (t *intType) literal() Type {
	return t;
}

func (t *intType) compatible(o Type) bool {
	return Type(t) == o;
}

func (t *intType) isInteger() bool {
	return true;
}

func (t *intType) String() string {
	return t.name;
}

func (t *intType) Zero() Value

func (t *intType) minVal() *bignum.Rational {
	return bignum.MakeRat(bignum.Int(-1).Shl(t.Bits - 1), bignum.Nat(1));
}

func (t *intType) maxVal() *bignum.Rational {
	return bignum.MakeRat(bignum.Int(1).Shl(t.Bits - 1).Add(bignum.Int(-1)), bignum.Nat(1));
}

type idealIntType struct {
	commonType;
}

var IdealIntType Type = &idealIntType{}

func (t *idealIntType) literal() Type {
	return t;
}

func (t *idealIntType) compatible(o Type) bool {
	return Type(t) == o;
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

type floatType struct {
	commonType;
	Bits uint;
}

var (
	Float32Type Type = &floatType{commonType{}, 32};
	Float64Type Type = &floatType{commonType{}, 64};
	FloatType   Type = &floatType{commonType{}, 64};
)

func (t *floatType) literal() Type {
	return t;
}

func (t *floatType) compatible(o Type) bool {
	return Type(t) == o;
}

func (t *floatType) isFloat() bool {
	return true;
}

func (t *floatType) String() string {
	return "float";
}

func (t *floatType) Zero() Value

var maxFloat32Val = bignum.MakeRat(bignum.Int(0xffffff).Shl(127-23), bignum.Nat(1));
var maxFloat64Val = bignum.MakeRat(bignum.Int(0x1fffffffffffff).Shl(1023-52), bignum.Nat(1));
var minFloat32Val = maxFloat32Val.Neg();
var minFloat64Val = maxFloat64Val.Neg();

func (t *floatType) minVal() *bignum.Rational {
	switch t.Bits {
	case 32:
		return minFloat32Val;
	case 64:
		return minFloat64Val;
	}
	log.Crashf("unexpected number of floating point bits: %d", t.Bits);
	panic();
}

func (t *floatType) maxVal() *bignum.Rational {
	switch t.Bits {
	case 32:
		return maxFloat32Val;
	case 64:
		return maxFloat64Val;
	}
	log.Crashf("unexpected number of floating point bits: %d", t.Bits);
	panic();
}

type idealFloatType struct {
	commonType;
}

var IdealFloatType Type = &idealFloatType{};

func (t *idealFloatType) literal() Type {
	return t;
}

func (t *idealFloatType) compatible(o Type) bool {
	return Type(t) == o;
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

type stringType struct {
	commonType;
}

var StringType Type = &stringType{};

func (t *stringType) literal() Type {
	return t;
}

func (t *stringType) compatible(o Type) bool {
	return Type(t) == o;
}

func (t *stringType) String() string {
	return "string";
}

func (t *stringType) Zero() Value

type ArrayType struct {
	commonType;
	Len int64;
	Elem Type;
	lit Type;
}

var arrayTypes = make(map[int64] map[Type] *ArrayType);

func NewArrayType(len int64, elem Type) *ArrayType {
	ts, ok := arrayTypes[len];
	if !ok {
		ts = make(map[Type] *ArrayType);
		arrayTypes[len] = ts;
	}
	t, ok := ts[elem];
	if !ok {
		t = &ArrayType{commonType{}, len, elem, nil};
		ts[elem] = t;
	}
	return t;
}

func (t *ArrayType) literal() Type {
	if t.lit == nil {
		t.lit = NewArrayType(t.Len, t.Elem.literal());
	}
	return t.lit;
}

func (t *ArrayType) compatible(o Type) bool {
	return t.literal() == o.literal();
}

func (t *ArrayType) String() string {
	return "[]" + t.Elem.String();
}

func (t *ArrayType) Zero() Value

/*
func (t *ArrayType) literal() Type {
	// TODO(austin)
}

type StructType struct {
	commonType;
	Names map[string] Name;
}
*/

type PtrType struct {
	commonType;
	Elem Type;
	lit Type;
}

var ptrTypes = make(map[Type] *PtrType)

func NewPtrType(elem Type) *PtrType {
	t, ok := ptrTypes[elem];
	if !ok {
		t = &PtrType{commonType{}, elem, nil};
		ptrTypes[elem] = t;
	}
	return t;
}

func (t *PtrType) literal() Type {
	if t.lit == nil {
		t.lit = NewPtrType(t.Elem.literal());
	}
	return t.lit;
}

func (t *PtrType) compatible(o Type) bool {
	return t.literal() == o.literal();
}

func (t *PtrType) String() string {
	return "*" + t.Elem.String();
}

func (t *PtrType) Zero() Value

/*
type FuncType struct {
	commonType;
	// TODO(austin)
}

func (t *FuncType) literal() Type {
	// TODO(austin)
}

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

type NamedType struct {
	// Declaration scope
	scope *Scope;
	name string;
	// Underlying type
	def Type;
	// TODO(austin) Methods can be on NamedType or *NamedType
	methods map[string] XXX;
}

func (t *NamedType) literal() Type {
	return t.def.literal();
}

func (t *NamedType) isInteger() bool {
	return t.isInteger();
}

*/
