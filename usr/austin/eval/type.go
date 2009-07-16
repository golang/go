// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
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

func (boolType) String() string {
	return "bool";
}

func (t *boolType) value(v bool) BoolValue

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

func (t *uintType) value(v uint64) UintValue

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

func (t *intType) value(v int64) IntValue

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

func (t *idealIntType) value(v *bignum.Integer) IdealIntValue

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

func (t *floatType) value(v float64) FloatValue

func (t *floatType) minVal() *bignum.Rational {
	panic("Not implemented");
}

func (t *floatType) maxVal() *bignum.Rational {
	panic("Not implemented");
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

func (t *idealFloatType) value(v *bignum.Rational) IdealFloatValue

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

func (t *stringType) value(v string) StringValue

/*
type ArrayType struct {
	commonType;
	elem Type;
}

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
	elem Type;
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

func (t *PtrType) Elem() Type {
	return t.elem;
}

func (t *PtrType) literal() Type {
	if t.lit == nil {
		t.lit = NewPtrType(t.elem.literal());
	}
	return t.lit;
}

func (t *PtrType) compatible(o Type) bool {
	return t.literal() == o.literal();
}

func (t *PtrType) String() string {
	return "*" + t.elem.String();
}

func (t *PtrType) value(v Value) PtrValue

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
