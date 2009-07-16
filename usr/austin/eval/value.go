// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
	"eval";
	"fmt";
)

/*
 * Bool
 */

type boolV bool

func (*boolV) Type() Type {
	return BoolType;
}

func (v *boolV) String() string {
	return fmt.Sprint(*v);
}

func (v *boolV) Get() bool {
	return bool(*v);
}

func (v *boolV) Set(x bool) {
	*v = boolV(x);
}

func (t *boolType) value(v bool) BoolValue {
	res := boolV(v);
	return &res;
}

/*
 * Uint
 */

type uint8V uint8

func (*uint8V) Type() Type {
	return Uint8Type;
}

func (v *uint8V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint8V) Get() uint64 {
	return uint64(*v);
}

func (v *uint8V) Set(x uint64) {
	*v = uint8V(x);
}

type uint16V uint16

func (*uint16V) Type() Type {
	return Uint16Type;
}

func (v *uint16V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint16V) Get() uint64 {
	return uint64(*v);
}

func (v *uint16V) Set(x uint64) {
	*v = uint16V(x);
}

type uint32V uint32

func (*uint32V) Type() Type {
	return Uint32Type;
}

func (v *uint32V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint32V) Get() uint64 {
	return uint64(*v);
}

func (v *uint32V) Set(x uint64) {
	*v = uint32V(x);
}

type uint64V uint64

func (*uint64V) Type() Type {
	return Uint64Type;
}

func (v *uint64V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint64V) Get() uint64 {
	return uint64(*v);
}

func (v *uint64V) Set(x uint64) {
	*v = uint64V(x);
}

type uintV uint

func (*uintV) Type() Type {
	return UintType;
}

func (v *uintV) String() string {
	return fmt.Sprint(*v);
}

func (v *uintV) Get() uint64 {
	return uint64(*v);
}

func (v *uintV) Set(x uint64) {
	*v = uintV(x);
}

type uintptrV uintptr

func (*uintptrV) Type() Type {
	return UintptrType;
}

func (v *uintptrV) String() string {
	return fmt.Sprint(*v);
}

func (v *uintptrV) Get() uint64 {
	return uint64(*v);
}

func (v *uintptrV) Set(x uint64) {
	*v = uintptrV(x);
}

func (t *uintType) value(v uint64) UintValue {
	// TODO(austin) The 'value' methods are only used for
	// testing right now.  Get rid of them.
	// TODO(austin) Deal with named types
	switch Type(t) {
	case Uint8Type:
		res := uint8V(v);
		return &res;
	case Uint16Type:
		res := uint16V(v);
		return &res;
	case Uint32Type:
		res := uint32V(v);
		return &res;
	case Uint64Type:
		res := uint64V(v);
		return &res;

	case UintType:
		res := uintV(v);
		return &res;
	case UintptrType:
		res := uintptrV(v);
		return &res;
	}
	panic("unknown uint type ", t.String());
}

/*
 * Int
 */

type int8V int8

func (*int8V) Type() Type {
	return Int8Type;
}

func (v *int8V) String() string {
	return fmt.Sprint(*v);
}

func (v *int8V) Get() int64 {
	return int64(*v);
}

func (v *int8V) Set(x int64) {
	*v = int8V(x);
}

type int16V int16

func (*int16V) Type() Type {
	return Int16Type;
}

func (v *int16V) String() string {
	return fmt.Sprint(*v);
}

func (v *int16V) Get() int64 {
	return int64(*v);
}

func (v *int16V) Set(x int64) {
	*v = int16V(x);
}

type int32V int32

func (*int32V) Type() Type {
	return Int32Type;
}

func (v *int32V) String() string {
	return fmt.Sprint(*v);
}

func (v *int32V) Get() int64 {
	return int64(*v);
}

func (v *int32V) Set(x int64) {
	*v = int32V(x);
}

type int64V int64

func (*int64V) Type() Type {
	return Int64Type;
}

func (v *int64V) String() string {
	return fmt.Sprint(*v);
}

func (v *int64V) Get() int64 {
	return int64(*v);
}

func (v *int64V) Set(x int64) {
	*v = int64V(x);
}

type intV int

func (*intV) Type() Type {
	return IntType;
}

func (v *intV) String() string {
	return fmt.Sprint(*v);
}

func (v *intV) Get() int64 {
	return int64(*v);
}

func (v *intV) Set(x int64) {
	*v = intV(x);
}

func (t *intType) value(v int64) IntValue {
	switch Type(t) {
	case Int8Type:
		res := int8V(v);
		return &res;
	case Int16Type:
		res := int16V(v);
		return &res;
	case Int32Type:
		res := int32V(v);
		return &res;
	case Int64Type:
		res := int64V(v);
		return &res;

	case IntType:
		res := intV(v);
		return &res;
	}
	panic("unknown int type ", t.String());
}

/*
 * Ideal int
 */

type idealIntV struct {
	V *bignum.Integer;
}

func (*idealIntV) Type() Type {
	return IdealIntType;
}

func (v *idealIntV) String() string {
	return v.V.String();
}

func (v *idealIntV) Get() *bignum.Integer {
	return v.V;
}

func (t *idealIntType) value(v *bignum.Integer) IdealIntValue {
	return &idealIntV{v};
}

/*
 * Float
 */

type float32V float32

func (*float32V) Type() Type {
	return Float32Type;
}

func (v *float32V) String() string {
	return fmt.Sprint(*v);
}

func (v *float32V) Get() float64 {
	return float64(*v);
}

func (v *float32V) Set(x float64) {
	*v = float32V(x);
}

type float64V float64

func (*float64V) Type() Type {
	return Float64Type;
}

func (v *float64V) String() string {
	return fmt.Sprint(*v);
}

func (v *float64V) Get() float64 {
	return float64(*v);
}

func (v *float64V) Set(x float64) {
	*v = float64V(x);
}

type floatV float

func (*floatV) Type() Type {
	return FloatType;
}

func (v *floatV) String() string {
	return fmt.Sprint(*v);
}

func (v *floatV) Get() float64 {
	return float64(*v);
}

func (v *floatV) Set(x float64) {
	*v = floatV(x);
}

func (t *floatType) value(v float64) FloatValue {
	switch Type(t) {
	case Float32Type:
		res := float32V(v);
		return &res;
	case Float64Type:
		res := float64V(v);
		return &res;
	case FloatType:
		res := floatV(v);
		return &res;
	}
	panic("unknown float type ", t.String());
}

/*
 * Ideal float
 */

type idealFloatV struct {
	V *bignum.Rational;
}

func (*idealFloatV) Type() Type {
	return IdealFloatType;
}

func (v *idealFloatV) String() string {
	return ratToString(v.V);
}

func (v *idealFloatV) Get() *bignum.Rational {
	return v.V;
}

func (t *idealFloatType) value(v *bignum.Rational) IdealFloatValue {
	return &idealFloatV{v};
}

/*
 * String
 */

type stringV string

func (*stringV) Type() Type {
	return StringType;
}

func (v *stringV) String() string {
	return fmt.Sprint(*v);
}

func (v *stringV) Get() string {
	return string(*v);
}

func (v *stringV) Set(x string) {
	*v = stringV(x);
}

func (t *stringType) value(v string) StringValue {
	res := stringV(v);
	return &res;
}

/*
 * Pointer
 */

type ptrV struct {
	// nil if the pointer is nil
	target Value;
}

func (v *ptrV) Type() Type {
	return NewPtrType(v.target.Type());
}

func (v *ptrV) String() string {
	return "&" + v.target.String();
}

func (v *ptrV) Get() Value {
	return v.target;
}

func (v *ptrV) Set(x Value) {
	v.target = x;
}

func (t *PtrType) value(v Value) PtrValue {
	res := ptrV{v};
	return &res;
}
