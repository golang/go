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

func (v *boolV) String() string {
	return fmt.Sprint(*v);
}

func (v *boolV) Assign(o Value) {
	*v = boolV(o.(BoolValue).Get());
}

func (v *boolV) Get() bool {
	return bool(*v);
}

func (v *boolV) Set(x bool) {
	*v = boolV(x);
}

func (t *boolType) Zero() Value {
	res := boolV(false);
	return &res;
}

/*
 * Uint
 */

type uint8V uint8

func (v *uint8V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint8V) Assign(o Value) {
	*v = uint8V(o.(UintValue).Get());
}

func (v *uint8V) Get() uint64 {
	return uint64(*v);
}

func (v *uint8V) Set(x uint64) {
	*v = uint8V(x);
}

type uint16V uint16

func (v *uint16V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint16V) Assign(o Value) {
	*v = uint16V(o.(UintValue).Get());
}

func (v *uint16V) Get() uint64 {
	return uint64(*v);
}

func (v *uint16V) Set(x uint64) {
	*v = uint16V(x);
}

type uint32V uint32

func (v *uint32V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint32V) Assign(o Value) {
	*v = uint32V(o.(UintValue).Get());
}

func (v *uint32V) Get() uint64 {
	return uint64(*v);
}

func (v *uint32V) Set(x uint64) {
	*v = uint32V(x);
}

type uint64V uint64

func (v *uint64V) String() string {
	return fmt.Sprint(*v);
}

func (v *uint64V) Assign(o Value) {
	*v = uint64V(o.(UintValue).Get());
}

func (v *uint64V) Get() uint64 {
	return uint64(*v);
}

func (v *uint64V) Set(x uint64) {
	*v = uint64V(x);
}

type uintV uint

func (v *uintV) String() string {
	return fmt.Sprint(*v);
}

func (v *uintV) Assign(o Value) {
	*v = uintV(o.(UintValue).Get());
}

func (v *uintV) Get() uint64 {
	return uint64(*v);
}

func (v *uintV) Set(x uint64) {
	*v = uintV(x);
}

type uintptrV uintptr

func (v *uintptrV) String() string {
	return fmt.Sprint(*v);
}

func (v *uintptrV) Assign(o Value) {
	*v = uintptrV(o.(UintValue).Get());
}

func (v *uintptrV) Get() uint64 {
	return uint64(*v);
}

func (v *uintptrV) Set(x uint64) {
	*v = uintptrV(x);
}

func (t *uintType) Zero() Value {
	switch t.Bits {
	case 0:
		if t.Ptr {
			res := uintptrV(0);
			return &res;
		} else {
			res := uintV(0);
			return &res;
		}
	case 8:
		res := uint8V(0);
		return &res;
	case 16:
		res := uint16V(0);
		return &res;
	case 32:
		res := uint32V(0);
		return &res;
	case 64:
		res := uint64V(0);
		return &res;
	}
	panic("unexpected uint bit count: ", t.Bits);
}

/*
 * Int
 */

type int8V int8

func (v *int8V) String() string {
	return fmt.Sprint(*v);
}

func (v *int8V) Assign(o Value) {
	*v = int8V(o.(IntValue).Get());
}

func (v *int8V) Get() int64 {
	return int64(*v);
}

func (v *int8V) Set(x int64) {
	*v = int8V(x);
}

type int16V int16

func (v *int16V) String() string {
	return fmt.Sprint(*v);
}

func (v *int16V) Assign(o Value) {
	*v = int16V(o.(IntValue).Get());
}

func (v *int16V) Get() int64 {
	return int64(*v);
}

func (v *int16V) Set(x int64) {
	*v = int16V(x);
}

type int32V int32

func (v *int32V) String() string {
	return fmt.Sprint(*v);
}

func (v *int32V) Assign(o Value) {
	*v = int32V(o.(IntValue).Get());
}

func (v *int32V) Get() int64 {
	return int64(*v);
}

func (v *int32V) Set(x int64) {
	*v = int32V(x);
}

type int64V int64

func (v *int64V) String() string {
	return fmt.Sprint(*v);
}

func (v *int64V) Assign(o Value) {
	*v = int64V(o.(IntValue).Get());
}

func (v *int64V) Get() int64 {
	return int64(*v);
}

func (v *int64V) Set(x int64) {
	*v = int64V(x);
}

type intV int

func (v *intV) String() string {
	return fmt.Sprint(*v);
}

func (v *intV) Assign(o Value) {
	*v = intV(o.(IntValue).Get());
}

func (v *intV) Get() int64 {
	return int64(*v);
}

func (v *intV) Set(x int64) {
	*v = intV(x);
}

func (t *intType) Zero() Value {
	switch t.Bits {
	case 8:
		res := int8V(0);
		return &res;
	case 16:
		res := int16V(0);
		return &res;
	case 32:
		res := int32V(0);
		return &res;
	case 64:
		res := int64V(0);
		return &res;

	case 0:
		res := intV(0);
		return &res;
	}
	panic("unexpected int bit count: ", t.Bits);
}

/*
 * Ideal int
 */

type idealIntV struct {
	V *bignum.Integer;
}

func (v *idealIntV) String() string {
	return v.V.String();
}

func (v *idealIntV) Assign(o Value) {
	v.V = o.(IdealIntValue).Get();
}

func (v *idealIntV) Get() *bignum.Integer {
	return v.V;
}

func (t *idealIntType) Zero() Value {
	return &idealIntV{bignum.Int(0)};
}

/*
 * Float
 */

type float32V float32

func (v *float32V) String() string {
	return fmt.Sprint(*v);
}

func (v *float32V) Assign(o Value) {
	*v = float32V(o.(FloatValue).Get());
}

func (v *float32V) Get() float64 {
	return float64(*v);
}

func (v *float32V) Set(x float64) {
	*v = float32V(x);
}

type float64V float64

func (v *float64V) String() string {
	return fmt.Sprint(*v);
}

func (v *float64V) Assign(o Value) {
	*v = float64V(o.(FloatValue).Get());
}

func (v *float64V) Get() float64 {
	return float64(*v);
}

func (v *float64V) Set(x float64) {
	*v = float64V(x);
}

type floatV float

func (v *floatV) String() string {
	return fmt.Sprint(*v);
}

func (v *floatV) Assign(o Value) {
	*v = floatV(o.(FloatValue).Get());
}

func (v *floatV) Get() float64 {
	return float64(*v);
}

func (v *floatV) Set(x float64) {
	*v = floatV(x);
}

func (t *floatType) Zero() Value {
	switch t.Bits {
	case 32:
		res := float32V(0);
		return &res;
	case 64:
		res := float64V(0);
		return &res;
	case 0:
		res := floatV(0);
		return &res;
	}
	panic("unexpected float bit count: ", t.Bits);
}

/*
 * Ideal float
 */

type idealFloatV struct {
	V *bignum.Rational;
}

func (v *idealFloatV) String() string {
	return ratToString(v.V);
}

func (v *idealFloatV) Assign(o Value) {
	v.V = o.(IdealFloatValue).Get();
}

func (v *idealFloatV) Get() *bignum.Rational {
	return v.V;
}

func (t *idealFloatType) Zero() Value {
	return &idealFloatV{bignum.Rat(1, 0)};
}

/*
 * String
 */

type stringV string

func (v *stringV) String() string {
	return fmt.Sprint(*v);
}

func (v *stringV) Assign(o Value) {
	*v = stringV(o.(StringValue).Get());
}

func (v *stringV) Get() string {
	return string(*v);
}

func (v *stringV) Set(x string) {
	*v = stringV(x);
}

func (t *stringType) Zero() Value {
	res := stringV("");
	return &res;
}

/*
 * Array
 */

type arrayV []Value

func (v *arrayV) String() string {
	return fmt.Sprint(*v);
}

func (v *arrayV) Assign(o Value) {
	oa := o.(ArrayValue);
	l := int64(len(*v));
	for i := int64(0); i < l; i++ {
		(*v)[i].Assign(oa.Elem(i));
	}
}

func (v *arrayV) Get() ArrayValue {
	return v;
}

func (v *arrayV) Elem(i int64) Value {
	return (*v)[i];
}

func (t *ArrayType) Zero() Value {
	res := arrayV(make([]Value, t.Len));
	// TODO(austin) It's unfortunate that each element is
	// separately heap allocated.  We could add ZeroArray to
	// everything, though that doesn't help with multidimensional
	// arrays.  Or we could do something unsafe.  We'll have this
	// same problem with structs.
	for i := int64(0); i < t.Len; i++ {
		res[i] = t.Elem.Zero();
	}
	return &res;
}

/*
 * Pointer
 */

type ptrV struct {
	// nil if the pointer is nil
	target Value;
}

func (v *ptrV) String() string {
	return "&" + v.target.String();
}

func (v *ptrV) Assign(o Value) {
	v.target = o.(PtrValue).Get();
}

func (v *ptrV) Get() Value {
	return v.target;
}

func (v *ptrV) Set(x Value) {
	v.target = x;
}

func (t *PtrType) Zero() Value {
	return &ptrV{nil};
}

/*
 * Functions
 */

type funcV struct {
	target Func;
}

func (v *funcV) String() string {
	// TODO(austin) Rob wants to see the definition
	return "func {...}";
}

func (v *funcV) Assign(o Value) {
	v.target = o.(FuncValue).Get();
}

func (v *funcV) Get() Func {
	return v.target;
}

func (v *funcV) Set(x Func) {
	v.target = x;
}

func (t *FuncType) Zero() Value {
	return &funcV{nil};
}

/*
 * Universal constants
 */

// TODO(austin) Nothing complains if I accidentally define init with
// arguments.  Is this intentional?
func init() {
	s := universe;

	true := boolV(true);
	s.DefineConst("true", BoolType, &true);
	false := boolV(false);
	s.DefineConst("false", BoolType, &false);
}
