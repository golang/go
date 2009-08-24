// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"bignum";
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

func (v *arrayV) From(i int64) ArrayValue {
	res := (*v)[i:len(*v)];
	return &res;
}

/*
 * Struct
 */

type structV []Value

// TODO(austin) Should these methods (and arrayV's) be on structV
// instead of *structV?
func (v *structV) String() string {
	res := "{";
	for i, v := range *v {
		if i > 0 {
			res += ", ";
		}
		res += v.String();
	}
	return res + "}";
}

func (v *structV) Assign(o Value) {
	oa := o.(StructValue);
	l := len(*v);
	for i := 0; i < l; i++ {
		(*v)[i].Assign(oa.Field(i));
	}
}

func (v *structV) Get() StructValue {
	return v;
}

func (v *structV) Field(i int) Value {
	return (*v)[i];
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

/*
 * Slices
 */

type sliceV struct {
	Slice;
}

func (v *sliceV) String() string {
	res := "{";
	for i := int64(0); i < v.Len; i++ {
		if i > 0 {
			res += ", ";
		}
		res += v.Base.Elem(i).String();
	}
	return res + "}";
}

func (v *sliceV) Assign(o Value) {
	v.Slice = o.(SliceValue).Get();
}

func (v *sliceV) Get() Slice {
	return v.Slice;
}

func (v *sliceV) Set(x Slice) {
	v.Slice = x;
}

/*
 * Maps
 */

type mapV struct {
	target Map;
}

func (v *mapV) String() string {
	res := "map[";
	i := 0;
	v.target.Iter(func(key interface{}, val Value) bool {
		if i > 0 {
			res += ", ";
		}
		i++;
		res += fmt.Sprint(key) + ":" + val.String();
		return true;
	});
	return res + "]";
}

func (v *mapV) Assign(o Value) {
	v.target = o.(MapValue).Get();
}

func (v *mapV) Get() Map {
	return v.target;
}

func (v *mapV) Set(x Map) {
	v.target = x;
}

type evalMap map[interface{}] Value

func (m evalMap) Len() int64 {
	return int64(len(m));
}

func (m evalMap) Elem(key interface{}) Value {
	if v, ok := m[key]; ok {
		return v;
	}
	return nil;
}

func (m evalMap) SetElem(key interface{}, val Value) {
	if val == nil {
		m[key] = nil, false;
	} else {
		m[key] = val;
	}
}

func (m evalMap) Iter(cb func(key interface{}, val Value) bool) {
	for k, v := range m {
		if !cb(k, v) {
			break;
		}
	}
}

/*
 * Multi-values
 */

type multiV []Value

func (v multiV) String() string {
	res := "(";
	for i, v := range v {
		if i > 0 {
			res += ", ";
		}
		res += v.String();
	}
	return res + ")";
}

func (v multiV) Assign(o Value) {
	omv := o.(multiV);
	for i := range v {
		v[i].Assign(omv[i]);
	}
}

/*
 * Universal constants
 */

// TODO(austin) Nothing complains if I accidentally define init with
// arguments.  Is this intentional?
func init() {
	s := universe;

	true := boolV(true);
	s.DefineConst("true", universePos, BoolType, &true);
	false := boolV(false);
	s.DefineConst("false", universePos, BoolType, &false);
}
