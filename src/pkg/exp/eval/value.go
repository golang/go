// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package eval

import (
	"big"
	"fmt"
)

type Value interface {
	String() string
	// Assign copies another value into this one.  It should
	// assume that the other value satisfies the same specific
	// value interface (BoolValue, etc.), but must not assume
	// anything about its specific type.
	Assign(t *Thread, o Value)
}

type BoolValue interface {
	Value
	Get(*Thread) bool
	Set(*Thread, bool)
}

type UintValue interface {
	Value
	Get(*Thread) uint64
	Set(*Thread, uint64)
}

type IntValue interface {
	Value
	Get(*Thread) int64
	Set(*Thread, int64)
}

// TODO(austin) IdealIntValue and IdealFloatValue should not exist
// because ideals are not l-values.
type IdealIntValue interface {
	Value
	Get() *big.Int
}

type FloatValue interface {
	Value
	Get(*Thread) float64
	Set(*Thread, float64)
}

type IdealFloatValue interface {
	Value
	Get() *big.Rat
}

type StringValue interface {
	Value
	Get(*Thread) string
	Set(*Thread, string)
}

type ArrayValue interface {
	Value
	// TODO(austin) Get() is here for uniformity, but is
	// completely useless.  If a lot of other types have similarly
	// useless Get methods, just special-case these uses.
	Get(*Thread) ArrayValue
	Elem(*Thread, int64) Value
	// Sub returns an ArrayValue backed by the same array that
	// starts from element i and has length len.
	Sub(i int64, len int64) ArrayValue
}

type StructValue interface {
	Value
	// TODO(austin) This is another useless Get()
	Get(*Thread) StructValue
	Field(*Thread, int) Value
}

type PtrValue interface {
	Value
	Get(*Thread) Value
	Set(*Thread, Value)
}

type Func interface {
	NewFrame() *Frame
	Call(*Thread)
}

type FuncValue interface {
	Value
	Get(*Thread) Func
	Set(*Thread, Func)
}

type Interface struct {
	Type  Type
	Value Value
}

type InterfaceValue interface {
	Value
	Get(*Thread) Interface
	Set(*Thread, Interface)
}

type Slice struct {
	Base     ArrayValue
	Len, Cap int64
}

type SliceValue interface {
	Value
	Get(*Thread) Slice
	Set(*Thread, Slice)
}

type Map interface {
	Len(*Thread) int64
	// Retrieve an element from the map, returning nil if it does
	// not exist.
	Elem(t *Thread, key interface{}) Value
	// Set an entry in the map.  If val is nil, delete the entry.
	SetElem(t *Thread, key interface{}, val Value)
	// TODO(austin)  Perhaps there should be an iterator interface instead.
	Iter(func(key interface{}, val Value) bool)
}

type MapValue interface {
	Value
	Get(*Thread) Map
	Set(*Thread, Map)
}

/*
 * Bool
 */

type boolV bool

func (v *boolV) String() string { return fmt.Sprint(*v) }

func (v *boolV) Assign(t *Thread, o Value) { *v = boolV(o.(BoolValue).Get(t)) }

func (v *boolV) Get(*Thread) bool { return bool(*v) }

func (v *boolV) Set(t *Thread, x bool) { *v = boolV(x) }

/*
 * Uint
 */

type uint8V uint8

func (v *uint8V) String() string { return fmt.Sprint(*v) }

func (v *uint8V) Assign(t *Thread, o Value) { *v = uint8V(o.(UintValue).Get(t)) }

func (v *uint8V) Get(*Thread) uint64 { return uint64(*v) }

func (v *uint8V) Set(t *Thread, x uint64) { *v = uint8V(x) }

type uint16V uint16

func (v *uint16V) String() string { return fmt.Sprint(*v) }

func (v *uint16V) Assign(t *Thread, o Value) { *v = uint16V(o.(UintValue).Get(t)) }

func (v *uint16V) Get(*Thread) uint64 { return uint64(*v) }

func (v *uint16V) Set(t *Thread, x uint64) { *v = uint16V(x) }

type uint32V uint32

func (v *uint32V) String() string { return fmt.Sprint(*v) }

func (v *uint32V) Assign(t *Thread, o Value) { *v = uint32V(o.(UintValue).Get(t)) }

func (v *uint32V) Get(*Thread) uint64 { return uint64(*v) }

func (v *uint32V) Set(t *Thread, x uint64) { *v = uint32V(x) }

type uint64V uint64

func (v *uint64V) String() string { return fmt.Sprint(*v) }

func (v *uint64V) Assign(t *Thread, o Value) { *v = uint64V(o.(UintValue).Get(t)) }

func (v *uint64V) Get(*Thread) uint64 { return uint64(*v) }

func (v *uint64V) Set(t *Thread, x uint64) { *v = uint64V(x) }

type uintV uint

func (v *uintV) String() string { return fmt.Sprint(*v) }

func (v *uintV) Assign(t *Thread, o Value) { *v = uintV(o.(UintValue).Get(t)) }

func (v *uintV) Get(*Thread) uint64 { return uint64(*v) }

func (v *uintV) Set(t *Thread, x uint64) { *v = uintV(x) }

type uintptrV uintptr

func (v *uintptrV) String() string { return fmt.Sprint(*v) }

func (v *uintptrV) Assign(t *Thread, o Value) { *v = uintptrV(o.(UintValue).Get(t)) }

func (v *uintptrV) Get(*Thread) uint64 { return uint64(*v) }

func (v *uintptrV) Set(t *Thread, x uint64) { *v = uintptrV(x) }

/*
 * Int
 */

type int8V int8

func (v *int8V) String() string { return fmt.Sprint(*v) }

func (v *int8V) Assign(t *Thread, o Value) { *v = int8V(o.(IntValue).Get(t)) }

func (v *int8V) Get(*Thread) int64 { return int64(*v) }

func (v *int8V) Set(t *Thread, x int64) { *v = int8V(x) }

type int16V int16

func (v *int16V) String() string { return fmt.Sprint(*v) }

func (v *int16V) Assign(t *Thread, o Value) { *v = int16V(o.(IntValue).Get(t)) }

func (v *int16V) Get(*Thread) int64 { return int64(*v) }

func (v *int16V) Set(t *Thread, x int64) { *v = int16V(x) }

type int32V int32

func (v *int32V) String() string { return fmt.Sprint(*v) }

func (v *int32V) Assign(t *Thread, o Value) { *v = int32V(o.(IntValue).Get(t)) }

func (v *int32V) Get(*Thread) int64 { return int64(*v) }

func (v *int32V) Set(t *Thread, x int64) { *v = int32V(x) }

type int64V int64

func (v *int64V) String() string { return fmt.Sprint(*v) }

func (v *int64V) Assign(t *Thread, o Value) { *v = int64V(o.(IntValue).Get(t)) }

func (v *int64V) Get(*Thread) int64 { return int64(*v) }

func (v *int64V) Set(t *Thread, x int64) { *v = int64V(x) }

type intV int

func (v *intV) String() string { return fmt.Sprint(*v) }

func (v *intV) Assign(t *Thread, o Value) { *v = intV(o.(IntValue).Get(t)) }

func (v *intV) Get(*Thread) int64 { return int64(*v) }

func (v *intV) Set(t *Thread, x int64) { *v = intV(x) }

/*
 * Ideal int
 */

type idealIntV struct {
	V *big.Int
}

func (v *idealIntV) String() string { return v.V.String() }

func (v *idealIntV) Assign(t *Thread, o Value) {
	v.V = o.(IdealIntValue).Get()
}

func (v *idealIntV) Get() *big.Int { return v.V }

/*
 * Float
 */

type float32V float32

func (v *float32V) String() string { return fmt.Sprint(*v) }

func (v *float32V) Assign(t *Thread, o Value) { *v = float32V(o.(FloatValue).Get(t)) }

func (v *float32V) Get(*Thread) float64 { return float64(*v) }

func (v *float32V) Set(t *Thread, x float64) { *v = float32V(x) }

type float64V float64

func (v *float64V) String() string { return fmt.Sprint(*v) }

func (v *float64V) Assign(t *Thread, o Value) { *v = float64V(o.(FloatValue).Get(t)) }

func (v *float64V) Get(*Thread) float64 { return float64(*v) }

func (v *float64V) Set(t *Thread, x float64) { *v = float64V(x) }

/*
 * Ideal float
 */

type idealFloatV struct {
	V *big.Rat
}

func (v *idealFloatV) String() string { return v.V.FloatString(6) }

func (v *idealFloatV) Assign(t *Thread, o Value) {
	v.V = o.(IdealFloatValue).Get()
}

func (v *idealFloatV) Get() *big.Rat { return v.V }

/*
 * String
 */

type stringV string

func (v *stringV) String() string { return fmt.Sprint(*v) }

func (v *stringV) Assign(t *Thread, o Value) { *v = stringV(o.(StringValue).Get(t)) }

func (v *stringV) Get(*Thread) string { return string(*v) }

func (v *stringV) Set(t *Thread, x string) { *v = stringV(x) }

/*
 * Array
 */

type arrayV []Value

func (v *arrayV) String() string {
	res := "{"
	for i, e := range *v {
		if i > 0 {
			res += ", "
		}
		res += e.String()
	}
	return res + "}"
}

func (v *arrayV) Assign(t *Thread, o Value) {
	oa := o.(ArrayValue)
	l := int64(len(*v))
	for i := int64(0); i < l; i++ {
		(*v)[i].Assign(t, oa.Elem(t, i))
	}
}

func (v *arrayV) Get(*Thread) ArrayValue { return v }

func (v *arrayV) Elem(t *Thread, i int64) Value {
	return (*v)[i]
}

func (v *arrayV) Sub(i int64, len int64) ArrayValue {
	res := (*v)[i : i+len]
	return &res
}

/*
 * Struct
 */

type structV []Value

// TODO(austin) Should these methods (and arrayV's) be on structV
// instead of *structV?
func (v *structV) String() string {
	res := "{"
	for i, v := range *v {
		if i > 0 {
			res += ", "
		}
		res += v.String()
	}
	return res + "}"
}

func (v *structV) Assign(t *Thread, o Value) {
	oa := o.(StructValue)
	l := len(*v)
	for i := 0; i < l; i++ {
		(*v)[i].Assign(t, oa.Field(t, i))
	}
}

func (v *structV) Get(*Thread) StructValue { return v }

func (v *structV) Field(t *Thread, i int) Value {
	return (*v)[i]
}

/*
 * Pointer
 */

type ptrV struct {
	// nil if the pointer is nil
	target Value
}

func (v *ptrV) String() string {
	if v.target == nil {
		return "<nil>"
	}
	return "&" + v.target.String()
}

func (v *ptrV) Assign(t *Thread, o Value) { v.target = o.(PtrValue).Get(t) }

func (v *ptrV) Get(*Thread) Value { return v.target }

func (v *ptrV) Set(t *Thread, x Value) { v.target = x }

/*
 * Functions
 */

type funcV struct {
	target Func
}

func (v *funcV) String() string {
	// TODO(austin) Rob wants to see the definition
	return "func {...}"
}

func (v *funcV) Assign(t *Thread, o Value) { v.target = o.(FuncValue).Get(t) }

func (v *funcV) Get(*Thread) Func { return v.target }

func (v *funcV) Set(t *Thread, x Func) { v.target = x }

/*
 * Interfaces
 */

type interfaceV struct {
	Interface
}

func (v *interfaceV) String() string {
	if v.Type == nil || v.Value == nil {
		return "<nil>"
	}
	return v.Value.String()
}

func (v *interfaceV) Assign(t *Thread, o Value) {
	v.Interface = o.(InterfaceValue).Get(t)
}

func (v *interfaceV) Get(*Thread) Interface { return v.Interface }

func (v *interfaceV) Set(t *Thread, x Interface) {
	v.Interface = x
}

/*
 * Slices
 */

type sliceV struct {
	Slice
}

func (v *sliceV) String() string {
	if v.Base == nil {
		return "<nil>"
	}
	return v.Base.Sub(0, v.Len).String()
}

func (v *sliceV) Assign(t *Thread, o Value) { v.Slice = o.(SliceValue).Get(t) }

func (v *sliceV) Get(*Thread) Slice { return v.Slice }

func (v *sliceV) Set(t *Thread, x Slice) { v.Slice = x }

/*
 * Maps
 */

type mapV struct {
	target Map
}

func (v *mapV) String() string {
	if v.target == nil {
		return "<nil>"
	}
	res := "map["
	i := 0
	v.target.Iter(func(key interface{}, val Value) bool {
		if i > 0 {
			res += ", "
		}
		i++
		res += fmt.Sprint(key) + ":" + val.String()
		return true
	})
	return res + "]"
}

func (v *mapV) Assign(t *Thread, o Value) { v.target = o.(MapValue).Get(t) }

func (v *mapV) Get(*Thread) Map { return v.target }

func (v *mapV) Set(t *Thread, x Map) { v.target = x }

type evalMap map[interface{}]Value

func (m evalMap) Len(t *Thread) int64 { return int64(len(m)) }

func (m evalMap) Elem(t *Thread, key interface{}) Value {
	return m[key]
}

func (m evalMap) SetElem(t *Thread, key interface{}, val Value) {
	if val == nil {
		m[key] = nil, false
	} else {
		m[key] = val
	}
}

func (m evalMap) Iter(cb func(key interface{}, val Value) bool) {
	for k, v := range m {
		if !cb(k, v) {
			break
		}
	}
}

/*
 * Multi-values
 */

type multiV []Value

func (v multiV) String() string {
	res := "("
	for i, v := range v {
		if i > 0 {
			res += ", "
		}
		res += v.String()
	}
	return res + ")"
}

func (v multiV) Assign(t *Thread, o Value) {
	omv := o.(multiV)
	for i := range v {
		v[i].Assign(t, omv[i])
	}
}

/*
 * Universal constants
 */

func init() {
	s := universe

	true := boolV(true)
	s.DefineConst("true", universePos, BoolType, &true)
	false := boolV(false)
	s.DefineConst("false", universePos, BoolType, &false)
}
