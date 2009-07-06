// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"reflect";
	"unsafe";
)

// Value is the common interface to reflection values.
// The implementations of Value (e.g., ArrayValue, StructValue)
// have additional type-specific methods.
type Value interface {
	// Type returns the value's type.
	Type()	Type;

	// Interface returns the value as an interface{}.
	Interface()	interface{};

	// CanSet returns whether the value can be changed.
	// If CanSet() returns false, calling the type-specific Set
	// will cause a crash.
	CanSet()	bool;

	// Addr returns a pointer to the underlying data.
	// It is for advanced clients that also
	// import the "unsafe" package.
	Addr()	uintptr;
}

/*
 * basic types
 */

// BoolValue represents a bool value.
type BoolValue struct {
}

// Get returns the underlying bool value.
func (v *BoolValue) Get() bool {
}

// Set sets v to the value x.
func (v *BoolValue) Set(x bool) {
}

// FloatValue represents a float value.
type FloatValue struct {
}

// Get returns the underlying float value.
func (v *FloatValue) Get() float {
}

// Set sets v to the value x.
func (v *FloatValue) Set(x float) {
}

// Float32Value represents a float32 value.
type Float32Value struct {
}

// Get returns the underlying float32 value.
func (v *Float32Value) Get() float32 {
}

// Set sets v to the value x.
func (v *Float32Value) Set(x float32) {
}

// Float64Value represents a float64 value.
type Float64Value struct {
}

// Get returns the underlying float64 value.
func (v *Float64Value) Get() float64 {
}

// Set sets v to the value x.
func (v *Float64Value) Set(x float64) {
}

// IntValue represents an int value.
type IntValue struct {
}

// Get returns the underlying int value.
func (v *IntValue) Get() int {
}

// Set sets v to the value x.
func (v *IntValue) Set(x int) {
}

// Int8Value represents an int8 value.
type Int8Value struct {
}

// Get returns the underlying int8 value.
func (v *Int8Value) Get() int8 {
}

// Set sets v to the value x.
func (v *Int8Value) Set(x int8) {
}

// Int16Value represents an int16 value.
type Int16Value struct {
}

// Get returns the underlying int16 value.
func (v *Int16Value) Get() int16 {
}

// Set sets v to the value x.
func (v *Int16Value) Set(x int16) {
}

// Int32Value represents an int32 value.
type Int32Value struct {
}

// Get returns the underlying int32 value.
func (v *Int32Value) Get() int32 {
}

// Set sets v to the value x.
func (v *Int32Value) Set(x int32) {
}

// Int64Value represents an int64 value.
type Int64Value struct {
}

// Get returns the underlying int64 value.
func (v *Int64Value) Get() int64 {
}

// Set sets v to the value x.
func (v *Int64Value) Set(x int64) {
}

// StringValue represents a string value.
type StringValue struct {
}

// Get returns the underlying string value.
func (v *StringValue) Get() string {
}

// Set sets v to the value x.
func (v *StringValue) Set(x string) {
}

// UintValue represents a uint value.
type UintValue struct {
}

// Get returns the underlying uint value.
func (v *UintValue) Get() uint {
}

// Set sets v to the value x.
func (v *UintValue) Set(x uint) {
}

// Uint8Value represents a uint8 value.
type Uint8Value struct {
}

// Get returns the underlying uint8 value.
func (v *Uint8Value) Get() uint8 {
}

// Set sets v to the value x.
func (v *Uint8Value) Set(x uint8) {
}

// Uint16Value represents a uint16 value.
type Uint16Value struct {
}

// Get returns the underlying uint16 value.
func (v *Uint16Value) Get() uint16 {
}

// Set sets v to the value x.
func (v *Uint16Value) Set(x uint16) {
}

// Uint32Value represents a uint32 value.
type Uint32Value struct {
}

// Get returns the underlying uint32 value.
func (v *Uint32Value) Get() uint32 {
}

// Set sets v to the value x.
func (v *Uint32Value) Set(x uint32) {
}

// Uint64Value represents a uint64 value.
type Uint64Value struct {
}

// Get returns the underlying uint64 value.
func (v *Uint64Value) Get() uint64 {
}

// Set sets v to the value x.
func (v *Uint64Value) Set(x uint64) {
}

// UintptrValue represents a uintptr value.
type UintptrValue struct {
}

// Get returns the underlying uintptr value.
func (v *UintptrValue) Get() uintptr {
}

// Set sets v to the value x.
func (v *UintptrValue) Set(x uintptr) {
}

// UnsafePointerValue represents an unsafe.Pointer value.
type UnsafePointerValue struct {
}

// Get returns the underlying uintptr value.
// Get returns uintptr, not unsafe.Pointer, so that
// programs that do not import "unsafe" cannot
// obtain a value of unsafe.Pointer type from "reflect".
func (v *UnsafePointerValue) Get() uintptr {
}

// Set sets v to the value x.
func (v *UnsafePointerValue) Set(x unsafe.Pointer) {
}

/*
 * array
 */

// ArrayOrSliceValue is the common interface
// implemented by both ArrayValue and SliceValue.
type ArrayOrSliceValue interface {
	Value;
	Len() int;
	Cap() int;
	Elem(i int) Value;
	addr() addr;
}

// ArrayCopy copies the contents of src into dst until either
// dst has been filled or src has been exhausted.
// It returns the number of elements copied.
// The arrays dst and src must have the same element type.
func ArrayCopy(dst, src ArrayOrSliceValue) int {
}

// An ArrayValue represents an array.
type ArrayValue struct {
}

// Len returns the length of the array.
func (v *ArrayValue) Len() int {
}

// Cap returns the capacity of the array (equal to Len()).
func (v *ArrayValue) Cap() int {
}

// addr returns the base address of the data in the array.
func (v *ArrayValue) addr() addr {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ArrayValue) Set(x *ArrayValue) {
}

// Elem returns the i'th element of v.
func (v *ArrayValue) Elem(i int) Value {
}

/*
 * slice
 */

// runtime representation of slice
type SliceHeader struct {
	Data uintptr;
	Len uint32;
	Cap uint32;
}

// A SliceValue represents a slice.
type SliceValue struct {
}

func (v *SliceValue) slice() *SliceHeader {
}

// IsNil returns whether v is a nil slice.
func (v *SliceValue) IsNil() bool {
}

// Len returns the length of the slice.
func (v *SliceValue) Len() int {
}

// Cap returns the capacity of the slice.
func (v *SliceValue) Cap() int {
}

// addr returns the base address of the data in the slice.
func (v *SliceValue) addr() addr {
}

// SetLen changes the length of v.
// The new length n must be between 0 and the capacity, inclusive.
func (v *SliceValue) SetLen(n int) {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *SliceValue) Set(x *SliceValue) {
}

// Slice returns a sub-slice of the slice v.
func (v *SliceValue) Slice(beg, end int) *SliceValue {
}

// Elem returns the i'th element of v.
func (v *SliceValue) Elem(i int) Value {
}

// MakeSlice creates a new zero-initialized slice value
// for the specified slice type, length, and capacity.
func MakeSlice(typ *SliceType, len, cap int) *SliceValue {
}

/*
 * chan
 */

// A ChanValue represents a chan.
type ChanValue struct {
}

// IsNil returns whether v is a nil channel.
func (v *ChanValue) IsNil() bool {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ChanValue) Set(x *ChanValue) {
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *ChanValue) Get() uintptr {
}

// Send sends x on the channel v.
func (v *ChanValue) Send(x Value) {
}

// Recv receives and returns a value from the channel v.
func (v *ChanValue) Recv() Value {
}

// TrySend attempts to sends x on the channel v but will not block.
// It returns true if the value was sent, false otherwise.
func (v *ChanValue) TrySend(x Value) bool {
}

// TryRecv attempts to receive a value from the channel v but will not block.
// It returns the value if one is received, nil otherwise.
func (v *ChanValue) TryRecv() Value {
}

/*
 * func
 */

// A FuncValue represents a function value.
type FuncValue struct {
}

// IsNil returns whether v is a nil function.
func (v *FuncValue) IsNil() bool {
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *FuncValue) Get() uintptr {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *FuncValue) Set(x *FuncValue) {
}

// Call calls the function v with input parameters in.
// It returns the function's output parameters as Values.
func (v *FuncValue) Call(in []Value) []Value {
}


/*
 * interface
 */

// An InterfaceValue represents an interface value.
type InterfaceValue struct {
}

// No Get because v.Interface() is available.

// IsNil returns whether v is a nil interface value.
func (v *InterfaceValue) IsNil() bool {
}

// Elem returns the concrete value stored in the interface value v.
func (v *InterfaceValue) Elem() Value {
}

// Set assigns x to v.
func (v *InterfaceValue) Set(x interface{}) {
}

/*
 * map
 */

// A MapValue represents a map value.
type MapValue struct {
}

// IsNil returns whether v is a nil map value.
func (v *MapValue) IsNil() bool {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *MapValue) Set(x *MapValue) {
}

// Elem returns the value associated with key in the map v.
// It returns nil if key is not found in the map.
func (v *MapValue) Elem(key Value) Value {
}

// Len returns the number of keys in the map v.
func (v *MapValue) Len() int {
}

// Keys returns a slice containing all the keys present in the map,
// in unspecified order.
func (v *MapValue) Keys() []Value {
}

/*
 * ptr
 */

// A PtrValue represents a pointer.
type PtrValue struct {
}

// IsNil returns whether v is a nil pointer.
func (v *PtrValue) IsNil() bool {
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *PtrValue) Get() uintptr {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *PtrValue) Set(x *PtrValue) {
}

// PointTo changes v to point to x.
func (v *PtrValue) PointTo(x Value) {
}

// Elem returns the value that v points to.
// If v is a nil pointer, Elem returns a nil Value.
func (v *PtrValue) Elem() Value {
}

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a nil Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
}

/*
 * struct
 */

// A StructValue represents a struct value.
type StructValue struct {
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *StructValue) Set(x *StructValue) {
}

// Field returns the i'th field of the struct.
func (v *StructValue) Field(i int) Value {
}

// NumField returns the number of fields in the struct.
func (v *StructValue) NumField() int {
}

/*
 * constructors
 */

// Typeof returns the reflection Type of the value in the interface{}.
func Typeof(i interface{}) Type {
}

// NewValue returns a new Value initialized to the concrete value
// stored in the interface i.  NewValue(nil) returns nil.
func NewValue(i interface{}) Value {
}

// MakeZeroValue returns a zero Value for the specified Type.
func MakeZero(typ Type) Value {
}
