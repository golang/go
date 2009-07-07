// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"reflect";
	"unsafe";
)

const cannotSet = "cannot set value obtained via unexported struct field"

// TODO: This will have to go away when
// the new gc goes in.
func memmove(dst, src, n uintptr) {
	var p uintptr;	// dummy for sizeof
	const ptrsize = uintptr(unsafe.Sizeof(p));
	switch {
	case src < dst && src+n > dst:
		// byte copy backward
		// careful: i is unsigned
		for i := n; i > 0; {
			i--;
			*(*byte)(addr(dst+i)) = *(*byte)(addr(src+i));
		}
	case (n|src|dst) & (ptrsize-1) != 0:
		// byte copy forward
		for i := uintptr(0); i < n; i++ {
			*(*byte)(addr(dst+i)) = *(*byte)(addr(src+i));
		}
	default:
		// word copy forward
		for i := uintptr(0); i < n; i += ptrsize {
			*(*uintptr)(addr(dst+i)) = *(*uintptr)(addr(src+i));
		}
	}
}

// Value is the common interface to reflection values.
// The implementations of Value (e.g., ArrayValue, StructValue)
// have additional type-specific methods.
type Value interface {
	// Type returns the value's type.
	Type()	Type;

	// Interface returns the value as an interface{}.
	Interface()	interface{};

	// CanSet returns whether the value can be changed.
	// Values obtained by the use of non-exported struct fields
	// can be used in Get but not Set.
	// If CanSet() returns false, calling the type-specific Set
	// will cause a crash.
	CanSet()	bool;

	// Addr returns a pointer to the underlying data.
	// It is for advanced clients that also
	// import the "unsafe" package.
	Addr()	uintptr;
}

type value struct {
	typ Type;
	addr addr;
	canSet bool;
}

func (v *value) Type() Type {
	return v.typ
}

func (v *value) Addr() uintptr {
	return uintptr(v.addr);
}

type InterfaceValue struct
type StructValue struct

func (v *value) Interface() interface{} {
	if typ, ok := v.typ.(*InterfaceType); ok {
		// There are two different representations of interface values,
		// one if the interface type has methods and one if it doesn't.
		// These two representations require different expressions
		// to extract correctly.
		if typ.NumMethod() == 0 {
			// Extract as interface value without methods.
			return *(*interface{})(v.addr)
		}
		// Extract from v.addr as interface value with methods.
		return *(*interface{ m() })(v.addr)
	}
	return unsafe.Unreflect(v.typ, unsafe.Pointer(v.addr));
}

func (v *value) CanSet() bool {
	return v.canSet;
}

func newValue(typ Type, addr addr, canSet bool) Value
func NewValue(i interface{}) Value

/*
 * basic types
 */

// BoolValue represents a bool value.
type BoolValue struct {
	value;
}

// Get returns the underlying bool value.
func (v *BoolValue) Get() bool {
	return *(*bool)(v.addr);
}

// Set sets v to the value x.
func (v *BoolValue) Set(x bool) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*bool)(v.addr) = x;
}

// FloatValue represents a float value.
type FloatValue struct {
	value;
}

// Get returns the underlying float value.
func (v *FloatValue) Get() float {
	return *(*float)(v.addr);
}

// Set sets v to the value x.
func (v *FloatValue) Set(x float) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*float)(v.addr) = x;
}

// Float32Value represents a float32 value.
type Float32Value struct {
	value;
}

// Get returns the underlying float32 value.
func (v *Float32Value) Get() float32 {
	return *(*float32)(v.addr);
}

// Set sets v to the value x.
func (v *Float32Value) Set(x float32) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*float32)(v.addr) = x;
}

// Float64Value represents a float64 value.
type Float64Value struct {
	value;
}

// Get returns the underlying float64 value.
func (v *Float64Value) Get() float64 {
	return *(*float64)(v.addr);
}

// Set sets v to the value x.
func (v *Float64Value) Set(x float64) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*float64)(v.addr) = x;
}

// IntValue represents an int value.
type IntValue struct {
	value;
}

// Get returns the underlying int value.
func (v *IntValue) Get() int {
	return *(*int)(v.addr);
}

// Set sets v to the value x.
func (v *IntValue) Set(x int) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int)(v.addr) = x;
}

// Int8Value represents an int8 value.
type Int8Value struct {
	value;
}

// Get returns the underlying int8 value.
func (v *Int8Value) Get() int8 {
	return *(*int8)(v.addr);
}

// Set sets v to the value x.
func (v *Int8Value) Set(x int8) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int8)(v.addr) = x;
}

// Int16Value represents an int16 value.
type Int16Value struct {
	value;
}

// Get returns the underlying int16 value.
func (v *Int16Value) Get() int16 {
	return *(*int16)(v.addr);
}

// Set sets v to the value x.
func (v *Int16Value) Set(x int16) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int16)(v.addr) = x;
}

// Int32Value represents an int32 value.
type Int32Value struct {
	value;
}

// Get returns the underlying int32 value.
func (v *Int32Value) Get() int32 {
	return *(*int32)(v.addr);
}

// Set sets v to the value x.
func (v *Int32Value) Set(x int32) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int32)(v.addr) = x;
}

// Int64Value represents an int64 value.
type Int64Value struct {
	value;
}

// Get returns the underlying int64 value.
func (v *Int64Value) Get() int64 {
	return *(*int64)(v.addr);
}

// Set sets v to the value x.
func (v *Int64Value) Set(x int64) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*int64)(v.addr) = x;
}

// StringValue represents a string value.
type StringValue struct {
	value;
}

// Get returns the underlying string value.
func (v *StringValue) Get() string {
	return *(*string)(v.addr);
}

// Set sets v to the value x.
func (v *StringValue) Set(x string) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*string)(v.addr) = x;
}

// UintValue represents a uint value.
type UintValue struct {
	value;
}

// Get returns the underlying uint value.
func (v *UintValue) Get() uint {
	return *(*uint)(v.addr);
}

// Set sets v to the value x.
func (v *UintValue) Set(x uint) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint)(v.addr) = x;
}

// Uint8Value represents a uint8 value.
type Uint8Value struct {
	value;
}

// Get returns the underlying uint8 value.
func (v *Uint8Value) Get() uint8 {
	return *(*uint8)(v.addr);
}

// Set sets v to the value x.
func (v *Uint8Value) Set(x uint8) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint8)(v.addr) = x;
}

// Uint16Value represents a uint16 value.
type Uint16Value struct {
	value;
}

// Get returns the underlying uint16 value.
func (v *Uint16Value) Get() uint16 {
	return *(*uint16)(v.addr);
}

// Set sets v to the value x.
func (v *Uint16Value) Set(x uint16) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint16)(v.addr) = x;
}

// Uint32Value represents a uint32 value.
type Uint32Value struct {
	value;
}

// Get returns the underlying uint32 value.
func (v *Uint32Value) Get() uint32 {
	return *(*uint32)(v.addr);
}

// Set sets v to the value x.
func (v *Uint32Value) Set(x uint32) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint32)(v.addr) = x;
}

// Uint64Value represents a uint64 value.
type Uint64Value struct {
	value;
}

// Get returns the underlying uint64 value.
func (v *Uint64Value) Get() uint64 {
	return *(*uint64)(v.addr);
}

// Set sets v to the value x.
func (v *Uint64Value) Set(x uint64) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uint64)(v.addr) = x;
}

// UintptrValue represents a uintptr value.
type UintptrValue struct {
	value;
}

// Get returns the underlying uintptr value.
func (v *UintptrValue) Get() uintptr {
	return *(*uintptr)(v.addr);
}

// Set sets v to the value x.
func (v *UintptrValue) Set(x uintptr) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*uintptr)(v.addr) = x;
}

// UnsafePointerValue represents an unsafe.Pointer value.
type UnsafePointerValue struct {
	value;
}

// Get returns the underlying uintptr value.
// Get returns uintptr, not unsafe.Pointer, so that
// programs that do not import "unsafe" cannot
// obtain a value of unsafe.Pointer type from "reflect".
func (v *UnsafePointerValue) Get() uintptr {
	return uintptr(*(*unsafe.Pointer)(v.addr));
}

// Set sets v to the value x.
func (v *UnsafePointerValue) Set(x unsafe.Pointer) {
	if !v.canSet {
		panic(cannotSet);
	}
	*(*unsafe.Pointer)(v.addr) = x;
}

func typesMustMatch(t1, t2 reflect.Type) {
	if t1 != t2 {
		panicln("type mismatch:", t1, "!=", t2);
	}
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
	// TODO: This will have to move into the runtime
	// once the real gc goes in.
	de := dst.Type().(ArrayOrSliceType).Elem();
	se := src.Type().(ArrayOrSliceType).Elem();
	typesMustMatch(de, se);
	n := dst.Len();
	if xn := src.Len(); n > xn {
		n = xn;
	}
	memmove(uintptr(dst.addr()), uintptr(src.addr()), uintptr(n) * de.Size());
	return n;
}

// An ArrayValue represents an array.
type ArrayValue struct {
	value
}

// Len returns the length of the array.
func (v *ArrayValue) Len() int {
	return v.typ.(*ArrayType).Len();
}

// Cap returns the capacity of the array (equal to Len()).
func (v *ArrayValue) Cap() int {
	return v.typ.(*ArrayType).Len();
}

// addr returns the base address of the data in the array.
func (v *ArrayValue) addr() addr {
	return v.value.addr;
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ArrayValue) Set(x *ArrayValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	ArrayCopy(v, x);
}

// Elem returns the i'th element of v.
func (v *ArrayValue) Elem(i int) Value {
	typ := v.typ.(*ArrayType).Elem();
	n := v.Len();
	if i < 0 || i >= n {
		panic("index", i, "in array len", n);
	}
	p := addr(uintptr(v.addr()) + uintptr(i)*typ.Size());
	return newValue(typ, p, v.canSet);
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
	value
}

func (v *SliceValue) slice() *SliceHeader {
	return (*SliceHeader)(v.value.addr);
}

// IsNil returns whether v is a nil slice.
func (v *SliceValue) IsNil() bool {
	return v.slice().Data == 0;
}

// Len returns the length of the slice.
func (v *SliceValue) Len() int {
	return int(v.slice().Len);
}

// Cap returns the capacity of the slice.
func (v *SliceValue) Cap() int {
	return int(v.slice().Cap);
}

// addr returns the base address of the data in the slice.
func (v *SliceValue) addr() addr {
	return addr(v.slice().Data);
}

// SetLen changes the length of v.
// The new length n must be between 0 and the capacity, inclusive.
func (v *SliceValue) SetLen(n int) {
	s := v.slice();
	if n < 0 || n > int(s.Cap) {
		panicln("SetLen", n, "with capacity", s.Cap);
	}
	s.Len = uint32(n);
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *SliceValue) Set(x *SliceValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*v.slice() = *x.slice();
}

// Slice returns a sub-slice of the slice v.
func (v *SliceValue) Slice(beg, end int) *SliceValue {
	cap := v.Cap();
	if beg < 0 || end < beg || end > cap {
		panic("slice bounds [", beg, ":", end, "] with capacity ", cap);
	}
	typ := v.typ.(*SliceType);
	s := new(SliceHeader);
	s.Data = uintptr(v.addr()) + uintptr(beg) * typ.Elem().Size();
	s.Len = uint32(end - beg);
	s.Cap = uint32(cap - beg);
	return newValue(typ, addr(s), v.canSet).(*SliceValue);
}

// Elem returns the i'th element of v.
func (v *SliceValue) Elem(i int) Value {
	typ := v.typ.(*SliceType).Elem();
	n := v.Len();
	if i < 0 || i >= n {
		panicln("index", i, "in array of length", n);
	}
	p := addr(uintptr(v.addr()) + uintptr(i)*typ.Size());
	return newValue(typ, p, v.canSet);
}

// MakeSlice creates a new zero-initialized slice value
// for the specified slice type, length, and capacity.
func MakeSlice(typ *SliceType, len, cap int) *SliceValue {
	s := new(SliceHeader);
	size := typ.Elem().Size() * uintptr(cap);
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	s.Data = uintptr(addr(&data[0]));
	s.Len = uint32(len);
	s.Cap = uint32(cap);
	return newValue(typ, addr(s), true).(*SliceValue);
}

/*
 * chan
 */

// A ChanValue represents a chan.
type ChanValue struct {
	value
}

// IsNil returns whether v is a nil channel.
func (v *ChanValue) IsNil() bool {
	return *(*uintptr)(v.addr) == 0;
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *ChanValue) Set(x *ChanValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *ChanValue) Get() uintptr {
	return *(*uintptr)(v.addr);
}

// Send sends x on the channel v.
func (v *ChanValue) Send(x Value) {
	panic("unimplemented: channel Send");
}

// Recv receives and returns a value from the channel v.
func (v *ChanValue) Recv() Value {
	panic("unimplemented: channel Receive");
}

// TrySend attempts to sends x on the channel v but will not block.
// It returns true if the value was sent, false otherwise.
func (v *ChanValue) TrySend(x Value) bool {
	panic("unimplemented: channel TrySend");
}

// TryRecv attempts to receive a value from the channel v but will not block.
// It returns the value if one is received, nil otherwise.
func (v *ChanValue) TryRecv() Value {
	panic("unimplemented: channel TryRecv");
}

/*
 * func
 */

// A FuncValue represents a function value.
type FuncValue struct {
	value
}

// IsNil returns whether v is a nil function.
func (v *FuncValue) IsNil() bool {
	return *(*uintptr)(v.addr) == 0;
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *FuncValue) Get() uintptr {
	return *(*uintptr)(v.addr);
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *FuncValue) Set(x *FuncValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Call calls the function v with input parameters in.
// It returns the function's output parameters as Values.
func (v *FuncValue) Call(in []Value) []Value {
	panic("unimplemented: function Call");
}


/*
 * interface
 */

// An InterfaceValue represents an interface value.
type InterfaceValue struct {
	value
}

// No Get because v.Interface() is available.

// IsNil returns whether v is a nil interface value.
func (v *InterfaceValue) IsNil() bool {
	return v.Interface() == nil;
}

// Elem returns the concrete value stored in the interface value v.
func (v *InterfaceValue) Elem() Value {
	return NewValue(v.Interface());
}

// Set assigns x to v.
func (v *InterfaceValue) Set(x interface{}) {
	if !v.canSet {
		panic(cannotSet);
	}
	// Two different representations; see comment in Get.
	// Empty interface is easy.
	if v.typ.(*InterfaceType).NumMethod() == 0 {
		*(*interface{})(v.addr) = x;
	}

	// Non-empty interface requires a runtime check.
	panic("unimplemented: interface Set");
//	unsafe.SetInterface(v.typ, v.addr, x);
}

/*
 * map
 */

// A MapValue represents a map value.
type MapValue struct {
	value
}

// IsNil returns whether v is a nil map value.
func (v *MapValue) IsNil() bool {
	return *(*uintptr)(v.addr) == 0;
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *MapValue) Set(x *MapValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// Elem returns the value associated with key in the map v.
// It returns nil if key is not found in the map.
func (v *MapValue) Elem(key Value) Value {
	panic("unimplemented: map Elem");
}

// Len returns the number of keys in the map v.
func (v *MapValue) Len() int {
	panic("unimplemented: map Len");
}

// Keys returns a slice containing all the keys present in the map,
// in unspecified order.
func (v *MapValue) Keys() []Value {
	panic("unimplemented: map Keys");
}

/*
 * ptr
 */

// A PtrValue represents a pointer.
type PtrValue struct {
	value
}

// IsNil returns whether v is a nil pointer.
func (v *PtrValue) IsNil() bool {
	return *(*uintptr)(v.addr) == 0;
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *PtrValue) Get() uintptr {
	return *(*uintptr)(v.addr);
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *PtrValue) Set(x *PtrValue) {
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	// TODO: This will have to move into the runtime
	// once the new gc goes in
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr);
}

// PointTo changes v to point to x.
func (v *PtrValue) PointTo(x Value) {
	if !x.CanSet() {
		panic("cannot set x; cannot point to x");
	}
	typesMustMatch(v.typ.(*PtrType).Elem(), x.Type());
	// TODO: This will have to move into the runtime
	// once the new gc goes in.
	*(*uintptr)(v.addr) = x.Addr();
}

// Elem returns the value that v points to.
// If v is a nil pointer, Elem returns a nil Value.
func (v *PtrValue) Elem() Value {
	if v.IsNil() {
		return nil;
	}
	return newValue(v.typ.(*PtrType).Elem(), *(*addr)(v.addr), v.canSet);
}

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a nil Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
	if pv, ok := v.(*PtrValue); ok {
		return pv.Elem();
	}
	return v;
}

/*
 * struct
 */

// A StructValue represents a struct value.
type StructValue struct {
	value
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *StructValue) Set(x *StructValue) {
	// TODO: This will have to move into the runtime
	// once the gc goes in.
	if !v.canSet {
		panic(cannotSet);
	}
	typesMustMatch(v.typ, x.typ);
	memmove(uintptr(v.addr), uintptr(x.addr), v.typ.Size());
}

// Field returns the i'th field of the struct.
func (v *StructValue) Field(i int) Value {
	t := v.typ.(*StructType);
	if i < 0 || i >= t.NumField() {
		return nil;
	}
	f := t.Field(i);
	return newValue(f.Type, addr(uintptr(v.addr)+f.Offset), v.canSet && f.PkgPath == "");
}

// NumField returns the number of fields in the struct.
func (v *StructValue) NumField() int {
	return v.typ.(*StructType).NumField();
}

/*
 * constructors
 */

// Typeof returns the reflection Type of the value in the interface{}.
func Typeof(i interface{}) Type {
	return toType(unsafe.Typeof(i));
}

// NewValue returns a new Value initialized to the concrete value
// stored in the interface i.  NewValue(nil) returns nil.
func NewValue(i interface{}) Value {
	if i == nil {
		return nil;
	}
	t, a := unsafe.Reflect(i);
	return newValue(toType(t), addr(a), true);
}

func newValue(typ Type, addr addr, canSet bool) Value {
	// All values have same memory layout;
	// build once and convert.
	v := &struct{value}{value{typ, addr, canSet}};
	switch t := typ.(type) {	// TODO(rsc): s/t := // ?
	case *ArrayType:
		// TODO(rsc): Something must prevent
		// clients of the package from doing
		// this same kind of cast.
		// We should be allowed because
		// they're our types.
		// Something about implicit assignment
		// to struct fields.
		return (*ArrayValue)(v);
	case *BoolType:
		return (*BoolValue)(v);
	case *ChanType:
		return (*ChanValue)(v);
	case *FloatType:
		return (*FloatValue)(v);
	case *Float32Type:
		return (*Float32Value)(v);
	case *Float64Type:
		return (*Float64Value)(v);
	case *FuncType:
		return (*FuncValue)(v);
	case *IntType:
		return (*IntValue)(v);
	case *Int8Type:
		return (*Int8Value)(v);
	case *Int16Type:
		return (*Int16Value)(v);
	case *Int32Type:
		return (*Int32Value)(v);
	case *Int64Type:
		return (*Int64Value)(v);
	case *InterfaceType:
		return (*InterfaceValue)(v);
	case *MapType:
		return (*MapValue)(v);
	case *PtrType:
		return (*PtrValue)(v);
	case *SliceType:
		return (*SliceValue)(v);
	case *StringType:
		return (*StringValue)(v);
	case *StructType:
		return (*StructValue)(v);
	case *UintType:
		return (*UintValue)(v);
	case *Uint8Type:
		return (*Uint8Value)(v);
	case *Uint16Type:
		return (*Uint16Value)(v);
	case *Uint32Type:
		return (*Uint32Value)(v);
	case *Uint64Type:
		return (*Uint64Value)(v);
	case *UintptrType:
		return (*UintptrValue)(v);
	case *UnsafePointerType:
		return (*UnsafePointerValue)(v);
	}
	panicln("newValue", typ.String());
}

func newFuncValue(typ Type, addr addr) *FuncValue {
	return newValue(typ, addr, true).(*FuncValue);
}

// MakeZeroValue returns a zero Value for the specified Type.
func MakeZero(typ Type) Value {
	// TODO: this will have to move into
	// the runtime proper in order to play nicely
	// with the garbage collector.
	size := typ.Size();
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	return newValue(typ, addr(&data[0]), true);
}
