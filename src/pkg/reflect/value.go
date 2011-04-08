// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"math"
	"runtime"
	"unsafe"
)

const ptrSize = uintptr(unsafe.Sizeof((*byte)(nil)))
const cannotSet = "cannot set value obtained from unexported struct field"

type addr unsafe.Pointer

// TODO: This will have to go away when
// the new gc goes in.
func memmove(adst, asrc addr, n uintptr) {
	dst := uintptr(adst)
	src := uintptr(asrc)
	switch {
	case src < dst && src+n > dst:
		// byte copy backward
		// careful: i is unsigned
		for i := n; i > 0; {
			i--
			*(*byte)(addr(dst + i)) = *(*byte)(addr(src + i))
		}
	case (n|src|dst)&(ptrSize-1) != 0:
		// byte copy forward
		for i := uintptr(0); i < n; i++ {
			*(*byte)(addr(dst + i)) = *(*byte)(addr(src + i))
		}
	default:
		// word copy forward
		for i := uintptr(0); i < n; i += ptrSize {
			*(*uintptr)(addr(dst + i)) = *(*uintptr)(addr(src + i))
		}
	}
}

// Value is the reflection interface to a Go value.
//
// Not all methods apply to all kinds of values.  Restrictions,
// if any, are noted in the documentation for each method.
// Use the Kind method to find out the kind of value before
// calling kind-specific methods.  Calling a method
// inappropriate to the kind of type causes a run time panic.
//
// The zero Value represents no value.
// Its IsValid method returns false, its Kind method returns Invalid,
// its String method returns "<invalid Value>", and all other methods panic.
// Most functions and methods never return an invalid value.
// If one does, its documentation states the conditions explicitly.
type Value struct {
	Internal valueInterface
}

// TODO(rsc): This implementation of Value is a just a façade
// in front of the old implementation, now called valueInterface.
// A future CL will change it to a real implementation.
// Changing the API is already a big enough step for one CL.

// A ValueError occurs when a Value method is invoked on
// a Value that does not support it.  Such cases are documented
// in the description of each method.
type ValueError struct {
	Method string
	Kind   Kind
}

func (e *ValueError) String() string {
	if e.Kind == 0 {
		return "reflect: call of " + e.Method + " on zero Value"
	}
	return "reflect: call of " + e.Method + " on " + e.Kind.String() + " Value"
}

// methodName returns the name of the calling method,
// assumed to be two stack frames above.
func methodName() string {
	pc, _, _, _ := runtime.Caller(2)
	f := runtime.FuncForPC(pc)
	if f == nil {
		return "unknown method"
	}
	return f.Name()
}

func (v Value) internal() valueInterface {
	vi := v.Internal
	if vi == nil {
		panic(&ValueError{methodName(), 0})
	}
	return vi
}

func (v Value) panicIfNot(want Kind) valueInterface {
	vi := v.Internal
	if vi == nil {
		panic(&ValueError{methodName(), 0})
	}
	if k := vi.Kind(); k != want {
		panic(&ValueError{methodName(), k})
	}
	return vi
}

func (v Value) panicIfNots(wants []Kind) valueInterface {
	vi := v.Internal
	if vi == nil {
		panic(&ValueError{methodName(), 0})
	}
	k := vi.Kind()
	for _, want := range wants {
		if k == want {
			return vi
		}
	}
	panic(&ValueError{methodName(), k})
}

// Addr returns a pointer value representing the address of v.
// It panics if CanAddr() returns false.
// Addr is typically used to obtain a pointer to a struct field
// or slice element in order to call a method that requires a
// pointer receiver.
func (v Value) Addr() Value {
	return v.internal().Addr()
}

// Bool returns v's underlying value.
// It panics if v's kind is not Bool.
func (v Value) Bool() bool {
	u := v.panicIfNot(Bool).(*boolValue)
	return u.Get()
}

// CanAddr returns true if the value's address can be obtained with Addr.
// Such values are called addressable.  A value is addressable if it is
// an element of a slice, an element of an addressable array,
// a field of an addressable struct, the result of dereferencing a pointer,
// or the result of a call to NewValue, MakeChan, MakeMap, or Zero.
// If CanAddr returns false, calling Addr will panic.
func (v Value) CanAddr() bool {
	return v.internal().CanAddr()
}

// CanSet returns true if the value of v can be changed.
// Values obtained by the use of unexported struct fields
// can be read but not set.
// If CanSet returns false, calling Set or any type-specific
// setter (e.g., SetBool, SetInt64) will panic.
func (v Value) CanSet() bool {
	return v.internal().CanSet()
}

// Call calls the function v with the input parameters in.
// It panics if v's Kind is not Func.
// It returns the output parameters as Values.
func (v Value) Call(in []Value) []Value {
	return v.panicIfNot(Func).(*funcValue).Call(in)
}

var capKinds = []Kind{Array, Chan, Slice}

type capper interface {
	Cap() int
}

// Cap returns v's capacity.
// It panics if v's Kind is not Array, Chan, or Slice.
func (v Value) Cap() int {
	return v.panicIfNots(capKinds).(capper).Cap()
}

// Close closes the channel v.
// It panics if v's Kind is not Chan.
func (v Value) Close() {
	v.panicIfNot(Chan).(*chanValue).Close()
}

var complexKinds = []Kind{Complex64, Complex128}

// Complex returns v's underlying value, as a complex128.
// It panics if v's Kind is not Complex64 or Complex128
func (v Value) Complex() complex128 {
	return v.panicIfNots(complexKinds).(*complexValue).Get()
}

var interfaceOrPtr = []Kind{Interface, Ptr}

type elemer interface {
	Elem() Value
}

// Elem returns the value that the interface v contains
// or that the pointer v points to.
// It panics if v's Kind is not Interface or Ptr.
// It returns the zero Value if v is nil.
func (v Value) Elem() Value {
	return v.panicIfNots(interfaceOrPtr).(elemer).Elem()
}

// Field returns the i'th field of the struct v.
// It panics if v's Kind is not Struct.
func (v Value) Field(i int) Value {
	return v.panicIfNot(Struct).(*structValue).Field(i)
}

// FieldByIndex returns the nested field corresponding to index.
// It panics if v's Kind is not struct.
func (v Value) FieldByIndex(index []int) Value {
	return v.panicIfNot(Struct).(*structValue).FieldByIndex(index)
}

// FieldByName returns the struct field with the given name.
// It returns the zero Value if no field was found.
// It panics if v's Kind is not struct.
func (v Value) FieldByName(name string) Value {
	return v.panicIfNot(Struct).(*structValue).FieldByName(name)
}

// FieldByNameFunc returns the struct field with a name
// that satisfies the match function.
// It panics if v's Kind is not struct.
// It returns the zero Value if no field was found.
func (v Value) FieldByNameFunc(match func(string) bool) Value {
	return v.panicIfNot(Struct).(*structValue).FieldByNameFunc(match)
}

var floatKinds = []Kind{Float32, Float64}

// Float returns v's underlying value, as an float64.
// It panics if v's Kind is not Float32 or Float64
func (v Value) Float() float64 {
	return v.panicIfNots(floatKinds).(*floatValue).Get()
}

var arrayOrSlice = []Kind{Array, Slice}

// Index returns v's i'th element.
// It panics if v's Kind is not Array or Slice.
func (v Value) Index(i int) Value {
	return v.panicIfNots(arrayOrSlice).(arrayOrSliceValue).Elem(i)
}

var intKinds = []Kind{Int, Int8, Int16, Int32, Int64}

// Int returns v's underlying value, as an int64.
// It panics if v's Kind is not a sized or unsized Int kind.
func (v Value) Int() int64 {
	return v.panicIfNots(intKinds).(*intValue).Get()
}

// Interface returns v's value as an interface{}.
// If v is a method obtained by invoking Value.Method
// (as opposed to Type.Method), Interface cannot return an
// interface value, so it panics.
func (v Value) Interface() interface{} {
	return v.internal().Interface()
}

// InterfaceData returns the interface v's value as a uintptr pair.
// It panics if v's Kind is not Interface.
func (v Value) InterfaceData() [2]uintptr {
	return v.panicIfNot(Interface).(*interfaceValue).Get()
}

var nilKinds = []Kind{Chan, Func, Interface, Map, Ptr, Slice}

type isNiller interface {
	IsNil() bool
}

// IsNil returns true if v is a nil value.
// It panics if v's Kind is not Chan, Func, Interface, Map, Ptr, or Slice.
func (v Value) IsNil() bool {
	return v.panicIfNots(nilKinds).(isNiller).IsNil()
}

// IsValid returns true if v represents a value.
// It returns false if v is the zero Value.
// If IsValid returns false, all other methods except String panic.
// Most functions and methods never return an invalid value.
// If one does, its documentation states the conditions explicitly.
func (v Value) IsValid() bool {
	return v.Internal != nil
}

// Kind returns v's Kind.
// If v is the zero Value (IsValid returns false), Kind returns Invalid.
func (v Value) Kind() Kind {
	if v.Internal == nil {
		return Invalid
	}
	return v.internal().Kind()
}

var lenKinds = []Kind{Array, Chan, Map, Slice}

type lenner interface {
	Len() int
}

// Len returns v's length.
// It panics if v's Kind is not Array, Chan, Map, or Slice.
func (v Value) Len() int {
	return v.panicIfNots(lenKinds).(lenner).Len()
}

// MapIndex returns the value associated with key in the map v.
// It panics if v's Kind is not Map.
// It returns the zero Value if key is not found in the map.
func (v Value) MapIndex(key Value) Value {
	return v.panicIfNot(Map).(*mapValue).Elem(key)
}

// MapKeys returns a slice containing all the keys present in the map,
// in unspecified order.
// It panics if v's Kind is not Map.
func (v Value) MapKeys() []Value {
	return v.panicIfNot(Map).(*mapValue).Keys()
}

// Method returns a function value corresponding to v's i'th method.
// The arguments to a Call on the returned function should not include
// a receiver; the returned function will always use v as the receiver.
func (v Value) Method(i int) Value {
	return v.internal().Method(i)
}

// NumField returns the number of fields in the struct v.
// It panics if v's Kind is not Struct.
func (v Value) NumField() int {
	return v.panicIfNot(Struct).(*structValue).NumField()
}

// OverflowComplex returns true if the complex128 x cannot be represented by v's type.
// It panics if v's Kind is not Complex64 or Complex128.
func (v Value) OverflowComplex(x complex128) bool {
	return v.panicIfNots(complexKinds).(*complexValue).Overflow(x)
}

// OverflowFloat returns true if the float64 x cannot be represented by v's type.
// It panics if v's Kind is not Float32 or Float64.
func (v Value) OverflowFloat(x float64) bool {
	return v.panicIfNots(floatKinds).(*floatValue).Overflow(x)
}

// OverflowInt returns true if the int64 x cannot be represented by v's type.
// It panics if v's Kind is not a sized or unsized Int kind.
func (v Value) OverflowInt(x int64) bool {
	return v.panicIfNots(intKinds).(*intValue).Overflow(x)
}

// OverflowUint returns true if the uint64 x cannot be represented by v's type.
// It panics if v's Kind is not a sized or unsized Uint kind.
func (v Value) OverflowUint(x uint64) bool {
	return v.panicIfNots(uintKinds).(*uintValue).Overflow(x)
}

var pointerKinds = []Kind{Chan, Func, Map, Ptr, Slice, UnsafePointer}

type uintptrGetter interface {
	Get() uintptr
}

// Pointer returns v's value as a uintptr.
// It returns uintptr instead of unsafe.Pointer so that
// code using reflect cannot obtain unsafe.Pointers
// without importing the unsafe package explicitly.
// It panics if v's Kind is not Chan, Func, Map, Ptr, Slice, or UnsafePointer.
func (v Value) Pointer() uintptr {
	return v.panicIfNots(pointerKinds).(uintptrGetter).Get()
}


// Recv receives and returns a value from the channel v.
// It panics if v's Kind is not Chan.
// The receive blocks until a value is ready.
// The boolean value ok is true if the value x corresponds to a send
// on the channel, false if it is a zero value received because the channel is closed.
func (v Value) Recv() (x Value, ok bool) {
	return v.panicIfNot(Chan).(*chanValue).Recv()
}

// Send sends x on the channel v.
// It panics if v's kind is not Chan or if x's type is not the same type as v's element type.
func (v Value) Send(x Value) {
	v.panicIfNot(Chan).(*chanValue).Send(x)
}

// Set assigns x to the value v; x must have the same type as v.
// It panics if CanSet() returns false or if x is the zero Value.
func (v Value) Set(x Value) {
	x.internal()
	v.internal().SetValue(x)
}

// SetBool sets v's underlying value.
// It panics if v's Kind is not Bool or if CanSet() is false.
func (v Value) SetBool(x bool) {
	v.panicIfNot(Bool).(*boolValue).Set(x)
}

// SetComplex sets v's underlying value to x.
// It panics if v's Kind is not Complex64 or Complex128, or if CanSet() is false.
func (v Value) SetComplex(x complex128) {
	v.panicIfNots(complexKinds).(*complexValue).Set(x)
}

// SetFloat sets v's underlying value to x.
// It panics if v's Kind is not Float32 or Float64, or if CanSet() is false.
func (v Value) SetFloat(x float64) {
	v.panicIfNots(floatKinds).(*floatValue).Set(x)
}

// SetInt sets v's underlying value to x.
// It panics if v's Kind is not a sized or unsized Int kind, or if CanSet() is false.
func (v Value) SetInt(x int64) {
	v.panicIfNots(intKinds).(*intValue).Set(x)
}

// SetLen sets v's length to n.
// It panics if v's Kind is not Slice.
func (v Value) SetLen(n int) {
	v.panicIfNot(Slice).(*sliceValue).SetLen(n)
}

// SetMapIndex sets the value associated with key in the map v to val.
// It panics if v's Kind is not Map.
// If val is the zero Value, SetMapIndex deletes the key from the map.
func (v Value) SetMapIndex(key, val Value) {
	v.panicIfNot(Map).(*mapValue).SetElem(key, val)
}

// SetUint sets v's underlying value to x.
// It panics if v's Kind is not a sized or unsized Uint kind, or if CanSet() is false.
func (v Value) SetUint(x uint64) {
	v.panicIfNots(uintKinds).(*uintValue).Set(x)
}

// SetPointer sets the unsafe.Pointer value v to x.
// It panics if v's Kind is not UnsafePointer.
func (v Value) SetPointer(x unsafe.Pointer) {
	v.panicIfNot(UnsafePointer).(*unsafePointerValue).Set(x)
}

// SetString sets v's underlying value to x.
// It panics if v's Kind is not String or if CanSet() is false.
func (v Value) SetString(x string) {
	v.panicIfNot(String).(*stringValue).Set(x)
}

// BUG(rsc): Value.Slice should allow slicing arrays.

// Slice returns a slice of v.
// It panics if v's Kind is not Slice.
func (v Value) Slice(beg, end int) Value {
	return v.panicIfNot(Slice).(*sliceValue).Slice(beg, end)
}

// String returns the string v's underlying value, as a string.
// String is a special case because of Go's String method convention.
// Unlike the other getters, it does not panic if v's Kind is not String.
// Instead, it returns a string of the form "<T value>" where T is v's type.
func (v Value) String() string {
	vi := v.Internal
	if vi == nil {
		return "<invalid Value>"
	}
	if vi.Kind() == String {
		return vi.(*stringValue).Get()
	}
	return "<" + vi.Type().String() + " Value>"
}

// TryRecv attempts to receive a value from the channel v but will not block.
// It panics if v's Kind is not Chan.
// If the receive cannot finish without blocking, x is the zero Value.
// The boolean ok is true if the value x corresponds to a send
// on the channel, false if it is a zero value received because the channel is closed.
func (v Value) TryRecv() (x Value, ok bool) {
	return v.panicIfNot(Chan).(*chanValue).TryRecv()
}

// TrySend attempts to send x on the channel v but will not block.
// It panics if v's Kind is not Chan.
// It returns true if the value was sent, false otherwise.
func (v Value) TrySend(x Value) bool {
	return v.panicIfNot(Chan).(*chanValue).TrySend(x)
}

// Type returns v's type.
func (v Value) Type() Type {
	return v.internal().Type()
}

var uintKinds = []Kind{Uint, Uint8, Uint16, Uint32, Uint64, Uintptr}

// Uint returns v's underlying value, as a uint64.
// It panics if v's Kind is not a sized or unsized Uint kind.
func (v Value) Uint() uint64 {
	return v.panicIfNots(uintKinds).(*uintValue).Get()
}

// UnsafeAddr returns a pointer to v's data.
// It is for advanced clients that also import the "unsafe" package.
func (v Value) UnsafeAddr() uintptr {
	return v.internal().UnsafeAddr()
}

// valueInterface is the common interface to reflection values.
// The implementations of Value (e.g., arrayValue, structValue)
// have additional type-specific methods.
type valueInterface interface {
	// Type returns the value's type.
	Type() Type

	// Interface returns the value as an interface{}.
	Interface() interface{}

	// CanSet returns true if the value can be changed.
	// Values obtained by the use of non-exported struct fields
	// can be used in Get but not Set.
	// If CanSet returns false, calling the type-specific Set will panic.
	CanSet() bool

	// SetValue assigns v to the value; v must have the same type as the value.
	SetValue(v Value)

	// CanAddr returns true if the value's address can be obtained with Addr.
	// Such values are called addressable.  A value is addressable if it is
	// an element of a slice, an element of an addressable array,
	// a field of an addressable struct, the result of dereferencing a pointer,
	// or the result of a call to NewValue, MakeChan, MakeMap, or Zero.
	// If CanAddr returns false, calling Addr will panic.
	CanAddr() bool

	// Addr returns the address of the value.
	// If the value is not addressable, Addr panics.
	// Addr is typically used to obtain a pointer to a struct field or slice element
	// in order to call a method that requires a pointer receiver.
	Addr() Value

	// UnsafeAddr returns a pointer to the underlying data.
	// It is for advanced clients that also import the "unsafe" package.
	UnsafeAddr() uintptr

	// Method returns a funcValue corresponding to the value's i'th method.
	// The arguments to a Call on the returned funcValue
	// should not include a receiver; the funcValue will use
	// the value as the receiver.
	Method(i int) Value

	Kind() Kind

	getAddr() addr
}

// flags for value
const (
	canSet   uint32 = 1 << iota // can set value (write to *v.addr)
	canAddr                     // can take address of value
	canStore                    // can store through value (write to **v.addr)
)

// value is the common implementation of most values.
// It is embedded in other, public struct types, but always
// with a unique tag like "uint" or "float" so that the client cannot
// convert from, say, *uintValue to *floatValue.
type value struct {
	typ  Type
	addr addr
	flag uint32
}

func (v *value) Type() Type { return v.typ }

func (v *value) Kind() Kind { return v.typ.Kind() }

func (v *value) Addr() Value {
	if !v.CanAddr() {
		panic("reflect: cannot take address of value")
	}
	a := v.addr
	flag := canSet
	if v.CanSet() {
		flag |= canStore
	}
	// We could safely set canAddr here too -
	// the caller would get the address of a -
	// but it doesn't match the Go model.
	// The language doesn't let you say &&v.
	return newValue(PtrTo(v.typ), addr(&a), flag)
}

func (v *value) UnsafeAddr() uintptr { return uintptr(v.addr) }

func (v *value) getAddr() addr { return v.addr }

func (v *value) Interface() interface{} {
	typ := v.typ
	if typ.Kind() == Interface {
		// There are two different representations of interface values,
		// one if the interface type has methods and one if it doesn't.
		// These two representations require different expressions
		// to extract correctly.
		if typ.NumMethod() == 0 {
			// Extract as interface value without methods.
			return *(*interface{})(v.addr)
		}
		// Extract from v.addr as interface value with methods.
		return *(*interface {
			m()
		})(v.addr)
	}
	return unsafe.Unreflect(v.typ, unsafe.Pointer(v.addr))
}

func (v *value) CanSet() bool { return v.flag&canSet != 0 }

func (v *value) CanAddr() bool { return v.flag&canAddr != 0 }


/*
 * basic types
 */

// boolValue represents a bool value.
type boolValue struct {
	value "bool"
}

// Get returns the underlying bool value.
func (v *boolValue) Get() bool { return *(*bool)(v.addr) }

// Set sets v to the value x.
func (v *boolValue) Set(x bool) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	*(*bool)(v.addr) = x
}

// Set sets v to the value x.
func (v *boolValue) SetValue(x Value) { v.Set(x.Bool()) }

// floatValue represents a float value.
type floatValue struct {
	value "float"
}

// Get returns the underlying int value.
func (v *floatValue) Get() float64 {
	switch v.typ.Kind() {
	case Float32:
		return float64(*(*float32)(v.addr))
	case Float64:
		return *(*float64)(v.addr)
	}
	panic("reflect: invalid float kind")
}

// Set sets v to the value x.
func (v *floatValue) Set(x float64) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	switch v.typ.Kind() {
	default:
		panic("reflect: invalid float kind")
	case Float32:
		*(*float32)(v.addr) = float32(x)
	case Float64:
		*(*float64)(v.addr) = x
	}
}

// Overflow returns true if x cannot be represented by the type of v.
func (v *floatValue) Overflow(x float64) bool {
	if v.typ.Size() == 8 {
		return false
	}
	if x < 0 {
		x = -x
	}
	return math.MaxFloat32 < x && x <= math.MaxFloat64
}

// Set sets v to the value x.
func (v *floatValue) SetValue(x Value) { v.Set(x.Float()) }

// complexValue represents a complex value.
type complexValue struct {
	value "complex"
}

// Get returns the underlying complex value.
func (v *complexValue) Get() complex128 {
	switch v.typ.Kind() {
	case Complex64:
		return complex128(*(*complex64)(v.addr))
	case Complex128:
		return *(*complex128)(v.addr)
	}
	panic("reflect: invalid complex kind")
}

// Set sets v to the value x.
func (v *complexValue) Set(x complex128) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	switch v.typ.Kind() {
	default:
		panic("reflect: invalid complex kind")
	case Complex64:
		*(*complex64)(v.addr) = complex64(x)
	case Complex128:
		*(*complex128)(v.addr) = x
	}
}

// How did we forget this one?
func (v *complexValue) Overflow(x complex128) bool {
	if v.typ.Size() == 16 {
		return false
	}
	r := real(x)
	i := imag(x)
	if r < 0 {
		r = -r
	}
	if i < 0 {
		i = -i
	}
	return math.MaxFloat32 <= r && r <= math.MaxFloat64 ||
		math.MaxFloat32 <= i && i <= math.MaxFloat64
}

// Set sets v to the value x.
func (v *complexValue) SetValue(x Value) { v.Set(x.Complex()) }

// intValue represents an int value.
type intValue struct {
	value "int"
}

// Get returns the underlying int value.
func (v *intValue) Get() int64 {
	switch v.typ.Kind() {
	case Int:
		return int64(*(*int)(v.addr))
	case Int8:
		return int64(*(*int8)(v.addr))
	case Int16:
		return int64(*(*int16)(v.addr))
	case Int32:
		return int64(*(*int32)(v.addr))
	case Int64:
		return *(*int64)(v.addr)
	}
	panic("reflect: invalid int kind")
}

// Set sets v to the value x.
func (v *intValue) Set(x int64) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	switch v.typ.Kind() {
	default:
		panic("reflect: invalid int kind")
	case Int:
		*(*int)(v.addr) = int(x)
	case Int8:
		*(*int8)(v.addr) = int8(x)
	case Int16:
		*(*int16)(v.addr) = int16(x)
	case Int32:
		*(*int32)(v.addr) = int32(x)
	case Int64:
		*(*int64)(v.addr) = x
	}
}

// Set sets v to the value x.
func (v *intValue) SetValue(x Value) { v.Set(x.Int()) }

// Overflow returns true if x cannot be represented by the type of v.
func (v *intValue) Overflow(x int64) bool {
	bitSize := uint(v.typ.Bits())
	trunc := (x << (64 - bitSize)) >> (64 - bitSize)
	return x != trunc
}

// StringHeader is the runtime representation of a string.
type StringHeader struct {
	Data uintptr
	Len  int
}

// stringValue represents a string value.
type stringValue struct {
	value "string"
}

// Get returns the underlying string value.
func (v *stringValue) Get() string { return *(*string)(v.addr) }

// Set sets v to the value x.
func (v *stringValue) Set(x string) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	*(*string)(v.addr) = x
}

// Set sets v to the value x.
func (v *stringValue) SetValue(x Value) {
	// Do the kind check explicitly, because x.String() does not.
	v.Set(x.panicIfNot(String).(*stringValue).Get())
}

// uintValue represents a uint value.
type uintValue struct {
	value "uint"
}

// Get returns the underlying uuint value.
func (v *uintValue) Get() uint64 {
	switch v.typ.Kind() {
	case Uint:
		return uint64(*(*uint)(v.addr))
	case Uint8:
		return uint64(*(*uint8)(v.addr))
	case Uint16:
		return uint64(*(*uint16)(v.addr))
	case Uint32:
		return uint64(*(*uint32)(v.addr))
	case Uint64:
		return *(*uint64)(v.addr)
	case Uintptr:
		return uint64(*(*uintptr)(v.addr))
	}
	panic("reflect: invalid uint kind")
}

// Set sets v to the value x.
func (v *uintValue) Set(x uint64) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	switch v.typ.Kind() {
	default:
		panic("reflect: invalid uint kind")
	case Uint:
		*(*uint)(v.addr) = uint(x)
	case Uint8:
		*(*uint8)(v.addr) = uint8(x)
	case Uint16:
		*(*uint16)(v.addr) = uint16(x)
	case Uint32:
		*(*uint32)(v.addr) = uint32(x)
	case Uint64:
		*(*uint64)(v.addr) = x
	case Uintptr:
		*(*uintptr)(v.addr) = uintptr(x)
	}
}

// Overflow returns true if x cannot be represented by the type of v.
func (v *uintValue) Overflow(x uint64) bool {
	bitSize := uint(v.typ.Bits())
	trunc := (x << (64 - bitSize)) >> (64 - bitSize)
	return x != trunc
}

// Set sets v to the value x.
func (v *uintValue) SetValue(x Value) { v.Set(x.Uint()) }

// unsafePointerValue represents an unsafe.Pointer value.
type unsafePointerValue struct {
	value "unsafe.Pointer"
}

// Get returns the underlying uintptr value.
// Get returns uintptr, not unsafe.Pointer, so that
// programs that do not import "unsafe" cannot
// obtain a value of unsafe.Pointer type from "reflect".
func (v *unsafePointerValue) Get() uintptr { return uintptr(*(*unsafe.Pointer)(v.addr)) }

// Set sets v to the value x.
func (v *unsafePointerValue) Set(x unsafe.Pointer) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	*(*unsafe.Pointer)(v.addr) = x
}

// Set sets v to the value x.
func (v *unsafePointerValue) SetValue(x Value) {
	// Do the kind check explicitly, because x.UnsafePointer
	// applies to more than just the UnsafePointer Kind.
	v.Set(unsafe.Pointer(x.panicIfNot(UnsafePointer).(*unsafePointerValue).Get()))
}

func typesMustMatch(t1, t2 Type) {
	if t1 != t2 {
		panic("type mismatch: " + t1.String() + " != " + t2.String())
	}
}

/*
 * array
 */

// ArrayOrSliceValue is the common interface
// implemented by both arrayValue and sliceValue.
type arrayOrSliceValue interface {
	valueInterface
	Len() int
	Cap() int
	Elem(i int) Value
	addr() addr
}

// grow grows the slice s so that it can hold extra more values, allocating
// more capacity if needed. It also returns the old and new slice lengths.
func grow(s Value, extra int) (Value, int, int) {
	i0 := s.Len()
	i1 := i0 + extra
	if i1 < i0 {
		panic("append: slice overflow")
	}
	m := s.Cap()
	if i1 <= m {
		return s.Slice(0, i1), i0, i1
	}
	if m == 0 {
		m = extra
	} else {
		for m < i1 {
			if i0 < 1024 {
				m += m
			} else {
				m += m / 4
			}
		}
	}
	t := MakeSlice(s.Type(), i1, m)
	Copy(t, s)
	return t, i0, i1
}

// Append appends the values x to a slice s and returns the resulting slice.
// Each x must have the same type as s' element type.
func Append(s Value, x ...Value) Value {
	s, i0, i1 := grow(s, len(x))
	sa := s.panicIfNot(Slice).(*sliceValue)
	for i, j := i0, 0; i < i1; i, j = i+1, j+1 {
		sa.Elem(i).Set(x[j])
	}
	return s
}

// AppendSlice appends a slice t to a slice s and returns the resulting slice.
// The slices s and t must have the same element type.
func AppendSlice(s, t Value) Value {
	s, i0, i1 := grow(s, t.Len())
	Copy(s.Slice(i0, i1), t)
	return s
}

// Copy copies the contents of src into dst until either
// dst has been filled or src has been exhausted.
// It returns the number of elements copied.
// Dst and src each must be a slice or array, and they
// must have the same element type.
func Copy(dst, src Value) int {
	// TODO: This will have to move into the runtime
	// once the real gc goes in.
	de := dst.Type().Elem()
	se := src.Type().Elem()
	typesMustMatch(de, se)
	n := dst.Len()
	if xn := src.Len(); n > xn {
		n = xn
	}
	memmove(dst.panicIfNots(arrayOrSlice).(arrayOrSliceValue).addr(),
		src.panicIfNots(arrayOrSlice).(arrayOrSliceValue).addr(),
		uintptr(n)*de.Size())
	return n
}

// An arrayValue represents an array.
type arrayValue struct {
	value "array"
}

// Len returns the length of the array.
func (v *arrayValue) Len() int { return v.typ.Len() }

// Cap returns the capacity of the array (equal to Len()).
func (v *arrayValue) Cap() int { return v.typ.Len() }

// addr returns the base address of the data in the array.
func (v *arrayValue) addr() addr { return v.value.addr }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *arrayValue) Set(x *arrayValue) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	Copy(Value{v}, Value{x})
}

// Set sets v to the value x.
func (v *arrayValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Array).(*arrayValue))
}

// Elem returns the i'th element of v.
func (v *arrayValue) Elem(i int) Value {
	typ := v.typ.Elem()
	n := v.Len()
	if i < 0 || i >= n {
		panic("array index out of bounds")
	}
	p := addr(uintptr(v.addr()) + uintptr(i)*typ.Size())
	return newValue(typ, p, v.flag)
}

/*
 * slice
 */

// runtime representation of slice
type SliceHeader struct {
	Data uintptr
	Len  int
	Cap  int
}

// A sliceValue represents a slice.
type sliceValue struct {
	value "slice"
}

func (v *sliceValue) slice() *SliceHeader { return (*SliceHeader)(v.value.addr) }

// IsNil returns whether v is a nil slice.
func (v *sliceValue) IsNil() bool { return v.slice().Data == 0 }

// Len returns the length of the slice.
func (v *sliceValue) Len() int { return int(v.slice().Len) }

// Cap returns the capacity of the slice.
func (v *sliceValue) Cap() int { return int(v.slice().Cap) }

// addr returns the base address of the data in the slice.
func (v *sliceValue) addr() addr { return addr(v.slice().Data) }

// SetLen changes the length of v.
// The new length n must be between 0 and the capacity, inclusive.
func (v *sliceValue) SetLen(n int) {
	s := v.slice()
	if n < 0 || n > int(s.Cap) {
		panic("reflect: slice length out of range in SetLen")
	}
	s.Len = n
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *sliceValue) Set(x *sliceValue) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	*v.slice() = *x.slice()
}

// Set sets v to the value x.
func (v *sliceValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Slice).(*sliceValue))
}

// Get returns the uintptr address of the v.Cap()'th element.  This gives
// the same result for all slices of the same array.
// It is mainly useful for printing.
func (v *sliceValue) Get() uintptr {
	typ := v.typ
	return uintptr(v.addr()) + uintptr(v.Cap())*typ.Elem().Size()
}

// Slice returns a sub-slice of the slice v.
func (v *sliceValue) Slice(beg, end int) Value {
	cap := v.Cap()
	if beg < 0 || end < beg || end > cap {
		panic("slice index out of bounds")
	}
	typ := v.typ
	s := new(SliceHeader)
	s.Data = uintptr(v.addr()) + uintptr(beg)*typ.Elem().Size()
	s.Len = end - beg
	s.Cap = cap - beg

	// Like the result of Addr, we treat Slice as an
	// unaddressable temporary, so don't set canAddr.
	flag := canSet
	if v.flag&canStore != 0 {
		flag |= canStore
	}
	return newValue(typ, addr(s), flag)
}

// Elem returns the i'th element of v.
func (v *sliceValue) Elem(i int) Value {
	typ := v.typ.Elem()
	n := v.Len()
	if i < 0 || i >= n {
		panic("reflect: slice index out of range")
	}
	p := addr(uintptr(v.addr()) + uintptr(i)*typ.Size())
	flag := canAddr
	if v.flag&canStore != 0 {
		flag |= canSet | canStore
	}
	return newValue(typ, p, flag)
}

// MakeSlice creates a new zero-initialized slice value
// for the specified slice type, length, and capacity.
func MakeSlice(typ Type, len, cap int) Value {
	if typ.Kind() != Slice {
		panic("reflect: MakeSlice of non-slice type")
	}
	s := &SliceHeader{
		Data: uintptr(unsafe.NewArray(typ.Elem(), cap)),
		Len:  len,
		Cap:  cap,
	}
	return newValue(typ, addr(s), canAddr|canSet|canStore)
}

/*
 * chan
 */

// A chanValue represents a chan.
type chanValue struct {
	value "chan"
}

// IsNil returns whether v is a nil channel.
func (v *chanValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *chanValue) Set(x *chanValue) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *chanValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Chan).(*chanValue))
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *chanValue) Get() uintptr { return *(*uintptr)(v.addr) }

// implemented in ../pkg/runtime/reflect.cgo
func makechan(typ *runtime.ChanType, size uint32) (ch *byte)
func chansend(ch, val *byte, selected *bool)
func chanrecv(ch, val *byte, selected *bool, ok *bool)
func chanclose(ch *byte)
func chanlen(ch *byte) int32
func chancap(ch *byte) int32

// Close closes the channel.
func (v *chanValue) Close() {
	ch := *(**byte)(v.addr)
	chanclose(ch)
}

func (v *chanValue) Len() int {
	ch := *(**byte)(v.addr)
	return int(chanlen(ch))
}

func (v *chanValue) Cap() int {
	ch := *(**byte)(v.addr)
	return int(chancap(ch))
}

// internal send; non-blocking if selected != nil
func (v *chanValue) send(x Value, selected *bool) {
	t := v.Type()
	if t.ChanDir()&SendDir == 0 {
		panic("send on recv-only channel")
	}
	typesMustMatch(t.Elem(), x.Type())
	ch := *(**byte)(v.addr)
	chansend(ch, (*byte)(x.internal().getAddr()), selected)
}

// internal recv; non-blocking if selected != nil
func (v *chanValue) recv(selected *bool) (Value, bool) {
	t := v.Type()
	if t.ChanDir()&RecvDir == 0 {
		panic("recv on send-only channel")
	}
	ch := *(**byte)(v.addr)
	x := Zero(t.Elem())
	var ok bool
	chanrecv(ch, (*byte)(x.internal().getAddr()), selected, &ok)
	return x, ok
}

// Send sends x on the channel v.
func (v *chanValue) Send(x Value) { v.send(x, nil) }

// Recv receives and returns a value from the channel v.
// The receive blocks until a value is ready.
// The boolean value ok is true if the value x corresponds to a send
// on the channel, false if it is a zero value received because the channel is closed.
func (v *chanValue) Recv() (x Value, ok bool) {
	return v.recv(nil)
}

// TrySend attempts to sends x on the channel v but will not block.
// It returns true if the value was sent, false otherwise.
func (v *chanValue) TrySend(x Value) bool {
	var selected bool
	v.send(x, &selected)
	return selected
}

// TryRecv attempts to receive a value from the channel v but will not block.
// If the receive cannot finish without blocking, TryRecv instead returns x == nil.
// If the receive can finish without blocking, TryRecv returns x != nil.
// The boolean value ok is true if the value x corresponds to a send
// on the channel, false if it is a zero value received because the channel is closed.
func (v *chanValue) TryRecv() (x Value, ok bool) {
	var selected bool
	x, ok = v.recv(&selected)
	if !selected {
		return Value{}, false
	}
	return x, ok
}

// MakeChan creates a new channel with the specified type and buffer size.
func MakeChan(typ Type, buffer int) Value {
	if typ.Kind() != Chan {
		panic("reflect: MakeChan of non-chan type")
	}
	if buffer < 0 {
		panic("MakeChan: negative buffer size")
	}
	if typ.ChanDir() != BothDir {
		panic("MakeChan: unidirectional channel type")
	}
	v := Zero(typ)
	ch := v.panicIfNot(Chan).(*chanValue)
	*(**byte)(ch.addr) = makechan((*runtime.ChanType)(unsafe.Pointer(typ.(*commonType))), uint32(buffer))
	return v
}

/*
 * func
 */

// A funcValue represents a function value.
type funcValue struct {
	value       "func"
	first       *value
	isInterface bool
}

// IsNil returns whether v is a nil function.
func (v *funcValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *funcValue) Get() uintptr { return *(*uintptr)(v.addr) }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *funcValue) Set(x *funcValue) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *funcValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Func).(*funcValue))
}

// Method returns a funcValue corresponding to v's i'th method.
// The arguments to a Call on the returned funcValue
// should not include a receiver; the funcValue will use v
// as the receiver.
func (v *value) Method(i int) Value {
	t := v.Type().uncommon()
	if t == nil || i < 0 || i >= len(t.methods) {
		panic("reflect: Method index out of range")
	}
	p := &t.methods[i]
	fn := p.tfn
	fv := &funcValue{value: value{toType(p.typ), addr(&fn), 0}, first: v, isInterface: false}
	return Value{fv}
}

// implemented in ../pkg/runtime/*/asm.s
func call(fn, arg *byte, n uint32)

type tiny struct {
	b byte
}

// Interface returns the fv as an interface value.
// If fv is a method obtained by invoking Value.Method
// (as opposed to Type.Method), Interface cannot return an
// interface value, so it panics.
func (fv *funcValue) Interface() interface{} {
	if fv.first != nil {
		panic("funcValue: cannot create interface value for method with bound receiver")
	}
	return fv.value.Interface()
}

// Call calls the function fv with input parameters in.
// It returns the function's output parameters as Values.
func (fv *funcValue) Call(in []Value) []Value {
	t := fv.Type()
	nin := len(in)
	if fv.first != nil && !fv.isInterface {
		nin++
	}
	if nin != t.NumIn() {
		panic("funcValue: wrong argument count")
	}
	nout := t.NumOut()

	// Compute arg size & allocate.
	// This computation is 6g/8g-dependent
	// and probably wrong for gccgo, but so
	// is most of this function.
	size := uintptr(0)
	if fv.isInterface {
		// extra word for interface value
		size += ptrSize
	}
	for i := 0; i < nin; i++ {
		tv := t.In(i)
		a := uintptr(tv.Align())
		size = (size + a - 1) &^ (a - 1)
		size += tv.Size()
	}
	size = (size + ptrSize - 1) &^ (ptrSize - 1)
	for i := 0; i < nout; i++ {
		tv := t.Out(i)
		a := uintptr(tv.Align())
		size = (size + a - 1) &^ (a - 1)
		size += tv.Size()
	}

	// size must be > 0 in order for &args[0] to be valid.
	// the argument copying is going to round it up to
	// a multiple of ptrSize anyway, so make it ptrSize to begin with.
	if size < ptrSize {
		size = ptrSize
	}

	// round to pointer size
	size = (size + ptrSize - 1) &^ (ptrSize - 1)

	// Copy into args.
	//
	// TODO(rsc): revisit when reference counting happens.
	// The values are holding up the in references for us,
	// but something must be done for the out references.
	// For now make everything look like a pointer by pretending
	// to allocate a []*int.
	args := make([]*int, size/ptrSize)
	ptr := uintptr(unsafe.Pointer(&args[0]))
	off := uintptr(0)
	delta := 0
	if v := fv.first; v != nil {
		// Hard-wired first argument.
		if fv.isInterface {
			// v is a single uninterpreted word
			memmove(addr(ptr), v.getAddr(), ptrSize)
			off = ptrSize
		} else {
			// v is a real value
			tv := v.Type()
			typesMustMatch(t.In(0), tv)
			n := tv.Size()
			memmove(addr(ptr), v.getAddr(), n)
			off = n
			delta = 1
		}
	}
	for i, v := range in {
		tv := v.Type()
		typesMustMatch(t.In(i+delta), tv)
		a := uintptr(tv.Align())
		off = (off + a - 1) &^ (a - 1)
		n := tv.Size()
		memmove(addr(ptr+off), v.internal().getAddr(), n)
		off += n
	}
	off = (off + ptrSize - 1) &^ (ptrSize - 1)

	// Call
	call(*(**byte)(fv.addr), (*byte)(addr(ptr)), uint32(size))

	// Copy return values out of args.
	//
	// TODO(rsc): revisit like above.
	ret := make([]Value, nout)
	for i := 0; i < nout; i++ {
		tv := t.Out(i)
		a := uintptr(tv.Align())
		off = (off + a - 1) &^ (a - 1)
		v := Zero(tv)
		n := tv.Size()
		memmove(v.internal().getAddr(), addr(ptr+off), n)
		ret[i] = v
		off += n
	}

	return ret
}

/*
 * interface
 */

// An interfaceValue represents an interface value.
type interfaceValue struct {
	value "interface"
}

// IsNil returns whether v is a nil interface value.
func (v *interfaceValue) IsNil() bool { return v.Interface() == nil }

// No single uinptr Get because v.Interface() is available.

// Get returns the two words that represent an interface in the runtime.
// Those words are useful only when playing unsafe games.
func (v *interfaceValue) Get() [2]uintptr {
	return *(*[2]uintptr)(v.addr)
}

// Elem returns the concrete value stored in the interface value v.
func (v *interfaceValue) Elem() Value { return NewValue(v.Interface()) }

// ../runtime/reflect.cgo
func setiface(typ *interfaceType, x *interface{}, addr addr)

// Set assigns x to v.
func (v *interfaceValue) Set(x Value) {
	i := x.Interface()
	if !v.CanSet() {
		panic(cannotSet)
	}
	// Two different representations; see comment in Get.
	// Empty interface is easy.
	t := (*interfaceType)(unsafe.Pointer(v.typ.(*commonType)))
	if t.NumMethod() == 0 {
		*(*interface{})(v.addr) = i
		return
	}

	// Non-empty interface requires a runtime check.
	setiface(t, &i, v.addr)
}

// Set sets v to the value x.
func (v *interfaceValue) SetValue(x Value) { v.Set(x) }

// Method returns a funcValue corresponding to v's i'th method.
// The arguments to a Call on the returned funcValue
// should not include a receiver; the funcValue will use v
// as the receiver.
func (v *interfaceValue) Method(i int) Value {
	t := (*interfaceType)(unsafe.Pointer(v.Type().(*commonType)))
	if t == nil || i < 0 || i >= len(t.methods) {
		panic("reflect: Method index out of range")
	}
	p := &t.methods[i]

	// Interface is two words: itable, data.
	tab := *(**runtime.Itable)(v.addr)
	data := &value{Typeof((*byte)(nil)), addr(uintptr(v.addr) + ptrSize), 0}

	// Function pointer is at p.perm in the table.
	fn := tab.Fn[i]
	fv := &funcValue{value: value{toType(p.typ), addr(&fn), 0}, first: data, isInterface: true}
	return Value{fv}
}

/*
 * map
 */

// A mapValue represents a map value.
type mapValue struct {
	value "map"
}

// IsNil returns whether v is a nil map value.
func (v *mapValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *mapValue) Set(x *mapValue) {
	if !v.CanSet() {
		panic(cannotSet)
	}
	if x == nil {
		*(**uintptr)(v.addr) = nil
		return
	}
	typesMustMatch(v.typ, x.typ)
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *mapValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Map).(*mapValue))
}

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *mapValue) Get() uintptr { return *(*uintptr)(v.addr) }

// implemented in ../pkg/runtime/reflect.cgo
func mapaccess(m, key, val *byte) bool
func mapassign(m, key, val *byte)
func maplen(m *byte) int32
func mapiterinit(m *byte) *byte
func mapiternext(it *byte)
func mapiterkey(it *byte, key *byte) bool
func makemap(t *runtime.MapType) *byte

// Elem returns the value associated with key in the map v.
// It returns nil if key is not found in the map.
func (v *mapValue) Elem(key Value) Value {
	t := v.Type()
	typesMustMatch(t.Key(), key.Type())
	m := *(**byte)(v.addr)
	if m == nil {
		return Value{}
	}
	newval := Zero(t.Elem())
	if !mapaccess(m, (*byte)(key.internal().getAddr()), (*byte)(newval.internal().getAddr())) {
		return Value{}
	}
	return newval
}

// SetElem sets the value associated with key in the map v to val.
// If val is nil, Put deletes the key from map.
func (v *mapValue) SetElem(key, val Value) {
	t := v.Type()
	typesMustMatch(t.Key(), key.Type())
	var vaddr *byte
	if val.IsValid() {
		typesMustMatch(t.Elem(), val.Type())
		vaddr = (*byte)(val.internal().getAddr())
	}
	m := *(**byte)(v.addr)
	mapassign(m, (*byte)(key.internal().getAddr()), vaddr)
}

// Len returns the number of keys in the map v.
func (v *mapValue) Len() int {
	m := *(**byte)(v.addr)
	if m == nil {
		return 0
	}
	return int(maplen(m))
}

// Keys returns a slice containing all the keys present in the map,
// in unspecified order.
func (v *mapValue) Keys() []Value {
	tk := v.Type().Key()
	m := *(**byte)(v.addr)
	mlen := int32(0)
	if m != nil {
		mlen = maplen(m)
	}
	it := mapiterinit(m)
	a := make([]Value, mlen)
	var i int
	for i = 0; i < len(a); i++ {
		k := Zero(tk)
		if !mapiterkey(it, (*byte)(k.internal().getAddr())) {
			break
		}
		a[i] = k
		mapiternext(it)
	}
	return a[0:i]
}

// MakeMap creates a new map of the specified type.
func MakeMap(typ Type) Value {
	if typ.Kind() != Map {
		panic("reflect: MakeMap of non-map type")
	}
	v := Zero(typ)
	m := v.panicIfNot(Map).(*mapValue)
	*(**byte)(m.addr) = makemap((*runtime.MapType)(unsafe.Pointer(typ.(*commonType))))
	return v
}

/*
 * ptr
 */

// A ptrValue represents a pointer.
type ptrValue struct {
	value "ptr"
}

// IsNil returns whether v is a nil pointer.
func (v *ptrValue) IsNil() bool { return *(*uintptr)(v.addr) == 0 }

// Get returns the uintptr value of v.
// It is mainly useful for printing.
func (v *ptrValue) Get() uintptr { return *(*uintptr)(v.addr) }

// Set assigns x to v.
// The new value x must have the same type as v, and x.Elem().CanSet() must be true.
func (v *ptrValue) Set(x *ptrValue) {
	if x == nil {
		*(**uintptr)(v.addr) = nil
		return
	}
	if !v.CanSet() {
		panic(cannotSet)
	}
	if x.flag&canStore == 0 {
		panic("cannot copy pointer obtained from unexported struct field")
	}
	typesMustMatch(v.typ, x.typ)
	// TODO: This will have to move into the runtime
	// once the new gc goes in
	*(*uintptr)(v.addr) = *(*uintptr)(x.addr)
}

// Set sets v to the value x.
func (v *ptrValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Ptr).(*ptrValue))
}

// PointTo changes v to point to x.
// If x is a nil Value, PointTo sets v to nil.
func (v *ptrValue) PointTo(x Value) {
	if !x.IsValid() {
		*(**uintptr)(v.addr) = nil
		return
	}
	if !x.CanSet() {
		panic("cannot set x; cannot point to x")
	}
	typesMustMatch(v.typ.Elem(), x.Type())
	// TODO: This will have to move into the runtime
	// once the new gc goes in.
	*(*uintptr)(v.addr) = x.UnsafeAddr()
}

// Elem returns the value that v points to.
// If v is a nil pointer, Elem returns a nil Value.
func (v *ptrValue) Elem() Value {
	if v.IsNil() {
		return Value{}
	}
	flag := canAddr
	if v.flag&canStore != 0 {
		flag |= canSet | canStore
	}
	return newValue(v.typ.Elem(), *(*addr)(v.addr), flag)
}

// Indirect returns the value that v points to.
// If v is a nil pointer, Indirect returns a nil Value.
// If v is not a pointer, Indirect returns v.
func Indirect(v Value) Value {
	if v.Kind() != Ptr {
		return v
	}
	return v.panicIfNot(Ptr).(*ptrValue).Elem()
}

/*
 * struct
 */

// A structValue represents a struct value.
type structValue struct {
	value "struct"
}

// Set assigns x to v.
// The new value x must have the same type as v.
func (v *structValue) Set(x *structValue) {
	// TODO: This will have to move into the runtime
	// once the gc goes in.
	if !v.CanSet() {
		panic(cannotSet)
	}
	typesMustMatch(v.typ, x.typ)
	memmove(v.addr, x.addr, v.typ.Size())
}

// Set sets v to the value x.
func (v *structValue) SetValue(x Value) {
	v.Set(x.panicIfNot(Struct).(*structValue))
}

// Field returns the i'th field of the struct.
func (v *structValue) Field(i int) Value {
	t := v.typ
	if i < 0 || i >= t.NumField() {
		panic("reflect: Field index out of range")
	}
	f := t.Field(i)
	flag := v.flag
	if f.PkgPath != "" {
		// unexported field
		flag &^= canSet | canStore
	}
	return newValue(f.Type, addr(uintptr(v.addr)+f.Offset), flag)
}

// FieldByIndex returns the nested field corresponding to index.
func (t *structValue) FieldByIndex(index []int) (v Value) {
	v = Value{t}
	for i, x := range index {
		if i > 0 {
			if v.Kind() == Ptr {
				v = v.Elem()
			}
			if v.Kind() != Struct {
				return Value{}
			}
		}
		v = v.Field(x)
	}
	return
}

// FieldByName returns the struct field with the given name.
// The result is nil if no field was found.
func (t *structValue) FieldByName(name string) Value {
	if f, ok := t.Type().FieldByName(name); ok {
		return t.FieldByIndex(f.Index)
	}
	return Value{}
}

// FieldByNameFunc returns the struct field with a name that satisfies the
// match function.
// The result is nil if no field was found.
func (t *structValue) FieldByNameFunc(match func(string) bool) Value {
	if f, ok := t.Type().FieldByNameFunc(match); ok {
		return t.FieldByIndex(f.Index)
	}
	return Value{}
}

// NumField returns the number of fields in the struct.
func (v *structValue) NumField() int { return v.typ.NumField() }

/*
 * constructors
 */

// NewValue returns a new Value initialized to the concrete value
// stored in the interface i.  NewValue(nil) returns the zero Value.
func NewValue(i interface{}) Value {
	if i == nil {
		return Value{}
	}
	_, a := unsafe.Reflect(i)
	return newValue(Typeof(i), addr(a), canSet|canAddr|canStore)
}

func newValue(typ Type, addr addr, flag uint32) Value {
	v := value{typ, addr, flag}
	switch typ.Kind() {
	case Array:
		return Value{&arrayValue{v}}
	case Bool:
		return Value{&boolValue{v}}
	case Chan:
		return Value{&chanValue{v}}
	case Float32, Float64:
		return Value{&floatValue{v}}
	case Func:
		return Value{&funcValue{value: v}}
	case Complex64, Complex128:
		return Value{&complexValue{v}}
	case Int, Int8, Int16, Int32, Int64:
		return Value{&intValue{v}}
	case Interface:
		return Value{&interfaceValue{v}}
	case Map:
		return Value{&mapValue{v}}
	case Ptr:
		return Value{&ptrValue{v}}
	case Slice:
		return Value{&sliceValue{v}}
	case String:
		return Value{&stringValue{v}}
	case Struct:
		return Value{&structValue{v}}
	case Uint, Uint8, Uint16, Uint32, Uint64, Uintptr:
		return Value{&uintValue{v}}
	case UnsafePointer:
		return Value{&unsafePointerValue{v}}
	}
	panic("newValue" + typ.String())
}

// Zero returns a Value representing a zero value for the specified type.
// The result is different from the zero value of the Value struct,
// which represents no value at all.
// For example, Zero(Typeof(42)) returns a Value with Kind Int and value 0.
func Zero(typ Type) Value {
	if typ == nil {
		panic("reflect: Zero(nil)")
	}
	return newValue(typ, addr(unsafe.New(typ)), canSet|canAddr|canStore)
}
