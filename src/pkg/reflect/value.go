// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Handling values.

package reflect

import (
	"reflect";
	"unsafe";
)

// Addr is shorthand for unsafe.Pointer and is used to represent the address of Values.
type Addr unsafe.Pointer

func equalType(a, b Type) bool {
	return a.String() == b.String()
}

// Value is the generic interface to reflection values.  Once its Kind is known,
// such as BoolKind, the Value can be narrowed to the appropriate, more
// specific interface, such as BoolValue.  Such narrowed values still implement
// the Value interface.
type Value interface {
	// The kind of thing described: ArrayKind, BoolKind, etc.
	Kind()	int;
	// The reflection Type of the value.
	Type()	Type;
	// The address of the value.
	Addr()	Addr;
	// The value itself is the dynamic value of an empty interface.
	Interface()	interface {};
}

func NewValue(e interface{}) Value;

// commonValue fields and functionality for all values

type commonValue struct {
	kind	int;
	typ	Type;
	addr	Addr;
}

func (c *commonValue) Kind() int {
	return c.kind
}

func (c *commonValue) Type() Type {
	return c.typ
}

func (c *commonValue) Addr() Addr {
	return c.addr
}

func (c *commonValue) Interface() interface {} {
	var i interface {};
	switch {
	case c.typ.Kind() == InterfaceKind:
		panic("not reached");	// InterfaceValue overrides this method
	case c.typ.Size() > unsafe.Sizeof(uintptr(0)):
		i = unsafe.Unreflect(uint64(uintptr(c.addr)), c.typ.String(), true);
	default:
		if uintptr(c.addr) == 0 {
			panicln("reflect: address 0 for", c.typ.String());
		}
		i = unsafe.Unreflect(uint64(uintptr(*(*Addr)(c.addr))), c.typ.String(), false);
	}
	return i;
}

func newValueAddr(typ Type, addr Addr) Value

type creatorFn func(typ Type, addr Addr) Value


// -- Missing

// MissingValue represents a value whose type is not known. It usually
// indicates an error.
type MissingValue interface {
	Value;
}

type missingValueStruct struct {
	commonValue
}

func missingCreator(typ Type, addr Addr) Value {
	return &missingValueStruct{ commonValue{MissingKind, typ, addr} }
}

// -- Int

// IntValue represents an int value.
type IntValue interface {
	Value;
	Get()	int;	// Get the underlying int.
	Set(int);	// Set the underlying int.
}

type intValueStruct struct {
	commonValue
}

func intCreator(typ Type, addr Addr) Value {
	return &intValueStruct{ commonValue{IntKind, typ, addr} }
}

func (v *intValueStruct) Get() int {
	return *(*int)(v.addr)
}

func (v *intValueStruct) Set(i int) {
	*(*int)(v.addr) = i
}

// -- Int8

// Int8Value represents an int8 value.
type Int8Value interface {
	Value;
	Get()	int8;	// Get the underlying int8.
	Set(int8);	// Set the underlying int8.
}

type int8ValueStruct struct {
	commonValue
}

func int8Creator(typ Type, addr Addr) Value {
	return &int8ValueStruct{ commonValue{Int8Kind, typ, addr} }
}

func (v *int8ValueStruct) Get() int8 {
	return *(*int8)(v.addr)
}

func (v *int8ValueStruct) Set(i int8) {
	*(*int8)(v.addr) = i
}

// -- Int16

// Int16Value represents an int16 value.
type Int16Value interface {
	Value;
	Get()	int16;	// Get the underlying int16.
	Set(int16);	// Set the underlying int16.
}

type int16ValueStruct struct {
	commonValue
}

func int16Creator(typ Type, addr Addr) Value {
	return &int16ValueStruct{ commonValue{Int16Kind, typ, addr} }
}

func (v *int16ValueStruct) Get() int16 {
	return *(*int16)(v.addr)
}

func (v *int16ValueStruct) Set(i int16) {
	*(*int16)(v.addr) = i
}

// -- Int32

// Int32Value represents an int32 value.
type Int32Value interface {
	Value;
	Get()	int32;	// Get the underlying int32.
	Set(int32);	// Set the underlying int32.
}

type int32ValueStruct struct {
	commonValue
}

func int32Creator(typ Type, addr Addr) Value {
	return &int32ValueStruct{ commonValue{Int32Kind, typ, addr} }
}

func (v *int32ValueStruct) Get() int32 {
	return *(*int32)(v.addr)
}

func (v *int32ValueStruct) Set(i int32) {
	*(*int32)(v.addr) = i
}

// -- Int64

// Int64Value represents an int64 value.
type Int64Value interface {
	Value;
	Get()	int64;	// Get the underlying int64.
	Set(int64);	// Set the underlying int64.
}

type int64ValueStruct struct {
	commonValue
}

func int64Creator(typ Type, addr Addr) Value {
	return &int64ValueStruct{ commonValue{Int64Kind, typ, addr} }
}

func (v *int64ValueStruct) Get() int64 {
	return *(*int64)(v.addr)
}

func (v *int64ValueStruct) Set(i int64) {
	*(*int64)(v.addr) = i
}

// -- Uint

// UintValue represents a uint value.
type UintValue interface {
	Value;
	Get()	uint;	// Get the underlying uint.
	Set(uint);	// Set the underlying uint.
}

type uintValueStruct struct {
	commonValue
}

func uintCreator(typ Type, addr Addr) Value {
	return &uintValueStruct{ commonValue{UintKind, typ, addr} }
}

func (v *uintValueStruct) Get() uint {
	return *(*uint)(v.addr)
}

func (v *uintValueStruct) Set(i uint) {
	*(*uint)(v.addr) = i
}

// -- Uint8

// Uint8Value represents a uint8 value.
type Uint8Value interface {
	Value;
	Get()	uint8;	// Get the underlying uint8.
	Set(uint8);	// Set the underlying uint8.
}

type uint8ValueStruct struct {
	commonValue
}

func uint8Creator(typ Type, addr Addr) Value {
	return &uint8ValueStruct{ commonValue{Uint8Kind, typ, addr} }
}

func (v *uint8ValueStruct) Get() uint8 {
	return *(*uint8)(v.addr)
}

func (v *uint8ValueStruct) Set(i uint8) {
	*(*uint8)(v.addr) = i
}

// -- Uint16

// Uint16Value represents a uint16 value.
type Uint16Value interface {
	Value;
	Get()	uint16;	// Get the underlying uint16.
	Set(uint16);	// Set the underlying uint16.
}

type uint16ValueStruct struct {
	commonValue
}

func uint16Creator(typ Type, addr Addr) Value {
	return &uint16ValueStruct{ commonValue{Uint16Kind, typ, addr} }
}

func (v *uint16ValueStruct) Get() uint16 {
	return *(*uint16)(v.addr)
}

func (v *uint16ValueStruct) Set(i uint16) {
	*(*uint16)(v.addr) = i
}

// -- Uint32

// Uint32Value represents a uint32 value.
type Uint32Value interface {
	Value;
	Get()	uint32;	// Get the underlying uint32.
	Set(uint32);	// Set the underlying uint32.
}

type uint32ValueStruct struct {
	commonValue
}

func uint32Creator(typ Type, addr Addr) Value {
	return &uint32ValueStruct{ commonValue{Uint32Kind, typ, addr} }
}

func (v *uint32ValueStruct) Get() uint32 {
	return *(*uint32)(v.addr)
}

func (v *uint32ValueStruct) Set(i uint32) {
	*(*uint32)(v.addr) = i
}

// -- Uint64

// Uint64Value represents a uint64 value.
type Uint64Value interface {
	Value;
	Get()	uint64;	// Get the underlying uint64.
	Set(uint64);	// Set the underlying uint64.
}

type uint64ValueStruct struct {
	commonValue
}

func uint64Creator(typ Type, addr Addr) Value {
	return &uint64ValueStruct{ commonValue{Uint64Kind, typ, addr} }
}

func (v *uint64ValueStruct) Get() uint64 {
	return *(*uint64)(v.addr)
}

func (v *uint64ValueStruct) Set(i uint64) {
	*(*uint64)(v.addr) = i
}

// -- Uintptr

// UintptrValue represents a uintptr value.
type UintptrValue interface {
	Value;
	Get()	uintptr;	// Get the underlying uintptr.
	Set(uintptr);	// Set the underlying uintptr.
}

type uintptrValueStruct struct {
	commonValue
}

func uintptrCreator(typ Type, addr Addr) Value {
	return &uintptrValueStruct{ commonValue{UintptrKind, typ, addr} }
}

func (v *uintptrValueStruct) Get() uintptr {
	return *(*uintptr)(v.addr)
}

func (v *uintptrValueStruct) Set(i uintptr) {
	*(*uintptr)(v.addr) = i
}

// -- Float

// FloatValue represents a float value.
type FloatValue interface {
	Value;
	Get()	float;	// Get the underlying float.
	Set(float);	// Get the underlying float.
}

type floatValueStruct struct {
	commonValue
}

func floatCreator(typ Type, addr Addr) Value {
	return &floatValueStruct{ commonValue{FloatKind, typ, addr} }
}

func (v *floatValueStruct) Get() float {
	return *(*float)(v.addr)
}

func (v *floatValueStruct) Set(f float) {
	*(*float)(v.addr) = f
}

// -- Float32

// Float32Value represents a float32 value.
type Float32Value interface {
	Value;
	Get()	float32;	// Get the underlying float32.
	Set(float32);	// Get the underlying float32.
}

type float32ValueStruct struct {
	commonValue
}

func float32Creator(typ Type, addr Addr) Value {
	return &float32ValueStruct{ commonValue{Float32Kind, typ, addr} }
}

func (v *float32ValueStruct) Get() float32 {
	return *(*float32)(v.addr)
}

func (v *float32ValueStruct) Set(f float32) {
	*(*float32)(v.addr) = f
}

// -- Float64

// Float64Value represents a float64 value.
type Float64Value interface {
	Value;
	Get()	float64;	// Get the underlying float64.
	Set(float64);	// Get the underlying float64.
}

type float64ValueStruct struct {
	commonValue
}

func float64Creator(typ Type, addr Addr) Value {
	return &float64ValueStruct{ commonValue{Float64Kind, typ, addr} }
}

func (v *float64ValueStruct) Get() float64 {
	return *(*float64)(v.addr)
}

func (v *float64ValueStruct) Set(f float64) {
	*(*float64)(v.addr) = f
}

// -- String

// StringValue represents a string value.
type StringValue interface {
	Value;
	Get()	string;	// Get the underlying string value.
	Set(string);	// Set the underlying string value.
}

type stringValueStruct struct {
	commonValue
}

func stringCreator(typ Type, addr Addr) Value {
	return &stringValueStruct{ commonValue{StringKind, typ, addr} }
}

func (v *stringValueStruct) Get() string {
	return *(*string)(v.addr)
}

func (v *stringValueStruct) Set(s string) {
	*(*string)(v.addr) = s
}

// -- Bool

// BoolValue represents a bool value.
type BoolValue interface {
	Value;
	Get()	bool;	// Get the underlying bool value.
	Set(bool);	// Set the underlying bool value.
}

type boolValueStruct struct {
	commonValue
}

func boolCreator(typ Type, addr Addr) Value {
	return &boolValueStruct{ commonValue{BoolKind, typ, addr} }
}

func (v *boolValueStruct) Get() bool {
	return *(*bool)(v.addr)
}

func (v *boolValueStruct) Set(b bool) {
	*(*bool)(v.addr) = b
}

// -- Pointer

// PtrValue represents a pointer value.
type PtrValue interface {
	Value;
	Sub()	Value;	// The Value pointed to.
	Get()	Addr;	// Get the address stored in the pointer.
	SetSub(Value);	// Set the the pointed-to Value.
	IsNil() bool;
}

type ptrValueStruct struct {
	commonValue
}

func (v *ptrValueStruct) Get() Addr {
	return *(*Addr)(v.addr)
}

func (v *ptrValueStruct) IsNil() bool {
	return uintptr(*(*Addr)(v.addr)) == 0
}

func (v *ptrValueStruct) Sub() Value {
	if v.IsNil() {
		return nil
	}
	return newValueAddr(v.typ.(PtrType).Sub(), v.Get());
}

func (v *ptrValueStruct) SetSub(subv Value) {
	a := v.typ.(PtrType).Sub();
	b := subv.Type();
	if !equalType(a, b) {
		panicln("reflect: incompatible types in PtrValue.SetSub:",
			a.String(), b.String());
	}
	*(*Addr)(v.addr) = subv.Addr();
}

func ptrCreator(typ Type, addr Addr) Value {
	return &ptrValueStruct{ commonValue{PtrKind, typ, addr} };
}

// -- Array
// Slices and arrays are represented by the same interface.

// ArrayValue represents an array or slice value.
type ArrayValue interface {
	Value;
	IsSlice()	bool;	// Is this a slice (true) or array (false)?
	Len()	int;	// The length of the array/slice.
	Cap() int;	// The capacity of the array/slice (==Len() for arrays).
	Elem(i int)	Value;	// The Value of the i'th element.
	SetLen(len int);	// Set the length; slice only.
	Set(src ArrayValue);	// Set the underlying Value; slice only for src and dest both.
	CopyFrom(src ArrayValue, n int);	// Copy the elements from src; lengths must match.
	IsNil() bool;
}

func copyArray(dst ArrayValue, src ArrayValue, n int);

/*
	Run-time representation of slices looks like this:
		struct	Slice {
			byte*	array;		// actual data
			uint32	nel;		// number of elements
			uint32	cap;
		};
*/
type runtimeSlice struct {
	data	Addr;
	len	uint32;
	cap	uint32;
}

type sliceValueStruct struct {
	commonValue;
	elemtype	Type;
	elemsize	int;
	slice *runtimeSlice;
}

func (v *sliceValueStruct) IsSlice() bool {
	return true
}

func (v *sliceValueStruct) Len() int {
	return int(v.slice.len);
}

func (v *sliceValueStruct) Cap() int {
	return int(v.slice.cap);
}

func (v *sliceValueStruct) SetLen(len int) {
	if len > v.Cap() {
		panicln("reflect: sliceValueStruct.SetLen", len, v.Cap());
	}
	v.slice.len = uint32(len);
}

func (v *sliceValueStruct) Set(src ArrayValue) {
	if !src.IsSlice() {
		panic("can't set slice from array");
	}
	s := src.(*sliceValueStruct);
	if !equalType(v.typ, s.typ) {
		panicln("incompatible types in ArrayValue.Set()");
	}
	*v.slice = *s.slice;
}

func (v *sliceValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.slice.data) + uintptr(i * v.elemsize);
	return newValueAddr(v.elemtype, Addr(data_uint));
}

func (v *sliceValueStruct) CopyFrom(src ArrayValue, n int) {
	copyArray(v, src, n);
}

func (v *sliceValueStruct) IsNil() bool {
	return uintptr(v.slice.data) == 0
}

type arrayValueStruct struct {
	commonValue;
	elemtype	Type;
	elemsize	int;
	len	int;
}

func (v *arrayValueStruct) IsSlice() bool {
	return false
}

func (v *arrayValueStruct) Len() int {
	return v.len
}

func (v *arrayValueStruct) Cap() int {
	return v.len
}

func (v *arrayValueStruct) SetLen(len int) {
	panicln("can't set len of array");
}

func (v *arrayValueStruct) Set(src ArrayValue) {
	panicln("can't set array");
}

func (v *arrayValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.addr) + uintptr(i * v.elemsize);
	return newValueAddr(v.elemtype, Addr(data_uint));
}

func (v *arrayValueStruct) CopyFrom(src ArrayValue, n int) {
	copyArray(v, src, n);
}

func (v *arrayValueStruct) IsNil() bool {
	return false
}

func arrayCreator(typ Type, addr Addr) Value {
	arraytype := typ.(ArrayType);
	if arraytype.IsSlice() {
		v := new(sliceValueStruct);
		v.kind = ArrayKind;
		v.addr = addr;
		v.typ = typ;
		v.elemtype = arraytype.Elem();
		v.elemsize = v.elemtype.Size();
		v.slice = (*runtimeSlice)(addr);
		return v;
	}
	v := new(arrayValueStruct);
	v.kind = ArrayKind;
	v.addr = addr;
	v.typ = typ;
	v.elemtype = arraytype.Elem();
	v.elemsize = v.elemtype.Size();
	v.len = arraytype.Len();
	return v;
}

// -- Map	TODO: finish and test

// MapValue represents a map value.
// Its implementation is incomplete.
type MapValue interface {
	Value;
	Len()	int;	// The number of elements; currently always returns 0.
	Elem(key Value)	Value;	// The value indexed by key; unimplemented.
	IsNil() bool;
}

type mapValueStruct struct {
	commonValue
}

func mapCreator(typ Type, addr Addr) Value {
	return &mapValueStruct{ commonValue{MapKind, typ, addr} }
}

func (v *mapValueStruct) Len() int {
	return 0	// TODO: probably want this to be dynamic
}

func (v *mapValueStruct) IsNil() bool {
	return false	// TODO: implement this properly
}

func (v *mapValueStruct) Elem(key Value) Value {
	panic("map value element");
	return nil
}

// -- Chan

// ChanValue represents a chan value.
// Its implementation is incomplete.
type ChanValue interface {
	Value;
	IsNil() bool;
}

type chanValueStruct struct {
	commonValue
}

func (v *chanValueStruct) IsNil() bool {
	return false	// TODO: implement this properly
}

func chanCreator(typ Type, addr Addr) Value {
	return &chanValueStruct{ commonValue{ChanKind, typ, addr} }
}

// -- Struct

// StructValue represents a struct value.
type StructValue interface {
	Value;
	Len()	int;	// The number of fields.
	Field(i int)	Value;	// The Value of field i.
}

type structValueStruct struct {
	commonValue;
	field	[]Value;
}

func (v *structValueStruct) Len() int {
	return len(v.field)
}

func (v *structValueStruct) Field(i int) Value {
	return v.field[i]
}

func structCreator(typ Type, addr Addr) Value {
	t := typ.(StructType);
	nfield := t.Len();
	v := &structValueStruct{ commonValue{StructKind, typ, addr}, make([]Value, nfield) };
	for i := 0; i < nfield; i++ {
		name, ftype, str, offset := t.Field(i);
		addr_uint := uintptr(addr) + uintptr(offset);
		v.field[i] = newValueAddr(ftype, Addr(addr_uint));
	}
	v.typ = typ;
	return v;
}

// -- Interface

// InterfaceValue represents an interface value.
type InterfaceValue interface {
	Value;
	Get()	interface {};	// Get the underlying interface{} value.
	Value() Value;
	IsNil() bool;
}

type interfaceValueStruct struct {
	commonValue
}

func (v *interfaceValueStruct) Get() interface{} {
	// There are two different representations of interface values,
	// one if the interface type has methods and one if it doesn't.
	// These two representations require different expressions
	// to extract correctly.
	if v.Type().(InterfaceType).Len() == 0 {
		// Extract as interface value without methods.
		return *(*interface{})(v.addr)
	}
	// Extract from v.addr as interface value with methods.
	return *(*interface{ m() })(v.addr)
}

func (v *interfaceValueStruct) Interface() interface{} {
	return v.Get();
}

func (v *interfaceValueStruct) Value() Value {
	i := v.Get();
	if i == nil {
		return nil;
	}
	return NewValue(i);
}

func (v *interfaceValueStruct) IsNil() bool {
	return *(*interface{})(v.addr) == nil
}

func interfaceCreator(typ Type, addr Addr) Value {
	return &interfaceValueStruct{ commonValue{InterfaceKind, typ, addr} }
}

// -- Func


// FuncValue represents a func value.
// Its implementation is incomplete.
type FuncValue interface {
	Value;
	Get()	Addr;	// The address of the function.
	IsNil() bool;
}

type funcValueStruct struct {
	commonValue
}

func (v *funcValueStruct) Get() Addr {
	return *(*Addr)(v.addr)
}

func (v *funcValueStruct) IsNil() bool {
	return *(*Addr)(v.addr) == nil
}

func funcCreator(typ Type, addr Addr) Value {
	return &funcValueStruct{ commonValue{FuncKind, typ, addr} }
}

var creator = map[int] creatorFn {
	MissingKind : missingCreator,
	IntKind : intCreator,
	Int8Kind : int8Creator,
	Int16Kind : int16Creator,
	Int32Kind : int32Creator,
	Int64Kind : int64Creator,
	UintKind : uintCreator,
	Uint8Kind : uint8Creator,
	Uint16Kind : uint16Creator,
	Uint32Kind : uint32Creator,
	Uint64Kind : uint64Creator,
	UintptrKind : uintptrCreator,
	FloatKind : floatCreator,
	Float32Kind : float32Creator,
	Float64Kind : float64Creator,
	StringKind : stringCreator,
	BoolKind : boolCreator,
	PtrKind : ptrCreator,
	ArrayKind : arrayCreator,
	MapKind : mapCreator,
	ChanKind : chanCreator,
	StructKind : structCreator,
	InterfaceKind : interfaceCreator,
	FuncKind : funcCreator,
}

var typecache = make(map[string] Type);

func newValueAddr(typ Type, addr Addr) Value {
	c, ok := creator[typ.Kind()];
	if !ok {
		panicln("no creator for type" , typ.String());
	}
	return c(typ, addr);
}

// NewZeroValue creates a new, zero-initialized Value for the specified Type.
func NewZeroValue(typ Type) Value {
	size := typ.Size();
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	return newValueAddr(typ, Addr(&data[0]));
}

// NewSliceValue creates a new, zero-initialized slice value (ArrayValue) for the specified
// slice type (ArrayType), length, and capacity.
func NewSliceValue(typ ArrayType, len, cap int) ArrayValue {
	if !typ.IsSlice() {
		return nil
	}

	array := new(runtimeSlice);
	size := typ.Elem().Size() * cap;
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	array.data = Addr(&data[0]);
	array.len = uint32(len);
	array.cap = uint32(cap);

	return newValueAddr(typ, Addr(array)).(ArrayValue);
}

// Works on both slices and arrays
func copyArray(dst ArrayValue, src ArrayValue, n int) {
	if n == 0 {
		return
	}
	dt := dst.Type().(ArrayType).Elem();
	st := src.Type().(ArrayType).Elem();
	if !equalType(dt, st) {
		panicln("reflect: incompatible types in CopyArray:",
			dt.String(), st.String());
	}
	if n < 0 || n > dst.Len() || n > src.Len() {
		panicln("reflect: CopyArray: invalid count", n);
	}
	dstp := uintptr(dst.Elem(0).Addr());
	srcp := uintptr(src.Elem(0).Addr());
	end := uintptr(n)*uintptr(dt.Size());
	if end % 8 == 0 {
		for i := uintptr(0); i < end; i += 8{
			di := Addr(dstp + i);
			si := Addr(srcp + i);
			*(*uint64)(di) = *(*uint64)(si);
		}
	} else {
		for i := uintptr(0); i < end; i++ {
			di := Addr(dstp + i);
			si := Addr(srcp + i);
			*(*byte)(di) = *(*byte)(si);
		}
	}
}

// NewValue creates a new Value from the interface{} object provided.
func NewValue(e interface {}) Value {
	value, typestring, indir := unsafe.Reflect(e);
	typ, ok := typecache[typestring];
	if !ok {
		typ = ParseTypeString("", typestring);
		if typ.Kind() == MissingKind {
			// This can not happen: unsafe.Reflect should only
			// ever tell us the names of types that exist.
			// Of course it does happen, and when it does
			// it is more helpful to catch it in action here than
			// to see $missing$ in a later print.
			panicln("missing type for", typestring);
		}
		typecache[typestring] = typ;
	}
	var ap Addr;
	if indir {
		// Content of interface is large and didn't
		// fit, so it's a pointer to the actual content.
		// We have an address, but we need to
		// make a copy to avoid letting the caller
		// edit the content inside the interface.
		n := uintptr(typ.Size());
		data := make([]byte, n);
		p1 := uintptr(Addr(&data[0]));
		p2 := uintptr(value);
		for i := uintptr(0); i < n; i++ {
			*(*byte)(Addr(p1+i)) = *(*byte)(Addr(p2+i));
		}
		ap = Addr(&data[0]);
	} else {
		// Content of interface is small and stored
		// inside the interface.  Make a copy so we
		// can take its address.
		x := new(uint64);
		*x = value;
		ap = Addr(x);
	}
	return newValueAddr(typ, ap);
}

// Indirect indirects one level through a value, if it is a pointer.
// If not a pointer, the value is returned unchanged.
// Useful when walking arbitrary data structures.
func Indirect(v Value) Value {
	if v.Kind() == PtrKind {
		p := v.(PtrValue);
		if p.Get() == nil {
			return nil
		}
		v = p.Sub()
	}
	return v
}
