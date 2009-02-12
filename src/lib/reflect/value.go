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

type Addr unsafe.Pointer

func equalType(a, b Type) bool {
	return a.String() == b.String()
}

type Value interface {
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};
}

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
	if c.typ.Size() > 8 {	// TODO(rsc): how do we know it is 8?
		i = sys.Unreflect(c.addr.(uintptr).(uint64), c.typ.String(), true);
	} else {
		if uintptr(c.addr) == 0 {
			panicln("reflect: address 0 for", c.typ.String());
		}
		i = sys.Unreflect(uint64(uintptr(*c.addr.(*Addr))), c.typ.String(), false);
	}
	return i;
}

func newValueAddr(typ Type, addr Addr) Value

type creatorFn func(typ Type, addr Addr) Value


// -- Missing

type MissingValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};
}

type missingValueStruct struct {
	commonValue
}

func missingCreator(typ Type, addr Addr) Value {
	return &missingValueStruct{ commonValue{MissingKind, typ, addr} }
}

// -- Int

type IntValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	int;
	Set(int);
}

type intValueStruct struct {
	commonValue
}

func intCreator(typ Type, addr Addr) Value {
	return &intValueStruct{ commonValue{IntKind, typ, addr} }
}

func (v *intValueStruct) Get() int {
	return *v.addr.(*int)
}

func (v *intValueStruct) Set(i int) {
	*v.addr.(*int) = i
}

// -- Int8

type Int8Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	int8;
	Set(int8);
}

type int8ValueStruct struct {
	commonValue
}

func int8Creator(typ Type, addr Addr) Value {
	return &int8ValueStruct{ commonValue{Int8Kind, typ, addr} }
}

func (v *int8ValueStruct) Get() int8 {
	return *v.addr.(*int8)
}

func (v *int8ValueStruct) Set(i int8) {
	*v.addr.(*int8) = i
}

// -- Int16

type Int16Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	int16;
	Set(int16);
}

type int16ValueStruct struct {
	commonValue
}

func int16Creator(typ Type, addr Addr) Value {
	return &int16ValueStruct{ commonValue{Int16Kind, typ, addr} }
}

func (v *int16ValueStruct) Get() int16 {
	return *v.addr.(*int16)
}

func (v *int16ValueStruct) Set(i int16) {
	*v.addr.(*int16) = i
}

// -- Int32

type Int32Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	int32;
	Set(int32);
}

type int32ValueStruct struct {
	commonValue
}

func int32Creator(typ Type, addr Addr) Value {
	return &int32ValueStruct{ commonValue{Int32Kind, typ, addr} }
}

func (v *int32ValueStruct) Get() int32 {
	return *v.addr.(*int32)
}

func (v *int32ValueStruct) Set(i int32) {
	*v.addr.(*int32) = i
}

// -- Int64

type Int64Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	int64;
	Set(int64);
}

type int64ValueStruct struct {
	commonValue
}

func int64Creator(typ Type, addr Addr) Value {
	return &int64ValueStruct{ commonValue{Int64Kind, typ, addr} }
}

func (v *int64ValueStruct) Get() int64 {
	return *v.addr.(*int64)
}

func (v *int64ValueStruct) Set(i int64) {
	*v.addr.(*int64) = i
}

// -- Uint

type UintValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	uint;
	Set(uint);
}

type uintValueStruct struct {
	commonValue
}

func uintCreator(typ Type, addr Addr) Value {
	return &uintValueStruct{ commonValue{UintKind, typ, addr} }
}

func (v *uintValueStruct) Get() uint {
	return *v.addr.(*uint)
}

func (v *uintValueStruct) Set(i uint) {
	*v.addr.(*uint) = i
}

// -- Uint8

type Uint8Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	uint8;
	Set(uint8);
}

type uint8ValueStruct struct {
	commonValue
}

func uint8Creator(typ Type, addr Addr) Value {
	return &uint8ValueStruct{ commonValue{Uint8Kind, typ, addr} }
}

func (v *uint8ValueStruct) Get() uint8 {
	return *v.addr.(*uint8)
}

func (v *uint8ValueStruct) Set(i uint8) {
	*v.addr.(*uint8) = i
}

// -- Uint16

type Uint16Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	uint16;
	Set(uint16);
}

type uint16ValueStruct struct {
	commonValue
}

func uint16Creator(typ Type, addr Addr) Value {
	return &uint16ValueStruct{ commonValue{Uint16Kind, typ, addr} }
}

func (v *uint16ValueStruct) Get() uint16 {
	return *v.addr.(*uint16)
}

func (v *uint16ValueStruct) Set(i uint16) {
	*v.addr.(*uint16) = i
}

// -- Uint32

type Uint32Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	uint32;
	Set(uint32);
}

type uint32ValueStruct struct {
	commonValue
}

func uint32Creator(typ Type, addr Addr) Value {
	return &uint32ValueStruct{ commonValue{Uint32Kind, typ, addr} }
}

func (v *uint32ValueStruct) Get() uint32 {
	return *v.addr.(*uint32)
}

func (v *uint32ValueStruct) Set(i uint32) {
	*v.addr.(*uint32) = i
}

// -- Uint64

type Uint64Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	uint64;
	Set(uint64);
}

type uint64ValueStruct struct {
	commonValue
}

func uint64Creator(typ Type, addr Addr) Value {
	return &uint64ValueStruct{ commonValue{Uint64Kind, typ, addr} }
}

func (v *uint64ValueStruct) Get() uint64 {
	return *v.addr.(*uint64)
}

func (v *uint64ValueStruct) Set(i uint64) {
	*v.addr.(*uint64) = i
}

// -- Uintptr

type UintptrValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	uintptr;
	Set(uintptr);
}

type uintptrValueStruct struct {
	commonValue
}

func uintptrCreator(typ Type, addr Addr) Value {
	return &uintptrValueStruct{ commonValue{UintptrKind, typ, addr} }
}

func (v *uintptrValueStruct) Get() uintptr {
	return *v.addr.(*uintptr)
}

func (v *uintptrValueStruct) Set(i uintptr) {
	*v.addr.(*uintptr) = i
}

// -- Float

type FloatValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	float;
	Set(float);
}

type floatValueStruct struct {
	commonValue
}

func floatCreator(typ Type, addr Addr) Value {
	return &floatValueStruct{ commonValue{FloatKind, typ, addr} }
}

func (v *floatValueStruct) Get() float {
	return *v.addr.(*float)
}

func (v *floatValueStruct) Set(f float) {
	*v.addr.(*float) = f
}

// -- Float32

type Float32Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	float32;
	Set(float32);
}

type float32ValueStruct struct {
	commonValue
}

func float32Creator(typ Type, addr Addr) Value {
	return &float32ValueStruct{ commonValue{Float32Kind, typ, addr} }
}

func (v *float32ValueStruct) Get() float32 {
	return *v.addr.(*float32)
}

func (v *float32ValueStruct) Set(f float32) {
	*v.addr.(*float32) = f
}

// -- Float64

type Float64Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	float64;
	Set(float64);
}

type float64ValueStruct struct {
	commonValue
}

func float64Creator(typ Type, addr Addr) Value {
	return &float64ValueStruct{ commonValue{Float64Kind, typ, addr} }
}

func (v *float64ValueStruct) Get() float64 {
	return *v.addr.(*float64)
}

func (v *float64ValueStruct) Set(f float64) {
	*v.addr.(*float64) = f
}

// -- Float80

type Float80Value interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	float80;
	Set(float80);
}

type float80ValueStruct struct {
	commonValue
}

func float80Creator(typ Type, addr Addr) Value {
	return &float80ValueStruct{ commonValue{Float80Kind, typ, addr} }
}

/*
BUG: can't gen code for float80s
func (v *Float80ValueStruct) Get() float80 {
	return *v.addr.(*float80)
}

func (v *Float80ValueStruct) Set(f float80) {
	*v.addr.(*float80) = f
}
*/

// -- String

type StringValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	string;
	Set(string);
}

type stringValueStruct struct {
	commonValue
}

func stringCreator(typ Type, addr Addr) Value {
	return &stringValueStruct{ commonValue{StringKind, typ, addr} }
}

func (v *stringValueStruct) Get() string {
	return *v.addr.(*string)
}

func (v *stringValueStruct) Set(s string) {
	*v.addr.(*string) = s
}

// -- Bool

type BoolValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	bool;
	Set(bool);
}

type boolValueStruct struct {
	commonValue
}

func boolCreator(typ Type, addr Addr) Value {
	return &boolValueStruct{ commonValue{BoolKind, typ, addr} }
}

func (v *boolValueStruct) Get() bool {
	return *v.addr.(*bool)
}

func (v *boolValueStruct) Set(b bool) {
	*v.addr.(*bool) = b
}

// -- Pointer

type PtrValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Sub()	Value;
	Get()	Addr;
	SetSub(Value);
}

type ptrValueStruct struct {
	commonValue
}

func (v *ptrValueStruct) Get() Addr {
	return *v.addr.(*Addr)
}

func (v *ptrValueStruct) Sub() Value {
	return newValueAddr(v.typ.(PtrType).Sub(), v.Get());
}

func (v *ptrValueStruct) SetSub(subv Value) {
	a := v.typ.(PtrType).Sub();
	b := subv.Type();
	if !equalType(a, b) {
		panicln("reflect: incompatible types in PtrValue.SetSub:",
			a.String(), b.String());
	}
	*v.addr.(*Addr) = subv.Addr();
}

func ptrCreator(typ Type, addr Addr) Value {
	return &ptrValueStruct{ commonValue{PtrKind, typ, addr} };
}

// -- Array
// Slices and arrays are represented by the same interface.

type ArrayValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	IsSlice()	bool;
	Len()	int;
	Cap() int;
	Elem(i int)	Value;
	SetLen(len int);
	Set(src ArrayValue);
	CopyFrom(src ArrayValue, n int)
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
		panic("can't set from fixed array");
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
}

func (v *arrayValueStruct) Set(src ArrayValue) {
	panicln("can't set fixed array");
}

func (v *arrayValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.addr) + uintptr(i * v.elemsize);
	return newValueAddr(v.elemtype, Addr(data_uint));
	return nil
}

func (v *arrayValueStruct) CopyFrom(src ArrayValue, n int) {
	copyArray(v, src, n);
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
		v.slice = addr.(*runtimeSlice);
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

type MapValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Len()	int;
	Elem(key Value)	Value;
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

func (v *mapValueStruct) Elem(key Value) Value {
	panic("map value element");
	return nil
}

// -- Chan

type ChanValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};
}

type chanValueStruct struct {
	commonValue
}

func chanCreator(typ Type, addr Addr) Value {
	return &chanValueStruct{ commonValue{ChanKind, typ, addr} }
}

// -- Struct

type StructValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Len()	int;
	Field(i int)	Value;
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

type InterfaceValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};

	Get()	interface {};
}

type interfaceValueStruct struct {
	commonValue
}

func (v *interfaceValueStruct) Get() interface{} {
	return *v.addr.(*interface{})
}

func interfaceCreator(typ Type, addr Addr) Value {
	return &interfaceValueStruct{ commonValue{InterfaceKind, typ, addr} }
}

// -- Func

type FuncValue interface {
	// TODO: Value;
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};
}

type funcValueStruct struct {
	commonValue
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
	Float80Kind : float80Creator,
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
		panicln("no creator for type" , typ.Kind());
	}
	return c(typ, addr);
}

func NewInitValue(typ Type) Value {
	// Some values cannot be made this way.
	switch typ.Kind() {
	case FuncKind:	// must be pointers, at least for now (TODO?)
		return nil;
	case ArrayKind:
		if typ.(ArrayType).IsSlice() {
			return nil
		}
	}
	size := typ.Size();
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	return newValueAddr(typ, Addr(&data[0]));
}

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
			*di.(*uint64) = *si.(*uint64);
		}
	} else {
		for i := uintptr(0); i < end; i++ {
			di := Addr(dstp + i);
			si := Addr(srcp + i);
			*di.(*byte) = *si.(*byte);
		}
	}
}


func NewValue(e interface {}) Value {
	value, typestring, indir := sys.Reflect(e);
	typ, ok := typecache[typestring];
	if !ok {
		typ = ParseTypeString("", typestring);
		typecache[typestring] = typ;
	}

	if indir {
		// Content of interface is a pointer.
		return newValueAddr(typ, value.(uintptr).(Addr));
	}

	// Content of interface is a value;
	// need a permanent copy to take its address.
	ap := new(uint64);
	*ap = value;
	return newValueAddr(typ, ap.(Addr));
}
