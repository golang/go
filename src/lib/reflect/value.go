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

export type Addr unsafe.pointer

func equalType(a, b Type) bool {
	return a.String() == b.String()
}

export type Value interface {
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
		i = sys.unreflect(c.addr.(uintptr).(uint64), c.typ.String(), true);
	} else {
		if uintptr(c.addr) == 0 {
			panicln("reflect: address 0 for", c.typ.String());
		}
		i = sys.unreflect(uint64(uintptr(*c.addr.(*Addr))), c.typ.String(), false);
	}
	return i;
}

func newValueAddr(typ Type, addr Addr) Value

type creatorFn *(typ Type, addr Addr) Value


// -- Missing

export type MissingValue interface {
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
}

type missingValueStruct struct {
	commonValue
}

func missingCreator(typ Type, addr Addr) Value {
	return &missingValueStruct{ commonValue{MissingKind, typ, addr} }
}

// -- Int

export type IntValue interface {
	Kind()	int;
	Get()	int;
	Set(int);
	Type()	Type;
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

export type Int8Value interface {
	Kind()	int;
	Get()	int8;
	Set(int8);
	Type()	Type;
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

export type Int16Value interface {
	Kind()	int;
	Get()	int16;
	Set(int16);
	Type()	Type;
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

export type Int32Value interface {
	Kind()	int;
	Get()	int32;
	Set(int32);
	Type()	Type;
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

export type Int64Value interface {
	Kind()	int;
	Get()	int64;
	Set(int64);
	Type()	Type;
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

export type UintValue interface {
	Kind()	int;
	Get()	uint;
	Set(uint);
	Type()	Type;
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

export type Uint8Value interface {
	Kind()	int;
	Get()	uint8;
	Set(uint8);
	Type()	Type;
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

export type Uint16Value interface {
	Kind()	int;
	Get()	uint16;
	Set(uint16);
	Type()	Type;
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

export type Uint32Value interface {
	Kind()	int;
	Get()	uint32;
	Set(uint32);
	Type()	Type;
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

export type Uint64Value interface {
	Kind()	int;
	Get()	uint64;
	Set(uint64);
	Type()	Type;
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

export type UintptrValue interface {
	Kind()	int;
	Get()	uintptr;
	Set(uintptr);
	Type()	Type;
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

export type FloatValue interface {
	Kind()	int;
	Get()	float;
	Set(float);
	Type()	Type;
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

export type Float32Value interface {
	Kind()	int;
	Get()	float32;
	Set(float32);
	Type()	Type;
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

export type Float64Value interface {
	Kind()	int;
	Get()	float64;
	Set(float64);
	Type()	Type;
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

export type Float80Value interface {
	Kind()	int;
	Get()	float80;
	Set(float80);
	Type()	Type;
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

export type StringValue interface {
	Kind()	int;
	Get()	string;
	Set(string);
	Type()	Type;
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

export type BoolValue interface {
	Kind()	int;
	Get()	bool;
	Set(bool);
	Type()	Type;
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

export type PtrValue interface {
	Kind()	int;
	Type()	Type;
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

export type ArrayValue interface {
	Kind()	int;
	Type()	Type;
	Open()	bool;
	Len()	int;
	Cap() int;
	Elem(i int)	Value;
	SetLen(len int);
}

/*
	Run-time representation of open arrays looks like this:
		struct	Array {
			byte*	array;		// actual data
			uint32	nel;		// number of elements
			uint32	cap;
		};
*/
type runtimeArray struct {
	data	Addr;
	len	uint32;
	cap	uint32;
}

type openArrayValueStruct struct {
	commonValue;
	elemtype	Type;
	elemsize	int;
	array *runtimeArray;
}

func (v *openArrayValueStruct) Open() bool {
	return true
}

func (v *openArrayValueStruct) Len() int {
	return int(v.array.len);
}

func (v *openArrayValueStruct) Cap() int {
	return int(v.array.cap);
}

func (v *openArrayValueStruct) SetLen(len int) {
	if len > v.Cap() {
		panicln("reflect: OpenArrayValueStruct.SetLen", len, v.Cap());
	}
	v.array.len = uint32(len);
}

func (v *openArrayValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.array.data) + uintptr(i * v.elemsize);
	return newValueAddr(v.elemtype, Addr(data_uint));
}

type fixedArrayValueStruct struct {
	commonValue;
	elemtype	Type;
	elemsize	int;
	len	int;
}

func (v *fixedArrayValueStruct) Open() bool {
	return false
}

func (v *fixedArrayValueStruct) Len() int {
	return v.len
}

func (v *fixedArrayValueStruct) Cap() int {
	return v.len
}

func (v *fixedArrayValueStruct) SetLen(len int) {
}

func (v *fixedArrayValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.addr) + uintptr(i * v.elemsize);
	return newValueAddr(v.elemtype, Addr(data_uint));
	return nil
}

func arrayCreator(typ Type, addr Addr) Value {
	arraytype := typ.(ArrayType);
	if arraytype.Open() {
		v := new(openArrayValueStruct);
		v.kind = ArrayKind;
		v.addr = addr;
		v.typ = typ;
		v.elemtype = arraytype.Elem();
		v.elemsize = v.elemtype.Size();
		v.array = addr.(*runtimeArray);
		return v;
	}
	v := new(fixedArrayValueStruct);
	v.kind = ArrayKind;
	v.addr = addr;
	v.typ = typ;
	v.elemtype = arraytype.Elem();
	v.elemsize = v.elemtype.Size();
	v.len = arraytype.Len();
	return v;
}

// -- Map	TODO: finish and test

export type MapValue interface {
	Kind()	int;
	Type()	Type;
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

export type ChanValue interface {
	Kind()	int;
	Type()	Type;
}

type chanValueStruct struct {
	commonValue
}

func chanCreator(typ Type, addr Addr) Value {
	return &chanValueStruct{ commonValue{ChanKind, typ, addr} }
}

// -- Struct

export type StructValue interface {
	Kind()	int;
	Type()	Type;
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

export type InterfaceValue interface {
	Kind()	int;
	Type()	Type;
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

export type FuncValue interface {
	Kind()	int;
	Type()	Type;
}

type funcValueStruct struct {
	commonValue
}

func funcCreator(typ Type, addr Addr) Value {
	return &funcValueStruct{ commonValue{FuncKind, typ, addr} }
}

var creator = map[int] creatorFn {
	MissingKind : &missingCreator,
	IntKind : &intCreator,
	Int8Kind : &int8Creator,
	Int16Kind : &int16Creator,
	Int32Kind : &int32Creator,
	Int64Kind : &int64Creator,
	UintKind : &uintCreator,
	Uint8Kind : &uint8Creator,
	Uint16Kind : &uint16Creator,
	Uint32Kind : &uint32Creator,
	Uint64Kind : &uint64Creator,
	UintptrKind : &uintptrCreator,
	FloatKind : &floatCreator,
	Float32Kind : &float32Creator,
	Float64Kind : &float64Creator,
	Float80Kind : &float80Creator,
	StringKind : &stringCreator,
	BoolKind : &boolCreator,
	PtrKind : &ptrCreator,
	ArrayKind : &arrayCreator,
	MapKind : &mapCreator,
	ChanKind : &chanCreator,
	StructKind : &structCreator,
	InterfaceKind : &interfaceCreator,
	FuncKind : &funcCreator,
}

var typecache = make(map[string] Type);

func newValueAddr(typ Type, addr Addr) Value {
	c, ok := creator[typ.Kind()];
	if !ok {
		panicln("no creator for type" , typ.Kind());
	}
	return c(typ, addr);
}

export func NewInitValue(typ Type) Value {
	// Some values cannot be made this way.
	switch typ.Kind() {
	case FuncKind:	// must be pointers, at least for now (TODO?)
		return nil;
	case ArrayKind:
		if typ.(ArrayType).Open() {
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

/*
	Run-time representation of open arrays looks like this:
		struct	Array {
			byte*	array;		// actual data
			uint32	nel;		// number of elements
			uint32	cap;		// allocated number of elements
		};
*/
export func NewOpenArrayValue(typ ArrayType, len, cap int) ArrayValue {
	if !typ.Open() {
		return nil
	}

	array := new(runtimeArray);
	size := typ.Elem().Size() * cap;
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	array.data = Addr(&data[0]);
	array.len = uint32(len);
	array.cap = uint32(cap);

	return newValueAddr(typ, Addr(array));
}

export func CopyArray(dst ArrayValue, src ArrayValue, n int) {
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


export func NewValue(e interface {}) Value {
	value, typestring, indir := sys.reflect(e);
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
