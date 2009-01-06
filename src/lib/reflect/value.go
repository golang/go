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

type Addr unsafe.pointer

func EqualType(a, b Type) bool {
	return a.String() == b.String()
}

export type Value interface {
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
	Interface()	interface {};
}

// Common fields and functionality for all values

type Common struct {
	kind	int;
	typ	Type;
	addr	Addr;
}

func (c *Common) Kind() int {
	return c.kind
}

func (c *Common) Type() Type {
	return c.typ
}

func (c *Common) Addr() Addr {
	return c.addr
}

func (c *Common) Interface() interface {} {
	if uintptr(c.addr) == 0 {
		panicln("reflect: address 0 for", c.typ.String());
	}
	return sys.unreflect(uint64(uintptr(*c.addr.(*Addr))), c.typ.String());
}

func NewValueAddr(typ Type, addr Addr) Value

type Creator *(typ Type, addr Addr) Value


// -- Missing

export type MissingValue interface {
	Kind()	int;
	Type()	Type;
	Addr()	Addr;
}

type MissingValueStruct struct {
	Common
}

func MissingCreator(typ Type, addr Addr) Value {
	return &MissingValueStruct{ Common{MissingKind, typ, addr} }
}

// -- Int

export type IntValue interface {
	Kind()	int;
	Get()	int;
	Set(int);
	Type()	Type;
}

type IntValueStruct struct {
	Common
}

func IntCreator(typ Type, addr Addr) Value {
	return &IntValueStruct{ Common{IntKind, typ, addr} }
}

func (v *IntValueStruct) Get() int {
	return *v.addr.(*int)
}

func (v *IntValueStruct) Set(i int) {
	*v.addr.(*int) = i
}

// -- Int8

export type Int8Value interface {
	Kind()	int;
	Get()	int8;
	Set(int8);
	Type()	Type;
}

type Int8ValueStruct struct {
	Common
}

func Int8Creator(typ Type, addr Addr) Value {
	return &Int8ValueStruct{ Common{Int8Kind, typ, addr} }
}

func (v *Int8ValueStruct) Get() int8 {
	return *v.addr.(*int8)
}

func (v *Int8ValueStruct) Set(i int8) {
	*v.addr.(*int8) = i
}

// -- Int16

export type Int16Value interface {
	Kind()	int;
	Get()	int16;
	Set(int16);
	Type()	Type;
}

type Int16ValueStruct struct {
	Common
}

func Int16Creator(typ Type, addr Addr) Value {
	return &Int16ValueStruct{ Common{Int16Kind, typ, addr} }
}

func (v *Int16ValueStruct) Get() int16 {
	return *v.addr.(*int16)
}

func (v *Int16ValueStruct) Set(i int16) {
	*v.addr.(*int16) = i
}

// -- Int32

export type Int32Value interface {
	Kind()	int;
	Get()	int32;
	Set(int32);
	Type()	Type;
}

type Int32ValueStruct struct {
	Common
}

func Int32Creator(typ Type, addr Addr) Value {
	return &Int32ValueStruct{ Common{Int32Kind, typ, addr} }
}

func (v *Int32ValueStruct) Get() int32 {
	return *v.addr.(*int32)
}

func (v *Int32ValueStruct) Set(i int32) {
	*v.addr.(*int32) = i
}

// -- Int64

export type Int64Value interface {
	Kind()	int;
	Get()	int64;
	Set(int64);
	Type()	Type;
}

type Int64ValueStruct struct {
	Common
}

func Int64Creator(typ Type, addr Addr) Value {
	return &Int64ValueStruct{ Common{Int64Kind, typ, addr} }
}

func (v *Int64ValueStruct) Get() int64 {
	return *v.addr.(*int64)
}

func (v *Int64ValueStruct) Set(i int64) {
	*v.addr.(*int64) = i
}

// -- Uint

export type UintValue interface {
	Kind()	int;
	Get()	uint;
	Set(uint);
	Type()	Type;
}

type UintValueStruct struct {
	Common
}

func UintCreator(typ Type, addr Addr) Value {
	return &UintValueStruct{ Common{UintKind, typ, addr} }
}

func (v *UintValueStruct) Get() uint {
	return *v.addr.(*uint)
}

func (v *UintValueStruct) Set(i uint) {
	*v.addr.(*uint) = i
}

// -- Uint8

export type Uint8Value interface {
	Kind()	int;
	Get()	uint8;
	Set(uint8);
	Type()	Type;
}

type Uint8ValueStruct struct {
	Common
}

func Uint8Creator(typ Type, addr Addr) Value {
	return &Uint8ValueStruct{ Common{Uint8Kind, typ, addr} }
}

func (v *Uint8ValueStruct) Get() uint8 {
	return *v.addr.(*uint8)
}

func (v *Uint8ValueStruct) Set(i uint8) {
	*v.addr.(*uint8) = i
}

// -- Uint16

export type Uint16Value interface {
	Kind()	int;
	Get()	uint16;
	Set(uint16);
	Type()	Type;
}

type Uint16ValueStruct struct {
	Common
}

func Uint16Creator(typ Type, addr Addr) Value {
	return &Uint16ValueStruct{ Common{Uint16Kind, typ, addr} }
}

func (v *Uint16ValueStruct) Get() uint16 {
	return *v.addr.(*uint16)
}

func (v *Uint16ValueStruct) Set(i uint16) {
	*v.addr.(*uint16) = i
}

// -- Uint32

export type Uint32Value interface {
	Kind()	int;
	Get()	uint32;
	Set(uint32);
	Type()	Type;
}

type Uint32ValueStruct struct {
	Common
}

func Uint32Creator(typ Type, addr Addr) Value {
	return &Uint32ValueStruct{ Common{Uint32Kind, typ, addr} }
}

func (v *Uint32ValueStruct) Get() uint32 {
	return *v.addr.(*uint32)
}

func (v *Uint32ValueStruct) Set(i uint32) {
	*v.addr.(*uint32) = i
}

// -- Uint64

export type Uint64Value interface {
	Kind()	int;
	Get()	uint64;
	Set(uint64);
	Type()	Type;
}

type Uint64ValueStruct struct {
	Common
}

func Uint64Creator(typ Type, addr Addr) Value {
	return &Uint64ValueStruct{ Common{Uint64Kind, typ, addr} }
}

func (v *Uint64ValueStruct) Get() uint64 {
	return *v.addr.(*uint64)
}

func (v *Uint64ValueStruct) Set(i uint64) {
	*v.addr.(*uint64) = i
}

// -- Uintptr

export type UintptrValue interface {
	Kind()	int;
	Get()	uintptr;
	Set(uintptr);
	Type()	Type;
}

type UintptrValueStruct struct {
	Common
}

func UintptrCreator(typ Type, addr Addr) Value {
	return &UintptrValueStruct{ Common{UintptrKind, typ, addr} }
}

func (v *UintptrValueStruct) Get() uintptr {
	return *v.addr.(*uintptr)
}

func (v *UintptrValueStruct) Set(i uintptr) {
	*v.addr.(*uintptr) = i
}

// -- Float

export type FloatValue interface {
	Kind()	int;
	Get()	float;
	Set(float);
	Type()	Type;
}

type FloatValueStruct struct {
	Common
}

func FloatCreator(typ Type, addr Addr) Value {
	return &FloatValueStruct{ Common{FloatKind, typ, addr} }
}

func (v *FloatValueStruct) Get() float {
	return *v.addr.(*float)
}

func (v *FloatValueStruct) Set(f float) {
	*v.addr.(*float) = f
}

// -- Float32

export type Float32Value interface {
	Kind()	int;
	Get()	float32;
	Set(float32);
	Type()	Type;
}

type Float32ValueStruct struct {
	Common
}

func Float32Creator(typ Type, addr Addr) Value {
	return &Float32ValueStruct{ Common{Float32Kind, typ, addr} }
}

func (v *Float32ValueStruct) Get() float32 {
	return *v.addr.(*float32)
}

func (v *Float32ValueStruct) Set(f float32) {
	*v.addr.(*float32) = f
}

// -- Float64

export type Float64Value interface {
	Kind()	int;
	Get()	float64;
	Set(float64);
	Type()	Type;
}

type Float64ValueStruct struct {
	Common
}

func Float64Creator(typ Type, addr Addr) Value {
	return &Float64ValueStruct{ Common{Float64Kind, typ, addr} }
}

func (v *Float64ValueStruct) Get() float64 {
	return *v.addr.(*float64)
}

func (v *Float64ValueStruct) Set(f float64) {
	*v.addr.(*float64) = f
}

// -- Float80

export type Float80Value interface {
	Kind()	int;
	Get()	float80;
	Set(float80);
	Type()	Type;
}

type Float80ValueStruct struct {
	Common
}

func Float80Creator(typ Type, addr Addr) Value {
	return &Float80ValueStruct{ Common{Float80Kind, typ, addr} }
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

type StringValueStruct struct {
	Common
}

func StringCreator(typ Type, addr Addr) Value {
	return &StringValueStruct{ Common{StringKind, typ, addr} }
}

func (v *StringValueStruct) Get() string {
	return *v.addr.(*string)
}

func (v *StringValueStruct) Set(s string) {
	*v.addr.(*string) = s
}

// -- Bool

export type BoolValue interface {
	Kind()	int;
	Get()	bool;
	Set(bool);
	Type()	Type;
}

type BoolValueStruct struct {
	Common
}

func BoolCreator(typ Type, addr Addr) Value {
	return &BoolValueStruct{ Common{BoolKind, typ, addr} }
}

func (v *BoolValueStruct) Get() bool {
	return *v.addr.(*bool)
}

func (v *BoolValueStruct) Set(b bool) {
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

type PtrValueStruct struct {
	Common
}

func (v *PtrValueStruct) Get() Addr {
	return *v.addr.(*Addr)
}

func (v *PtrValueStruct) Sub() Value {
	return NewValueAddr(v.typ.(PtrType).Sub(), v.Get());
}

func (v *PtrValueStruct) SetSub(subv Value) {
	a := v.typ.(PtrType).Sub();
	b := subv.Type();
	if !EqualType(a, b) {
		panicln("reflect: incompatible types in PtrValue.SetSub:",
			a.String(), b.String());
	}
	*v.addr.(*Addr) = subv.Addr();
}

func PtrCreator(typ Type, addr Addr) Value {
	return &PtrValueStruct{ Common{PtrKind, typ, addr} };
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
type RuntimeArray struct {
	data	Addr;
	len	uint32;
	cap	uint32;
}

type OpenArrayValueStruct struct {
	Common;
	elemtype	Type;
	elemsize	int;
	array *RuntimeArray;
}

func (v *OpenArrayValueStruct) Open() bool {
	return true
}

func (v *OpenArrayValueStruct) Len() int {
	return int(v.array.len);
}

func (v *OpenArrayValueStruct) Cap() int {
	return int(v.array.cap);
}

func (v *OpenArrayValueStruct) SetLen(len int) {
	if len > v.Cap() {
		panicln("reflect: OpenArrayValueStruct.SetLen", len, v.Cap());
	}
	v.array.len = uint32(len);
}

func (v *OpenArrayValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.array.data) + uintptr(i * v.elemsize);
	return NewValueAddr(v.elemtype, Addr(data_uint));
}

type FixedArrayValueStruct struct {
	Common;
	elemtype	Type;
	elemsize	int;
	len	int;
}

func (v *FixedArrayValueStruct) Open() bool {
	return false
}

func (v *FixedArrayValueStruct) Len() int {
	return v.len
}

func (v *FixedArrayValueStruct) Cap() int {
	return v.len
}

func (v *FixedArrayValueStruct) SetLen(len int) {
}

func (v *FixedArrayValueStruct) Elem(i int) Value {
	data_uint := uintptr(v.addr) + uintptr(i * v.elemsize);
	return NewValueAddr(v.elemtype, Addr(data_uint));
	return nil
}

func ArrayCreator(typ Type, addr Addr) Value {
	arraytype := typ.(ArrayType);
	if arraytype.Open() {
		v := new(OpenArrayValueStruct);
		v.kind = ArrayKind;
		v.addr = addr;
		v.typ = typ;
		v.elemtype = arraytype.Elem();
		v.elemsize = v.elemtype.Size();
		v.array = addr.(*RuntimeArray);
		return v;
	}
	v := new(FixedArrayValueStruct);
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

type MapValueStruct struct {
	Common
}

func MapCreator(typ Type, addr Addr) Value {
	return &MapValueStruct{ Common{MapKind, typ, addr} }
}

func (v *MapValueStruct) Len() int {
	return 0	// TODO: probably want this to be dynamic
}

func (v *MapValueStruct) Elem(key Value) Value {
	panic("map value element");
	return nil
}

// -- Chan

export type ChanValue interface {
	Kind()	int;
	Type()	Type;
}

type ChanValueStruct struct {
	Common
}

func ChanCreator(typ Type, addr Addr) Value {
	return &ChanValueStruct{ Common{ChanKind, typ, addr} }
}

// -- Struct

export type StructValue interface {
	Kind()	int;
	Type()	Type;
	Len()	int;
	Field(i int)	Value;
}

type StructValueStruct struct {
	Common;
	field	[]Value;
}

func (v *StructValueStruct) Len() int {
	return len(v.field)
}

func (v *StructValueStruct) Field(i int) Value {
	return v.field[i]
}

func StructCreator(typ Type, addr Addr) Value {
	t := typ.(StructType);
	nfield := t.Len();
	v := &StructValueStruct{ Common{StructKind, typ, addr}, make([]Value, nfield) };
	for i := 0; i < nfield; i++ {
		name, ftype, str, offset := t.Field(i);
		addr_uint := uintptr(addr) + uintptr(offset);
		v.field[i] = NewValueAddr(ftype, Addr(addr_uint));
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

type InterfaceValueStruct struct {
	Common
}

func (v *InterfaceValueStruct) Get() interface{} {
	return *v.addr.(*interface{})
}

func InterfaceCreator(typ Type, addr Addr) Value {
	return &InterfaceValueStruct{ Common{InterfaceKind, typ, addr} }
}

// -- Func

export type FuncValue interface {
	Kind()	int;
	Type()	Type;
}

type FuncValueStruct struct {
	Common
}

func FuncCreator(typ Type, addr Addr) Value {
	return &FuncValueStruct{ Common{FuncKind, typ, addr} }
}

var creator = map[int] Creator {
	MissingKind : &MissingCreator,
	IntKind : &IntCreator,
	Int8Kind : &Int8Creator,
	Int16Kind : &Int16Creator,
	Int32Kind : &Int32Creator,
	Int64Kind : &Int64Creator,
	UintKind : &UintCreator,
	Uint8Kind : &Uint8Creator,
	Uint16Kind : &Uint16Creator,
	Uint32Kind : &Uint32Creator,
	Uint64Kind : &Uint64Creator,
	UintptrKind : &UintptrCreator,
	FloatKind : &FloatCreator,
	Float32Kind : &Float32Creator,
	Float64Kind : &Float64Creator,
	Float80Kind : &Float80Creator,
	StringKind : &StringCreator,
	BoolKind : &BoolCreator,
	PtrKind : &PtrCreator,
	ArrayKind : &ArrayCreator,
	MapKind : &MapCreator,
	ChanKind : &ChanCreator,
	StructKind : &StructCreator,
	InterfaceKind : &InterfaceCreator,
	FuncKind : &FuncCreator,
}

var typecache = make(map[string] *Type);

func NewValueAddr(typ Type, addr Addr) Value {
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
	return NewValueAddr(typ, Addr(&data[0]));
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

	array := new(RuntimeArray);
	size := typ.Elem().Size() * cap;
	if size == 0 {
		size = 1;
	}
	data := make([]uint8, size);
	array.data = Addr(&data[0]);
	array.len = uint32(len);
	array.cap = uint32(cap);

	return NewValueAddr(typ, Addr(array));
}

export func CopyArray(dst ArrayValue, src ArrayValue, n int) {
	if n == 0 {
		return
	}
	dt := dst.Type().(ArrayType).Elem();
	st := src.Type().(ArrayType).Elem();
	if !EqualType(dt, st) {
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
	value, typestring  := sys.reflect(e);
	p, ok := typecache[typestring];
	if !ok {
		typ := ParseTypeString("", typestring);
		p = new(Type);
		*p = typ;
		typecache[typestring] = p;
	}
	// Content of interface is a value; need a permanent copy to take its address
	// so we can modify the contents. Values contain pointers to 'values'.
	ap := new(uint64);
	*ap = value;
	return NewValueAddr(*p, ap.(Addr));
}
