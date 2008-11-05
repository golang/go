// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Handling values.

package reflect

import (
	"reflect";
)

type Addr uint64	// TODO: where are ptrint/intptr etc?

// Conversion functions, implemented in assembler
func AddrToPtrAddr(Addr) *Addr
func AddrToPtrInt(Addr) *int
func AddrToPtrInt8(Addr) *int8
func AddrToPtrInt16(Addr) *int16
func AddrToPtrInt32(Addr) *int32
func AddrToPtrInt64(Addr) *int64
func AddrToPtrUint(Addr) *uint
func AddrToPtrUint8(Addr) *uint8
func PtrUint8ToAddr(*uint8) Addr
func AddrToPtrUint16(Addr) *uint16
func AddrToPtrUint32(Addr) *uint32
func AddrToPtrUint64(Addr) *uint64
func PtrUint64ToAddr(*uint64) Addr
func AddrToPtrFloat(Addr) *float
func AddrToPtrFloat32(Addr) *float32
func AddrToPtrFloat64(Addr) *float64
func AddrToPtrFloat80(Addr) *float80
func AddrToPtrString(Addr) *string
func AddrToPtrBool(Addr) *bool

export type Empty interface {}	// TODO(r): Delete when no longer needed?

export type Value interface {
	Kind()	int;
	Type()	Type;
	Unreflect()	Empty;
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

func (c *Common) Unreflect() Empty {
	return sys.unreflect(*AddrToPtrAddr(c.addr), c.typ.String());
}

func NewValueAddr(typ Type, addr Addr) Value

type Creator *(typ Type, addr Addr) Value


// -- Missing

export type MissingValue interface {
	Kind()	int;
	Type()	Type;
}

type MissingValueStruct struct {
	Common
}

func MissingCreator(typ Type, addr Addr) Value {
	return &MissingValueStruct{ Common{IntKind, typ, addr} }
}

// -- Int

export type IntValue interface {
	Kind()	int;
	Get()	int;
	Put(int);
	Type()	Type;
}

type IntValueStruct struct {
	Common
}

func IntCreator(typ Type, addr Addr) Value {
	return &IntValueStruct{ Common{IntKind, typ, addr} }
}

func (v *IntValueStruct) Get() int {
	return *AddrToPtrInt(v.addr)
}

func (v *IntValueStruct) Put(i int) {
	*AddrToPtrInt(v.addr) = i
}

// -- Int8

export type Int8Value interface {
	Kind()	int;
	Get()	int8;
	Put(int8);
	Type()	Type;
}

type Int8ValueStruct struct {
	Common
}

func Int8Creator(typ Type, addr Addr) Value {
	return &Int8ValueStruct{ Common{Int8Kind, typ, addr} }
}

func (v *Int8ValueStruct) Get() int8 {
	return *AddrToPtrInt8(v.addr)
}

func (v *Int8ValueStruct) Put(i int8) {
	*AddrToPtrInt8(v.addr) = i
}

// -- Int16

export type Int16Value interface {
	Kind()	int;
	Get()	int16;
	Put(int16);
	Type()	Type;
}

type Int16ValueStruct struct {
	Common
}

func Int16Creator(typ Type, addr Addr) Value {
	return &Int16ValueStruct{ Common{Int16Kind, typ, addr} }
}

func (v *Int16ValueStruct) Get() int16 {
	return *AddrToPtrInt16(v.addr)
}

func (v *Int16ValueStruct) Put(i int16) {
	*AddrToPtrInt16(v.addr) = i
}

// -- Int32

export type Int32Value interface {
	Kind()	int;
	Get()	int32;
	Put(int32);
	Type()	Type;
}

type Int32ValueStruct struct {
	Common
}

func Int32Creator(typ Type, addr Addr) Value {
	return &Int32ValueStruct{ Common{Int32Kind, typ, addr} }
}

func (v *Int32ValueStruct) Get() int32 {
	return *AddrToPtrInt32(v.addr)
}

func (v *Int32ValueStruct) Put(i int32) {
	*AddrToPtrInt32(v.addr) = i
}

// -- Int64

export type Int64Value interface {
	Kind()	int;
	Get()	int64;
	Put(int64);
	Type()	Type;
}

type Int64ValueStruct struct {
	Common
}

func Int64Creator(typ Type, addr Addr) Value {
	return &Int64ValueStruct{ Common{Int64Kind, typ, addr} }
}

func (v *Int64ValueStruct) Get() int64 {
	return *AddrToPtrInt64(v.addr)
}

func (v *Int64ValueStruct) Put(i int64) {
	*AddrToPtrInt64(v.addr) = i
}

// -- Uint

export type UintValue interface {
	Kind()	int;
	Get()	uint;
	Put(uint);
	Type()	Type;
}

type UintValueStruct struct {
	Common
}

func UintCreator(typ Type, addr Addr) Value {
	return &UintValueStruct{ Common{UintKind, typ, addr} }
}

func (v *UintValueStruct) Get() uint {
	return *AddrToPtrUint(v.addr)
}

func (v *UintValueStruct) Put(i uint) {
	*AddrToPtrUint(v.addr) = i
}

// -- Uint8

export type Uint8Value interface {
	Kind()	int;
	Get()	uint8;
	Put(uint8);
	Type()	Type;
}

type Uint8ValueStruct struct {
	Common
}

func Uint8Creator(typ Type, addr Addr) Value {
	return &Uint8ValueStruct{ Common{Uint8Kind, typ, addr} }
}

func (v *Uint8ValueStruct) Get() uint8 {
	return *AddrToPtrUint8(v.addr)
}

func (v *Uint8ValueStruct) Put(i uint8) {
	*AddrToPtrUint8(v.addr) = i
}

// -- Uint16

export type Uint16Value interface {
	Kind()	int;
	Get()	uint16;
	Put(uint16);
	Type()	Type;
}

type Uint16ValueStruct struct {
	Common
}

func Uint16Creator(typ Type, addr Addr) Value {
	return &Uint16ValueStruct{ Common{Uint16Kind, typ, addr} }
}

func (v *Uint16ValueStruct) Get() uint16 {
	return *AddrToPtrUint16(v.addr)
}

func (v *Uint16ValueStruct) Put(i uint16) {
	*AddrToPtrUint16(v.addr) = i
}

// -- Uint32

export type Uint32Value interface {
	Kind()	int;
	Get()	uint32;
	Put(uint32);
	Type()	Type;
}

type Uint32ValueStruct struct {
	Common
}

func Uint32Creator(typ Type, addr Addr) Value {
	return &Uint32ValueStruct{ Common{Uint32Kind, typ, addr} }
}

func (v *Uint32ValueStruct) Get() uint32 {
	return *AddrToPtrUint32(v.addr)
}

func (v *Uint32ValueStruct) Put(i uint32) {
	*AddrToPtrUint32(v.addr) = i
}

// -- Uint64

export type Uint64Value interface {
	Kind()	int;
	Get()	uint64;
	Put(uint64);
	Type()	Type;
}

type Uint64ValueStruct struct {
	Common
}

func Uint64Creator(typ Type, addr Addr) Value {
	return &Uint64ValueStruct{ Common{Uint64Kind, typ, addr} }
}

func (v *Uint64ValueStruct) Get() uint64 {
	return *AddrToPtrUint64(v.addr)
}

func (v *Uint64ValueStruct) Put(i uint64) {
	*AddrToPtrUint64(v.addr) = i
}

// -- Float

export type FloatValue interface {
	Kind()	int;
	Get()	float;
	Put(float);
	Type()	Type;
}

type FloatValueStruct struct {
	Common
}

func FloatCreator(typ Type, addr Addr) Value {
	return &FloatValueStruct{ Common{FloatKind, typ, addr} }
}

func (v *FloatValueStruct) Get() float {
	return *AddrToPtrFloat(v.addr)
}

func (v *FloatValueStruct) Put(f float) {
	*AddrToPtrFloat(v.addr) = f
}

// -- Float32

export type Float32Value interface {
	Kind()	int;
	Get()	float32;
	Put(float32);
	Type()	Type;
}

type Float32ValueStruct struct {
	Common
}

func Float32Creator(typ Type, addr Addr) Value {
	return &Float32ValueStruct{ Common{Float32Kind, typ, addr} }
}

func (v *Float32ValueStruct) Get() float32 {
	return *AddrToPtrFloat32(v.addr)
}

func (v *Float32ValueStruct) Put(f float32) {
	*AddrToPtrFloat32(v.addr) = f
}

// -- Float64

export type Float64Value interface {
	Kind()	int;
	Get()	float64;
	Put(float64);
	Type()	Type;
}

type Float64ValueStruct struct {
	Common
}

func Float64Creator(typ Type, addr Addr) Value {
	return &Float64ValueStruct{ Common{Float64Kind, typ, addr} }
}

func (v *Float64ValueStruct) Get() float64 {
	return *AddrToPtrFloat64(v.addr)
}

func (v *Float64ValueStruct) Put(f float64) {
	*AddrToPtrFloat64(v.addr) = f
}

// -- Float80

export type Float80Value interface {
	Kind()	int;
	Get()	float80;
	Put(float80);
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
	return *AddrToPtrFloat80(v.addr)
	return 0;
}

func (v *Float80ValueStruct) Put(f float80) {
	*AddrToPtrFloat80(v.addr) = f
}
*/

// -- String

export type StringValue interface {
	Kind()	int;
	Get()	string;
	Put(string);
	Type()	Type;
}

type StringValueStruct struct {
	Common
}

func StringCreator(typ Type, addr Addr) Value {
	return &StringValueStruct{ Common{StringKind, typ, addr} }
}

func (v *StringValueStruct) Get() string {
	return *AddrToPtrString(v.addr)
}

func (v *StringValueStruct) Put(s string) {
	*AddrToPtrString(v.addr) = s
}

// -- Bool

export type BoolValue interface {
	Kind()	int;
	Get()	bool;
	Put(bool);
	Type()	Type;
}

type BoolValueStruct struct {
	Common
}

func BoolCreator(typ Type, addr Addr) Value {
	return &BoolValueStruct{ Common{BoolKind, typ, addr} }
}

func (v *BoolValueStruct) Get() bool {
	return *AddrToPtrBool(v.addr)
}

func (v *BoolValueStruct) Put(b bool) {
	*AddrToPtrBool(v.addr) = b
}

// -- Pointer

export type PtrValue interface {
	Kind()	int;
	Type()	Type;
	Sub()	Value;
	Get()	Addr;
}

type PtrValueStruct struct {
	Common
}

func (v *PtrValueStruct) Get() Addr {
	return *AddrToPtrAddr(v.addr)
}

func (v *PtrValueStruct) Sub() Value {
	return NewValueAddr(v.typ.(PtrType).Sub(), v.Get());
}

func PtrCreator(typ Type, addr Addr) Value {
	return &PtrValueStruct{ Common{PtrKind, typ, addr} };
}

// -- Array

export type ArrayValue interface {
	Kind()	int;
	Type()	Type;
	Open()	bool;
	Len()	uint64;
	Elem(i uint64)	Value;
}

type OpenArrayValueStruct struct {
	Common;
	elemtype	Type;
	elemsize	uint64;
}

/*
	Run-time representation of open arrays looks like this:
		struct	Array {
			byte*	array;		// actual data
			uint32	nel;		// number of elements
		};
*/

func (v *OpenArrayValueStruct) Open() bool {
	return true
}

func (v *OpenArrayValueStruct) Len() uint64 {
	return uint64(*AddrToPtrInt32(v.addr+8));
}

func (v *OpenArrayValueStruct) Elem(i uint64) Value {
	base := *AddrToPtrAddr(v.addr);
	return NewValueAddr(v.elemtype, base + i * v.elemsize);
}

type FixedArrayValueStruct struct {
	Common;
	elemtype	Type;
	elemsize	uint64;
	len	uint64;
}

func (v *FixedArrayValueStruct) Open() bool {
	return false
}

func (v *FixedArrayValueStruct) Len() uint64 {
	return v.len
}

func (v *FixedArrayValueStruct) Elem(i uint64) Value {
	return NewValueAddr(v.elemtype, v.addr + i * v.elemsize);
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
	field	*[]Value;
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
	v := &StructValueStruct{ Common{StructKind, typ, addr}, new([]Value, nfield) };
	for i := 0; i < nfield; i++ {
		name, ftype, str, offset := t.Field(i);
		v.field[i] = NewValueAddr(ftype, addr + offset);
	}
	v.typ = typ;
	return v;
}

// -- Interface

export type InterfaceValue interface {
	Kind()	int;
	Type()	Type;
}

type InterfaceValueStruct struct {
	Common
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

var creator *map[int] Creator

func init() {
	creator = new(map[int] Creator);
	creator[MissingKind] = &MissingCreator;
	creator[IntKind] = &IntCreator;
	creator[Int8Kind] = &Int8Creator;
	creator[Int16Kind] = &Int16Creator;
	creator[Int32Kind] = &Int32Creator;
	creator[Int64Kind] = &Int64Creator;
	creator[UintKind] = &UintCreator;
	creator[Uint8Kind] = &Uint8Creator;
	creator[Uint16Kind] = &Uint16Creator;
	creator[Uint32Kind] = &Uint32Creator;
	creator[Uint64Kind] = &Uint64Creator;
	creator[FloatKind] = &FloatCreator;
	creator[Float32Kind] = &Float32Creator;
	creator[Float64Kind] = &Float64Creator;
	creator[Float80Kind] = &Float80Creator;
	creator[StringKind] = &StringCreator;
	creator[BoolKind] = &BoolCreator;
	creator[PtrKind] = &PtrCreator;
	creator[ArrayKind] = &ArrayCreator;
	creator[MapKind] = &MapCreator;
	creator[ChanKind] = &ChanCreator;
	creator[StructKind] = &StructCreator;
	creator[InterfaceKind] = &InterfaceCreator;
	creator[FuncKind] = &FuncCreator;
}

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
	case FuncKind, ChanKind, MapKind:	// must be pointers, at least for now (TODO?)
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
	data := new([]uint8, size);
	return NewValueAddr(typ, PtrUint8ToAddr(&data[0]));
}

export func NewValue(e Empty) Value {
	value, typestring  := sys.reflect(e);
	typ := ParseTypeString("", typestring);
	// Content of interface is a value; need a permanent copy to take its address
	// so we can modify the contents. Values contain pointers to 'values'.
	ap := new(uint64);
	*ap = value;
	return NewValueAddr(typ, PtrUint64ToAddr(ap));
}
