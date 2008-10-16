// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

export type Type interface
export type Value interface{}	// TODO: define this

export func LookupTypeName(name string) Type

//export var GlobalTypeStrings = sys.typestrings;

// Cache of types keyed by type name
var types = new(map[string] *Type)	// BUG TODO: should be Type not *Type
// Cache of type strings keyed by type name
var strings = new(map[string] string)

export const (
	ArrayKind = iota;
	ChanKind;
	Float32Kind;
	Float64Kind;
	Float80Kind;
	FuncKind;
	Int16Kind;
	Int32Kind;
	Int64Kind;
	Int8Kind;
	MapKind;
	PtrKind;
	StringKind;
	StructKind;
	Uint16Kind;
	Uint32Kind;
	Uint64Kind;
	Uint8Kind;
)

type Type interface {
	Kind()	int;
}

type BasicType struct{
	kind	int
}

func (t *BasicType) Kind() int {
	return t.kind
}

func NewBasicType(k int) Type {
	t := new(BasicType);
	t.kind = k;
	return t;
}

// Basic types
export var (
	Int8 = NewBasicType(Int8Kind);
	Int16 = NewBasicType(Int16Kind);
	Int32 = NewBasicType(Int32Kind);
	Int64 = NewBasicType(Int64Kind);
	Uint8 = NewBasicType(Uint8Kind);
	Uint16 = NewBasicType(Uint16Kind);
	Uint32 = NewBasicType(Uint32Kind);
	Uint64 = NewBasicType(Uint64Kind);
	Float32 = NewBasicType(Float32Kind);
	Float64 = NewBasicType(Float64Kind);
	Float80 = NewBasicType(Float80Kind);
	String = NewBasicType(StringKind);
)

type StubType struct {
	name	string;
	typ		Type;
}

func (t *StubType) Get() Type {
	if t.typ == nil {
		t.typ = LookupTypeName(t.name)
	}
	return t.typ
}

export type PtrType interface {
	Sub()	Type
}

type PtrTypeStruct struct {
	sub	*StubType
}

func (t *PtrTypeStruct) Kind() int {
	return PtrKind
}

func (t *PtrTypeStruct) Sub() Type {
	return t.sub.Get()
}

func NewPtrTypeStruct(sub *StubType) *PtrTypeStruct {
	t := new(PtrTypeStruct);
	t.sub = sub;
	return t;
}

export type ArrayType interface {
	Len()	int;
	Elem()	Type;
}

type ArrayTypeStruct struct {
	elem	*StubType;
	len	int;
}

func (t *ArrayTypeStruct) Kind() int {
	return ArrayKind
}

func (t *ArrayTypeStruct) Len() int {
	// -1 is open array?  TODO
	return t.len
}

func (t *ArrayTypeStruct) Elem() Type {
	return t.elem.Get()
}

func NewArrayTypeStruct(len int, elem *StubType) *ArrayTypeStruct {
	t := new(ArrayTypeStruct);
	t.len = len;
	t.elem = elem;
	return t;
}

export type MapType interface {
	Key()	Type;
	Elem()	Type;
}

type MapTypeStruct struct {
	key	*StubType;
	elem	*StubType;
}

func (t *MapTypeStruct) Kind() int {
	return MapKind
}

func (t *MapTypeStruct) Key() Type {
	return t.key.Get()
}

func (t *MapTypeStruct) Elem() Type {
	return t.elem.Get()
}

func NewMapTypeStruct(key, elem *StubType) *MapTypeStruct {
	t := new(MapTypeStruct);
	t.key = key;
	t.elem = elem;
	return t;
}

export type ChanType interface {
	Dir()	int;
	Elem()	Type;
}

export const (	// channel direction
	SendDir = 1 << iota;
	RecvDir;
	BothDir = SendDir | RecvDir;
)

type ChanTypeStruct struct {
	elem	*StubType;
	dir	int;
}

func (t *ChanTypeStruct) Kind() int {
	return ChanKind
}

func (t *ChanTypeStruct) Dir() int {
	// -1 is open array?  TODO
	return t.dir
}

func (t *ChanTypeStruct) Elem() Type {
	return t.elem.Get()
}

func NewChanTypeStruct(dir int, elem *StubType) *ChanTypeStruct {
	t := new(ChanTypeStruct);
	t.dir = dir;
	t.elem = elem;
	return t;
}

export type StructType interface {
	Field(int)	(name string, typ Type);
	Len()	int;
}

type Field struct {
	name	string;
	typ	*StubType;
}

type StructTypeStruct struct {
	field	*[]Field;
}

func (t *StructTypeStruct) Kind() int {
	return StructKind
}

func (t *StructTypeStruct) Field(i int) (name string, typ Type) {
	return t.field[i].name, t.field[i].typ.Get()
}

func (t *StructTypeStruct) Len() int {
	return len(t.field)
}

func Struct(field *[]Field) *StructTypeStruct {
	t := new(StructTypeStruct);
	t.field = field;
	return t;
}

func NewStructTypeStruct(field *[]Field) *StructTypeStruct {
	t := new(StructTypeStruct);
	t.field = field;
	return t;
}

export type FuncType interface {
	Receiver()	StructType;
	In()	StructType;
	Out()	StructType;
}

type FuncTypeStruct struct {
	receiver	*StructTypeStruct;
	in	*StructTypeStruct;
	out	*StructTypeStruct;
}

func (t *FuncTypeStruct) Kind() int {
	return FuncKind
}

func (t *FuncTypeStruct) Receiver() StructType {
	return t.receiver
}

func (t *FuncTypeStruct) In() StructType {
	return t.in
}

func (t *FuncTypeStruct) Out() StructType {
	return t.out
}

func NewFuncTypeStruct(receiver, in, out *StructTypeStruct) *FuncTypeStruct {
	t := new(FuncTypeStruct);
	t.receiver = receiver;
	t.in = in;
	t.out = out;
	return t;
}

//helpers for early bootstrap and debugging
export func LookupTypeName(name string) Type { return Int8 }
func Stub(n string, t Type) *StubType {
	s := new(StubType);
	s.name = n;
	s.typ = t;
	return s;
}
export var PtrInt8 Type = NewPtrTypeStruct(Stub("i", Int8));
export var ArrayFloat32 Type = NewArrayTypeStruct(100, Stub("f", Float32));
export var MapStringInt16 Type = NewMapTypeStruct(Stub("s", String), Stub("i", Int16));
export var ChanArray Type = NewChanTypeStruct(RecvDir, Stub("a", ArrayFloat32));
var F1 = Field{"i", Stub("i", Int64)};
var Fields = []Field{F1};
export var Structure = NewStructTypeStruct(&Fields);
export var Function Type = NewFuncTypeStruct(Structure, Structure, Structure);
