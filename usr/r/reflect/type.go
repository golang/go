// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Types and parsing of type strings.

package reflect

export type Type interface

export func ExpandType(name string) Type

//export var GlobalTypeStrings = sys.typestrings;

export const (
	MissingKind = iota;
	ArrayKind;
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

var MissingString = "missing"	// syntactic name for undefined type names

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
	Missing = NewBasicType(MissingKind);
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

// Stub types allow us to defer evaluating type names until needed.
// If the name is empty, the type must be non-nil.
type StubType struct {
	name	string;
	typ		Type;
}

func (t *StubType) Get() Type {
	if t.typ == nil {
		t.typ = ExpandType(t.name)
	}
	return t.typ
}

func NewStubType(t Type) *StubType {
	s := new(StubType);
	s.typ = t;
	return s;
}

func NewNamedStubType(n string) *StubType {
	s := new(StubType);
	s.name = n;
	return s;
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

////////////////////////
//helpers for early bootstrap and debugging
func Stub(n string, t Type) *StubType {
	s := new(StubType);
	s.name = n;
	s.typ = t;
	return s;
}
export var PtrInt8 Type = NewPtrTypeStruct(Stub("i", Int8));
export var PtrPtrInt8 Type = NewPtrTypeStruct(Stub("i", PtrInt8));
export var ArrayFloat32 Type = NewArrayTypeStruct(100, Stub("f", Float32));
export var MapStringInt16 Type = NewMapTypeStruct(Stub("s", String), Stub("i", Int16));
export var ChanArray Type = NewChanTypeStruct(RecvDir, Stub("a", ArrayFloat32));
var F1 = Field{"i", Stub("i", Int64)};
var Fields = []Field{F1};
export var Structure = NewStructTypeStruct(&Fields);
export var Function Type = NewFuncTypeStruct(Structure, Structure, Structure);
////////////////////////

// Cache of expanded types keyed by type name.
var types *map[string] *Type	// BUG TODO: should be Type not *Type
// List of typename, typestring pairs
var typestrings *map[string] string
// Map of basic types to prebuilt StubTypes
var basicstubs *map[string] *StubType

var MissingStub *StubType;

func init() {
	types = new(map[string] *Type);
	typestrings = new(map[string] string);
	basicstubs = new(map[string] *StubType);

	// Basics go into types table
	types["missing"] = &Missing;
	types["int8"] = &Int8;
	types["int16"] = &Int16;
	types["int32"] = &Int32;
	types["int64"] = &Int64;
	types["uint8"] = &Uint8;
	types["uint16"] = &Uint16;
	types["uint32"] = &Uint32;
	types["uint64"] = &Uint64;
	types["float32"] = &Float32;
	types["float64"] = &Float64;
	types["float80"] = &Float80;
	types["string"] = &String;

	// Basics get prebuilt stubs
	MissingStub = NewStubType(Missing);
	basicstubs["missing"] = MissingStub;
	basicstubs["int8"] = NewStubType(Int8);
	basicstubs["int16"] = NewStubType(Int16);
	basicstubs["int32"] = NewStubType(Int32);
	basicstubs["int64"] = NewStubType(Int64);
	basicstubs["uint8"] = NewStubType(Uint8);
	basicstubs["uint16"] = NewStubType(Uint16);
	basicstubs["uint32"] = NewStubType(Uint32);
	basicstubs["uint64"] = NewStubType(Uint64);
	basicstubs["float32"] = NewStubType(Float32);
	basicstubs["float64"] = NewStubType(Float64);
	basicstubs["float80"] = NewStubType(Float80);
	basicstubs["string"] = NewStubType(String);

	typestrings["P.integer"] = "int32";
	return;
	typestrings["P.S"] =  "struct {t *P.T}";
	typestrings["P.T"] = "struct {c *(? *chan P.S, *int)}";
}

/*
	Grammar

	stubtype =	- represent as StubType when possible
		type
	identifier =
		name
		'?'
	type =
		basictypename	- int8, string, etc.
		typename
		arraytype
		structtype
		interfacetype
		chantype
		maptype
		pointertype
		functiontype
	typename =
		name '.' name
	fieldlist =
		[ field { ',' field } ]
	field =
		identifier stubtype
	arraytype =
		'[' [ number ] ']' stubtype
	structtype =
		'struct' '{' fieldlist '}'
	interfacetype =
		'interface' '{' fieldlist '}'
	chantype =
		'<-' chan stubtype
		chan '<-' stubtype
		chan stubtype
	maptype =
		'map' '[' stubtype ']' stubtype
	pointertype =
		'*' stubtype
	functiontype =
		'(' fieldlist ')'

*/

func isdigit(c uint8) bool {
	return '0' <= c && c <= '9'
}

func special(c uint8) bool {
	s := "*[](){}<";	// Note: '.' is not in this list.  "P.T" is an identifer, as is "?".
	for i := 0; i < len(s); i++ {
		if c == s[i] {
			return true
		}
	}
	return false;
}

type Parser struct {
	str	string;
	index	int;
	token	string;
}

func (p *Parser) Next() {
	token := "";
	for ; p.index < len(p.str) && p.str[p.index] == ' '; p.index++ {
	}
	if p.index >= len(p.str) {
		p.token = "";
		return;
	}
	start := p.index;
	c, w := sys.stringtorune(p.str, p.index);
	p.index += w;
	switch {
	case c == '*':
		p.token = "*";
		return;
	case c == '[':
		p.token = "[";
		return;
	case c == ']':
		p.token = "]";
		return;
	case c == '(':
		p.token = "(";
		return;
	case c == ')':
		p.token = ")";
		return;
	case c == '<':
		if p.index < len(p.str) && p.str[p.index+1] == '-' {
			p.index++;
			p.token = "<-";
			return;
		}
		p.token = "<";	// shouldn't happen but let the parser figure it out
		return;
	case isdigit(uint8(c)):
		for p.index < len(p.str) && isdigit(p.str[p.index]) {
			p.index++
		}
		p.token = p.str[start : p.index];
		return;
	}
	for p.index < len(p.str) && !special(p.str[p.index]) {
		p.index++
	}
	p.token = p.str[start : p.index];
}

func (p *Parser) Type() *StubType

func (p *Parser) Array() *StubType {
	size := -1;
	if p.token != "]" {
		if len(p.token) == 0 || !isdigit(p.token[0]) {
			return MissingStub
		}
		// write our own (trivial and simpleminded) atoi to avoid dependency
		size = 0;
		for i := 0; i < len(p.token); i++ {
			size = size * 10 + int(p.token[i]) - '0'
		}
		p.Next();
	}
	if p.token != "]" {
		return MissingStub
	}
	p.Next();
	elemtype := p.Type();
	return NewStubType(NewArrayTypeStruct(size, elemtype));
}

func (p *Parser) Map() *StubType {
	if p.token != "[" {
		return MissingStub
	}
	p.Next();
	keytype := p.Type();
	if p.token != "]" {
		return MissingStub
	}
	p.Next();
	elemtype := p.Type();
	return NewStubType(NewMapTypeStruct(keytype, elemtype));
}

func (p *Parser) Simple() *StubType {
	switch {
	case p.token == "":
		return nil;
	case p.token == "*":
		p.Next();
		return NewStubType(NewPtrTypeStruct(p.Simple()));
	case p.token == "[":
		p.Next();
		return p.Array();
	case p.token == "map":
		p.Next();
		return p.Map();
	case isdigit(p.token[0]):
		p.Next();
		print("reflect.Simple: number encountered\n");	// TODO: remove
		return MissingStub;
	case special(p.token[0]):
		// TODO: get chans right
		p.Next();
		print("reflect.Simple: special character encountered\n");	// TODO: remove
		return MissingStub;
	}
	// must be an identifier. is it basic? if so, we have a stub
	if s, ok := basicstubs[p.token]; ok {
		p.Next();
		return s
	}
	// not a basic - must be of the form "P.T"
	ndot := 0;
	for i := 0; i < len(p.token); i++ {
		if p.token[i] == '.' {
			ndot++
		}
	}
	if ndot != 1 {
		print("reflect.Simple: illegal identifier ", p.token, "\n");	// TODO: remove
		p.Next();
		return MissingStub;
	}
	s := new(StubType);
	s.name = p.token;
	p.Next();
	return s;
}

func (p *Parser) Type() *StubType {
	return p.Simple();
}

export func ParseTypeString(str string) Type {
	p := new(Parser);
	p.str = str;
	p.Next();
	return p.Type().Get();
}

// Look up type string associated with name.
func TypeNameToTypeString(name string) string {
	s, ok := typestrings[name];
	if !ok {
		s = MissingString;
		typestrings[name] = s;
	}
	return s
}

// Type is known by name.  Find (and create if necessary) its real type.
func ExpandType(name string) Type {
	t, ok := types[name];
	if ok {
		return *t
	}
	types[name] = &Missing;	// prevent recursion; will overwrite
	t1 := ParseTypeString(TypeNameToTypeString(name));
	p := new(Type);
	*p = t1;
	types[name] = p;
	return t1;
}
