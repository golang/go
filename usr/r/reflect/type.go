// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Types and parsing of type strings.

package reflect

export type Type interface

export func ExpandType(name string) Type

export func typestrings() string	// implemented in C; declared here

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
	InterfaceKind;
	MapKind;
	PtrKind;
	StringKind;
	StructKind;
	Uint16Kind;
	Uint32Kind;
	Uint64Kind;
	Uint8Kind;
)

var ptrsize uint64
var interfacesize uint64

var MissingString = "$missing$"	// syntactic name for undefined type names

export type Type interface {
	Kind()	int;
	Size()	uint64;
}

// -- Basic

type BasicType struct{
	name	string;
	kind	int;
	size	uint64;
}

func (t *BasicType) Name() string {
	return t.name
}

func (t *BasicType) Kind() int {
	return t.kind
}

func (t *BasicType) Size() uint64 {
	return t.size
}

func NewBasicType(n string, k int, size uint64) Type {
	t := new(BasicType);
	t.name = n;
	t.kind = k;
	t.size = size;
	return t;
}

// Prebuilt basic types
export var (
	Missing = NewBasicType(MissingString, MissingKind, 1);
	Int8 = NewBasicType("int8", Int8Kind, 1);
	Int16 = NewBasicType("int16", Int16Kind, 2);
	Int32 = NewBasicType("int32", Int32Kind, 4);
	Int64 = NewBasicType("int64", Int64Kind, 8);
	Uint8 = NewBasicType("uint8", Uint8Kind, 1);
	Uint16 = NewBasicType("uint16", Uint16Kind, 2);
	Uint32 = NewBasicType("uint32", Uint32Kind, 4);
	Uint64 = NewBasicType("uint64", Uint64Kind, 8);
	Float32 = NewBasicType("float32", Float32Kind, 4);
	Float64 = NewBasicType("float64", Float64Kind, 8);
	Float80 = NewBasicType("float80", Float80Kind, 10);	// TODO: strange size?
	String = NewBasicType("string", StringKind, 8);	// implemented as a pointer
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

// -- Pointer

export type PtrType interface {
	Sub()	Type
}

type PtrTypeStruct struct {
	sub	*StubType
}

func (t *PtrTypeStruct) Kind() int {
	return PtrKind
}

func (t *PtrTypeStruct) Size() uint64 {
	return ptrsize
}

func (t *PtrTypeStruct) Sub() Type {
	return t.sub.Get()
}

func NewPtrTypeStruct(sub *StubType) *PtrTypeStruct {
	t := new(PtrTypeStruct);
	t.sub = sub;
	return t;
}

// -- Array

export type ArrayType interface {
	Open()	bool;
	Len()	uint64;
	Elem()	Type;
}

type ArrayTypeStruct struct {
	elem	*StubType;
	open	bool;	// otherwise fixed size
	len	uint64;
}

func (t *ArrayTypeStruct) Kind() int {
	return ArrayKind
}

func (t *ArrayTypeStruct) Size() uint64 {
	if t.open {
		return ptrsize	// open arrays are pointers to structures
	}
	return t.len * t.elem.Get().Size();
}

func (t *ArrayTypeStruct) Open() bool {
	return t.open
}

func (t *ArrayTypeStruct) Len() uint64 {
	// what about open array?  TODO
	return t.len
}

func (t *ArrayTypeStruct) Elem() Type {
	return t.elem.Get()
}

func NewArrayTypeStruct(open bool, len uint64, elem *StubType) *ArrayTypeStruct {
	t := new(ArrayTypeStruct);
	t.open = open;
	t.len = len;
	t.elem = elem;
	return t;
}

// -- Map

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

func (t *MapTypeStruct) Size() uint64 {
	panic("reflect.type: map.Size(): cannot happen");
	return 0
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

// -- Chan

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

func (t *ChanTypeStruct) Size() uint64 {
	panic("reflect.type: chan.Size(): cannot happen");
	return 0
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

// -- Struct

export type StructType interface {
	Field(int)	(name string, typ Type, offset uint64);
	Len()	int;
}

type Field struct {
	name	string;
	typ	*StubType;
	size	uint64;
	offset	uint64;
}

type StructTypeStruct struct {
	field	*[]Field;
}

func (t *StructTypeStruct) Kind() int {
	return StructKind
}

// TODO: not portable; depends on 6g
func (t *StructTypeStruct) Size() uint64 {
	size := uint64(0);
	for i := 0; i < len(t.field); i++ {
		elemsize := t.field[i].typ.Get().Size();
		// pad until at (elemsize mod 8) boundary
		align := elemsize - 1;
		if align > 7 {	// BUG: we know structs are 8-aligned
			align = 7
		}
		if align > 0 {
			size = (size + align) & ^align;
		}
		t.field[i].offset = size;
		size += elemsize;
	}
	size = (size + 7) & ((1<<64 - 1) & ^7);
	return size;
}

func (t *StructTypeStruct) Field(i int) (name string, typ Type, offset uint64) {
	if t.field[i].offset == 0 {
		t.Size();	// will compute offsets
	}
	return t.field[i].name, t.field[i].typ.Get(), t.field[i].offset
}

func (t *StructTypeStruct) Len() int {
	return len(t.field)
}

func NewStructTypeStruct(field *[]Field) *StructTypeStruct {
	t := new(StructTypeStruct);
	t.field = field;
	return t;
}

// -- Interface

export type InterfaceType interface {
	Field(int)	(name string, typ Type, offset uint64);
	Len()	int;
}

type InterfaceTypeStruct struct {
	field	*[]Field;
}

func (t *InterfaceTypeStruct) Field(i int) (name string, typ Type, offset uint64) {
	return t.field[i].name, t.field[i].typ.Get(), 0
}

func (t *InterfaceTypeStruct) Len() int {
	return len(t.field)
}

func NewInterfaceTypeStruct(field *[]Field) *InterfaceTypeStruct {
	t := new(InterfaceTypeStruct);
	t.field = field;
	return t;
}

func (t *InterfaceTypeStruct) Kind() int {
	return InterfaceKind
}

func (t *InterfaceTypeStruct) Size() uint64 {
	return interfacesize
}

// -- Func

export type FuncType interface {
	In()	StructType;
	Out()	StructType;
}

type FuncTypeStruct struct {
	in	*StructTypeStruct;
	out	*StructTypeStruct;
}

func (t *FuncTypeStruct) Kind() int {
	return FuncKind
}

func (t *FuncTypeStruct) Size() uint64 {
	panic("reflect.type: func.Size(): cannot happen");
	return 0
}

func (t *FuncTypeStruct) In() StructType {
	return t.in
}

func (t *FuncTypeStruct) Out() StructType {
	if t.out == nil {	// nil.(StructType) != nil so make sure caller sees real nil
		return nil
	}
	return t.out
}

func NewFuncTypeStruct(in, out *StructTypeStruct) *FuncTypeStruct {
	t := new(FuncTypeStruct);
	t.in = in;
	t.out = out;
	return t;
}

// Cache of expanded types keyed by type name.
var types *map[string] *Type	// BUG TODO: should be Type not *Type

// List of typename, typestring pairs
var typestring *map[string] string
var initialized bool = false

// Map of basic types to prebuilt StubTypes
var basicstub *map[string] *StubType

var MissingStub *StubType;

// The database stored in the maps is global; use locking to guarantee safety.
var lockchan *chan bool  // Channel with buffer of 1, used as a mutex

func Lock() {
	lockchan <- true	// block if buffer is full
}

func Unlock() {
	<-lockchan	// release waiters
}

func init() {
	ptrsize = 8;	// TODO: compute this
	interfacesize = 2*ptrsize;	// TODO: compute this

	lockchan = new(chan bool, 1);	// unlocked at creation - buffer is empty
	Lock();	// not necessary because of init ordering but be safe.

	types = new(map[string] *Type);
	typestring = new(map[string] string);
	basicstub = new(map[string] *StubType);

	// Basics go into types table
	types[MissingString] = &Missing;
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
	basicstub[MissingString] = MissingStub;
	basicstub["int8"] = NewStubType(Int8);
	basicstub["int16"] = NewStubType(Int16);
	basicstub["int32"] = NewStubType(Int32);
	basicstub["int64"] = NewStubType(Int64);
	basicstub["uint8"] = NewStubType(Uint8);
	basicstub["uint16"] = NewStubType(Uint16);
	basicstub["uint32"] = NewStubType(Uint32);
	basicstub["uint64"] = NewStubType(Uint64);
	basicstub["float32"] = NewStubType(Float32);
	basicstub["float64"] = NewStubType(Float64);
	basicstub["float80"] = NewStubType(Float80);
	basicstub["string"] = NewStubType(String);

	Unlock();
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
		[ field { [ ',' | ';' ] field } ]
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

// Helper functions for token scanning
func isdigit(c uint8) bool {
	return '0' <= c && c <= '9'
}

func special(c uint8) bool {
	s := "*[](){}<;,";	// Note: '.' is not in this list.  "P.T" is an identifer, as is "?".
	for i := 0; i < len(s); i++ {
		if c == s[i] {
			return true
		}
	}
	return false;
}

// Simple parser for type strings
type Parser struct {
	str	string;	// string being parsed
	token	string;	// the token being parsed now
	index	int;	// next character position in str
}

// Load next token into p.token
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
	case c == '<':
		if p.index < len(p.str) && p.str[p.index] == '-' {
			p.index++;
			p.token = "<-";
			return;
		}
		fallthrough;	// shouldn't happen but let the parser figure it out
	case special(uint8(c)):
		p.token = string(c);
		return;
	case isdigit(uint8(c)):
		for p.index < len(p.str) && isdigit(p.str[p.index]) {
			p.index++
		}
		p.token = p.str[start : p.index];
		return;
	}
	for p.index < len(p.str) && p.str[p.index] != ' ' && !special(p.str[p.index]) {
		p.index++
	}
	p.token = p.str[start : p.index];
}

func (p *Parser) Type() *StubType

func (p *Parser) Array() *StubType {
	size := uint64(0);
	open := true;
	if p.token != "]" {
		if len(p.token) == 0 || !isdigit(p.token[0]) {
			return MissingStub
		}
		// write our own (trivial and simpleminded) atoi to avoid dependency
		size = 0;
		for i := 0; i < len(p.token); i++ {
			size = size * 10 + uint64(p.token[i]) - '0'
		}
		p.Next();
		open = false;
	}
	if p.token != "]" {
		return MissingStub
	}
	p.Next();
	elemtype := p.Type();
	return NewStubType(NewArrayTypeStruct(open, size, elemtype));
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

func (p *Parser) Chan(dir int) *StubType {
	if p.token == "<-" {
		if dir != BothDir {
			return MissingStub
		}
		p.Next();
		dir = SendDir;
	}
	elemtype := p.Type();
	return NewStubType(NewChanTypeStruct(dir, elemtype));
}

// Parse array of fields for struct, interface, and func arguments
func (p *Parser) Fields(sep string) *[]Field {
	a := new([]Field, 10);
	nf := 0;
	for p.token != "" && !special(p.token[0]) {
		if nf == len(a) {
			a1 := new([]Field, 2*nf);
			for i := 0; i < nf; i++ {
				a1[i] = a[i];
			}
			a = a1;
		}
		a[nf].name = p.token;
		p.Next();
		a[nf].typ = p.Type();
		nf++;
		if p.token != sep {
			break;
		}
		p.Next();	// skip separator
	}
	return a[0:nf];
}

func (p *Parser) Struct() *StubType {
	f := p.Fields(";");
	if p.token != "}" {
		return MissingStub;
	}
	p.Next();
	return NewStubType(NewStructTypeStruct(f));
}

func (p *Parser) Interface() *StubType {
	f := p.Fields(";");
	if p.token != "}" {
		return MissingStub;
	}
	p.Next();
	return NewStubType(NewInterfaceTypeStruct(f));
}

func (p *Parser) Func() *StubType {
	// may be 1 or 2 parenthesized lists
	f1 := NewStructTypeStruct(p.Fields(","));
	if p.token != ")" {
		return MissingStub;
	}
	p.Next();
	if p.token != "(" {
		// 1 list: the in parameters only
		return NewStubType(NewFuncTypeStruct(f1, nil));
	}
	p.Next();
	f2 := NewStructTypeStruct(p.Fields(","));
	if p.token != ")" {
		return MissingStub;
	}
	p.Next();
	// 2 lists: the in and out parameters are present
	return NewStubType(NewFuncTypeStruct(f1, f2));
}

func (p *Parser) Type() *StubType {
	dir := BothDir;
	switch {
	case p.token == "":
		return nil;
	case p.token == "*":
		p.Next();
		return NewStubType(NewPtrTypeStruct(p.Type()));
	case p.token == "[":
		p.Next();
		return p.Array();
	case p.token == "map":
		p.Next();
		return p.Map();
	case p.token == "<-":
		p.Next();
		dir = RecvDir;
		if p.token != "chan" {
			return MissingStub;
		}
		fallthrough;
	case p.token == "chan":
		p.Next();
		return p.Chan(dir);
	case p.token == "struct":
		p.Next();
		if p.token != "{" {
			return MissingStub
		}
		p.Next();
		return p.Struct();
	case p.token == "interface":
		p.Next();
		if p.token != "{" {
			return MissingStub
		}
		p.Next();
		return p.Interface();
	case p.token == "(":
		p.Next();
		return p.Func();
	case isdigit(p.token[0]):
		p.Next();
		return MissingStub;
	case special(p.token[0]):
		p.Next();
		return MissingStub;
	}
	// must be an identifier. is it basic? if so, we have a stub
	if s, ok := basicstub[p.token]; ok {
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
		p.Next();
		return MissingStub;
	}
	s := new(StubType);
	s.name = p.token;
	p.Next();
	return s;
}

export func ParseTypeString(str string) Type {
	p := new(Parser);
	p.str = str;
	p.Next();
	return p.Type().Get();
}

// Create typestring map from reflect.typestrings() data.  Lock is held.
func InitializeTypeStrings() {
	if initialized {
		return
	}
	initialized = true;
	s := typestrings();
	slen := len(s);
	for i := 0; i < slen; {
		// "reflect.PtrType interface { Sub () (? reflect.Type) }\n"
		// find the identifier
		idstart := i;
		for ; i < slen && s[i] != ' '; i++ {
		}
		if i == slen {
			print("reflect.InitializeTypeStrings: bad identifier\n");
			return;
		}
		idend := i;
		i++;
		// find the end of the line, terminating the type
		typestart := i;
		for ; i < slen && s[i] != '\n'; i++ {
		}
		if i == slen {
			print("reflect.InitializeTypeStrings: bad type string\n");
			return;
		}
		typeend := i;
		i++;	//skip newline
		typestring[s[idstart:idend]] = s[typestart:typeend];
	}
}

// Look up type string associated with name.  Lock is held.
func TypeNameToTypeString(name string) string {
	s, ok := typestring[name];
	if !ok {
		InitializeTypeStrings();
		s, ok = typestring[name];
		if !ok {
			s = MissingString;
			typestring[name] = s;
		}
	}
	return s
}

// Type is known by name.  Find (and create if necessary) its real type.
func ExpandType(name string) Type {
	Lock();
	t, ok := types[name];
	if ok {
		Unlock();
		return *t
	}
	types[name] = &Missing;	// prevent recursion; will overwrite
	t1 := ParseTypeString(TypeNameToTypeString(name));
	p := new(Type);
	*p = t1;
	types[name] = p;
	Unlock();
	return t1;
}
