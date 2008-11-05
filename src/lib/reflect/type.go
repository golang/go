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
	BoolKind;
	ChanKind;
	DotDotDotKind;
	FloatKind;
	Float32Kind;
	Float64Kind;
	Float80Kind;
	FuncKind;
	IntKind;
	Int16Kind;
	Int32Kind;
	Int64Kind;
	Int8Kind;
	InterfaceKind;
	MapKind;
	PtrKind;
	StringKind;
	StructKind;
	UintKind;
	Uint16Kind;
	Uint32Kind;
	Uint64Kind;
	Uint8Kind;
)

// Int is guaranteed large enough to store a size.
var ptrsize int
var interfacesize int

var MissingString = "$missing$"	// syntactic name for undefined type names
var DotDotDotString = "..."

export type Type interface {
	Kind()	int;
	Name()	string;
	String()	string;
	SetString(string);	// TODO: remove when no longer needed
	Size()	int;
}

// Fields and methods common to all types
type Common struct {
	kind	int;
	str	string;
	name	string;
	size	int;
}

func (c *Common) Kind() int {
	return c.kind
}

func (c *Common) Name() string {
	return c.name
}

func (c *Common) String() string {
	return c.str
}

func (c *Common) SetString(s string) {
	c.str = s
}

func (c *Common) Size() int {
	return c.size
}

// -- Basic

type BasicType struct {
	Common
}

func NewBasicType(name string, kind int, size int) Type {
	return &BasicType{ Common{kind, name, name, size} }
}

// Prebuilt basic types
export var (
	Missing = NewBasicType(MissingString, MissingKind, 1);
	DotDotDot = NewBasicType(DotDotDotString, DotDotDotKind, 16);	// TODO(r): size of interface?
	Bool = NewBasicType("bool", BoolKind, 1); // TODO: need to know how big a bool is
	Int = NewBasicType("int", IntKind, 4);	// TODO: need to know how big an int is
	Int8 = NewBasicType("int8", Int8Kind, 1);
	Int16 = NewBasicType("int16", Int16Kind, 2);
	Int32 = NewBasicType("int32", Int32Kind, 4);
	Int64 = NewBasicType("int64", Int64Kind, 8);
	Uint = NewBasicType("uint", UintKind, 4);	// TODO: need to know how big a uint is
	Uint8 = NewBasicType("uint8", Uint8Kind, 1);
	Uint16 = NewBasicType("uint16", Uint16Kind, 2);
	Uint32 = NewBasicType("uint32", Uint32Kind, 4);
	Uint64 = NewBasicType("uint64", Uint64Kind, 8);
	Float = NewBasicType("float", FloatKind, 4);	// TODO: need to know how big a float is
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

func NewStubType(name string, typ Type) *StubType {
	return &StubType{name, typ}
}

func (t *StubType) Get() Type {
	if t.typ == nil {
		t.typ = ExpandType(t.name)
	}
	return t.typ
}

// -- Pointer

export type PtrType interface {
	Sub()	Type
}

type PtrTypeStruct struct {
	Common;
	sub	*StubType;
}

func NewPtrTypeStruct(name, typestring string, sub *StubType) *PtrTypeStruct {
	return &PtrTypeStruct{ Common{PtrKind, typestring, name, ptrsize}, sub}
}

func (t *PtrTypeStruct) Sub() Type {
	return t.sub.Get()
}

// -- Array

export type ArrayType interface {
	Open()	bool;
	Len()	int;
	Elem()	Type;
}

type ArrayTypeStruct struct {
	Common;
	elem	*StubType;
	open	bool;	// otherwise fixed size
	len	int;
}

func NewArrayTypeStruct(name, typestring string, open bool, len int, elem *StubType) *ArrayTypeStruct {
	return &ArrayTypeStruct{ Common{ArrayKind, typestring, name, 0}, elem, open, len}
}

func (t *ArrayTypeStruct) Size() int {
	if t.open {
		return ptrsize	// open arrays are pointers to structures
	}
	return t.len * t.elem.Get().Size();
}

func (t *ArrayTypeStruct) Open() bool {
	return t.open
}

func (t *ArrayTypeStruct) Len() int {
	// what about open array?  TODO
	return t.len
}

func (t *ArrayTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Map

export type MapType interface {
	Key()	Type;
	Elem()	Type;
}

type MapTypeStruct struct {
	Common;
	key	*StubType;
	elem	*StubType;
}

func NewMapTypeStruct(name, typestring string, key, elem *StubType) *MapTypeStruct {
	return &MapTypeStruct{ Common{MapKind, typestring, name, 0}, key, elem}
}

func (t *MapTypeStruct) Size() int {
	panic("reflect.type: map.Size(): cannot happen");
	return 0
}

func (t *MapTypeStruct) Key() Type {
	return t.key.Get()
}

func (t *MapTypeStruct) Elem() Type {
	return t.elem.Get()
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
	Common;
	elem	*StubType;
	dir	int;
}

func NewChanTypeStruct(name, typestring string, dir int, elem *StubType) *ChanTypeStruct {
	return &ChanTypeStruct{ Common{ChanKind, typestring, name, 0}, elem, dir}
}

func (t *ChanTypeStruct) Size() int {
	panic("reflect.type: chan.Size(): cannot happen");
	return 0
}

func (t *ChanTypeStruct) Dir() int {
	return t.dir
}

func (t *ChanTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Struct

export type StructType interface {
	Field(int)	(name string, typ Type, tag string, offset int);
	Len()	int;
}

type Field struct {
	name	string;
	typ	*StubType;
	tag	string;
	size	int;
	offset	int;
}

type StructTypeStruct struct {
	Common;
	field	*[]Field;
}

func NewStructTypeStruct(name, typestring string, field *[]Field) *StructTypeStruct {
	return &StructTypeStruct{ Common{StructKind, typestring, name, 0}, field}
}

// TODO: not portable; depends on 6g
func (t *StructTypeStruct) Size() int {
	if t.size > 0 {
		return t.size
	}
	size := 0;
	structalignmask := 7;	// BUG: we know structs are 8-aligned
	for i := 0; i < len(t.field); i++ {
		elemsize := t.field[i].typ.Get().Size();
		// pad until at (elemsize mod 8) boundary
		align := elemsize - 1;
		if align > structalignmask {
			align = structalignmask
		}
		if align > 0 {
			size = (size + align) & ^align;
		}
		t.field[i].offset = size;
		size += elemsize;
	}
	size = (size + structalignmask) & ^(structalignmask);
	t.size = size;
	return size;
}

func (t *StructTypeStruct) Field(i int) (name string, typ Type, tag string, offset int) {
	if t.field[i].offset == 0 {
		t.Size();	// will compute offsets
	}
	return t.field[i].name, t.field[i].typ.Get(), t.field[i].tag, t.field[i].offset
}

func (t *StructTypeStruct) Len() int {
	return len(t.field)
}

// -- Interface

export type InterfaceType interface {
	Field(int)	(name string, typ Type, tag string, offset int);
	Len()	int;
}

type InterfaceTypeStruct struct {
	Common;
	field	*[]Field;
}

func NewInterfaceTypeStruct(name, typestring string, field *[]Field) *InterfaceTypeStruct {
	return &InterfaceTypeStruct{ Common{InterfaceKind, typestring, name, interfacesize}, field }
}

func (t *InterfaceTypeStruct) Field(i int) (name string, typ Type, tag string, offset int) {
	return t.field[i].name, t.field[i].typ.Get(), "", 0
}

func (t *InterfaceTypeStruct) Len() int {
	return len(t.field)
}

// -- Func

export type FuncType interface {
	In()	StructType;
	Out()	StructType;
}

type FuncTypeStruct struct {
	Common;
	in	*StructTypeStruct;
	out	*StructTypeStruct;
}

func NewFuncTypeStruct(name, typestring string, in, out *StructTypeStruct) *FuncTypeStruct {
	return &FuncTypeStruct{ Common{FuncKind, typestring, name, 0}, in, out }
}

func (t *FuncTypeStruct) Size() int {
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

// Cache of expanded types keyed by type name.
var types *map[string] *Type	// BUG TODO: should be Type not *Type

// List of typename, typestring pairs
var typestring *map[string] string
var initialized bool = false

// Map of basic types to prebuilt StubTypes
var basicstub *map[string] *StubType

var MissingStub *StubType;
var DotDotDotStub *StubType;

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
	types[DotDotDotString] = &DotDotDot;
	types["int"] = &Int;
	types["int8"] = &Int8;
	types["int16"] = &Int16;
	types["int32"] = &Int32;
	types["int64"] = &Int64;
	types["uint"] = &Uint;
	types["uint8"] = &Uint8;
	types["uint16"] = &Uint16;
	types["uint32"] = &Uint32;
	types["uint64"] = &Uint64;
	types["float"] = &Float;
	types["float32"] = &Float32;
	types["float64"] = &Float64;
	types["float80"] = &Float80;
	types["string"] = &String;
	types["bool"] = &Bool;

	// Basics get prebuilt stubs
	MissingStub = NewStubType(MissingString, Missing);
	DotDotDotStub = NewStubType(DotDotDotString, DotDotDot);
	basicstub[MissingString] = MissingStub;
	basicstub[DotDotDotString] = DotDotDotStub;
	basicstub["int"] = NewStubType("int", Int);
	basicstub["int8"] = NewStubType("int8", Int8);
	basicstub["int16"] = NewStubType("int16", Int16);
	basicstub["int32"] = NewStubType("int32", Int32);
	basicstub["int64"] = NewStubType("int64", Int64);
	basicstub["uint"] = NewStubType("uint", Uint);
	basicstub["uint8"] = NewStubType("uint8", Uint8);
	basicstub["uint16"] = NewStubType("uint16", Uint16);
	basicstub["uint32"] = NewStubType("uint32", Uint32);
	basicstub["uint64"] = NewStubType("uint64", Uint64);
	basicstub["float"] = NewStubType("float", Float);
	basicstub["float32"] = NewStubType("float32", Float32);
	basicstub["float64"] = NewStubType("float64", Float64);
	basicstub["float80"] = NewStubType("float80", Float80);
	basicstub["string"] = NewStubType("string", String);
	basicstub["bool"] = NewStubType("bool", Bool);

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
	doublequotedstring = 
		string in " ";  escapes are \x00 (NUL) \n \t \" \\
	fieldlist =
		[ field { [ ',' | ';' ] field } ]
	field =
		identifier stubtype [ doublequotedstring ]
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

func hex00(s string, i int) bool {
	return i + 2 < len(s) && s[i] == '0' && s[i+1] == '0'
}

// Process backslashes.  String known to be well-formed.
// Initial double-quote is left in, as an indication this token is a string.
func unescape(s string, backslash bool) string {
	if !backslash {
		return s
	}
	out := "\"";
	for i := 1; i < len(s); i++ {
		c := s[i];
		if c == '\\' {
			i++;
			c = s[i];
			switch c {
			case 'n':
				c = '\n';
			case 't':
				c = '\t';
			case 'x':
				if hex00(s, i+1) {
					i += 2;
					c = 0;
					break;
				}
			// otherwise just put an 'x'; erroneous but safe.
			// default is correct already; \\ is \; \" is "
			}
		}
		out += string(c);
	}
	return out;
}

// Simple parser for type strings
type Parser struct {
	str	string;	// string being parsed
	token	string;	// the token being parsed now
	tokstart	int;	// starting position of token
	prevend	int;	// (one after) ending position of previous token
	index	int;	// next character position in str
}

// Return typestring starting at position i.
// Trim trailing blanks.
func (p *Parser) TypeString(i int) string {
	return p.str[i:p.prevend];
}

// Load next token into p.token
func (p *Parser) Next() {
	p.prevend = p.index;
	token := "";
	for ; p.index < len(p.str) && p.str[p.index] == ' '; p.index++ {
	}
	p.tokstart = p.index;
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
	case c == '.':
		if p.index < len(p.str)+2 && p.str[p.index-1:p.index+2] == DotDotDotString {
			p.index += 2;
			p.token = DotDotDotString;
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
	case c == '"':	// double-quoted string for struct field annotation
		backslash := false;
		for p.index < len(p.str) && p.str[p.index] != '"' {
			if p.str[p.index] == '\\' {
				if p.index+1 == len(p.str) {	// bad final backslash
					break;
				}
				p.index++;	// skip (and accept) backslash
				backslash = true;
			}
			p.index++
		}
		p.token = unescape(p.str[start : p.index], backslash);
		if p.index < len(p.str) {	// properly terminated string
			p.index++;	// skip the terminating double-quote
		}
		return;
	}
	for p.index < len(p.str) && p.str[p.index] != ' ' && !special(p.str[p.index]) {
		p.index++
	}
	p.token = p.str[start : p.index];
}

func (p *Parser) Type(name string) *StubType

func (p *Parser) Array(name string, tokstart int) *StubType {
	size := 0;
	open := true;
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
		open = false;
	}
	if p.token != "]" {
		return MissingStub
	}
	p.Next();
	elemtype := p.Type("");
	return NewStubType(name, NewArrayTypeStruct(name, p.TypeString(tokstart), open, size, elemtype));
}

func (p *Parser) Map(name string, tokstart int) *StubType {
	if p.token != "[" {
		return MissingStub
	}
	p.Next();
	keytype := p.Type("");
	if p.token != "]" {
		return MissingStub
	}
	p.Next();
	elemtype := p.Type("");
	return NewStubType(name, NewMapTypeStruct(name, p.TypeString(tokstart), keytype, elemtype));
}

func (p *Parser) Chan(name string, tokstart, dir int) *StubType {
	if p.token == "<-" {
		if dir != BothDir {
			return MissingStub
		}
		p.Next();
		dir = SendDir;
	}
	elemtype := p.Type("");
	return NewStubType(name, NewChanTypeStruct(name, p.TypeString(tokstart), dir, elemtype));
}

// Parse array of fields for struct, interface, and func arguments
func (p *Parser) Fields(sep, term string) *[]Field {
	a := new([]Field, 10);
	nf := 0;
	for p.token != "" && p.token != term {
		if nf == len(a) {
			a1 := new([]Field, 2*nf);
			for i := 0; i < nf; i++ {
				a1[i] = a[i];
			}
			a = a1;
		}
		a[nf].name = p.token;
		p.Next();
		a[nf].typ = p.Type("");
		if p.token != "" && p.token[0] == '"' {
			a[nf].tag = p.token[1:len(p.token)];
			p.Next();
		}
		nf++;
		if p.token != sep {
			break;
		}
		p.Next();	// skip separator
	}
	return a[0:nf];
}

// A single type packaged as a field for a function return
func (p *Parser) OneField() *[]Field {
	a := new([]Field, 1);
	a[0].name = "";
	a[0].typ = p.Type("");
	return a;
}

func (p *Parser) Struct(name string, tokstart int) *StubType {
	f := p.Fields(";", "}");
	if p.token != "}" {
		return MissingStub;
	}
	p.Next();
	return NewStubType(name, NewStructTypeStruct(name, p.TypeString(tokstart), f));
}

func (p *Parser) Interface(name string, tokstart int) *StubType {
	f := p.Fields(";", "}");
	if p.token != "}" {
		return MissingStub;
	}
	p.Next();
	return NewStubType(name, NewInterfaceTypeStruct(name, p.TypeString(tokstart), f));
}

func (p *Parser) Func(name string, tokstart int) *StubType {
	// may be 1 or 2 parenthesized lists
	f1 := NewStructTypeStruct("", "", p.Fields(",", ")"));
	if p.token != ")" {
		return MissingStub;
	}
	p.Next();
	if p.token != "(" {
		// 1 list: the in parameters are a list.  Is there a single out parameter?
		if p.token == "" || p.token == "}" || p.token == "," || p.token == ";" {
			return NewStubType(name, NewFuncTypeStruct(name, p.TypeString(tokstart), f1, nil));
		}
		// A single out parameter.
		f2 := NewStructTypeStruct("", "", p.OneField());
		return NewStubType(name, NewFuncTypeStruct(name, p.TypeString(tokstart), f1, f2));
	} else {
		p.Next();
	}
	f2 := NewStructTypeStruct("", "", p.Fields(",", ")"));
	if p.token != ")" {
		return MissingStub;
	}
	p.Next();
	// 2 lists: the in and out parameters are present
	return NewStubType(name, NewFuncTypeStruct(name, p.TypeString(tokstart), f1, f2));
}

func (p *Parser) Type(name string) *StubType {
	dir := BothDir;
	tokstart := p.tokstart;
	switch {
	case p.token == "":
		return nil;
	case p.token == "*":
		p.Next();
		sub := p.Type("");
		return NewStubType(name, NewPtrTypeStruct(name, p.TypeString(tokstart), sub));
	case p.token == "[":
		p.Next();
		return p.Array(name, tokstart);
	case p.token == "map":
		p.Next();
		return p.Map(name, tokstart);
	case p.token == "<-":
		p.Next();
		dir = RecvDir;
		if p.token != "chan" {
			return MissingStub;
		}
		fallthrough;
	case p.token == "chan":
		p.Next();
		return p.Chan(name, tokstart, dir);
	case p.token == "struct":
		p.Next();
		if p.token != "{" {
			return MissingStub
		}
		p.Next();
		return p.Struct(name, tokstart);
	case p.token == "interface":
		p.Next();
		if p.token != "{" {
			return MissingStub
		}
		p.Next();
		return p.Interface(name, tokstart);
	case p.token == "(":
		p.Next();
		return p.Func(name, tokstart);
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
		if name != "" {
			// Need to make a copy because we are renaming a basic type
			b := s.Get();
			s = NewStubType(name, NewBasicType(name, b.Kind(), b.Size()));
		}
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
	s := NewStubType(p.token, nil);
	p.Next();
	return s;
}

export func ParseTypeString(name, typestring string) Type {
	p := new(Parser);
	p.str = typestring;
	p.Next();
	return p.Type(name).Get();
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
	t1 := ParseTypeString(name, TypeNameToTypeString(name));
	p := new(Type);
	*p = t1;
	types[name] = p;
	Unlock();
	return t1;
}
