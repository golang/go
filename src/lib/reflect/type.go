// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Types and parsing of type strings.

package reflect

import (
	"utf8";
	"sync";
	"unsafe";
)

type Type interface

func ExpandType(name string) Type

func typestrings() string	// implemented in C; declared here

const (
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
	UintptrKind;
)

var tmp_interface interface{}	// used just to compute sizes of these constants
const (
	ptrsize = unsafe.Sizeof(&tmp_interface);
	interfacesize = unsafe.Sizeof(tmp_interface);
)

var missingString = "$missing$"	// syntactic name for undefined type names
var dotDotDotString = "..."

type Type interface {
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;
}

// Fields and methods common to all types
type commonType struct {
	kind	int;
	str	string;
	name	string;
	size	int;
}

func (c *commonType) Kind() int {
	return c.kind
}

func (c *commonType) Name() string {
	return c.name
}

func (c *commonType) String() string {
	// If there is a name, show that instead of its expansion.
	// This is important for reflection: a named type
	// might have methods that the unnamed type does not.
	if c.name != "" {
		return c.name
	}
	return c.str
}

func (c *commonType) Size() int {
	return c.size
}

// -- Basic

type basicType struct {
	commonType
}

func newBasicType(name string, kind int, size int) Type {
	return &basicType{ commonType{kind, name, name, size} }
}

// Prebuilt basic types
var (
	Missing = newBasicType(missingString, MissingKind, 1);
	DotDotDot = newBasicType(dotDotDotString, DotDotDotKind, 16);	// TODO(r): size of interface?
	Bool = newBasicType("bool", BoolKind, 1); // TODO: need to know how big a bool is
	Int = newBasicType("int", IntKind, 4);	// TODO: need to know how big an int is
	Int8 = newBasicType("int8", Int8Kind, 1);
	Int16 = newBasicType("int16", Int16Kind, 2);
	Int32 = newBasicType("int32", Int32Kind, 4);
	Int64 = newBasicType("int64", Int64Kind, 8);
	Uint = newBasicType("uint", UintKind, 4);	// TODO: need to know how big a uint is
	Uint8 = newBasicType("uint8", Uint8Kind, 1);
	Uint16 = newBasicType("uint16", Uint16Kind, 2);
	Uint32 = newBasicType("uint32", Uint32Kind, 4);
	Uint64 = newBasicType("uint64", Uint64Kind, 8);
	Uintptr = newBasicType("uintptr", UintptrKind, 8);	// TODO: need to know how big a uintptr is
	Float = newBasicType("float", FloatKind, 4);	// TODO: need to know how big a float is
	Float32 = newBasicType("float32", Float32Kind, 4);
	Float64 = newBasicType("float64", Float64Kind, 8);
	Float80 = newBasicType("float80", Float80Kind, 10);	// TODO: strange size?
	String = newBasicType("string", StringKind, 8);	// implemented as a pointer
)

// Stub types allow us to defer evaluating type names until needed.
// If the name is empty, the type must be non-nil.

type stubType struct {
	name	string;
	typ		Type;
}

func newStubType(name string, typ Type) *stubType {
	return &stubType{name, typ}
}

func (t *stubType) Get() Type {
	if t.typ == nil {
		t.typ = ExpandType(t.name)
	}
	return t.typ
}

// -- Pointer

type PtrType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	Sub()	Type
}

type ptrTypeStruct struct {
	commonType;
	sub	*stubType;
}

func newPtrTypeStruct(name, typestring string, sub *stubType) *ptrTypeStruct {
	return &ptrTypeStruct{ commonType{PtrKind, typestring, name, ptrsize}, sub}
}

func (t *ptrTypeStruct) Sub() Type {
	return t.sub.Get()
}

// -- Array

type ArrayType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	IsSlice()	bool;
	Len()	int;
	Elem()	Type;
}

type arrayTypeStruct struct {
	commonType;
	elem	*stubType;
	isslice	bool;	// otherwise fixed array
	len	int;
}

func newArrayTypeStruct(name, typestring string, open bool, len int, elem *stubType) *arrayTypeStruct {
	return &arrayTypeStruct{ commonType{ArrayKind, typestring, name, 0}, elem, open, len}
}

func (t *arrayTypeStruct) Size() int {
	if t.isslice {
		return ptrsize*2	// open arrays are 2-word headers
	}
	return t.len * t.elem.Get().Size();
}

func (t *arrayTypeStruct) IsSlice() bool {
	return t.isslice
}

func (t *arrayTypeStruct) Len() int {
	// what about open array?  TODO
	return t.len
}

func (t *arrayTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Map

type MapType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	Key()	Type;
	Elem()	Type;
}

type mapTypeStruct struct {
	commonType;
	key	*stubType;
	elem	*stubType;
}

func newMapTypeStruct(name, typestring string, key, elem *stubType) *mapTypeStruct {
	return &mapTypeStruct{ commonType{MapKind, typestring, name, ptrsize}, key, elem}
}

func (t *mapTypeStruct) Key() Type {
	return t.key.Get()
}

func (t *mapTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Chan

type ChanType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	Dir()	int;
	Elem()	Type;
}

const (	// channel direction
	SendDir = 1 << iota;
	RecvDir;
	BothDir = SendDir | RecvDir;
)

type chanTypeStruct struct {
	commonType;
	elem	*stubType;
	dir	int;
}

func newChanTypeStruct(name, typestring string, dir int, elem *stubType) *chanTypeStruct {
	return &chanTypeStruct{ commonType{ChanKind, typestring, name, ptrsize}, elem, dir}
}

func (t *chanTypeStruct) Dir() int {
	return t.dir
}

func (t *chanTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Struct

type StructType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	Field(int)	(name string, typ Type, tag string, offset int);
	Len()	int;
}

type structField struct {
	name	string;
	typ	*stubType;
	tag	string;
	size	int;
	offset	int;
}

type structTypeStruct struct {
	commonType;
	field	[]structField;
}

func newStructTypeStruct(name, typestring string, field []structField) *structTypeStruct {
	return &structTypeStruct{ commonType{StructKind, typestring, name, 0}, field}
}

// TODO: not portable; depends on 6g
func (t *structTypeStruct) Size() int {
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

func (t *structTypeStruct) Field(i int) (name string, typ Type, tag string, offset int) {
	if t.field[i].offset == 0 {
		t.Size();	// will compute offsets
	}
	return t.field[i].name, t.field[i].typ.Get(), t.field[i].tag, t.field[i].offset
}

func (t *structTypeStruct) Len() int {
	return len(t.field)
}

// -- Interface

type InterfaceType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	Field(int)	(name string, typ Type, tag string, offset int);
	Len()	int;
}

type interfaceTypeStruct struct {
	commonType;
	field	[]structField;
}

func newInterfaceTypeStruct(name, typestring string, field []structField) *interfaceTypeStruct {
	return &interfaceTypeStruct{ commonType{InterfaceKind, typestring, name, interfacesize}, field }
}

func (t *interfaceTypeStruct) Field(i int) (name string, typ Type, tag string, offset int) {
	return t.field[i].name, t.field[i].typ.Get(), "", 0
}

func (t *interfaceTypeStruct) Len() int {
	return len(t.field)
}

var nilInterface = newInterfaceTypeStruct("nil", "", make([]structField, 0));

// -- Func

type FuncType interface {
	// TODO: Type;
	Kind()	int;
	Name()	string;
	String()	string;
	Size()	int;

	In()	StructType;
	Out()	StructType;
}

type funcTypeStruct struct {
	commonType;
	in	*structTypeStruct;
	out	*structTypeStruct;
}

func newFuncTypeStruct(name, typestring string, in, out *structTypeStruct) *funcTypeStruct {
	return &funcTypeStruct{ commonType{FuncKind, typestring, name, 0}, in, out }
}

func (t *funcTypeStruct) Size() int {
	panic("reflect.type: func.Size(): cannot happen");
	return 0
}

func (t *funcTypeStruct) In() StructType {
	return t.in
}

func (t *funcTypeStruct) Out() StructType {
	if t.out == nil {	// nil.(StructType) != nil so make sure caller sees real nil
		return nil
	}
	return t.out
}

// Cache of expanded types keyed by type name.
var types map[string] Type

// List of typename, typestring pairs
var typestring map[string] string
var initialized bool = false

// Map of basic types to prebuilt stubTypes
var basicstub map[string] *stubType

var missingStub *stubType;
var dotDotDotStub *stubType;

// The database stored in the maps is global; use locking to guarantee safety.
var typestringlock sync.Mutex

func lock() {
	typestringlock.Lock()
}

func unlock() {
	typestringlock.Unlock()
}

func init() {
	lock();	// not necessary because of init ordering but be safe.

	types = make(map[string] Type);
	typestring = make(map[string] string);
	basicstub = make(map[string] *stubType);

	// Basics go into types table
	types[missingString] = Missing;
	types[dotDotDotString] = DotDotDot;
	types["int"] = Int;
	types["int8"] = Int8;
	types["int16"] = Int16;
	types["int32"] = Int32;
	types["int64"] = Int64;
	types["uint"] = Uint;
	types["uint8"] = Uint8;
	types["uint16"] = Uint16;
	types["uint32"] = Uint32;
	types["uint64"] = Uint64;
	types["uintptr"] = Uintptr;
	types["float"] = Float;
	types["float32"] = Float32;
	types["float64"] = Float64;
	types["float80"] = Float80;
	types["string"] = String;
	types["bool"] = Bool;

	// Basics get prebuilt stubs
	missingStub = newStubType(missingString, Missing);
	dotDotDotStub = newStubType(dotDotDotString, DotDotDot);
	basicstub[missingString] = missingStub;
	basicstub[dotDotDotString] = dotDotDotStub;
	basicstub["int"] = newStubType("int", Int);
	basicstub["int8"] = newStubType("int8", Int8);
	basicstub["int16"] = newStubType("int16", Int16);
	basicstub["int32"] = newStubType("int32", Int32);
	basicstub["int64"] = newStubType("int64", Int64);
	basicstub["uint"] = newStubType("uint", Uint);
	basicstub["uint8"] = newStubType("uint8", Uint8);
	basicstub["uint16"] = newStubType("uint16", Uint16);
	basicstub["uint32"] = newStubType("uint32", Uint32);
	basicstub["uint64"] = newStubType("uint64", Uint64);
	basicstub["uintptr"] = newStubType("uintptr", Uintptr);
	basicstub["float"] = newStubType("float", Float);
	basicstub["float32"] = newStubType("float32", Float32);
	basicstub["float64"] = newStubType("float64", Float64);
	basicstub["float80"] = newStubType("float80", Float80);
	basicstub["string"] = newStubType("string", String);
	basicstub["bool"] = newStubType("bool", Bool);

	unlock();
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
type typeParser struct {
	str	string;	// string being parsed
	token	string;	// the token being parsed now
	tokstart	int;	// starting position of token
	prevend	int;	// (one after) ending position of previous token
	index	int;	// next character position in str
}

// Return typestring starting at position i.  It will finish at the
// end of the previous token (before trailing white space).
func (p *typeParser) TypeString(i int) string {
	return p.str[i:p.prevend];
}

// Load next token into p.token
func (p *typeParser) Next() {
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
	c, w := utf8.DecodeRuneInString(p.str, p.index);
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
		if p.index < len(p.str)+2 && p.str[p.index-1:p.index+2] == dotDotDotString {
			p.index += 2;
			p.token = dotDotDotString;
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

func (p *typeParser) Type(name string) *stubType

func (p *typeParser) Array(name string, tokstart int) *stubType {
	size := 0;
	open := true;
	if p.token != "]" {
		if len(p.token) == 0 || !isdigit(p.token[0]) {
			return missingStub
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
		return missingStub
	}
	p.Next();
	elemtype := p.Type("");
	return newStubType(name, newArrayTypeStruct(name, p.TypeString(tokstart), open, size, elemtype));
}

func (p *typeParser) Map(name string, tokstart int) *stubType {
	if p.token != "[" {
		return missingStub
	}
	p.Next();
	keytype := p.Type("");
	if p.token != "]" {
		return missingStub
	}
	p.Next();
	elemtype := p.Type("");
	return newStubType(name, newMapTypeStruct(name, p.TypeString(tokstart), keytype, elemtype));
}

func (p *typeParser) Chan(name string, tokstart, dir int) *stubType {
	if p.token == "<-" {
		if dir != BothDir {
			return missingStub
		}
		p.Next();
		dir = SendDir;
	}
	elemtype := p.Type("");
	return newStubType(name, newChanTypeStruct(name, p.TypeString(tokstart), dir, elemtype));
}

// Parse array of fields for struct, interface, and func arguments
func (p *typeParser) Fields(sep, term string) []structField {
	a := make([]structField, 10);
	nf := 0;
	for p.token != "" && p.token != term {
		if nf == len(a) {
			a1 := make([]structField, 2*nf);
			for i := 0; i < nf; i++ {
				a1[i] = a[i];
			}
			a = a1;
		}
		name := p.token;
		if name == "?" {	// used to represent a missing name
			name = ""
		}
		a[nf].name = name;
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
func (p *typeParser) OneField() []structField {
	a := make([]structField, 1);
	a[0].name = "";
	a[0].typ = p.Type("");
	return a;
}

func (p *typeParser) Struct(name string, tokstart int) *stubType {
	f := p.Fields(";", "}");
	if p.token != "}" {
		return missingStub;
	}
	p.Next();
	return newStubType(name, newStructTypeStruct(name, p.TypeString(tokstart), f));
}

func (p *typeParser) Interface(name string, tokstart int) *stubType {
	f := p.Fields(";", "}");
	if p.token != "}" {
		return missingStub;
	}
	p.Next();
	return newStubType(name, newInterfaceTypeStruct(name, p.TypeString(tokstart), f));
}

func (p *typeParser) Func(name string, tokstart int) *stubType {
	// may be 1 or 2 parenthesized lists
	f1 := newStructTypeStruct("", "", p.Fields(",", ")"));
	if p.token != ")" {
		return missingStub;
	}
	p.Next();
	if p.token != "(" {
		// 1 list: the in parameters are a list.  Is there a single out parameter?
		if p.token == "" || p.token == "}" || p.token == "," || p.token == ";" {
			return newStubType(name, newFuncTypeStruct(name, p.TypeString(tokstart), f1, nil));
		}
		// A single out parameter.
		f2 := newStructTypeStruct("", "", p.OneField());
		return newStubType(name, newFuncTypeStruct(name, p.TypeString(tokstart), f1, f2));
	} else {
		p.Next();
	}
	f2 := newStructTypeStruct("", "", p.Fields(",", ")"));
	if p.token != ")" {
		return missingStub;
	}
	p.Next();
	// 2 lists: the in and out parameters are present
	return newStubType(name, newFuncTypeStruct(name, p.TypeString(tokstart), f1, f2));
}

func (p *typeParser) Type(name string) *stubType {
	dir := BothDir;
	tokstart := p.tokstart;
	switch {
	case p.token == "":
		return nil;
	case p.token == "*":
		p.Next();
		sub := p.Type("");
		return newStubType(name, newPtrTypeStruct(name, p.TypeString(tokstart), sub));
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
			return missingStub;
		}
		fallthrough;
	case p.token == "chan":
		p.Next();
		return p.Chan(name, tokstart, dir);
	case p.token == "struct":
		p.Next();
		if p.token != "{" {
			return missingStub
		}
		p.Next();
		return p.Struct(name, tokstart);
	case p.token == "interface":
		p.Next();
		if p.token != "{" {
			return missingStub
		}
		p.Next();
		return p.Interface(name, tokstart);
	case p.token == "(":
		p.Next();
		return p.Func(name, tokstart);
	case isdigit(p.token[0]):
		p.Next();
		return missingStub;
	case special(p.token[0]):
		p.Next();
		return missingStub;
	}
	// must be an identifier. is it basic? if so, we have a stub
	if s, ok := basicstub[p.token]; ok {
		p.Next();
		if name != "" {
			// Need to make a copy because we are renaming a basic type
			b := s.Get();
			s = newStubType(name, newBasicType(name, b.Kind(), b.Size()));
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
		return missingStub;
	}
	s := newStubType(p.token, nil);
	p.Next();
	return s;
}

func ParseTypeString(name, typestring string) Type {
	if typestring == "" {
		// If the typestring is empty, it represents (the type of) a nil interface value
		return nilInterface
	}
	p := new(typeParser);
	p.str = typestring;
	p.Next();
	return p.Type(name).Get();
}

// Create typestring map from reflect.typestrings() data.  Lock is held.
func initializeTypeStrings() {
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
func typeNameToTypeString(name string) string {
	s, ok := typestring[name];
	if !ok {
		initializeTypeStrings();
		s, ok = typestring[name];
		if !ok {
			s = missingString;
			typestring[name] = s;
		}
	}
	return s
}

// Type is known by name.  Find (and create if necessary) its real type.
func ExpandType(name string) Type {
	lock();
	t, ok := types[name];
	if ok {
		unlock();
		return t
	}
	types[name] = Missing;	// prevent recursion; will overwrite
	t1 := ParseTypeString(name, typeNameToTypeString(name));
	types[name] = t1;
	unlock();
	return t1;
}
