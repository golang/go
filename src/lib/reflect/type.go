// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reflection library.
// Types and parsing of type strings.

// This package implements data ``reflection''.  A program can use it to analyze types
// and values it does not know at compile time, such as the values passed in a call
// to a function with a ... parameter.  This is achieved by extracting the dynamic
// contents of an interface value.
package reflect

import (
	"utf8";
	"sync";
	"unsafe";
)

type Type interface

func ExpandType(name string) Type

func typestrings() string	// implemented in C; declared here

// These constants identify what kind of thing a Type represents: an int, struct, etc.
const (
	MissingKind = iota;
	ArrayKind;
	BoolKind;
	ChanKind;
	DotDotDotKind;
	FloatKind;
	Float32Kind;
	Float64Kind;
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

// For sizes and alignments.

type allTypes struct {
	xarray		[]byte;
	xbool		bool;
	xchan		chan byte;
	xfloat		float;
	xfloat32	float32;
	xfloat64	float64;
	xfunc		func();
	xint		int;
	xint16		int16;
	xint32		int32;
	xint64		int64;
	xint8		int8;
	xinterface	interface {};
	xmap		map[byte]byte;
	xptr		*byte;
	xslice		[]byte;
	xstring		string;
	xuint		uint;
	xuint16		uint16;
	xuint32		uint32;
	xuint64		uint64;
	xuint8		uint8;
	xuintptr	uintptr;
}

var x allTypes

const (
	ptrsize = unsafe.Sizeof(&x);
	interfacesize = unsafe.Sizeof(x.xinterface);
)

var missingString = "$missing$"	// syntactic name for undefined type names
var dotDotDotString = "..."

// Type is the generic interface to reflection types.  Once its Kind is known,
// such as BoolKind, the Type can be narrowed to the appropriate, more
// specific interface, such as BoolType.  Such narrowed types still implement
// the Type interface.
type Type interface {
	// The kind of thing described: ArrayKind, BoolKind, etc.
	Kind()	int;
	// The name declared for the type ("int", "BoolArray", etc.).
	Name()	string;
	// For a named type, same as Name(); otherwise a representation of the type such as "[]int".
	String()	string;
	// The number of bytes needed to store a value; analogous to unsafe.Sizeof().
	Size()	int;
	// The alignment of a value of this type when used as a field in a struct.
	FieldAlign()	int;
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
	commonType;
	fieldAlign	int;
}

func newBasicType(name string, kind int, size int, fieldAlign int) Type {
	return &basicType{ commonType{kind, name, name, size}, fieldAlign }
}

func (t *basicType) FieldAlign() int {
	return t.fieldAlign
}

// Prebuilt basic Type objects representing the predeclared basic types.
// Most are self-evident except:
//	Missing represents types whose representation cannot be discovered; usually an error.
//	DotDotDot represents the pseudo-type of a ... parameter.
var (
	Missing = newBasicType(missingString, MissingKind, 1, 1);
	DotDotDot = newBasicType(dotDotDotString, DotDotDotKind, unsafe.Sizeof(x.xinterface), unsafe.Alignof(x.xinterface));
	Bool = newBasicType("bool", BoolKind, unsafe.Sizeof(x.xbool), unsafe.Alignof(x.xbool));
	Int = newBasicType("int", IntKind, unsafe.Sizeof(x.xint), unsafe.Alignof(x.xint));
	Int8 = newBasicType("int8", Int8Kind, unsafe.Sizeof(x.xint8), unsafe.Alignof(x.xint8));
	Int16 = newBasicType("int16", Int16Kind, unsafe.Sizeof(x.xint16), unsafe.Alignof(x.xint16));
	Int32 = newBasicType("int32", Int32Kind, unsafe.Sizeof(x.xint32), unsafe.Alignof(x.xint32));
	Int64 = newBasicType("int64", Int64Kind, unsafe.Sizeof(x.xint64), unsafe.Alignof(x.xint64));
	Uint = newBasicType("uint", UintKind, unsafe.Sizeof(x.xuint), unsafe.Alignof(x.xuint));
	Uint8 = newBasicType("uint8", Uint8Kind, unsafe.Sizeof(x.xuint8), unsafe.Alignof(x.xuint8));
	Uint16 = newBasicType("uint16", Uint16Kind, unsafe.Sizeof(x.xuint16), unsafe.Alignof(x.xuint16));
	Uint32 = newBasicType("uint32", Uint32Kind, unsafe.Sizeof(x.xuint32), unsafe.Alignof(x.xuint32));
	Uint64 = newBasicType("uint64", Uint64Kind, unsafe.Sizeof(x.xuint64), unsafe.Alignof(x.xuint64));
	Uintptr = newBasicType("uintptr", UintptrKind, unsafe.Sizeof(x.xuintptr), unsafe.Alignof(x.xuintptr));
	Float = newBasicType("float", FloatKind, unsafe.Sizeof(x.xfloat), unsafe.Alignof(x.xfloat));
	Float32 = newBasicType("float32", Float32Kind, unsafe.Sizeof(x.xfloat32), unsafe.Alignof(x.xfloat32));
	Float64 = newBasicType("float64", Float64Kind, unsafe.Sizeof(x.xfloat64), unsafe.Alignof(x.xfloat64));
	String = newBasicType("string", StringKind, unsafe.Sizeof(x.xstring), unsafe.Alignof(x.xstring));
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

// PtrType represents a pointer.
type PtrType interface {
	Type;
	Sub()	Type	// The type of the pointed-to item; for "*int", it will be "int".
}

type ptrTypeStruct struct {
	commonType;
	sub	*stubType;
}

func newPtrTypeStruct(name, typestring string, sub *stubType) *ptrTypeStruct {
	return &ptrTypeStruct{ commonType{PtrKind, typestring, name, ptrsize}, sub}
}

func (t *ptrTypeStruct) FieldAlign() int {
	return unsafe.Alignof(x.xptr);
}

func (t *ptrTypeStruct) Sub() Type {
	return t.sub.Get()
}

// -- Array

// ArrayType represents an array or slice type.
type ArrayType interface {
	Type;
	IsSlice()	bool;	// True for slices, false for arrays.
	Len()	int;	// 0 for slices, the length for array types.
	Elem()	Type;	// The type of the elements.
}

type arrayTypeStruct struct {
	commonType;
	elem	*stubType;
	isslice	bool;	// otherwise fixed array
	len	int;
}

func newArrayTypeStruct(name, typestring string, open bool, len int, elem *stubType) *arrayTypeStruct {
	return &arrayTypeStruct{ commonType{ArrayKind, typestring, name, 0 }, elem, open, len}
}

func (t *arrayTypeStruct) Size() int {
	if t.isslice {
		return unsafe.Sizeof(x.xslice);
	}
	return t.len * t.elem.Get().Size();
}

func (t *arrayTypeStruct) FieldAlign() int {
	 if t.isslice {
		return unsafe.Alignof(x.xslice);
	}
	return t.elem.Get().FieldAlign();
}

func (t *arrayTypeStruct) IsSlice() bool {
	return t.isslice
}

func (t *arrayTypeStruct) Len() int {
	if t.isslice {
		return 0
	}
	return t.len
}

func (t *arrayTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Map

// MapType represents a map type.
type MapType interface {
	Type;
	Key()	Type;	// The type of the keys.
	Elem()	Type;	// The type of the elements/values.
}

type mapTypeStruct struct {
	commonType;
	key	*stubType;
	elem	*stubType;
}

func newMapTypeStruct(name, typestring string, key, elem *stubType) *mapTypeStruct {
	return &mapTypeStruct{ commonType{MapKind, typestring, name, ptrsize}, key, elem}
}

func (t *mapTypeStruct) FieldAlign() int {
	return unsafe.Alignof(x.xmap);
}

func (t *mapTypeStruct) Key() Type {
	return t.key.Get()
}

func (t *mapTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Chan

// ChanType represents a chan type.
type ChanType interface {
	Type;
	Dir()	int;	// The direction of the channel.
	Elem()	Type;	// The type of the elements.
}

// Channel direction.
const (
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

func (t *chanTypeStruct) FieldAlign() int {
	return unsafe.Alignof(x.xchan);
}

func (t *chanTypeStruct) Dir() int {
	return t.dir
}

func (t *chanTypeStruct) Elem() Type {
	return t.elem.Get()
}

// -- Struct

// StructType represents a struct type.
type StructType interface {
	Type;
	// Field returns, for field i, its name, Type, tag information, and byte offset.
	// The indices are in declaration order starting at 0.
	Field(i int)	(name string, typ Type, tag string, offset int);
	// Len is the number of fields.
	Len()	int;
}

type structField struct {
	name	string;
	typ	*stubType;
	tag	string;
	offset	int;
}

type structTypeStruct struct {
	commonType;
	field	[]structField;
	fieldAlign	int;
}

func newStructTypeStruct(name, typestring string, field []structField) *structTypeStruct {
	return &structTypeStruct{ commonType{StructKind, typestring, name, 0}, field, 0}
}

func (t *structTypeStruct) FieldAlign() int {
	t.Size();	// Compute size and alignment.
	return t.fieldAlign
}

func (t *structTypeStruct) Size() int {
	if t.size > 0 {
		return t.size
	}
	size := 0;
	structalign := 0;
	for i := 0; i < len(t.field); i++ {
		typ := t.field[i].typ.Get();
		elemsize := typ.Size();
		align := typ.FieldAlign() - 1;
		if align > structalign {
			structalign = align
		}
		if align > 0 {
			size = (size + align) &^ align;
		}
		t.field[i].offset = size;
		size += elemsize;
	}
	if (structalign > 0) {
		// TODO: In the PPC64 ELF ABI, floating point fields
		// in a struct are aligned to a 4-byte boundary, but
		// if the first field in the struct is a 64-bit float,
		// the whole struct is aligned to an 8-byte boundary.
		size = (size + structalign) &^ structalign;
		t.fieldAlign = structalign + 1;
	}
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

// InterfaceType represents an interface type.
// It behaves much like a StructType, treating the methods as fields.
type InterfaceType interface {
	Type;
	// Field returns, for method i, its name, Type, the empty string, and 0.
	// The indices are in declaration order starting at 0.  TODO: is this true?
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

func (t *interfaceTypeStruct) FieldAlign() int {
	return unsafe.Alignof(x.xinterface);
}

func (t *interfaceTypeStruct) Field(i int) (name string, typ Type, tag string, offset int) {
	return t.field[i].name, t.field[i].typ.Get(), "", 0
}

func (t *interfaceTypeStruct) Len() int {
	return len(t.field)
}

var nilInterface = newInterfaceTypeStruct("nil", "", make([]structField, 0));

// -- Func

// FuncType represents a function type.
type FuncType interface {
	Type;
	In()	StructType;	// The parameters in the form of a StructType.
	Out()	StructType;	// The results in the form of a StructType.
}

type funcTypeStruct struct {
	commonType;
	in	*structTypeStruct;
	out	*structTypeStruct;
}

func newFuncTypeStruct(name, typestring string, in, out *structTypeStruct) *funcTypeStruct {
	return &funcTypeStruct{ commonType{FuncKind, typestring, name, ptrsize}, in, out }
}

func (t *funcTypeStruct) FieldAlign() int {
	return unsafe.Alignof(x.xfunc);
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
	basicstub["string"] = newStubType("string", String);
	basicstub["bool"] = newStubType("bool", Bool);

	unlock();
}

/*
	Parsing of type strings.  These strings are how the run-time recovers type
	information dynamically.

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
		'<-' 'chan' stubtype
		'chan' '<-' stubtype
		'chan' stubtype
	maptype =
		'map' '[' stubtype ']' stubtype
	pointertype =
		'*' stubtype
	functiontype =
		[ 'func' ] '(' fieldlist ')' [ '(' fieldlist ')' | stubtype ]

	In functiontype 'func' is optional because it is omitted in
	the reflection string for interface types.

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
		switch p.token {
		case "", "}", ")", ",", ";":
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
	case p.token == "func":
		p.Next();
		if p.token != "(" {
			return missingStub
		}
		p.Next();
		return p.Func(name, tokstart);
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
			s = newStubType(name, newBasicType(name, b.Kind(), b.Size(), b.FieldAlign()));
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

// ParseTypeString takes a type name and type string (such as "[]int") and
// returns the Type structure representing a type name specifying the corresponding
// type.  An empty typestring represents (the type of) a nil interface value.
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

// ExpandType takes the name of a type and returns its Type structure,
// unpacking the associated type string if necessary.
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
