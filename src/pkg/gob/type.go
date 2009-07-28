// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"fmt";
	"os";
	"reflect";
	"strings";
	"sync";
	"unicode";
)

type kind reflect.Type

// Reflection types are themselves interface values holding structs
// describing the type.  Each type has a different struct so that struct can
// be the kind.  For example, if typ is the reflect type for an int8, typ is
// a pointer to a reflect.Int8Type struct; if typ is the reflect type for a
// function, typ is a pointer to a reflect.FuncType struct; we use the type
// of that pointer as the kind.

// typeKind returns a reflect.Type representing typ's kind.  The kind is the
// general kind of type:
//	int8, int16, int, uint, float, func, chan, struct, and so on.
// That is, all struct types have the same kind, all func types have the same
// kind, all int8 types have the same kind, and so on.
func typeKind(typ reflect.Type) kind {
	return kind(reflect.Typeof(typ));
}

// valueKind returns the kind of the value type
// stored inside the interface v.
func valueKind(v interface{}) reflect.Type {
	return typeKind(reflect.Typeof(v));
}

// A typeId represents a gob Type as an integer that can be passed on the wire.
// Internally, typeIds are used as keys to a map to recover the underlying type info.
type typeId int32

var nextId	typeId	// incremented for each new type we build
var typeLock	sync.Mutex	// set while building a type

type gobType interface {
	id()	typeId;
	setId(id typeId);
	Name()	string;
	String()	string;
	safeString(seen map[typeId] bool)	string;
}

var types = make(map[reflect.Type] gobType)
var idToType = make(map[typeId] gobType)

func setTypeId(typ gobType) {
	nextId++;
	typ.setId(nextId);
	idToType[nextId] = typ;
}

func (t typeId) gobType() gobType {
	if t == 0 {
		return nil
	}
	return idToType[t]
}

// String returns the string representation of the type associated with the typeId.
func (t typeId) String() string {
	return t.gobType().String()
}

// Name returns the name of the type associated with the typeId.
func (t typeId) Name() string {
	return t.gobType().Name()
}

// Common elements of all types.
type commonType struct {
	name	string;
	_id	typeId;
}

func (t *commonType) id() typeId {
	return t._id
}

func (t *commonType) setId(id typeId) {
	t._id = id
}

func (t *commonType) String() string {
	return t.name
}

func (t *commonType) safeString(seen map[uint32] bool) string {
	return t.name
}

func (t *commonType) Name() string {
	return t.name
}

// Basic type identifiers, predefined.
var tBool typeId
var tInt typeId
var tUint typeId
var tFloat typeId
var tString typeId
var tBytes typeId

// Predefined because it's needed by the Decoder
var tWireType typeId

// Array type
type arrayType struct {
	commonType;
	Elem	typeId;
	Len	int;
}

func newArrayType(name string, elem gobType, length int) *arrayType {
	a := &arrayType{ commonType{ name: name }, elem.id(), length };
	setTypeId(a);
	return a;
}

func (a *arrayType) safeString(seen map[typeId] bool) string {
	if _, ok := seen[a._id]; ok {
		return a.name
	}
	seen[a._id] = true;
	return fmt.Sprintf("[%d]%s", a.Len, a.Elem.gobType().safeString(seen));
}

func (a *arrayType) String() string {
	return a.safeString(make(map[uint32] bool))
}

// Slice type
type sliceType struct {
	commonType;
	Elem	typeId;
}

func newSliceType(name string, elem gobType) *sliceType {
	s := &sliceType{ commonType{ name: name }, elem.id() };
	setTypeId(s);
	return s;
}

func (s *sliceType) safeString(seen map[typeId] bool) string {
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true;
	return fmt.Sprintf("[]%s", s.Elem.gobType().safeString(seen));
}

func (s *sliceType) String() string {
	return s.safeString(make(map[typeId] bool))
}

// Struct type
type fieldType struct {
	name	string;
	id	typeId;
}

type structType struct {
	commonType;
	field	[]*fieldType;
}

func (s *structType) safeString(seen map[typeId] bool) string {
	if s == nil {
		return "<nil>"
	}
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true;
	str := s.name + " = struct { ";
	for _, f := range s.field {
		str += fmt.Sprintf("%s %s; ", f.name, f.id.gobType().safeString(seen));
	}
	str += "}";
	return str;
}

func (s *structType) String() string {
	return s.safeString(make(map[typeId] bool))
}

func newStructType(name string) *structType {
	s := &structType{ commonType{ name: name }, nil };
	setTypeId(s);
	return s;
}

// Construction
func newType(name string, rt reflect.Type) gobType

// Step through the indirections on a type to discover the base type.
// Return the number of indirections.
func indirect(t reflect.Type) (rt reflect.Type, count int) {
	rt = t;
	for {
		pt, ok := rt.(*reflect.PtrType);
		if !ok {
			break;
		}
		rt = pt.Elem();
		count++;
	}
	return;
}

func newTypeObject(name string, rt reflect.Type) gobType {
	switch t := rt.(type) {
	// All basic types are easy: they are predefined.
	case *reflect.BoolType:
		return tBool.gobType()

	case *reflect.IntType:
		return tInt.gobType()
	case *reflect.Int8Type:
		return tInt.gobType()
	case *reflect.Int16Type:
		return tInt.gobType()
	case *reflect.Int32Type:
		return tInt.gobType()
	case *reflect.Int64Type:
		return tInt.gobType()

	case *reflect.UintType:
		return tUint.gobType()
	case *reflect.Uint8Type:
		return tUint.gobType()
	case *reflect.Uint16Type:
		return tUint.gobType()
	case *reflect.Uint32Type:
		return tUint.gobType()
	case *reflect.Uint64Type:
		return tUint.gobType()
	case *reflect.UintptrType:
		return tUint.gobType()

	case *reflect.FloatType:
		return tFloat.gobType()
	case *reflect.Float32Type:
		return tFloat.gobType()
	case *reflect.Float64Type:
		return tFloat.gobType()

	case *reflect.StringType:
		return tString.gobType()

	case *reflect.ArrayType:
		return newArrayType(name, newType("", t.Elem()), t.Len());

	case *reflect.SliceType:
		// []byte == []uint8 is a special case
		if _, ok := t.Elem().(*reflect.Uint8Type); ok {
			return tBytes.gobType()
		}
		return newSliceType(name, newType(t.Elem().Name(), t.Elem()));

	case *reflect.StructType:
		// Install the struct type itself before the fields so recursive
		// structures can be constructed safely.
		strType := newStructType(name);
		types[rt] = strType;
		idToType[strType.id()] = strType;
		field := make([]*fieldType, t.NumField());
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i);
			typ, _indir := indirect(f.Type);
			tname := typ.Name();
			if tname == "" {
				tname = f.Type.String();
			}
			field[i] =  &fieldType{ f.Name, newType(tname, f.Type).id() };
		}
		strType.field = field;
		return strType;

	default:
		panicln("gob NewTypeObject can't handle type", rt.String());	// TODO(r): panic?
	}
	return nil
}

func newType(name string, rt reflect.Type) gobType {
	// Flatten the data structure by collapsing out pointers
	for {
		pt, ok := rt.(*reflect.PtrType);
		if !ok {
			break;
		}
		rt = pt.Elem();
	}
	typ, present := types[rt];
	if present {
		return typ
	}
	typ = newTypeObject(name, rt);
	types[rt] = typ;
	return typ
}

// getType returns the Gob type describing the given reflect.Type.
// typeLock must be held.
func getType(name string, rt reflect.Type) gobType {
	// Set lock; all code running under here is synchronized.
	t := newType(name, rt);
	return t;
}

func checkId(want, got typeId) {
	if want != got {
		panicln("bootstrap type wrong id:", got.Name(), got, "not", want);
	}
}

// used for building the basic types; called only from init()
func bootstrapType(name string, e interface{}, expect typeId) typeId {
	rt := reflect.Typeof(e);
	_, present := types[rt];
	if present {
		panicln("bootstrap type already present:", name);
	}
	typ := &commonType{ name: name };
	types[rt] = typ;
	setTypeId(typ);
	checkId(expect, nextId);
	return nextId
}

// Representation of the information we send and receive about this type.
// Each value we send is preceded by its type definition: an encoded int.
// However, the very first time we send the value, we first send the pair
// (-id, wireType).
// For bootstrapping purposes, we assume that the recipient knows how
// to decode a wireType; it is exactly the wireType struct here, interpreted
// using the gob rules for sending a structure, except that we assume the
// ids for wireType and structType are known.  The relevant pieces
// are built in encode.go's init() function.

type wireType struct {
	s	*structType;
}

func (w *wireType) name() string {
	// generalize once we can have non-struct types on the wire.
	return w.s.name
}

type decEngine struct	// defined in decode.go
type encEngine struct	// defined in encode.go
type typeInfo struct {
	id	typeId;
	encoder	*encEngine;
	wire	*wireType;
}

var typeInfoMap = make(map[reflect.Type] *typeInfo)	// protected by typeLock

// The reflection type must have all its indirections processed out.
// typeLock must be held.
func getTypeInfo(rt reflect.Type) *typeInfo {
	if pt, ok := rt.(*reflect.PtrType); ok {
		panicln("pointer type in getTypeInfo:", rt.String())
	}
	info, ok := typeInfoMap[rt];
	if !ok {
		info = new(typeInfo);
		name := rt.Name();
		info.id = getType(name, rt).id();
		// assume it's a struct type
		info.wire = &wireType{info.id.gobType().(*structType)};
		typeInfoMap[rt] = info;
	}
	return info;
}

func init() {
	// Create and check predefined types
	tBool = bootstrapType("bool", false, 1);
	tInt = bootstrapType("int", int(0), 2);
	tUint = bootstrapType("uint", uint(0), 3);
	tFloat = bootstrapType("float", float64(0), 4);
	// The string for tBytes is "bytes" not "[]byte" to signify its specialness.
	tBytes = bootstrapType("bytes", make([]byte, 0), 5);
	tString= bootstrapType("string", "", 6);
	tWireType = getTypeInfo(reflect.Typeof(wireType{})).id;
	checkId(7, tWireType);
	checkId(8, getTypeInfo(reflect.Typeof(structType{})).id);
	checkId(9, getTypeInfo(reflect.Typeof(commonType{})).id);
	checkId(10, getTypeInfo(reflect.Typeof(fieldType{})).id);
}
