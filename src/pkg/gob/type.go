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

// Types are identified by an integer TypeId.  These can be passed on the wire.
// Internally, they are used as keys to a map to recover the underlying type info.
type TypeId int32

var nextId	TypeId	// incremented for each new type we build
var typeLock	sync.Mutex	// set while building a type

type gobType interface {
	id()	TypeId;
	setId(id TypeId);
	String()	string;
	safeString(seen map[TypeId] bool)	string;
}

var types = make(map[reflect.Type] gobType)
var idToType = make(map[TypeId] gobType)

func setTypeId(typ gobType) {
	nextId++;
	typ.setId(nextId);
	idToType[nextId] = typ;
}

func (t TypeId) gobType() gobType {
	if t == 0 {
		return nil
	}
	return idToType[t]
}

func (t TypeId) String() string {
	return t.gobType().String()
}

// Common elements of all types.
type commonType struct {
	name	string;
	_id	TypeId;
}

func (t *commonType) id() TypeId {
	return t._id
}

func (t *commonType) setId(id TypeId) {
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
var tBool TypeId
var tInt TypeId
var tUint TypeId
var tFloat TypeId
var tString TypeId
var tBytes TypeId

// Predefined because it's needed by the Decoder
var tWireType TypeId

// Array type
type arrayType struct {
	commonType;
	Elem	TypeId;
	Len	int;
}

func newArrayType(name string, elem gobType, length int) *arrayType {
	a := &arrayType{ commonType{ name: name }, elem.id(), length };
	setTypeId(a);
	return a;
}

func (a *arrayType) safeString(seen map[TypeId] bool) string {
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
	Elem	TypeId;
}

func newSliceType(name string, elem gobType) *sliceType {
	s := &sliceType{ commonType{ name: name }, elem.id() };
	setTypeId(s);
	return s;
}

func (s *sliceType) safeString(seen map[TypeId] bool) string {
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true;
	return fmt.Sprintf("[]%s", s.Elem.gobType().safeString(seen));
}

func (s *sliceType) String() string {
	return s.safeString(make(map[TypeId] bool))
}

// Struct type
type fieldType struct {
	name	string;
	typeId	TypeId;
}

type structType struct {
	commonType;
	field	[]*fieldType;
}

func (s *structType) safeString(seen map[TypeId] bool) string {
	if s == nil {
		return "<nil>"
	}
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true;
	str := s.name + " = struct { ";
	for _, f := range s.field {
		str += fmt.Sprintf("%s %s; ", f.name, f.typeId.gobType().safeString(seen));
	}
	str += "}";
	return str;
}

func (s *structType) String() string {
	return s.safeString(make(map[TypeId] bool))
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
		return newSliceType(name, newType("", t.Elem()));

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
			_pkg, tname := typ.Name();
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

// used for building the basic types; called only from init()
func bootstrapType(name string, e interface{}) TypeId {
	rt := reflect.Typeof(e);
	_, present := types[rt];
	if present {
		panicln("bootstrap type already present:", name);
	}
	typ := &commonType{ name: name };
	types[rt] = typ;
	setTypeId(typ);
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
	typeId	TypeId;
	// Decoder engine to convert TypeId.Type() to this type.  Stored as a pointer to a
	// pointer to aid construction of recursive types.  Protected by typeLock.
	decoderPtr	map[TypeId] **decEngine;
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
		path, name := rt.Name();
		info.typeId = getType(name, rt).id();
		info.decoderPtr = make(map[TypeId] **decEngine);
		// assume it's a struct type
		info.wire = &wireType{info.typeId.gobType().(*structType)};
		typeInfoMap[rt] = info;
	}
	return info;
}

func init() {
	tBool = bootstrapType("bool", false);
	tInt = bootstrapType("int", int(0));
	tUint = bootstrapType("uint", uint(0));
	tFloat = bootstrapType("float", float64(0));
	// The string for tBytes is "bytes" not "[]byte" to signify its specialness.
	tBytes = bootstrapType("bytes", make([]byte, 0));
	tString= bootstrapType("string", "");
	tWireType = getTypeInfo(reflect.Typeof(wireType{})).typeId;
}
