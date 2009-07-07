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

var id	uint32	// incremented for each new type we build
var typeLock	sync.Mutex	// set while building a type

type Type interface {
	id()	uint32;
	setId(id uint32);
	String()	string;
	safeString(seen map[uint32] bool)	string;
}
var types = make(map[reflect.Type] Type)

// Common elements of all types.
type commonType struct {
	name	string;
	_id	uint32;
}

func (t *commonType) id() uint32 {
	return t._id
}

func (t *commonType) setId(id uint32) {
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
var tBool Type
var tInt Type
var tUint Type
var tFloat Type
var tString Type
var tBytes Type

// Array type
type arrayType struct {
	commonType;
	Elem	Type;
	Len	int;
}

func newArrayType(name string, elem Type, length int) *arrayType {
	a := &arrayType{ commonType{ name: name }, elem, length };
	return a;
}

func (a *arrayType) safeString(seen map[uint32] bool) string {
	if _, ok := seen[a._id]; ok {
		return a.name
	}
	seen[a._id] = true;
	return fmt.Sprintf("[%d]%s", a.Len, a.Elem.safeString(seen));
}

func (a *arrayType) String() string {
	return a.safeString(make(map[uint32] bool))
}

// Slice type
type sliceType struct {
	commonType;
	Elem	Type;
}

func newSliceType(name string, elem Type) *sliceType {
	s := &sliceType{ commonType{ name: name }, elem };
	return s;
}

func (s *sliceType) safeString(seen map[uint32] bool) string {
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true;
	return fmt.Sprintf("[]%s", s.Elem.safeString(seen));
}

func (s *sliceType) String() string {
	return s.safeString(make(map[uint32] bool))
}

// Struct type
type fieldType struct {
	name	string;
	typ	Type;
}

type structType struct {
	commonType;
	field	[]*fieldType;
}

func (s *structType) safeString(seen map[uint32] bool) string {
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true;
	str := s.name + " = struct { ";
	for _, f := range s.field {
		str += fmt.Sprintf("%s %s; ", f.name, f.typ.safeString(seen));
	}
	str += "}";
	return str;
}

func (s *structType) String() string {
	return s.safeString(make(map[uint32] bool))
}

func newStructType(name string) *structType {
	s := &structType{ commonType{ name: name }, nil };
	return s;
}

// Construction
func newType(name string, rt reflect.Type) Type

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

func newTypeObject(name string, rt reflect.Type) Type {
	switch t := rt.(type) {
	// All basic types are easy: they are predefined.
	case *reflect.BoolType:
		return tBool

	case *reflect.IntType:
		return tInt
	case *reflect.Int32Type:
		return tInt
	case *reflect.Int64Type:
		return tInt

	case *reflect.UintType:
		return tUint
	case *reflect.Uint32Type:
		return tUint
	case *reflect.Uint64Type:
		return tUint

	case *reflect.FloatType:
		return tFloat
	case *reflect.Float32Type:
		return tFloat
	case *reflect.Float64Type:
		return tFloat

	case *reflect.StringType:
		return tString

	case *reflect.ArrayType:
		return newArrayType(name, newType("", t.Elem()), t.Len());

	case *reflect.SliceType:
		// []byte == []uint8 is a special case
		if _, ok := t.Elem().(*reflect.Uint8Type); ok {
			return tBytes
		}
		return newSliceType(name, newType("", t.Elem()));

	case *reflect.StructType:
		// Install the struct type itself before the fields so recursive
		// structures can be constructed safely.
		strType := newStructType(name);
		types[rt] = strType;
		field := make([]*fieldType, t.NumField());
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i);
			typ, _indir := indirect(f.Type);
			_pkg, tname := typ.Name();
			if tname == "" {
				tname = f.Type.String();
			}
			field[i] =  &fieldType{ f.Name, newType(tname, f.Type) };
		}
		strType.field = field;
		return strType;

	default:
		panicln("gob NewTypeObject can't handle type", rt.String());	// TODO(r): panic?
	}
	return nil
}

func newType(name string, rt reflect.Type) Type {
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
	id++;
	typ.setId(id);
	types[rt] = typ;
	return typ
}

// GetType returns the Gob type describing the interface value.
func GetType(name string, e interface{}) Type {
	rt := reflect.Typeof(e);
	// Set lock; all code running under here is synchronized.
	typeLock.Lock();
	t := newType(name, rt);
	typeLock.Unlock();
	return t;
}

// used for building the basic types; called only from init()
func bootstrapType(name string, e interface{}) Type {
	rt := reflect.Typeof(e);
	_, present := types[rt];
	if present {
		panicln("bootstrap type already present:", name);
	}
	typ := &commonType{ name: name };
	id++;
	typ.setId(id);
	types[rt] = typ;
	return typ
}

func init() {
	tBool= bootstrapType("bool", false);
	tInt = bootstrapType("int", int(0));
	tUint = bootstrapType("uint", uint(0));
	tFloat = bootstrapType("float", float64(0));
	// The string for tBytes is "bytes" not "[]byte" to signify its specialness.
	tBytes = bootstrapType("bytes", make([]byte, 0));
	tString= bootstrapType("string", "");
}
