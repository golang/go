// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"fmt"
	"os"
	"reflect"
	"sync"
)

// Reflection types are themselves interface values holding structs
// describing the type.  Each type has a different struct so that struct can
// be the kind.  For example, if typ is the reflect type for an int8, typ is
// a pointer to a reflect.Int8Type struct; if typ is the reflect type for a
// function, typ is a pointer to a reflect.FuncType struct; we use the type
// of that pointer as the kind.

// A typeId represents a gob Type as an integer that can be passed on the wire.
// Internally, typeIds are used as keys to a map to recover the underlying type info.
type typeId int32

var nextId typeId       // incremented for each new type we build
var typeLock sync.Mutex // set while building a type
const firstUserId = 64  // lowest id number granted to user

type gobType interface {
	id() typeId
	setId(id typeId)
	Name() string
	string() string // not public; only for debugging
	safeString(seen map[typeId]bool) string
}

var types = make(map[reflect.Type]gobType)
var idToType = make(map[typeId]gobType)
var builtinIdToType map[typeId]gobType // set in init() after builtins are established

func setTypeId(typ gobType) {
	nextId++
	typ.setId(nextId)
	idToType[nextId] = typ
}

func (t typeId) gobType() gobType {
	if t == 0 {
		return nil
	}
	return idToType[t]
}

// string returns the string representation of the type associated with the typeId.
func (t typeId) string() string {
	if t.gobType() == nil {
		return "<nil>"
	}
	return t.gobType().string()
}

// Name returns the name of the type associated with the typeId.
func (t typeId) Name() string {
	if t.gobType() == nil {
		return "<nil>"
	}
	return t.gobType().Name()
}

// Common elements of all types.
type commonType struct {
	name string
	_id  typeId
}

func (t *commonType) id() typeId { return t._id }

func (t *commonType) setId(id typeId) { t._id = id }

func (t *commonType) string() string { return t.name }

func (t *commonType) safeString(seen map[typeId]bool) string {
	return t.name
}

func (t *commonType) Name() string { return t.name }

// Create and check predefined types
// The string for tBytes is "bytes" not "[]byte" to signify its specialness.

var (
	// Primordial types, needed during initialization.
	tBool   = bootstrapType("bool", false, 1)
	tInt    = bootstrapType("int", int(0), 2)
	tUint   = bootstrapType("uint", uint(0), 3)
	tFloat  = bootstrapType("float", float64(0), 4)
	tBytes  = bootstrapType("bytes", make([]byte, 0), 5)
	tString = bootstrapType("string", "", 6)
	// Types added to the language later, not needed during initialization.
	tComplex typeId
)

// Predefined because it's needed by the Decoder
var tWireType = mustGetTypeInfo(reflect.Typeof(wireType{})).id

func init() {
	// Some magic numbers to make sure there are no surprises.
	checkId(7, tWireType)
	checkId(9, mustGetTypeInfo(reflect.Typeof(commonType{})).id)
	checkId(11, mustGetTypeInfo(reflect.Typeof(structType{})).id)
	checkId(12, mustGetTypeInfo(reflect.Typeof(fieldType{})).id)

	// Complex was added after gob was written, so appears after the
	// fundamental types are built.
	tComplex = bootstrapType("complex", 0+0i, 15)
	decIgnoreOpMap[tComplex] = ignoreTwoUints

	builtinIdToType = make(map[typeId]gobType)
	for k, v := range idToType {
		builtinIdToType[k] = v
	}

	// Move the id space upwards to allow for growth in the predefined world
	// without breaking existing files.
	if nextId > firstUserId {
		panic(fmt.Sprintln("nextId too large:", nextId))
	}
	nextId = firstUserId
}

// Array type
type arrayType struct {
	commonType
	Elem typeId
	Len  int
}

func newArrayType(name string, elem gobType, length int) *arrayType {
	a := &arrayType{commonType{name: name}, elem.id(), length}
	setTypeId(a)
	return a
}

func (a *arrayType) safeString(seen map[typeId]bool) string {
	if seen[a._id] {
		return a.name
	}
	seen[a._id] = true
	return fmt.Sprintf("[%d]%s", a.Len, a.Elem.gobType().safeString(seen))
}

func (a *arrayType) string() string { return a.safeString(make(map[typeId]bool)) }

// Map type
type mapType struct {
	commonType
	Key  typeId
	Elem typeId
}

func newMapType(name string, key, elem gobType) *mapType {
	m := &mapType{commonType{name: name}, key.id(), elem.id()}
	setTypeId(m)
	return m
}

func (m *mapType) safeString(seen map[typeId]bool) string {
	if seen[m._id] {
		return m.name
	}
	seen[m._id] = true
	key := m.Key.gobType().safeString(seen)
	elem := m.Elem.gobType().safeString(seen)
	return fmt.Sprintf("map[%s]%s", key, elem)
}

func (m *mapType) string() string { return m.safeString(make(map[typeId]bool)) }

// Slice type
type sliceType struct {
	commonType
	Elem typeId
}

func newSliceType(name string, elem gobType) *sliceType {
	s := &sliceType{commonType{name: name}, elem.id()}
	setTypeId(s)
	return s
}

func (s *sliceType) safeString(seen map[typeId]bool) string {
	if seen[s._id] {
		return s.name
	}
	seen[s._id] = true
	return fmt.Sprintf("[]%s", s.Elem.gobType().safeString(seen))
}

func (s *sliceType) string() string { return s.safeString(make(map[typeId]bool)) }

// Struct type
type fieldType struct {
	name string
	id   typeId
}

type structType struct {
	commonType
	field []*fieldType
}

func (s *structType) safeString(seen map[typeId]bool) string {
	if s == nil {
		return "<nil>"
	}
	if _, ok := seen[s._id]; ok {
		return s.name
	}
	seen[s._id] = true
	str := s.name + " = struct { "
	for _, f := range s.field {
		str += fmt.Sprintf("%s %s; ", f.name, f.id.gobType().safeString(seen))
	}
	str += "}"
	return str
}

func (s *structType) string() string { return s.safeString(make(map[typeId]bool)) }

func newStructType(name string) *structType {
	s := &structType{commonType{name: name}, nil}
	setTypeId(s)
	return s
}

// Step through the indirections on a type to discover the base type.
// Return the number of indirections.
func indirect(t reflect.Type) (rt reflect.Type, count int) {
	rt = t
	for {
		pt, ok := rt.(*reflect.PtrType)
		if !ok {
			break
		}
		rt = pt.Elem()
		count++
	}
	return
}

func newTypeObject(name string, rt reflect.Type) (gobType, os.Error) {
	switch t := rt.(type) {
	// All basic types are easy: they are predefined.
	case *reflect.BoolType:
		return tBool.gobType(), nil

	case *reflect.IntType:
		return tInt.gobType(), nil

	case *reflect.UintType:
		return tUint.gobType(), nil

	case *reflect.FloatType:
		return tFloat.gobType(), nil

	case *reflect.ComplexType:
		return tComplex.gobType(), nil

	case *reflect.StringType:
		return tString.gobType(), nil

	case *reflect.ArrayType:
		gt, err := getType("", t.Elem())
		if err != nil {
			return nil, err
		}
		return newArrayType(name, gt, t.Len()), nil

	case *reflect.MapType:
		kt, err := getType("", t.Key())
		if err != nil {
			return nil, err
		}
		vt, err := getType("", t.Elem())
		if err != nil {
			return nil, err
		}
		return newMapType(name, kt, vt), nil

	case *reflect.SliceType:
		// []byte == []uint8 is a special case
		if t.Elem().Kind() == reflect.Uint8 {
			return tBytes.gobType(), nil
		}
		gt, err := getType(t.Elem().Name(), t.Elem())
		if err != nil {
			return nil, err
		}
		return newSliceType(name, gt), nil

	case *reflect.StructType:
		// Install the struct type itself before the fields so recursive
		// structures can be constructed safely.
		strType := newStructType(name)
		types[rt] = strType
		idToType[strType.id()] = strType
		field := make([]*fieldType, t.NumField())
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			typ, _ := indirect(f.Type)
			tname := typ.Name()
			if tname == "" {
				tname = f.Type.String()
			}
			gt, err := getType(tname, f.Type)
			if err != nil {
				return nil, err
			}
			field[i] = &fieldType{f.Name, gt.id()}
		}
		strType.field = field
		return strType, nil

	default:
		return nil, os.ErrorString("gob NewTypeObject can't handle type: " + rt.String())
	}
	return nil, nil
}

// getType returns the Gob type describing the given reflect.Type.
// typeLock must be held.
func getType(name string, rt reflect.Type) (gobType, os.Error) {
	// Flatten the data structure by collapsing out pointers
	for {
		pt, ok := rt.(*reflect.PtrType)
		if !ok {
			break
		}
		rt = pt.Elem()
	}
	typ, present := types[rt]
	if present {
		return typ, nil
	}
	typ, err := newTypeObject(name, rt)
	if err == nil {
		types[rt] = typ
	}
	return typ, err
}

func checkId(want, got typeId) {
	if want != got {
		panic("bootstrap type wrong id: " + got.Name() + " " + got.string() + " not " + want.string())
	}
}

// used for building the basic types; called only from init()
func bootstrapType(name string, e interface{}, expect typeId) typeId {
	rt := reflect.Typeof(e)
	_, present := types[rt]
	if present {
		panic("bootstrap type already present: " + name + ", " + rt.String())
	}
	typ := &commonType{name: name}
	types[rt] = typ
	setTypeId(typ)
	checkId(expect, nextId)
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
// To maintain binary compatibility, if you extend this type, always put
// the new fields last.
type wireType struct {
	arrayT  *arrayType
	sliceT  *sliceType
	structT *structType
	mapT    *mapType
}

func (w *wireType) name() string {
	if w.structT != nil {
		return w.structT.name
	}
	return "unknown"
}

type typeInfo struct {
	id      typeId
	encoder *encEngine
	wire    *wireType
}

var typeInfoMap = make(map[reflect.Type]*typeInfo) // protected by typeLock

// The reflection type must have all its indirections processed out.
// typeLock must be held.
func getTypeInfo(rt reflect.Type) (*typeInfo, os.Error) {
	if rt.Kind() == reflect.Ptr {
		panic("pointer type in getTypeInfo: " + rt.String())
	}
	info, ok := typeInfoMap[rt]
	if !ok {
		info = new(typeInfo)
		name := rt.Name()
		gt, err := getType(name, rt)
		if err != nil {
			return nil, err
		}
		info.id = gt.id()
		t := info.id.gobType()
		switch typ := rt.(type) {
		case *reflect.ArrayType:
			info.wire = &wireType{arrayT: t.(*arrayType)}
		case *reflect.MapType:
			info.wire = &wireType{mapT: t.(*mapType)}
		case *reflect.SliceType:
			// []byte == []uint8 is a special case handled separately
			if typ.Elem().Kind() != reflect.Uint8 {
				info.wire = &wireType{sliceT: t.(*sliceType)}
			}
		case *reflect.StructType:
			info.wire = &wireType{structT: t.(*structType)}
		}
		typeInfoMap[rt] = info
	}
	return info, nil
}

// Called only when a panic is acceptable and unexpected.
func mustGetTypeInfo(rt reflect.Type) *typeInfo {
	t, err := getTypeInfo(rt)
	if err != nil {
		panic("getTypeInfo: " + err.String())
	}
	return t
}
