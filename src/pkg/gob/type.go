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
	name() string
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
func (t typeId) name() string {
	if t.gobType() == nil {
		return "<nil>"
	}
	return t.gobType().name()
}

// Common elements of all types.
type CommonType struct {
	Name string
	Id   typeId
}

func (t *CommonType) id() typeId { return t.Id }

func (t *CommonType) setId(id typeId) { t.Id = id }

func (t *CommonType) string() string { return t.Name }

func (t *CommonType) safeString(seen map[typeId]bool) string {
	return t.Name
}

func (t *CommonType) name() string { return t.Name }

// Create and check predefined types
// The string for tBytes is "bytes" not "[]byte" to signify its specialness.

var (
	// Primordial types, needed during initialization.
	tBool      = bootstrapType("bool", false, 1)
	tInt       = bootstrapType("int", int(0), 2)
	tUint      = bootstrapType("uint", uint(0), 3)
	tFloat     = bootstrapType("float", 0.0, 4)
	tBytes     = bootstrapType("bytes", make([]byte, 0), 5)
	tString    = bootstrapType("string", "", 6)
	tComplex   = bootstrapType("complex", 0+0i, 7)
	tInterface = bootstrapType("interface", interface{}(nil), 8)
	// Reserve some Ids for compatible expansion
	tReserved7 = bootstrapType("_reserved1", struct{ r7 int }{}, 9)
	tReserved6 = bootstrapType("_reserved1", struct{ r6 int }{}, 10)
	tReserved5 = bootstrapType("_reserved1", struct{ r5 int }{}, 11)
	tReserved4 = bootstrapType("_reserved1", struct{ r4 int }{}, 12)
	tReserved3 = bootstrapType("_reserved1", struct{ r3 int }{}, 13)
	tReserved2 = bootstrapType("_reserved1", struct{ r2 int }{}, 14)
	tReserved1 = bootstrapType("_reserved1", struct{ r1 int }{}, 15)
)

// Predefined because it's needed by the Decoder
var tWireType = mustGetTypeInfo(reflect.Typeof(wireType{})).id

func init() {
	// Some magic numbers to make sure there are no surprises.
	checkId(16, tWireType)
	checkId(17, mustGetTypeInfo(reflect.Typeof(arrayType{})).id)
	checkId(18, mustGetTypeInfo(reflect.Typeof(CommonType{})).id)
	checkId(19, mustGetTypeInfo(reflect.Typeof(sliceType{})).id)
	checkId(20, mustGetTypeInfo(reflect.Typeof(structType{})).id)
	checkId(21, mustGetTypeInfo(reflect.Typeof(fieldType{})).id)
	checkId(23, mustGetTypeInfo(reflect.Typeof(mapType{})).id)

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
	registerBasics()
}

// Array type
type arrayType struct {
	CommonType
	Elem typeId
	Len  int
}

func newArrayType(name string, elem gobType, length int) *arrayType {
	a := &arrayType{CommonType{Name: name}, elem.id(), length}
	setTypeId(a)
	return a
}

func (a *arrayType) safeString(seen map[typeId]bool) string {
	if seen[a.Id] {
		return a.Name
	}
	seen[a.Id] = true
	return fmt.Sprintf("[%d]%s", a.Len, a.Elem.gobType().safeString(seen))
}

func (a *arrayType) string() string { return a.safeString(make(map[typeId]bool)) }

// Map type
type mapType struct {
	CommonType
	Key  typeId
	Elem typeId
}

func newMapType(name string, key, elem gobType) *mapType {
	m := &mapType{CommonType{Name: name}, key.id(), elem.id()}
	setTypeId(m)
	return m
}

func (m *mapType) safeString(seen map[typeId]bool) string {
	if seen[m.Id] {
		return m.Name
	}
	seen[m.Id] = true
	key := m.Key.gobType().safeString(seen)
	elem := m.Elem.gobType().safeString(seen)
	return fmt.Sprintf("map[%s]%s", key, elem)
}

func (m *mapType) string() string { return m.safeString(make(map[typeId]bool)) }

// Slice type
type sliceType struct {
	CommonType
	Elem typeId
}

func newSliceType(name string, elem gobType) *sliceType {
	s := &sliceType{CommonType{Name: name}, elem.id()}
	setTypeId(s)
	return s
}

func (s *sliceType) safeString(seen map[typeId]bool) string {
	if seen[s.Id] {
		return s.Name
	}
	seen[s.Id] = true
	return fmt.Sprintf("[]%s", s.Elem.gobType().safeString(seen))
}

func (s *sliceType) string() string { return s.safeString(make(map[typeId]bool)) }

// Struct type
type fieldType struct {
	Name string
	Id   typeId
}

type structType struct {
	CommonType
	Field []*fieldType
}

func (s *structType) safeString(seen map[typeId]bool) string {
	if s == nil {
		return "<nil>"
	}
	if _, ok := seen[s.Id]; ok {
		return s.Name
	}
	seen[s.Id] = true
	str := s.Name + " = struct { "
	for _, f := range s.Field {
		str += fmt.Sprintf("%s %s; ", f.Name, f.Id.gobType().safeString(seen))
	}
	str += "}"
	return str
}

func (s *structType) string() string { return s.safeString(make(map[typeId]bool)) }

func newStructType(name string) *structType {
	s := &structType{CommonType{Name: name}, nil}
	setTypeId(s)
	return s
}

// Step through the indirections on a type to discover the base type.
// Return the base type and the number of indirections.
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

	case *reflect.InterfaceType:
		return tInterface.gobType(), nil

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
				t, _ := indirect(f.Type)
				tname = t.String()
			}
			gt, err := getType(tname, f.Type)
			if err != nil {
				return nil, err
			}
			field[i] = &fieldType{f.Name, gt.id()}
		}
		strType.Field = field
		return strType, nil

	default:
		return nil, os.ErrorString("gob NewTypeObject can't handle type: " + rt.String())
	}
	return nil, nil
}

// getType returns the Gob type describing the given reflect.Type.
// typeLock must be held.
func getType(name string, rt reflect.Type) (gobType, os.Error) {
	rt, _ = indirect(rt)
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
		fmt.Fprintf(os.Stderr, "checkId: %d should be %d\n", int(want), int(got))
		panic("bootstrap type wrong id: " + got.name() + " " + got.string() + " not " + want.string())
	}
}

// used for building the basic types; called only from init()
func bootstrapType(name string, e interface{}, expect typeId) typeId {
	rt := reflect.Typeof(e)
	_, present := types[rt]
	if present {
		panic("bootstrap type already present: " + name + ", " + rt.String())
	}
	typ := &CommonType{Name: name}
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
	ArrayT  *arrayType
	SliceT  *sliceType
	StructT *structType
	MapT    *mapType
}

func (w *wireType) string() string {
	const unknown = "unknown type"
	if w == nil {
		return unknown
	}
	switch {
	case w.ArrayT != nil:
		return w.ArrayT.Name
	case w.SliceT != nil:
		return w.SliceT.Name
	case w.StructT != nil:
		return w.StructT.Name
	case w.MapT != nil:
		return w.MapT.Name
	}
	return unknown
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
			info.wire = &wireType{ArrayT: t.(*arrayType)}
		case *reflect.MapType:
			info.wire = &wireType{MapT: t.(*mapType)}
		case *reflect.SliceType:
			// []byte == []uint8 is a special case handled separately
			if typ.Elem().Kind() != reflect.Uint8 {
				info.wire = &wireType{SliceT: t.(*sliceType)}
			}
		case *reflect.StructType:
			info.wire = &wireType{StructT: t.(*structType)}
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

var (
	nameToConcreteType = make(map[string]reflect.Type)
	concreteTypeToName = make(map[reflect.Type]string)
)

// RegisterName is like Register but uses the provided name rather than the
// type's default.
func RegisterName(name string, value interface{}) {
	if name == "" {
		// reserved for nil
		panic("attempt to register empty name")
	}
	rt, _ := indirect(reflect.Typeof(value))
	// Check for incompatible duplicates.
	if t, ok := nameToConcreteType[name]; ok && t != rt {
		panic("gob: registering duplicate types for " + name)
	}
	if n, ok := concreteTypeToName[rt]; ok && n != name {
		panic("gob: registering duplicate names for " + rt.String())
	}
	// Store the name and type provided by the user....
	nameToConcreteType[name] = reflect.Typeof(value)
	// but the flattened type in the type table, since that's what decode needs.
	concreteTypeToName[rt] = name
}

// Register records a type, identified by a value for that type, under its
// internal type name.  That name will identify the concrete type of a value
// sent or received as an interface variable.  Only types that will be
// transferred as implementations of interface values need to be registered.
// Expecting to be used only during initialization, it panics if the mapping
// between types and names is not a bijection.
func Register(value interface{}) {
	// Default to printed representation for unnamed types
	rt := reflect.Typeof(value)
	name := rt.String()

	// But for named types (or pointers to them), qualify with import path.
	// Dereference one pointer looking for a named type.
	star := ""
	if rt.Name() == "" {
		if pt, ok := rt.(*reflect.PtrType); ok {
			star = "*"
			rt = pt
		}
	}
	if rt.Name() != "" {
		if rt.PkgPath() == "" {
			name = star + rt.Name()
		} else {
			name = star + rt.PkgPath() + "." + rt.Name()
		}
	}

	RegisterName(name, value)
}

func registerBasics() {
	Register(int(0))
	Register(int8(0))
	Register(int16(0))
	Register(int32(0))
	Register(int64(0))
	Register(uint(0))
	Register(uint8(0))
	Register(uint16(0))
	Register(uint32(0))
	Register(uint64(0))
	Register(float32(0))
	Register(0.0)
	Register(complex64(0i))
	Register(complex128(0i))
	Register(false)
	Register("")
	Register([]byte(nil))
}
