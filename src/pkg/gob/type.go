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

// userTypeInfo stores the information associated with a type the user has handed
// to the package.  It's computed once and stored in a map keyed by reflection
// type.
type userTypeInfo struct {
	user  reflect.Type // the type the user handed us
	base  reflect.Type // the base type after all indirections
	indir int          // number of indirections to reach the base type
}

var (
	// Protected by an RWMutex because we read it a lot and write
	// it only when we see a new type, typically when compiling.
	userTypeLock  sync.RWMutex
	userTypeCache = make(map[reflect.Type]*userTypeInfo)
)

// validType returns, and saves, the information associated with user-provided type rt.
// If the user type is not valid, err will be non-nil.  To be used when the error handler
// is not set up.
func validUserType(rt reflect.Type) (ut *userTypeInfo, err os.Error) {
	userTypeLock.RLock()
	ut = userTypeCache[rt]
	userTypeLock.RUnlock()
	if ut != nil {
		return
	}
	// Now set the value under the write lock.
	userTypeLock.Lock()
	defer userTypeLock.Unlock()
	if ut = userTypeCache[rt]; ut != nil {
		// Lost the race; not a problem.
		return
	}
	ut = new(userTypeInfo)
	ut.base = rt
	ut.user = rt
	// A type that is just a cycle of pointers (such as type T *T) cannot
	// be represented in gobs, which need some concrete data.  We use a
	// cycle detection algorithm from Knuth, Vol 2, Section 3.1, Ex 6,
	// pp 539-540.  As we step through indirections, run another type at
	// half speed. If they meet up, there's a cycle.
	slowpoke := ut.base // walks half as fast as ut.base
	for {
		pt, ok := ut.base.(*reflect.PtrType)
		if !ok {
			break
		}
		ut.base = pt.Elem()
		if ut.base == slowpoke { // ut.base lapped slowpoke
			// recursive pointer type.
			return nil, os.ErrorString("can't represent recursive pointer type " + ut.base.String())
		}
		if ut.indir%2 == 0 {
			slowpoke = slowpoke.(*reflect.PtrType).Elem()
		}
		ut.indir++
	}
	userTypeCache[rt] = ut
	return
}

// userType returns, and saves, the information associated with user-provided type rt.
// If the user type is not valid, it calls error.
func userType(rt reflect.Type) *userTypeInfo {
	ut, err := validUserType(rt)
	if err != nil {
		error(err)
	}
	return ut
}
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
	tFloat     = bootstrapType("float", float64(0), 4)
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
var wireTypeUserInfo *userTypeInfo // userTypeInfo of (*wireType)

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
	wireTypeUserInfo = userType(reflect.Typeof((*wireType)(nil)))
}

// Array type
type arrayType struct {
	CommonType
	Elem typeId
	Len  int
}

func newArrayType(name string) *arrayType {
	a := &arrayType{CommonType{Name: name}, 0, 0}
	return a
}

func (a *arrayType) init(elem gobType, len int) {
	// Set our type id before evaluating the element's, in case it's our own.
	setTypeId(a)
	a.Elem = elem.id()
	a.Len = len
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

func newMapType(name string) *mapType {
	m := &mapType{CommonType{Name: name}, 0, 0}
	return m
}

func (m *mapType) init(key, elem gobType) {
	// Set our type id before evaluating the element's, in case it's our own.
	setTypeId(m)
	m.Key = key.id()
	m.Elem = elem.id()
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

func newSliceType(name string) *sliceType {
	s := &sliceType{CommonType{Name: name}, 0}
	return s
}

func (s *sliceType) init(elem gobType) {
	// Set our type id before evaluating the element's, in case it's our own.
	setTypeId(s)
	s.Elem = elem.id()
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
	// For historical reasons we set the id here rather than init.
	// Se the comment in newTypeObject for details.
	setTypeId(s)
	return s
}

func (s *structType) init(field []*fieldType) {
	s.Field = field
}

func newTypeObject(name string, rt reflect.Type) (gobType, os.Error) {
	var err os.Error
	var type0, type1 gobType
	defer func() {
		if err != nil {
			types[rt] = nil, false
		}
	}()
	// Install the top-level type before the subtypes (e.g. struct before
	// fields) so recursive types can be constructed safely.
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
		at := newArrayType(name)
		types[rt] = at
		type0, err = getType("", t.Elem())
		if err != nil {
			return nil, err
		}
		// Historical aside:
		// For arrays, maps, and slices, we set the type id after the elements
		// are constructed. This is to retain the order of type id allocation after
		// a fix made to handle recursive types, which changed the order in
		// which types are built.  Delaying the setting in this way preserves
		// type ids while allowing recursive types to be described. Structs,
		// done below, were already handling recursion correctly so they
		// assign the top-level id before those of the field.
		at.init(type0, t.Len())
		return at, nil

	case *reflect.MapType:
		mt := newMapType(name)
		types[rt] = mt
		type0, err = getType("", t.Key())
		if err != nil {
			return nil, err
		}
		type1, err = getType("", t.Elem())
		if err != nil {
			return nil, err
		}
		mt.init(type0, type1)
		return mt, nil

	case *reflect.SliceType:
		// []byte == []uint8 is a special case
		if t.Elem().Kind() == reflect.Uint8 {
			return tBytes.gobType(), nil
		}
		st := newSliceType(name)
		types[rt] = st
		type0, err = getType(t.Elem().Name(), t.Elem())
		if err != nil {
			return nil, err
		}
		st.init(type0)
		return st, nil

	case *reflect.StructType:
		st := newStructType(name)
		types[rt] = st
		idToType[st.id()] = st
		field := make([]*fieldType, t.NumField())
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			typ := userType(f.Type).base
			tname := typ.Name()
			if tname == "" {
				t := userType(f.Type).base
				tname = t.String()
			}
			gt, err := getType(tname, f.Type)
			if err != nil {
				return nil, err
			}
			field[i] = &fieldType{f.Name, gt.id()}
		}
		st.init(field)
		return st, nil

	default:
		return nil, os.ErrorString("gob NewTypeObject can't handle type: " + rt.String())
	}
	return nil, nil
}

// getType returns the Gob type describing the given reflect.Type.
// typeLock must be held.
func getType(name string, rt reflect.Type) (gobType, os.Error) {
	rt = userType(rt).base
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
	userType(rt) // might as well cache it now
	return nextId
}

// Representation of the information we send and receive about this type.
// Each value we send is preceded by its type definition: an encoded int.
// However, the very first time we send the value, we first send the pair
// (-id, wireType).
// For bootstrapping purposes, we assume that the recipient knows how
// to decode a wireType; it is exactly the wireType struct here, interpreted
// using the gob rules for sending a structure, except that we assume the
// ids for wireType and structType etc. are known.  The relevant pieces
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
	base := userType(reflect.Typeof(value)).base
	// Check for incompatible duplicates.
	if t, ok := nameToConcreteType[name]; ok && t != base {
		panic("gob: registering duplicate types for " + name)
	}
	if n, ok := concreteTypeToName[base]; ok && n != name {
		panic("gob: registering duplicate names for " + base.String())
	}
	// Store the name and type provided by the user....
	nameToConcreteType[name] = reflect.Typeof(value)
	// but the flattened type in the type table, since that's what decode needs.
	concreteTypeToName[base] = name
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
	Register(float64(0))
	Register(complex64(0i))
	Register(complex128(0i))
	Register(false)
	Register("")
	Register([]byte(nil))
}
