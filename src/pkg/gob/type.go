// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gob

import (
	"fmt"
	"os"
	"reflect"
	"sync"
	"unicode"
	"utf8"
)

// userTypeInfo stores the information associated with a type the user has handed
// to the package.  It's computed once and stored in a map keyed by reflection
// type.
type userTypeInfo struct {
	user         reflect.Type // the type the user handed us
	base         reflect.Type // the base type after all indirections
	indir        int          // number of indirections to reach the base type
	isGobEncoder bool         // does the type implement GobEncoder?
	isGobDecoder bool         // does the type implement GobDecoder?
	encIndir     int8         // number of indirections to reach the receiver type; may be negative
	decIndir     int8         // number of indirections to reach the receiver type; may be negative
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
		pt := ut.base
		if pt.Kind() != reflect.Ptr {
			break
		}
		ut.base = pt.Elem()
		if ut.base == slowpoke { // ut.base lapped slowpoke
			// recursive pointer type.
			return nil, os.NewError("can't represent recursive pointer type " + ut.base.String())
		}
		if ut.indir%2 == 0 {
			slowpoke = slowpoke.Elem()
		}
		ut.indir++
	}
	ut.isGobEncoder, ut.encIndir = implementsInterface(ut.user, gobEncoderInterfaceType)
	ut.isGobDecoder, ut.decIndir = implementsInterface(ut.user, gobDecoderInterfaceType)
	userTypeCache[rt] = ut
	return
}

var (
	gobEncoderInterfaceType = reflect.TypeOf((*GobEncoder)(nil)).Elem()
	gobDecoderInterfaceType = reflect.TypeOf((*GobDecoder)(nil)).Elem()
)

// implementsInterface reports whether the type implements the
// gobEncoder/gobDecoder interface.
// It also returns the number of indirections required to get to the
// implementation.
func implementsInterface(typ, gobEncDecType reflect.Type) (success bool, indir int8) {
	if typ == nil {
		return
	}
	rt := typ
	// The type might be a pointer and we need to keep
	// dereferencing to the base type until we find an implementation.
	for {
		if rt.Implements(gobEncDecType) {
			return true, indir
		}
		if p := rt; p.Kind() == reflect.Ptr {
			indir++
			if indir > 100 { // insane number of indirections
				return false, 0
			}
			rt = p.Elem()
			continue
		}
		break
	}
	// No luck yet, but if this is a base type (non-pointer), the pointer might satisfy.
	if typ.Kind() != reflect.Ptr {
		// Not a pointer, but does the pointer work?
		if reflect.PtrTo(typ).Implements(gobEncDecType) {
			return true, -1
		}
	}
	return false, 0
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
	// Always passed as pointers so the interface{} type
	// goes through without losing its interfaceness.
	tBool      = bootstrapType("bool", (*bool)(nil), 1)
	tInt       = bootstrapType("int", (*int)(nil), 2)
	tUint      = bootstrapType("uint", (*uint)(nil), 3)
	tFloat     = bootstrapType("float", (*float64)(nil), 4)
	tBytes     = bootstrapType("bytes", (*[]byte)(nil), 5)
	tString    = bootstrapType("string", (*string)(nil), 6)
	tComplex   = bootstrapType("complex", (*complex128)(nil), 7)
	tInterface = bootstrapType("interface", (*interface{})(nil), 8)
	// Reserve some Ids for compatible expansion
	tReserved7 = bootstrapType("_reserved1", (*struct{ r7 int })(nil), 9)
	tReserved6 = bootstrapType("_reserved1", (*struct{ r6 int })(nil), 10)
	tReserved5 = bootstrapType("_reserved1", (*struct{ r5 int })(nil), 11)
	tReserved4 = bootstrapType("_reserved1", (*struct{ r4 int })(nil), 12)
	tReserved3 = bootstrapType("_reserved1", (*struct{ r3 int })(nil), 13)
	tReserved2 = bootstrapType("_reserved1", (*struct{ r2 int })(nil), 14)
	tReserved1 = bootstrapType("_reserved1", (*struct{ r1 int })(nil), 15)
)

// Predefined because it's needed by the Decoder
var tWireType = mustGetTypeInfo(reflect.TypeOf(wireType{})).id
var wireTypeUserInfo *userTypeInfo // userTypeInfo of (*wireType)

func init() {
	// Some magic numbers to make sure there are no surprises.
	checkId(16, tWireType)
	checkId(17, mustGetTypeInfo(reflect.TypeOf(arrayType{})).id)
	checkId(18, mustGetTypeInfo(reflect.TypeOf(CommonType{})).id)
	checkId(19, mustGetTypeInfo(reflect.TypeOf(sliceType{})).id)
	checkId(20, mustGetTypeInfo(reflect.TypeOf(structType{})).id)
	checkId(21, mustGetTypeInfo(reflect.TypeOf(fieldType{})).id)
	checkId(23, mustGetTypeInfo(reflect.TypeOf(mapType{})).id)

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
	wireTypeUserInfo = userType(reflect.TypeOf((*wireType)(nil)))
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

// GobEncoder type (something that implements the GobEncoder interface)
type gobEncoderType struct {
	CommonType
}

func newGobEncoderType(name string) *gobEncoderType {
	g := &gobEncoderType{CommonType{Name: name}}
	setTypeId(g)
	return g
}

func (g *gobEncoderType) safeString(seen map[typeId]bool) string {
	return g.Name
}

func (g *gobEncoderType) string() string { return g.Name }

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
	// See the comment in newTypeObject for details.
	setTypeId(s)
	return s
}

// newTypeObject allocates a gobType for the reflection type rt.
// Unless ut represents a GobEncoder, rt should be the base type
// of ut.
// This is only called from the encoding side. The decoding side
// works through typeIds and userTypeInfos alone.
func newTypeObject(name string, ut *userTypeInfo, rt reflect.Type) (gobType, os.Error) {
	// Does this type implement GobEncoder?
	if ut.isGobEncoder {
		return newGobEncoderType(name), nil
	}
	var err os.Error
	var type0, type1 gobType
	defer func() {
		if err != nil {
			delete(types, rt)
		}
	}()
	// Install the top-level type before the subtypes (e.g. struct before
	// fields) so recursive types can be constructed safely.
	switch t := rt; t.Kind() {
	// All basic types are easy: they are predefined.
	case reflect.Bool:
		return tBool.gobType(), nil

	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return tInt.gobType(), nil

	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return tUint.gobType(), nil

	case reflect.Float32, reflect.Float64:
		return tFloat.gobType(), nil

	case reflect.Complex64, reflect.Complex128:
		return tComplex.gobType(), nil

	case reflect.String:
		return tString.gobType(), nil

	case reflect.Interface:
		return tInterface.gobType(), nil

	case reflect.Array:
		at := newArrayType(name)
		types[rt] = at
		type0, err = getBaseType("", t.Elem())
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

	case reflect.Map:
		mt := newMapType(name)
		types[rt] = mt
		type0, err = getBaseType("", t.Key())
		if err != nil {
			return nil, err
		}
		type1, err = getBaseType("", t.Elem())
		if err != nil {
			return nil, err
		}
		mt.init(type0, type1)
		return mt, nil

	case reflect.Slice:
		// []byte == []uint8 is a special case
		if t.Elem().Kind() == reflect.Uint8 {
			return tBytes.gobType(), nil
		}
		st := newSliceType(name)
		types[rt] = st
		type0, err = getBaseType(t.Elem().Name(), t.Elem())
		if err != nil {
			return nil, err
		}
		st.init(type0)
		return st, nil

	case reflect.Struct:
		st := newStructType(name)
		types[rt] = st
		idToType[st.id()] = st
		for i := 0; i < t.NumField(); i++ {
			f := t.Field(i)
			if !isExported(f.Name) {
				continue
			}
			typ := userType(f.Type).base
			tname := typ.Name()
			if tname == "" {
				t := userType(f.Type).base
				tname = t.String()
			}
			gt, err := getBaseType(tname, f.Type)
			if err != nil {
				return nil, err
			}
			st.Field = append(st.Field, &fieldType{f.Name, gt.id()})
		}
		return st, nil

	default:
		return nil, os.NewError("gob NewTypeObject can't handle type: " + rt.String())
	}
	return nil, nil
}

// isExported reports whether this is an exported - upper case - name.
func isExported(name string) bool {
	rune, _ := utf8.DecodeRuneInString(name)
	return unicode.IsUpper(rune)
}

// getBaseType returns the Gob type describing the given reflect.Type's base type.
// typeLock must be held.
func getBaseType(name string, rt reflect.Type) (gobType, os.Error) {
	ut := userType(rt)
	return getType(name, ut, ut.base)
}

// getType returns the Gob type describing the given reflect.Type.
// Should be called only when handling GobEncoders/Decoders,
// which may be pointers.  All other types are handled through the
// base type, never a pointer.
// typeLock must be held.
func getType(name string, ut *userTypeInfo, rt reflect.Type) (gobType, os.Error) {
	typ, present := types[rt]
	if present {
		return typ, nil
	}
	typ, err := newTypeObject(name, ut, rt)
	if err == nil {
		types[rt] = typ
	}
	return typ, err
}

func checkId(want, got typeId) {
	if want != got {
		fmt.Fprintf(os.Stderr, "checkId: %d should be %d\n", int(got), int(want))
		panic("bootstrap type wrong id: " + got.name() + " " + got.string() + " not " + want.string())
	}
}

// used for building the basic types; called only from init().  the incoming
// interface always refers to a pointer.
func bootstrapType(name string, e interface{}, expect typeId) typeId {
	rt := reflect.TypeOf(e).Elem()
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
	ArrayT      *arrayType
	SliceT      *sliceType
	StructT     *structType
	MapT        *mapType
	GobEncoderT *gobEncoderType
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
	case w.GobEncoderT != nil:
		return w.GobEncoderT.Name
	}
	return unknown
}

type typeInfo struct {
	id      typeId
	encoder *encEngine
	wire    *wireType
}

var typeInfoMap = make(map[reflect.Type]*typeInfo) // protected by typeLock

// typeLock must be held.
func getTypeInfo(ut *userTypeInfo) (*typeInfo, os.Error) {
	rt := ut.base
	if ut.isGobEncoder {
		// We want the user type, not the base type.
		rt = ut.user
	}
	info, ok := typeInfoMap[rt]
	if ok {
		return info, nil
	}
	info = new(typeInfo)
	gt, err := getBaseType(rt.Name(), rt)
	if err != nil {
		return nil, err
	}
	info.id = gt.id()

	if ut.isGobEncoder {
		userType, err := getType(rt.Name(), ut, rt)
		if err != nil {
			return nil, err
		}
		info.wire = &wireType{GobEncoderT: userType.id().gobType().(*gobEncoderType)}
		typeInfoMap[ut.user] = info
		return info, nil
	}

	t := info.id.gobType()
	switch typ := rt; typ.Kind() {
	case reflect.Array:
		info.wire = &wireType{ArrayT: t.(*arrayType)}
	case reflect.Map:
		info.wire = &wireType{MapT: t.(*mapType)}
	case reflect.Slice:
		// []byte == []uint8 is a special case handled separately
		if typ.Elem().Kind() != reflect.Uint8 {
			info.wire = &wireType{SliceT: t.(*sliceType)}
		}
	case reflect.Struct:
		info.wire = &wireType{StructT: t.(*structType)}
	}
	typeInfoMap[rt] = info
	return info, nil
}

// Called only when a panic is acceptable and unexpected.
func mustGetTypeInfo(rt reflect.Type) *typeInfo {
	t, err := getTypeInfo(userType(rt))
	if err != nil {
		panic("getTypeInfo: " + err.String())
	}
	return t
}

// GobEncoder is the interface describing data that provides its own
// representation for encoding values for transmission to a GobDecoder.
// A type that implements GobEncoder and GobDecoder has complete
// control over the representation of its data and may therefore
// contain things such as private fields, channels, and functions,
// which are not usually transmissible in gob streams.
//
// Note: Since gobs can be stored permanently, It is good design
// to guarantee the encoding used by a GobEncoder is stable as the
// software evolves.  For instance, it might make sense for GobEncode
// to include a version number in the encoding.
type GobEncoder interface {
	// GobEncode returns a byte slice representing the encoding of the
	// receiver for transmission to a GobDecoder, usually of the same
	// concrete type.
	GobEncode() ([]byte, os.Error)
}

// GobDecoder is the interface describing data that provides its own
// routine for decoding transmitted values sent by a GobEncoder.
type GobDecoder interface {
	// GobDecode overwrites the receiver, which must be a pointer,
	// with the value represented by the byte slice, which was written
	// by GobEncode, usually for the same concrete type.
	GobDecode([]byte) os.Error
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
	base := userType(reflect.TypeOf(value)).base
	// Check for incompatible duplicates.
	if t, ok := nameToConcreteType[name]; ok && t != base {
		panic("gob: registering duplicate types for " + name)
	}
	if n, ok := concreteTypeToName[base]; ok && n != name {
		panic("gob: registering duplicate names for " + base.String())
	}
	// Store the name and type provided by the user....
	nameToConcreteType[name] = reflect.TypeOf(value)
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
	rt := reflect.TypeOf(value)
	name := rt.String()

	// But for named types (or pointers to them), qualify with import path.
	// Dereference one pointer looking for a named type.
	star := ""
	if rt.Name() == "" {
		if pt := rt; pt.Kind() == reflect.Ptr {
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
	Register(uintptr(0))
	Register(false)
	Register("")
	Register([]byte(nil))
	Register([]int(nil))
	Register([]int8(nil))
	Register([]int16(nil))
	Register([]int32(nil))
	Register([]int64(nil))
	Register([]uint(nil))
	Register([]uint8(nil))
	Register([]uint16(nil))
	Register([]uint32(nil))
	Register([]uint64(nil))
	Register([]float32(nil))
	Register([]float64(nil))
	Register([]complex64(nil))
	Register([]complex128(nil))
	Register([]uintptr(nil))
	Register([]bool(nil))
	Register([]string(nil))
}
