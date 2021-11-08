// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflectlite implements lightweight version of reflect, not using
// any package except for "runtime" and "unsafe".
package reflectlite

import (
	"internal/unsafeheader"
	"unsafe"
)

// Type is the representation of a Go type.
//
// Not all methods apply to all kinds of types. Restrictions,
// if any, are noted in the documentation for each method.
// Use the Kind method to find out the kind of type before
// calling kind-specific methods. Calling a method
// inappropriate to the kind of type causes a run-time panic.
//
// Type values are comparable, such as with the == operator,
// so they can be used as map keys.
// Two Type values are equal if they represent identical types.
type Type interface {
	// Methods applicable to all types.

	// Name returns the type's name within its package for a defined type.
	// For other (non-defined) types it returns the empty string.
	Name() string

	// PkgPath returns a defined type's package path, that is, the import path
	// that uniquely identifies the package, such as "encoding/base64".
	// If the type was predeclared (string, error) or not defined (*T, struct{},
	// []int, or A where A is an alias for a non-defined type), the package path
	// will be the empty string.
	PkgPath() string

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size() uintptr

	// Kind returns the specific kind of this type.
	Kind() Kind

	// Implements reports whether the type implements the interface type u.
	Implements(u Type) bool

	// AssignableTo reports whether a value of the type is assignable to type u.
	AssignableTo(u Type) bool

	// Comparable reports whether values of this type are comparable.
	Comparable() bool

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., base64 instead of "encoding/base64") and is not
	// guaranteed to be unique among types. To test for type identity,
	// compare the Types directly.
	String() string

	// Elem returns a type's element type.
	// It panics if the type's Kind is not Ptr.
	Elem() Type

	common() *rtype
	uncommon() *uncommonType
}

/*
 * These data structures are known to the compiler (../../cmd/internal/reflectdata/reflect.go).
 * A few are known to ../runtime/type.go to convey to debuggers.
 * They are also known to ../runtime/type.go.
 */

// A Kind represents the specific kind of type that a Type represents.
// The zero Kind is not a valid kind.
type Kind uint

const (
	Invalid Kind = iota
	Bool
	Int
	Int8
	Int16
	Int32
	Int64
	Uint
	Uint8
	Uint16
	Uint32
	Uint64
	Uintptr
	Float32
	Float64
	Complex64
	Complex128
	Array
	Chan
	Func
	Interface
	Map
	Pointer
	Slice
	String
	Struct
	UnsafePointer
)

const Ptr = Pointer

// tflag is used by an rtype to signal what extra type information is
// available in the memory directly following the rtype value.
//
// tflag values must be kept in sync with copies in:
//	cmd/compile/internal/reflectdata/reflect.go
//	cmd/link/internal/ld/decodesym.go
//	runtime/type.go
type tflag uint8

const (
	// tflagUncommon means that there is a pointer, *uncommonType,
	// just beyond the outer type structure.
	//
	// For example, if t.Kind() == Struct and t.tflag&tflagUncommon != 0,
	// then t has uncommonType data and it can be accessed as:
	//
	//	type tUncommon struct {
	//		structType
	//		u uncommonType
	//	}
	//	u := &(*tUncommon)(unsafe.Pointer(t)).u
	tflagUncommon tflag = 1 << 0

	// tflagExtraStar means the name in the str field has an
	// extraneous '*' prefix. This is because for most types T in
	// a program, the type *T also exists and reusing the str data
	// saves binary size.
	tflagExtraStar tflag = 1 << 1

	// tflagNamed means the type has a name.
	tflagNamed tflag = 1 << 2

	// tflagRegularMemory means that equal and hash functions can treat
	// this type as a single region of t.size bytes.
	tflagRegularMemory tflag = 1 << 3
)

// rtype is the common implementation of most values.
// It is embedded in other struct types.
//
// rtype must be kept in sync with ../runtime/type.go:/^type._type.
type rtype struct {
	size       uintptr
	ptrdata    uintptr // number of bytes in the type that can contain pointers
	hash       uint32  // hash of type; avoids computation in hash tables
	tflag      tflag   // extra type information flags
	align      uint8   // alignment of variable with this type
	fieldAlign uint8   // alignment of struct field with this type
	kind       uint8   // enumeration for C
	// function for comparing objects of this type
	// (ptr to object A, ptr to object B) -> ==?
	equal     func(unsafe.Pointer, unsafe.Pointer) bool
	gcdata    *byte   // garbage collection data
	str       nameOff // string form
	ptrToThis typeOff // type for pointer to this type, may be zero
}

// Method on non-interface type
type method struct {
	name nameOff // name of method
	mtyp typeOff // method type (without receiver)
	ifn  textOff // fn used in interface call (one-word receiver)
	tfn  textOff // fn used for normal method call
}

// uncommonType is present only for defined types or types with methods
// (if T is a defined type, the uncommonTypes for T and *T have methods).
// Using a pointer to this struct reduces the overall size required
// to describe a non-defined type with no methods.
type uncommonType struct {
	pkgPath nameOff // import path; empty for built-in types like int, string
	mcount  uint16  // number of methods
	xcount  uint16  // number of exported methods
	moff    uint32  // offset from this uncommontype to [mcount]method
	_       uint32  // unused
}

// chanDir represents a channel type's direction.
type chanDir int

const (
	recvDir chanDir             = 1 << iota // <-chan
	sendDir                                 // chan<-
	bothDir = recvDir | sendDir             // chan
)

// arrayType represents a fixed array type.
type arrayType struct {
	rtype
	elem  *rtype // array element type
	slice *rtype // slice type
	len   uintptr
}

// chanType represents a channel type.
type chanType struct {
	rtype
	elem *rtype  // channel element type
	dir  uintptr // channel direction (chanDir)
}

// funcType represents a function type.
//
// A *rtype for each in and out parameter is stored in an array that
// directly follows the funcType (and possibly its uncommonType). So
// a function type with one method, one input, and one output is:
//
//	struct {
//		funcType
//		uncommonType
//		[2]*rtype    // [0] is in, [1] is out
//	}
type funcType struct {
	rtype
	inCount  uint16
	outCount uint16 // top bit is set if last input parameter is ...
}

// imethod represents a method on an interface type
type imethod struct {
	name nameOff // name of method
	typ  typeOff // .(*FuncType) underneath
}

// interfaceType represents an interface type.
type interfaceType struct {
	rtype
	pkgPath name      // import path
	methods []imethod // sorted by hash
}

// mapType represents a map type.
type mapType struct {
	rtype
	key    *rtype // map key type
	elem   *rtype // map element (value) type
	bucket *rtype // internal bucket structure
	// function for hashing keys (ptr to key, seed) -> hash
	hasher     func(unsafe.Pointer, uintptr) uintptr
	keysize    uint8  // size of key slot
	valuesize  uint8  // size of value slot
	bucketsize uint16 // size of bucket
	flags      uint32
}

// ptrType represents a pointer type.
type ptrType struct {
	rtype
	elem *rtype // pointer element (pointed at) type
}

// sliceType represents a slice type.
type sliceType struct {
	rtype
	elem *rtype // slice element type
}

// Struct field
type structField struct {
	name        name    // name is always non-empty
	typ         *rtype  // type of field
	offsetEmbed uintptr // byte offset of field<<1 | isEmbedded
}

func (f *structField) offset() uintptr {
	return f.offsetEmbed >> 1
}

func (f *structField) embedded() bool {
	return f.offsetEmbed&1 != 0
}

// structType represents a struct type.
type structType struct {
	rtype
	pkgPath name
	fields  []structField // sorted by offset
}

// name is an encoded type name with optional extra data.
//
// The first byte is a bit field containing:
//
//	1<<0 the name is exported
//	1<<1 tag data follows the name
//	1<<2 pkgPath nameOff follows the name and tag
//
// The next two bytes are the data length:
//
//	 l := uint16(data[1])<<8 | uint16(data[2])
//
// Bytes [3:3+l] are the string data.
//
// If tag data follows then bytes 3+l and 3+l+1 are the tag length,
// with the data following.
//
// If the import path follows, then 4 bytes at the end of
// the data form a nameOff. The import path is only set for concrete
// methods that are defined in a different package than their type.
//
// If a name starts with "*", then the exported bit represents
// whether the pointed to type is exported.
type name struct {
	bytes *byte
}

func (n name) data(off int, whySafe string) *byte {
	return (*byte)(add(unsafe.Pointer(n.bytes), uintptr(off), whySafe))
}

func (n name) isExported() bool {
	return (*n.bytes)&(1<<0) != 0
}

func (n name) hasTag() bool {
	return (*n.bytes)&(1<<1) != 0
}

// readVarint parses a varint as encoded by encoding/binary.
// It returns the number of encoded bytes and the encoded value.
func (n name) readVarint(off int) (int, int) {
	v := 0
	for i := 0; ; i++ {
		x := *n.data(off+i, "read varint")
		v += int(x&0x7f) << (7 * i)
		if x&0x80 == 0 {
			return i + 1, v
		}
	}
}

func (n name) name() (s string) {
	if n.bytes == nil {
		return
	}
	i, l := n.readVarint(1)
	hdr := (*unsafeheader.String)(unsafe.Pointer(&s))
	hdr.Data = unsafe.Pointer(n.data(1+i, "non-empty string"))
	hdr.Len = l
	return
}

func (n name) tag() (s string) {
	if !n.hasTag() {
		return ""
	}
	i, l := n.readVarint(1)
	i2, l2 := n.readVarint(1 + i + l)
	hdr := (*unsafeheader.String)(unsafe.Pointer(&s))
	hdr.Data = unsafe.Pointer(n.data(1+i+l+i2, "non-empty string"))
	hdr.Len = l2
	return
}

func (n name) pkgPath() string {
	if n.bytes == nil || *n.data(0, "name flag field")&(1<<2) == 0 {
		return ""
	}
	i, l := n.readVarint(1)
	off := 1 + i + l
	if n.hasTag() {
		i2, l2 := n.readVarint(off)
		off += i2 + l2
	}
	var nameOff int32
	// Note that this field may not be aligned in memory,
	// so we cannot use a direct int32 assignment here.
	copy((*[4]byte)(unsafe.Pointer(&nameOff))[:], (*[4]byte)(unsafe.Pointer(n.data(off, "name offset field")))[:])
	pkgPathName := name{(*byte)(resolveTypeOff(unsafe.Pointer(n.bytes), nameOff))}
	return pkgPathName.name()
}

/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

const (
	kindDirectIface = 1 << 5
	kindGCProg      = 1 << 6 // Type.gc points to GC program
	kindMask        = (1 << 5) - 1
)

// String returns the name of k.
func (k Kind) String() string {
	if int(k) < len(kindNames) {
		return kindNames[k]
	}
	return kindNames[0]
}

var kindNames = []string{
	Invalid:       "invalid",
	Bool:          "bool",
	Int:           "int",
	Int8:          "int8",
	Int16:         "int16",
	Int32:         "int32",
	Int64:         "int64",
	Uint:          "uint",
	Uint8:         "uint8",
	Uint16:        "uint16",
	Uint32:        "uint32",
	Uint64:        "uint64",
	Uintptr:       "uintptr",
	Float32:       "float32",
	Float64:       "float64",
	Complex64:     "complex64",
	Complex128:    "complex128",
	Array:         "array",
	Chan:          "chan",
	Func:          "func",
	Interface:     "interface",
	Map:           "map",
	Ptr:           "ptr",
	Slice:         "slice",
	String:        "string",
	Struct:        "struct",
	UnsafePointer: "unsafe.Pointer",
}

func (t *uncommonType) methods() []method {
	if t.mcount == 0 {
		return nil
	}
	return (*[1 << 16]method)(add(unsafe.Pointer(t), uintptr(t.moff), "t.mcount > 0"))[:t.mcount:t.mcount]
}

func (t *uncommonType) exportedMethods() []method {
	if t.xcount == 0 {
		return nil
	}
	return (*[1 << 16]method)(add(unsafe.Pointer(t), uintptr(t.moff), "t.xcount > 0"))[:t.xcount:t.xcount]
}

// resolveNameOff resolves a name offset from a base pointer.
// The (*rtype).nameOff method is a convenience wrapper for this function.
// Implemented in the runtime package.
func resolveNameOff(ptrInModule unsafe.Pointer, off int32) unsafe.Pointer

// resolveTypeOff resolves an *rtype offset from a base type.
// The (*rtype).typeOff method is a convenience wrapper for this function.
// Implemented in the runtime package.
func resolveTypeOff(rtype unsafe.Pointer, off int32) unsafe.Pointer

type nameOff int32 // offset to a name
type typeOff int32 // offset to an *rtype
type textOff int32 // offset from top of text section

func (t *rtype) nameOff(off nameOff) name {
	return name{(*byte)(resolveNameOff(unsafe.Pointer(t), int32(off)))}
}

func (t *rtype) typeOff(off typeOff) *rtype {
	return (*rtype)(resolveTypeOff(unsafe.Pointer(t), int32(off)))
}

func (t *rtype) uncommon() *uncommonType {
	if t.tflag&tflagUncommon == 0 {
		return nil
	}
	switch t.Kind() {
	case Struct:
		return &(*structTypeUncommon)(unsafe.Pointer(t)).u
	case Ptr:
		type u struct {
			ptrType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	case Func:
		type u struct {
			funcType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	case Slice:
		type u struct {
			sliceType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	case Array:
		type u struct {
			arrayType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	case Chan:
		type u struct {
			chanType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	case Map:
		type u struct {
			mapType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	case Interface:
		type u struct {
			interfaceType
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	default:
		type u struct {
			rtype
			u uncommonType
		}
		return &(*u)(unsafe.Pointer(t)).u
	}
}

func (t *rtype) String() string {
	s := t.nameOff(t.str).name()
	if t.tflag&tflagExtraStar != 0 {
		return s[1:]
	}
	return s
}

func (t *rtype) Size() uintptr { return t.size }

func (t *rtype) Kind() Kind { return Kind(t.kind & kindMask) }

func (t *rtype) pointers() bool { return t.ptrdata != 0 }

func (t *rtype) common() *rtype { return t }

func (t *rtype) exportedMethods() []method {
	ut := t.uncommon()
	if ut == nil {
		return nil
	}
	return ut.exportedMethods()
}

func (t *rtype) NumMethod() int {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.NumMethod()
	}
	return len(t.exportedMethods())
}

func (t *rtype) PkgPath() string {
	if t.tflag&tflagNamed == 0 {
		return ""
	}
	ut := t.uncommon()
	if ut == nil {
		return ""
	}
	return t.nameOff(ut.pkgPath).name()
}

func (t *rtype) hasName() bool {
	return t.tflag&tflagNamed != 0
}

func (t *rtype) Name() string {
	if !t.hasName() {
		return ""
	}
	s := t.String()
	i := len(s) - 1
	for i >= 0 && s[i] != '.' {
		i--
	}
	return s[i+1:]
}

func (t *rtype) chanDir() chanDir {
	if t.Kind() != Chan {
		panic("reflect: chanDir of non-chan type")
	}
	tt := (*chanType)(unsafe.Pointer(t))
	return chanDir(tt.dir)
}

func (t *rtype) Elem() Type {
	switch t.Kind() {
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Chan:
		tt := (*chanType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Map:
		tt := (*mapType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Ptr:
		tt := (*ptrType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Slice:
		tt := (*sliceType)(unsafe.Pointer(t))
		return toType(tt.elem)
	}
	panic("reflect: Elem of invalid type")
}

func (t *rtype) In(i int) Type {
	if t.Kind() != Func {
		panic("reflect: In of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.in()[i])
}

func (t *rtype) Key() Type {
	if t.Kind() != Map {
		panic("reflect: Key of non-map type")
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.key)
}

func (t *rtype) Len() int {
	if t.Kind() != Array {
		panic("reflect: Len of non-array type")
	}
	tt := (*arrayType)(unsafe.Pointer(t))
	return int(tt.len)
}

func (t *rtype) NumField() int {
	if t.Kind() != Struct {
		panic("reflect: NumField of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return len(tt.fields)
}

func (t *rtype) NumIn() int {
	if t.Kind() != Func {
		panic("reflect: NumIn of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return int(tt.inCount)
}

func (t *rtype) NumOut() int {
	if t.Kind() != Func {
		panic("reflect: NumOut of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return len(tt.out())
}

func (t *rtype) Out(i int) Type {
	if t.Kind() != Func {
		panic("reflect: Out of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.out()[i])
}

func (t *funcType) in() []*rtype {
	uadd := unsafe.Sizeof(*t)
	if t.tflag&tflagUncommon != 0 {
		uadd += unsafe.Sizeof(uncommonType{})
	}
	if t.inCount == 0 {
		return nil
	}
	return (*[1 << 20]*rtype)(add(unsafe.Pointer(t), uadd, "t.inCount > 0"))[:t.inCount:t.inCount]
}

func (t *funcType) out() []*rtype {
	uadd := unsafe.Sizeof(*t)
	if t.tflag&tflagUncommon != 0 {
		uadd += unsafe.Sizeof(uncommonType{})
	}
	outCount := t.outCount & (1<<15 - 1)
	if outCount == 0 {
		return nil
	}
	return (*[1 << 20]*rtype)(add(unsafe.Pointer(t), uadd, "outCount > 0"))[t.inCount : t.inCount+outCount : t.inCount+outCount]
}

// add returns p+x.
//
// The whySafe string is ignored, so that the function still inlines
// as efficiently as p+x, but all call sites should use the string to
// record why the addition is safe, which is to say why the addition
// does not cause x to advance to the very end of p's allocation
// and therefore point incorrectly at the next block in memory.
func add(p unsafe.Pointer, x uintptr, whySafe string) unsafe.Pointer {
	return unsafe.Pointer(uintptr(p) + x)
}

// NumMethod returns the number of interface methods in the type's method set.
func (t *interfaceType) NumMethod() int { return len(t.methods) }

// TypeOf returns the reflection Type that represents the dynamic type of i.
// If i is a nil interface value, TypeOf returns nil.
func TypeOf(i interface{}) Type {
	eface := *(*emptyInterface)(unsafe.Pointer(&i))
	return toType(eface.typ)
}

func (t *rtype) Implements(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.Implements")
	}
	if u.Kind() != Interface {
		panic("reflect: non-interface type passed to Type.Implements")
	}
	return implements(u.(*rtype), t)
}

func (t *rtype) AssignableTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.AssignableTo")
	}
	uu := u.(*rtype)
	return directlyAssignable(uu, t) || implements(uu, t)
}

func (t *rtype) Comparable() bool {
	return t.equal != nil
}

// implements reports whether the type V implements the interface type T.
func implements(T, V *rtype) bool {
	if T.Kind() != Interface {
		return false
	}
	t := (*interfaceType)(unsafe.Pointer(T))
	if len(t.methods) == 0 {
		return true
	}

	// The same algorithm applies in both cases, but the
	// method tables for an interface type and a concrete type
	// are different, so the code is duplicated.
	// In both cases the algorithm is a linear scan over the two
	// lists - T's methods and V's methods - simultaneously.
	// Since method tables are stored in a unique sorted order
	// (alphabetical, with no duplicate method names), the scan
	// through V's methods must hit a match for each of T's
	// methods along the way, or else V does not implement T.
	// This lets us run the scan in overall linear time instead of
	// the quadratic time  a naive search would require.
	// See also ../runtime/iface.go.
	if V.Kind() == Interface {
		v := (*interfaceType)(unsafe.Pointer(V))
		i := 0
		for j := 0; j < len(v.methods); j++ {
			tm := &t.methods[i]
			tmName := t.nameOff(tm.name)
			vm := &v.methods[j]
			vmName := V.nameOff(vm.name)
			if vmName.name() == tmName.name() && V.typeOff(vm.typ) == t.typeOff(tm.typ) {
				if !tmName.isExported() {
					tmPkgPath := tmName.pkgPath()
					if tmPkgPath == "" {
						tmPkgPath = t.pkgPath.name()
					}
					vmPkgPath := vmName.pkgPath()
					if vmPkgPath == "" {
						vmPkgPath = v.pkgPath.name()
					}
					if tmPkgPath != vmPkgPath {
						continue
					}
				}
				if i++; i >= len(t.methods) {
					return true
				}
			}
		}
		return false
	}

	v := V.uncommon()
	if v == nil {
		return false
	}
	i := 0
	vmethods := v.methods()
	for j := 0; j < int(v.mcount); j++ {
		tm := &t.methods[i]
		tmName := t.nameOff(tm.name)
		vm := vmethods[j]
		vmName := V.nameOff(vm.name)
		if vmName.name() == tmName.name() && V.typeOff(vm.mtyp) == t.typeOff(tm.typ) {
			if !tmName.isExported() {
				tmPkgPath := tmName.pkgPath()
				if tmPkgPath == "" {
					tmPkgPath = t.pkgPath.name()
				}
				vmPkgPath := vmName.pkgPath()
				if vmPkgPath == "" {
					vmPkgPath = V.nameOff(v.pkgPath).name()
				}
				if tmPkgPath != vmPkgPath {
					continue
				}
			}
			if i++; i >= len(t.methods) {
				return true
			}
		}
	}
	return false
}

// directlyAssignable reports whether a value x of type V can be directly
// assigned (using memmove) to a value of type T.
// https://golang.org/doc/go_spec.html#Assignability
// Ignoring the interface rules (implemented elsewhere)
// and the ideal constant rules (no ideal constants at run time).
func directlyAssignable(T, V *rtype) bool {
	// x's type V is identical to T?
	if T == V {
		return true
	}

	// Otherwise at least one of T and V must not be defined
	// and they must have the same kind.
	if T.hasName() && V.hasName() || T.Kind() != V.Kind() {
		return false
	}

	// x's type T and V must  have identical underlying types.
	return haveIdenticalUnderlyingType(T, V, true)
}

func haveIdenticalType(T, V Type, cmpTags bool) bool {
	if cmpTags {
		return T == V
	}

	if T.Name() != V.Name() || T.Kind() != V.Kind() {
		return false
	}

	return haveIdenticalUnderlyingType(T.common(), V.common(), false)
}

func haveIdenticalUnderlyingType(T, V *rtype, cmpTags bool) bool {
	if T == V {
		return true
	}

	kind := T.Kind()
	if kind != V.Kind() {
		return false
	}

	// Non-composite types of equal kind have same underlying type
	// (the predefined instance of the type).
	if Bool <= kind && kind <= Complex128 || kind == String || kind == UnsafePointer {
		return true
	}

	// Composite types.
	switch kind {
	case Array:
		return T.Len() == V.Len() && haveIdenticalType(T.Elem(), V.Elem(), cmpTags)

	case Chan:
		// Special case:
		// x is a bidirectional channel value, T is a channel type,
		// and x's type V and T have identical element types.
		if V.chanDir() == bothDir && haveIdenticalType(T.Elem(), V.Elem(), cmpTags) {
			return true
		}

		// Otherwise continue test for identical underlying type.
		return V.chanDir() == T.chanDir() && haveIdenticalType(T.Elem(), V.Elem(), cmpTags)

	case Func:
		t := (*funcType)(unsafe.Pointer(T))
		v := (*funcType)(unsafe.Pointer(V))
		if t.outCount != v.outCount || t.inCount != v.inCount {
			return false
		}
		for i := 0; i < t.NumIn(); i++ {
			if !haveIdenticalType(t.In(i), v.In(i), cmpTags) {
				return false
			}
		}
		for i := 0; i < t.NumOut(); i++ {
			if !haveIdenticalType(t.Out(i), v.Out(i), cmpTags) {
				return false
			}
		}
		return true

	case Interface:
		t := (*interfaceType)(unsafe.Pointer(T))
		v := (*interfaceType)(unsafe.Pointer(V))
		if len(t.methods) == 0 && len(v.methods) == 0 {
			return true
		}
		// Might have the same methods but still
		// need a run time conversion.
		return false

	case Map:
		return haveIdenticalType(T.Key(), V.Key(), cmpTags) && haveIdenticalType(T.Elem(), V.Elem(), cmpTags)

	case Ptr, Slice:
		return haveIdenticalType(T.Elem(), V.Elem(), cmpTags)

	case Struct:
		t := (*structType)(unsafe.Pointer(T))
		v := (*structType)(unsafe.Pointer(V))
		if len(t.fields) != len(v.fields) {
			return false
		}
		if t.pkgPath.name() != v.pkgPath.name() {
			return false
		}
		for i := range t.fields {
			tf := &t.fields[i]
			vf := &v.fields[i]
			if tf.name.name() != vf.name.name() {
				return false
			}
			if !haveIdenticalType(tf.typ, vf.typ, cmpTags) {
				return false
			}
			if cmpTags && tf.name.tag() != vf.name.tag() {
				return false
			}
			if tf.offsetEmbed != vf.offsetEmbed {
				return false
			}
		}
		return true
	}

	return false
}

type structTypeUncommon struct {
	structType
	u uncommonType
}

// toType converts from a *rtype to a Type that can be returned
// to the client of package reflect. In gc, the only concern is that
// a nil *rtype must be replaced by a nil Type, but in gccgo this
// function takes care of ensuring that multiple *rtype for the same
// type are coalesced into a single Type.
func toType(t *rtype) Type {
	if t == nil {
		return nil
	}
	return t
}

// ifaceIndir reports whether t is stored indirectly in an interface value.
func ifaceIndir(t *rtype) bool {
	return t.kind&kindDirectIface == 0
}
