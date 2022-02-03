// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflect implements run-time reflection, allowing a program to
// manipulate objects with arbitrary types. The typical use is to take a value
// with static type interface{} and extract its dynamic type information by
// calling TypeOf, which returns a Type.
//
// A call to ValueOf returns a Value representing the run-time data.
// Zero takes a Type and returns a Value representing a zero value
// for that type.
//
// See "The Laws of Reflection" for an introduction to reflection in Go:
// https://golang.org/doc/articles/laws_of_reflection.html
package reflect

import (
	"internal/goarch"
	"internal/unsafeheader"
	"strconv"
	"sync"
	"unicode"
	"unicode/utf8"
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

	// Align returns the alignment in bytes of a value of
	// this type when allocated in memory.
	Align() int

	// FieldAlign returns the alignment in bytes of a value of
	// this type when used as a field in a struct.
	FieldAlign() int

	// Method returns the i'th method in the type's method set.
	// It panics if i is not in the range [0, NumMethod()).
	//
	// For a non-interface type T or *T, the returned Method's Type and Func
	// fields describe a function whose first argument is the receiver,
	// and only exported methods are accessible.
	//
	// For an interface type, the returned Method's Type field gives the
	// method signature, without a receiver, and the Func field is nil.
	//
	// Methods are sorted in lexicographic order.
	Method(int) Method

	// MethodByName returns the method with that name in the type's
	// method set and a boolean indicating if the method was found.
	//
	// For a non-interface type T or *T, the returned Method's Type and Func
	// fields describe a function whose first argument is the receiver.
	//
	// For an interface type, the returned Method's Type field gives the
	// method signature, without a receiver, and the Func field is nil.
	MethodByName(string) (Method, bool)

	// NumMethod returns the number of methods accessible using Method.
	//
	// Note that NumMethod counts unexported methods only for interface types.
	NumMethod() int

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

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., base64 instead of "encoding/base64") and is not
	// guaranteed to be unique among types. To test for type identity,
	// compare the Types directly.
	String() string

	// Kind returns the specific kind of this type.
	Kind() Kind

	// Implements reports whether the type implements the interface type u.
	Implements(u Type) bool

	// AssignableTo reports whether a value of the type is assignable to type u.
	AssignableTo(u Type) bool

	// ConvertibleTo reports whether a value of the type is convertible to type u.
	// Even if ConvertibleTo returns true, the conversion may still panic.
	// For example, a slice of type []T is convertible to *[N]T,
	// but the conversion will panic if its length is less than N.
	ConvertibleTo(u Type) bool

	// Comparable reports whether values of this type are comparable.
	// Even if Comparable returns true, the comparison may still panic.
	// For example, values of interface type are comparable,
	// but the comparison will panic if their dynamic type is not comparable.
	Comparable() bool

	// Methods applicable only to some types, depending on Kind.
	// The methods allowed for each kind are:
	//
	//	Int*, Uint*, Float*, Complex*: Bits
	//	Array: Elem, Len
	//	Chan: ChanDir, Elem
	//	Func: In, NumIn, Out, NumOut, IsVariadic.
	//	Map: Key, Elem
	//	Pointer: Elem
	//	Slice: Elem
	//	Struct: Field, FieldByIndex, FieldByName, FieldByNameFunc, NumField

	// Bits returns the size of the type in bits.
	// It panics if the type's Kind is not one of the
	// sized or unsized Int, Uint, Float, or Complex kinds.
	Bits() int

	// ChanDir returns a channel type's direction.
	// It panics if the type's Kind is not Chan.
	ChanDir() ChanDir

	// IsVariadic reports whether a function type's final input parameter
	// is a "..." parameter. If so, t.In(t.NumIn() - 1) returns the parameter's
	// implicit actual type []T.
	//
	// For concreteness, if t represents func(x int, y ... float64), then
	//
	//	t.NumIn() == 2
	//	t.In(0) is the reflect.Type for "int"
	//	t.In(1) is the reflect.Type for "[]float64"
	//	t.IsVariadic() == true
	//
	// IsVariadic panics if the type's Kind is not Func.
	IsVariadic() bool

	// Elem returns a type's element type.
	// It panics if the type's Kind is not Array, Chan, Map, Pointer, or Slice.
	Elem() Type

	// Field returns a struct type's i'th field.
	// It panics if the type's Kind is not Struct.
	// It panics if i is not in the range [0, NumField()).
	Field(i int) StructField

	// FieldByIndex returns the nested field corresponding
	// to the index sequence. It is equivalent to calling Field
	// successively for each index i.
	// It panics if the type's Kind is not Struct.
	FieldByIndex(index []int) StructField

	// FieldByName returns the struct field with the given name
	// and a boolean indicating if the field was found.
	FieldByName(name string) (StructField, bool)

	// FieldByNameFunc returns the struct field with a name
	// that satisfies the match function and a boolean indicating if
	// the field was found.
	//
	// FieldByNameFunc considers the fields in the struct itself
	// and then the fields in any embedded structs, in breadth first order,
	// stopping at the shallowest nesting depth containing one or more
	// fields satisfying the match function. If multiple fields at that depth
	// satisfy the match function, they cancel each other
	// and FieldByNameFunc returns no match.
	// This behavior mirrors Go's handling of name lookup in
	// structs containing embedded fields.
	FieldByNameFunc(match func(string) bool) (StructField, bool)

	// In returns the type of a function type's i'th input parameter.
	// It panics if the type's Kind is not Func.
	// It panics if i is not in the range [0, NumIn()).
	In(i int) Type

	// Key returns a map type's key type.
	// It panics if the type's Kind is not Map.
	Key() Type

	// Len returns an array type's length.
	// It panics if the type's Kind is not Array.
	Len() int

	// NumField returns a struct type's field count.
	// It panics if the type's Kind is not Struct.
	NumField() int

	// NumIn returns a function type's input parameter count.
	// It panics if the type's Kind is not Func.
	NumIn() int

	// NumOut returns a function type's output parameter count.
	// It panics if the type's Kind is not Func.
	NumOut() int

	// Out returns the type of a function type's i'th output parameter.
	// It panics if the type's Kind is not Func.
	// It panics if i is not in the range [0, NumOut()).
	Out(i int) Type

	common() *rtype
	uncommon() *uncommonType
}

// BUG(rsc): FieldByName and related functions consider struct field names to be equal
// if the names are equal, even if they are unexported names originating
// in different packages. The practical effect of this is that the result of
// t.FieldByName("x") is not well defined if the struct type t contains
// multiple fields named x (embedded from different packages).
// FieldByName may return one of the fields named x or may report that there are none.
// See https://golang.org/issue/4876 for more details.

/*
 * These data structures are known to the compiler (../cmd/compile/internal/reflectdata/reflect.go).
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

// Ptr is the old name for the Pointer kind.
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

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir ChanDir             = 1 << iota // <-chan
	SendDir                                 // chan<-
	BothDir = RecvDir | SendDir             // chan
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
	dir  uintptr // channel direction (ChanDir)
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
// Following that, there is a varint-encoded length of the name,
// followed by the name itself.
//
// If tag data is present, it also has a varint-encoded length
// followed by the tag itself.
//
// If the import path follows, then 4 bytes at the end of
// the data form a nameOff. The import path is only set for concrete
// methods that are defined in a different package than their type.
//
// If a name starts with "*", then the exported bit represents
// whether the pointed to type is exported.
//
// Note: this encoding must match here and in:
//   cmd/compile/internal/reflectdata/reflect.go
//   runtime/type.go
//   internal/reflectlite/type.go
//   cmd/link/internal/ld/decodesym.go

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

// writeVarint writes n to buf in varint form. Returns the
// number of bytes written. n must be nonnegative.
// Writes at most 10 bytes.
func writeVarint(buf []byte, n int) int {
	for i := 0; ; i++ {
		b := byte(n & 0x7f)
		n >>= 7
		if n == 0 {
			buf[i] = b
			return i + 1
		}
		buf[i] = b | 0x80
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

func newName(n, tag string, exported bool) name {
	if len(n) >= 1<<29 {
		panic("reflect.nameFrom: name too long: " + n[:1024] + "...")
	}
	if len(tag) >= 1<<29 {
		panic("reflect.nameFrom: tag too long: " + tag[:1024] + "...")
	}
	var nameLen [10]byte
	var tagLen [10]byte
	nameLenLen := writeVarint(nameLen[:], len(n))
	tagLenLen := writeVarint(tagLen[:], len(tag))

	var bits byte
	l := 1 + nameLenLen + len(n)
	if exported {
		bits |= 1 << 0
	}
	if len(tag) > 0 {
		l += tagLenLen + len(tag)
		bits |= 1 << 1
	}

	b := make([]byte, l)
	b[0] = bits
	copy(b[1:], nameLen[:nameLenLen])
	copy(b[1+nameLenLen:], n)
	if len(tag) > 0 {
		tb := b[1+nameLenLen+len(n):]
		copy(tb, tagLen[:tagLenLen])
		copy(tb[tagLenLen:], tag)
	}

	return name{bytes: &b[0]}
}

/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

// Method represents a single method.
type Method struct {
	// Name is the method name.
	Name string

	// PkgPath is the package path that qualifies a lower case (unexported)
	// method name. It is empty for upper case (exported) method names.
	// The combination of PkgPath and Name uniquely identifies a method
	// in a method set.
	// See https://golang.org/ref/spec#Uniqueness_of_identifiers
	PkgPath string

	Type  Type  // method type
	Func  Value // func with receiver as first argument
	Index int   // index for Type.Method
}

// IsExported reports whether the method is exported.
func (m Method) IsExported() bool {
	return m.PkgPath == ""
}

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
	return "kind" + strconv.Itoa(int(k))
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
	Pointer:       "ptr",
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

// resolveTextOff resolves a function pointer offset from a base type.
// The (*rtype).textOff method is a convenience wrapper for this function.
// Implemented in the runtime package.
func resolveTextOff(rtype unsafe.Pointer, off int32) unsafe.Pointer

// addReflectOff adds a pointer to the reflection lookup map in the runtime.
// It returns a new ID that can be used as a typeOff or textOff, and will
// be resolved correctly. Implemented in the runtime package.
func addReflectOff(ptr unsafe.Pointer) int32

// resolveReflectName adds a name to the reflection lookup map in the runtime.
// It returns a new nameOff that can be used to refer to the pointer.
func resolveReflectName(n name) nameOff {
	return nameOff(addReflectOff(unsafe.Pointer(n.bytes)))
}

// resolveReflectType adds a *rtype to the reflection lookup map in the runtime.
// It returns a new typeOff that can be used to refer to the pointer.
func resolveReflectType(t *rtype) typeOff {
	return typeOff(addReflectOff(unsafe.Pointer(t)))
}

// resolveReflectText adds a function pointer to the reflection lookup map in
// the runtime. It returns a new textOff that can be used to refer to the
// pointer.
func resolveReflectText(ptr unsafe.Pointer) textOff {
	return textOff(addReflectOff(ptr))
}

type nameOff int32 // offset to a name
type typeOff int32 // offset to an *rtype
type textOff int32 // offset from top of text section

func (t *rtype) nameOff(off nameOff) name {
	return name{(*byte)(resolveNameOff(unsafe.Pointer(t), int32(off)))}
}

func (t *rtype) typeOff(off typeOff) *rtype {
	return (*rtype)(resolveTypeOff(unsafe.Pointer(t), int32(off)))
}

func (t *rtype) textOff(off textOff) unsafe.Pointer {
	return resolveTextOff(unsafe.Pointer(t), int32(off))
}

func (t *rtype) uncommon() *uncommonType {
	if t.tflag&tflagUncommon == 0 {
		return nil
	}
	switch t.Kind() {
	case Struct:
		return &(*structTypeUncommon)(unsafe.Pointer(t)).u
	case Pointer:
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

func (t *rtype) Bits() int {
	if t == nil {
		panic("reflect: Bits of nil Type")
	}
	k := t.Kind()
	if k < Int || k > Complex128 {
		panic("reflect: Bits of non-arithmetic Type " + t.String())
	}
	return int(t.size) * 8
}

func (t *rtype) Align() int { return int(t.align) }

func (t *rtype) FieldAlign() int { return int(t.fieldAlign) }

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

func (t *rtype) Method(i int) (m Method) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.Method(i)
	}
	methods := t.exportedMethods()
	if i < 0 || i >= len(methods) {
		panic("reflect: Method index out of range")
	}
	p := methods[i]
	pname := t.nameOff(p.name)
	m.Name = pname.name()
	fl := flag(Func)
	mtyp := t.typeOff(p.mtyp)
	ft := (*funcType)(unsafe.Pointer(mtyp))
	in := make([]Type, 0, 1+len(ft.in()))
	in = append(in, t)
	for _, arg := range ft.in() {
		in = append(in, arg)
	}
	out := make([]Type, 0, len(ft.out()))
	for _, ret := range ft.out() {
		out = append(out, ret)
	}
	mt := FuncOf(in, out, ft.IsVariadic())
	m.Type = mt
	tfn := t.textOff(p.tfn)
	fn := unsafe.Pointer(&tfn)
	m.Func = Value{mt.(*rtype), fn, fl}

	m.Index = i
	return m
}

func (t *rtype) MethodByName(name string) (m Method, ok bool) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.MethodByName(name)
	}
	ut := t.uncommon()
	if ut == nil {
		return Method{}, false
	}
	// TODO(mdempsky): Binary search.
	for i, p := range ut.exportedMethods() {
		if t.nameOff(p.name).name() == name {
			return t.Method(i), true
		}
	}
	return Method{}, false
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
	sqBrackets := 0
	for i >= 0 && (s[i] != '.' || sqBrackets != 0) {
		switch s[i] {
		case ']':
			sqBrackets++
		case '[':
			sqBrackets--
		}
		i--
	}
	return s[i+1:]
}

func (t *rtype) ChanDir() ChanDir {
	if t.Kind() != Chan {
		panic("reflect: ChanDir of non-chan type " + t.String())
	}
	tt := (*chanType)(unsafe.Pointer(t))
	return ChanDir(tt.dir)
}

func (t *rtype) IsVariadic() bool {
	if t.Kind() != Func {
		panic("reflect: IsVariadic of non-func type " + t.String())
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return tt.outCount&(1<<15) != 0
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
	case Pointer:
		tt := (*ptrType)(unsafe.Pointer(t))
		return toType(tt.elem)
	case Slice:
		tt := (*sliceType)(unsafe.Pointer(t))
		return toType(tt.elem)
	}
	panic("reflect: Elem of invalid type " + t.String())
}

func (t *rtype) Field(i int) StructField {
	if t.Kind() != Struct {
		panic("reflect: Field of non-struct type " + t.String())
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.Field(i)
}

func (t *rtype) FieldByIndex(index []int) StructField {
	if t.Kind() != Struct {
		panic("reflect: FieldByIndex of non-struct type " + t.String())
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByIndex(index)
}

func (t *rtype) FieldByName(name string) (StructField, bool) {
	if t.Kind() != Struct {
		panic("reflect: FieldByName of non-struct type " + t.String())
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByName(name)
}

func (t *rtype) FieldByNameFunc(match func(string) bool) (StructField, bool) {
	if t.Kind() != Struct {
		panic("reflect: FieldByNameFunc of non-struct type " + t.String())
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByNameFunc(match)
}

func (t *rtype) In(i int) Type {
	if t.Kind() != Func {
		panic("reflect: In of non-func type " + t.String())
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.in()[i])
}

func (t *rtype) Key() Type {
	if t.Kind() != Map {
		panic("reflect: Key of non-map type " + t.String())
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.key)
}

func (t *rtype) Len() int {
	if t.Kind() != Array {
		panic("reflect: Len of non-array type " + t.String())
	}
	tt := (*arrayType)(unsafe.Pointer(t))
	return int(tt.len)
}

func (t *rtype) NumField() int {
	if t.Kind() != Struct {
		panic("reflect: NumField of non-struct type " + t.String())
	}
	tt := (*structType)(unsafe.Pointer(t))
	return len(tt.fields)
}

func (t *rtype) NumIn() int {
	if t.Kind() != Func {
		panic("reflect: NumIn of non-func type " + t.String())
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return int(tt.inCount)
}

func (t *rtype) NumOut() int {
	if t.Kind() != Func {
		panic("reflect: NumOut of non-func type " + t.String())
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return len(tt.out())
}

func (t *rtype) Out(i int) Type {
	if t.Kind() != Func {
		panic("reflect: Out of non-func type " + t.String())
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

func (d ChanDir) String() string {
	switch d {
	case SendDir:
		return "chan<-"
	case RecvDir:
		return "<-chan"
	case BothDir:
		return "chan"
	}
	return "ChanDir" + strconv.Itoa(int(d))
}

// Method returns the i'th method in the type's method set.
func (t *interfaceType) Method(i int) (m Method) {
	if i < 0 || i >= len(t.methods) {
		return
	}
	p := &t.methods[i]
	pname := t.nameOff(p.name)
	m.Name = pname.name()
	if !pname.isExported() {
		m.PkgPath = pname.pkgPath()
		if m.PkgPath == "" {
			m.PkgPath = t.pkgPath.name()
		}
	}
	m.Type = toType(t.typeOff(p.typ))
	m.Index = i
	return
}

// NumMethod returns the number of interface methods in the type's method set.
func (t *interfaceType) NumMethod() int { return len(t.methods) }

// MethodByName method with the given name in the type's method set.
func (t *interfaceType) MethodByName(name string) (m Method, ok bool) {
	if t == nil {
		return
	}
	var p *imethod
	for i := range t.methods {
		p = &t.methods[i]
		if t.nameOff(p.name).name() == name {
			return t.Method(i), true
		}
	}
	return
}

// A StructField describes a single field in a struct.
type StructField struct {
	// Name is the field name.
	Name string

	// PkgPath is the package path that qualifies a lower case (unexported)
	// field name. It is empty for upper case (exported) field names.
	// See https://golang.org/ref/spec#Uniqueness_of_identifiers
	PkgPath string

	Type      Type      // field type
	Tag       StructTag // field tag string
	Offset    uintptr   // offset within struct, in bytes
	Index     []int     // index sequence for Type.FieldByIndex
	Anonymous bool      // is an embedded field
}

// IsExported reports whether the field is exported.
func (f StructField) IsExported() bool {
	return f.PkgPath == ""
}

// A StructTag is the tag string in a struct field.
//
// By convention, tag strings are a concatenation of
// optionally space-separated key:"value" pairs.
// Each key is a non-empty string consisting of non-control
// characters other than space (U+0020 ' '), quote (U+0022 '"'),
// and colon (U+003A ':').  Each value is quoted using U+0022 '"'
// characters and Go string literal syntax.
type StructTag string

// Get returns the value associated with key in the tag string.
// If there is no such key in the tag, Get returns the empty string.
// If the tag does not have the conventional format, the value
// returned by Get is unspecified. To determine whether a tag is
// explicitly set to the empty string, use Lookup.
func (tag StructTag) Get(key string) string {
	v, _ := tag.Lookup(key)
	return v
}

// Lookup returns the value associated with key in the tag string.
// If the key is present in the tag the value (which may be empty)
// is returned. Otherwise the returned value will be the empty string.
// The ok return value reports whether the value was explicitly set in
// the tag string. If the tag does not have the conventional format,
// the value returned by Lookup is unspecified.
func (tag StructTag) Lookup(key string) (value string, ok bool) {
	// When modifying this code, also update the validateStructTag code
	// in cmd/vet/structtag.go.

	for tag != "" {
		// Skip leading space.
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		if tag == "" {
			break
		}

		// Scan to colon. A space, a quote or a control character is a syntax error.
		// Strictly speaking, control chars include the range [0x7f, 0x9f], not just
		// [0x00, 0x1f], but in practice, we ignore the multi-byte control characters
		// as it is simpler to inspect the tag's bytes than the tag's runes.
		i = 0
		for i < len(tag) && tag[i] > ' ' && tag[i] != ':' && tag[i] != '"' && tag[i] != 0x7f {
			i++
		}
		if i == 0 || i+1 >= len(tag) || tag[i] != ':' || tag[i+1] != '"' {
			break
		}
		name := string(tag[:i])
		tag = tag[i+1:]

		// Scan quoted string to find value.
		i = 1
		for i < len(tag) && tag[i] != '"' {
			if tag[i] == '\\' {
				i++
			}
			i++
		}
		if i >= len(tag) {
			break
		}
		qvalue := string(tag[:i+1])
		tag = tag[i+1:]

		if key == name {
			value, err := strconv.Unquote(qvalue)
			if err != nil {
				break
			}
			return value, true
		}
	}
	return "", false
}

// Field returns the i'th struct field.
func (t *structType) Field(i int) (f StructField) {
	if i < 0 || i >= len(t.fields) {
		panic("reflect: Field index out of bounds")
	}
	p := &t.fields[i]
	f.Type = toType(p.typ)
	f.Name = p.name.name()
	f.Anonymous = p.embedded()
	if !p.name.isExported() {
		f.PkgPath = t.pkgPath.name()
	}
	if tag := p.name.tag(); tag != "" {
		f.Tag = StructTag(tag)
	}
	f.Offset = p.offset()

	// NOTE(rsc): This is the only allocation in the interface
	// presented by a reflect.Type. It would be nice to avoid,
	// at least in the common cases, but we need to make sure
	// that misbehaving clients of reflect cannot affect other
	// uses of reflect. One possibility is CL 5371098, but we
	// postponed that ugliness until there is a demonstrated
	// need for the performance. This is issue 2320.
	f.Index = []int{i}
	return
}

// TODO(gri): Should there be an error/bool indicator if the index
//            is wrong for FieldByIndex?

// FieldByIndex returns the nested field corresponding to index.
func (t *structType) FieldByIndex(index []int) (f StructField) {
	f.Type = toType(&t.rtype)
	for i, x := range index {
		if i > 0 {
			ft := f.Type
			if ft.Kind() == Pointer && ft.Elem().Kind() == Struct {
				ft = ft.Elem()
			}
			f.Type = ft
		}
		f = f.Type.Field(x)
	}
	return
}

// A fieldScan represents an item on the fieldByNameFunc scan work list.
type fieldScan struct {
	typ   *structType
	index []int
}

// FieldByNameFunc returns the struct field with a name that satisfies the
// match function and a boolean to indicate if the field was found.
func (t *structType) FieldByNameFunc(match func(string) bool) (result StructField, ok bool) {
	// This uses the same condition that the Go language does: there must be a unique instance
	// of the match at a given depth level. If there are multiple instances of a match at the
	// same depth, they annihilate each other and inhibit any possible match at a lower level.
	// The algorithm is breadth first search, one depth level at a time.

	// The current and next slices are work queues:
	// current lists the fields to visit on this depth level,
	// and next lists the fields on the next lower level.
	current := []fieldScan{}
	next := []fieldScan{{typ: t}}

	// nextCount records the number of times an embedded type has been
	// encountered and considered for queueing in the 'next' slice.
	// We only queue the first one, but we increment the count on each.
	// If a struct type T can be reached more than once at a given depth level,
	// then it annihilates itself and need not be considered at all when we
	// process that next depth level.
	var nextCount map[*structType]int

	// visited records the structs that have been considered already.
	// Embedded pointer fields can create cycles in the graph of
	// reachable embedded types; visited avoids following those cycles.
	// It also avoids duplicated effort: if we didn't find the field in an
	// embedded type T at level 2, we won't find it in one at level 4 either.
	visited := map[*structType]bool{}

	for len(next) > 0 {
		current, next = next, current[:0]
		count := nextCount
		nextCount = nil

		// Process all the fields at this depth, now listed in 'current'.
		// The loop queues embedded fields found in 'next', for processing during the next
		// iteration. The multiplicity of the 'current' field counts is recorded
		// in 'count'; the multiplicity of the 'next' field counts is recorded in 'nextCount'.
		for _, scan := range current {
			t := scan.typ
			if visited[t] {
				// We've looked through this type before, at a higher level.
				// That higher level would shadow the lower level we're now at,
				// so this one can't be useful to us. Ignore it.
				continue
			}
			visited[t] = true
			for i := range t.fields {
				f := &t.fields[i]
				// Find name and (for embedded field) type for field f.
				fname := f.name.name()
				var ntyp *rtype
				if f.embedded() {
					// Embedded field of type T or *T.
					ntyp = f.typ
					if ntyp.Kind() == Pointer {
						ntyp = ntyp.Elem().common()
					}
				}

				// Does it match?
				if match(fname) {
					// Potential match
					if count[t] > 1 || ok {
						// Name appeared multiple times at this level: annihilate.
						return StructField{}, false
					}
					result = t.Field(i)
					result.Index = nil
					result.Index = append(result.Index, scan.index...)
					result.Index = append(result.Index, i)
					ok = true
					continue
				}

				// Queue embedded struct fields for processing with next level,
				// but only if we haven't seen a match yet at this level and only
				// if the embedded types haven't already been queued.
				if ok || ntyp == nil || ntyp.Kind() != Struct {
					continue
				}
				styp := (*structType)(unsafe.Pointer(ntyp))
				if nextCount[styp] > 0 {
					nextCount[styp] = 2 // exact multiple doesn't matter
					continue
				}
				if nextCount == nil {
					nextCount = map[*structType]int{}
				}
				nextCount[styp] = 1
				if count[t] > 1 {
					nextCount[styp] = 2 // exact multiple doesn't matter
				}
				var index []int
				index = append(index, scan.index...)
				index = append(index, i)
				next = append(next, fieldScan{styp, index})
			}
		}
		if ok {
			break
		}
	}
	return
}

// FieldByName returns the struct field with the given name
// and a boolean to indicate if the field was found.
func (t *structType) FieldByName(name string) (f StructField, present bool) {
	// Quick check for top-level name, or struct without embedded fields.
	hasEmbeds := false
	if name != "" {
		for i := range t.fields {
			tf := &t.fields[i]
			if tf.name.name() == name {
				return t.Field(i), true
			}
			if tf.embedded() {
				hasEmbeds = true
			}
		}
	}
	if !hasEmbeds {
		return
	}
	return t.FieldByNameFunc(func(s string) bool { return s == name })
}

// TypeOf returns the reflection Type that represents the dynamic type of i.
// If i is a nil interface value, TypeOf returns nil.
func TypeOf(i any) Type {
	eface := *(*emptyInterface)(unsafe.Pointer(&i))
	return toType(eface.typ)
}

// ptrMap is the cache for PointerTo.
var ptrMap sync.Map // map[*rtype]*ptrType

// PtrTo returns the pointer type with element t.
// For example, if t represents type Foo, PtrTo(t) represents *Foo.
//
// PtrTo is the old spelling of PointerTo.
// The two functions behave identically.
func PtrTo(t Type) Type { return PointerTo(t) }

// PointerTo returns the pointer type with element t.
// For example, if t represents type Foo, PointerTo(t) represents *Foo.
func PointerTo(t Type) Type {
	return t.(*rtype).ptrTo()
}

func (t *rtype) ptrTo() *rtype {
	if t.ptrToThis != 0 {
		return t.typeOff(t.ptrToThis)
	}

	// Check the cache.
	if pi, ok := ptrMap.Load(t); ok {
		return &pi.(*ptrType).rtype
	}

	// Look in known types.
	s := "*" + t.String()
	for _, tt := range typesByString(s) {
		p := (*ptrType)(unsafe.Pointer(tt))
		if p.elem != t {
			continue
		}
		pi, _ := ptrMap.LoadOrStore(t, p)
		return &pi.(*ptrType).rtype
	}

	// Create a new ptrType starting with the description
	// of an *unsafe.Pointer.
	var iptr any = (*unsafe.Pointer)(nil)
	prototype := *(**ptrType)(unsafe.Pointer(&iptr))
	pp := *prototype

	pp.str = resolveReflectName(newName(s, "", false))
	pp.ptrToThis = 0

	// For the type structures linked into the binary, the
	// compiler provides a good hash of the string.
	// Create a good hash for the new string by using
	// the FNV-1 hash's mixing function to combine the
	// old hash and the new "*".
	pp.hash = fnv1(t.hash, '*')

	pp.elem = t

	pi, _ := ptrMap.LoadOrStore(t, &pp)
	return &pi.(*ptrType).rtype
}

// fnv1 incorporates the list of bytes into the hash x using the FNV-1 hash function.
func fnv1(x uint32, list ...byte) uint32 {
	for _, b := range list {
		x = x*16777619 ^ uint32(b)
	}
	return x
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

func (t *rtype) ConvertibleTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.ConvertibleTo")
	}
	uu := u.(*rtype)
	return convertOp(uu, t) != nil
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

// specialChannelAssignability reports whether a value x of channel type V
// can be directly assigned (using memmove) to another channel type T.
// https://golang.org/doc/go_spec.html#Assignability
// T and V must be both of Chan kind.
func specialChannelAssignability(T, V *rtype) bool {
	// Special case:
	// x is a bidirectional channel value, T is a channel type,
	// x's type V and T have identical element types,
	// and at least one of V or T is not a defined type.
	return V.ChanDir() == BothDir && (T.Name() == "" || V.Name() == "") && haveIdenticalType(T.Elem(), V.Elem(), true)
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

	if T.Kind() == Chan && specialChannelAssignability(T, V) {
		return true
	}

	// x's type T and V must have identical underlying types.
	return haveIdenticalUnderlyingType(T, V, true)
}

func haveIdenticalType(T, V Type, cmpTags bool) bool {
	if cmpTags {
		return T == V
	}

	if T.Name() != V.Name() || T.Kind() != V.Kind() || T.PkgPath() != V.PkgPath() {
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
		return V.ChanDir() == T.ChanDir() && haveIdenticalType(T.Elem(), V.Elem(), cmpTags)

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

	case Pointer, Slice:
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

// typelinks is implemented in package runtime.
// It returns a slice of the sections in each module,
// and a slice of *rtype offsets in each module.
//
// The types in each module are sorted by string. That is, the first
// two linked types of the first module are:
//
//	d0 := sections[0]
//	t1 := (*rtype)(add(d0, offset[0][0]))
//	t2 := (*rtype)(add(d0, offset[0][1]))
//
// and
//
//	t1.String() < t2.String()
//
// Note that strings are not unique identifiers for types:
// there can be more than one with a given string.
// Only types we might want to look up are included:
// pointers, channels, maps, slices, and arrays.
func typelinks() (sections []unsafe.Pointer, offset [][]int32)

func rtypeOff(section unsafe.Pointer, off int32) *rtype {
	return (*rtype)(add(section, uintptr(off), "sizeof(rtype) > 0"))
}

// typesByString returns the subslice of typelinks() whose elements have
// the given string representation.
// It may be empty (no known types with that string) or may have
// multiple elements (multiple types with that string).
func typesByString(s string) []*rtype {
	sections, offset := typelinks()
	var ret []*rtype

	for offsI, offs := range offset {
		section := sections[offsI]

		// We are looking for the first index i where the string becomes >= s.
		// This is a copy of sort.Search, with f(h) replaced by (*typ[h].String() >= s).
		i, j := 0, len(offs)
		for i < j {
			h := i + (j-i)>>1 // avoid overflow when computing h
			// i ≤ h < j
			if !(rtypeOff(section, offs[h]).String() >= s) {
				i = h + 1 // preserves f(i-1) == false
			} else {
				j = h // preserves f(j) == true
			}
		}
		// i == j, f(i-1) == false, and f(j) (= f(i)) == true  =>  answer is i.

		// Having found the first, linear scan forward to find the last.
		// We could do a second binary search, but the caller is going
		// to do a linear scan anyway.
		for j := i; j < len(offs); j++ {
			typ := rtypeOff(section, offs[j])
			if typ.String() != s {
				break
			}
			ret = append(ret, typ)
		}
	}
	return ret
}

// The lookupCache caches ArrayOf, ChanOf, MapOf and SliceOf lookups.
var lookupCache sync.Map // map[cacheKey]*rtype

// A cacheKey is the key for use in the lookupCache.
// Four values describe any of the types we are looking for:
// type kind, one or two subtypes, and an extra integer.
type cacheKey struct {
	kind  Kind
	t1    *rtype
	t2    *rtype
	extra uintptr
}

// The funcLookupCache caches FuncOf lookups.
// FuncOf does not share the common lookupCache since cacheKey is not
// sufficient to represent functions unambiguously.
var funcLookupCache struct {
	sync.Mutex // Guards stores (but not loads) on m.

	// m is a map[uint32][]*rtype keyed by the hash calculated in FuncOf.
	// Elements of m are append-only and thus safe for concurrent reading.
	m sync.Map
}

// ChanOf returns the channel type with the given direction and element type.
// For example, if t represents int, ChanOf(RecvDir, t) represents <-chan int.
//
// The gc runtime imposes a limit of 64 kB on channel element types.
// If t's size is equal to or exceeds this limit, ChanOf panics.
func ChanOf(dir ChanDir, t Type) Type {
	typ := t.(*rtype)

	// Look in cache.
	ckey := cacheKey{Chan, typ, nil, uintptr(dir)}
	if ch, ok := lookupCache.Load(ckey); ok {
		return ch.(*rtype)
	}

	// This restriction is imposed by the gc compiler and the runtime.
	if typ.size >= 1<<16 {
		panic("reflect.ChanOf: element size too large")
	}

	// Look in known types.
	var s string
	switch dir {
	default:
		panic("reflect.ChanOf: invalid dir")
	case SendDir:
		s = "chan<- " + typ.String()
	case RecvDir:
		s = "<-chan " + typ.String()
	case BothDir:
		typeStr := typ.String()
		if typeStr[0] == '<' {
			// typ is recv chan, need parentheses as "<-" associates with leftmost
			// chan possible, see:
			// * https://golang.org/ref/spec#Channel_types
			// * https://github.com/golang/go/issues/39897
			s = "chan (" + typeStr + ")"
		} else {
			s = "chan " + typeStr
		}
	}
	for _, tt := range typesByString(s) {
		ch := (*chanType)(unsafe.Pointer(tt))
		if ch.elem == typ && ch.dir == uintptr(dir) {
			ti, _ := lookupCache.LoadOrStore(ckey, tt)
			return ti.(Type)
		}
	}

	// Make a channel type.
	var ichan any = (chan unsafe.Pointer)(nil)
	prototype := *(**chanType)(unsafe.Pointer(&ichan))
	ch := *prototype
	ch.tflag = tflagRegularMemory
	ch.dir = uintptr(dir)
	ch.str = resolveReflectName(newName(s, "", false))
	ch.hash = fnv1(typ.hash, 'c', byte(dir))
	ch.elem = typ

	ti, _ := lookupCache.LoadOrStore(ckey, &ch.rtype)
	return ti.(Type)
}

// MapOf returns the map type with the given key and element types.
// For example, if k represents int and e represents string,
// MapOf(k, e) represents map[int]string.
//
// If the key type is not a valid map key type (that is, if it does
// not implement Go's == operator), MapOf panics.
func MapOf(key, elem Type) Type {
	ktyp := key.(*rtype)
	etyp := elem.(*rtype)

	if ktyp.equal == nil {
		panic("reflect.MapOf: invalid key type " + ktyp.String())
	}

	// Look in cache.
	ckey := cacheKey{Map, ktyp, etyp, 0}
	if mt, ok := lookupCache.Load(ckey); ok {
		return mt.(Type)
	}

	// Look in known types.
	s := "map[" + ktyp.String() + "]" + etyp.String()
	for _, tt := range typesByString(s) {
		mt := (*mapType)(unsafe.Pointer(tt))
		if mt.key == ktyp && mt.elem == etyp {
			ti, _ := lookupCache.LoadOrStore(ckey, tt)
			return ti.(Type)
		}
	}

	// Make a map type.
	// Note: flag values must match those used in the TMAP case
	// in ../cmd/compile/internal/reflectdata/reflect.go:writeType.
	var imap any = (map[unsafe.Pointer]unsafe.Pointer)(nil)
	mt := **(**mapType)(unsafe.Pointer(&imap))
	mt.str = resolveReflectName(newName(s, "", false))
	mt.tflag = 0
	mt.hash = fnv1(etyp.hash, 'm', byte(ktyp.hash>>24), byte(ktyp.hash>>16), byte(ktyp.hash>>8), byte(ktyp.hash))
	mt.key = ktyp
	mt.elem = etyp
	mt.bucket = bucketOf(ktyp, etyp)
	mt.hasher = func(p unsafe.Pointer, seed uintptr) uintptr {
		return typehash(ktyp, p, seed)
	}
	mt.flags = 0
	if ktyp.size > maxKeySize {
		mt.keysize = uint8(goarch.PtrSize)
		mt.flags |= 1 // indirect key
	} else {
		mt.keysize = uint8(ktyp.size)
	}
	if etyp.size > maxValSize {
		mt.valuesize = uint8(goarch.PtrSize)
		mt.flags |= 2 // indirect value
	} else {
		mt.valuesize = uint8(etyp.size)
	}
	mt.bucketsize = uint16(mt.bucket.size)
	if isReflexive(ktyp) {
		mt.flags |= 4
	}
	if needKeyUpdate(ktyp) {
		mt.flags |= 8
	}
	if hashMightPanic(ktyp) {
		mt.flags |= 16
	}
	mt.ptrToThis = 0

	ti, _ := lookupCache.LoadOrStore(ckey, &mt.rtype)
	return ti.(Type)
}

// TODO(crawshaw): as these funcTypeFixedN structs have no methods,
// they could be defined at runtime using the StructOf function.
type funcTypeFixed4 struct {
	funcType
	args [4]*rtype
}
type funcTypeFixed8 struct {
	funcType
	args [8]*rtype
}
type funcTypeFixed16 struct {
	funcType
	args [16]*rtype
}
type funcTypeFixed32 struct {
	funcType
	args [32]*rtype
}
type funcTypeFixed64 struct {
	funcType
	args [64]*rtype
}
type funcTypeFixed128 struct {
	funcType
	args [128]*rtype
}

// FuncOf returns the function type with the given argument and result types.
// For example if k represents int and e represents string,
// FuncOf([]Type{k}, []Type{e}, false) represents func(int) string.
//
// The variadic argument controls whether the function is variadic. FuncOf
// panics if the in[len(in)-1] does not represent a slice and variadic is
// true.
func FuncOf(in, out []Type, variadic bool) Type {
	if variadic && (len(in) == 0 || in[len(in)-1].Kind() != Slice) {
		panic("reflect.FuncOf: last arg of variadic func must be slice")
	}

	// Make a func type.
	var ifunc any = (func())(nil)
	prototype := *(**funcType)(unsafe.Pointer(&ifunc))
	n := len(in) + len(out)

	var ft *funcType
	var args []*rtype
	switch {
	case n <= 4:
		fixed := new(funcTypeFixed4)
		args = fixed.args[:0:len(fixed.args)]
		ft = &fixed.funcType
	case n <= 8:
		fixed := new(funcTypeFixed8)
		args = fixed.args[:0:len(fixed.args)]
		ft = &fixed.funcType
	case n <= 16:
		fixed := new(funcTypeFixed16)
		args = fixed.args[:0:len(fixed.args)]
		ft = &fixed.funcType
	case n <= 32:
		fixed := new(funcTypeFixed32)
		args = fixed.args[:0:len(fixed.args)]
		ft = &fixed.funcType
	case n <= 64:
		fixed := new(funcTypeFixed64)
		args = fixed.args[:0:len(fixed.args)]
		ft = &fixed.funcType
	case n <= 128:
		fixed := new(funcTypeFixed128)
		args = fixed.args[:0:len(fixed.args)]
		ft = &fixed.funcType
	default:
		panic("reflect.FuncOf: too many arguments")
	}
	*ft = *prototype

	// Build a hash and minimally populate ft.
	var hash uint32
	for _, in := range in {
		t := in.(*rtype)
		args = append(args, t)
		hash = fnv1(hash, byte(t.hash>>24), byte(t.hash>>16), byte(t.hash>>8), byte(t.hash))
	}
	if variadic {
		hash = fnv1(hash, 'v')
	}
	hash = fnv1(hash, '.')
	for _, out := range out {
		t := out.(*rtype)
		args = append(args, t)
		hash = fnv1(hash, byte(t.hash>>24), byte(t.hash>>16), byte(t.hash>>8), byte(t.hash))
	}
	if len(args) > 50 {
		panic("reflect.FuncOf does not support more than 50 arguments")
	}
	ft.tflag = 0
	ft.hash = hash
	ft.inCount = uint16(len(in))
	ft.outCount = uint16(len(out))
	if variadic {
		ft.outCount |= 1 << 15
	}

	// Look in cache.
	if ts, ok := funcLookupCache.m.Load(hash); ok {
		for _, t := range ts.([]*rtype) {
			if haveIdenticalUnderlyingType(&ft.rtype, t, true) {
				return t
			}
		}
	}

	// Not in cache, lock and retry.
	funcLookupCache.Lock()
	defer funcLookupCache.Unlock()
	if ts, ok := funcLookupCache.m.Load(hash); ok {
		for _, t := range ts.([]*rtype) {
			if haveIdenticalUnderlyingType(&ft.rtype, t, true) {
				return t
			}
		}
	}

	addToCache := func(tt *rtype) Type {
		var rts []*rtype
		if rti, ok := funcLookupCache.m.Load(hash); ok {
			rts = rti.([]*rtype)
		}
		funcLookupCache.m.Store(hash, append(rts, tt))
		return tt
	}

	// Look in known types for the same string representation.
	str := funcStr(ft)
	for _, tt := range typesByString(str) {
		if haveIdenticalUnderlyingType(&ft.rtype, tt, true) {
			return addToCache(tt)
		}
	}

	// Populate the remaining fields of ft and store in cache.
	ft.str = resolveReflectName(newName(str, "", false))
	ft.ptrToThis = 0
	return addToCache(&ft.rtype)
}

// funcStr builds a string representation of a funcType.
func funcStr(ft *funcType) string {
	repr := make([]byte, 0, 64)
	repr = append(repr, "func("...)
	for i, t := range ft.in() {
		if i > 0 {
			repr = append(repr, ", "...)
		}
		if ft.IsVariadic() && i == int(ft.inCount)-1 {
			repr = append(repr, "..."...)
			repr = append(repr, (*sliceType)(unsafe.Pointer(t)).elem.String()...)
		} else {
			repr = append(repr, t.String()...)
		}
	}
	repr = append(repr, ')')
	out := ft.out()
	if len(out) == 1 {
		repr = append(repr, ' ')
	} else if len(out) > 1 {
		repr = append(repr, " ("...)
	}
	for i, t := range out {
		if i > 0 {
			repr = append(repr, ", "...)
		}
		repr = append(repr, t.String()...)
	}
	if len(out) > 1 {
		repr = append(repr, ')')
	}
	return string(repr)
}

// isReflexive reports whether the == operation on the type is reflexive.
// That is, x == x for all values x of type t.
func isReflexive(t *rtype) bool {
	switch t.Kind() {
	case Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr, Chan, Pointer, String, UnsafePointer:
		return true
	case Float32, Float64, Complex64, Complex128, Interface:
		return false
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return isReflexive(tt.elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.fields {
			if !isReflexive(f.typ) {
				return false
			}
		}
		return true
	default:
		// Func, Map, Slice, Invalid
		panic("isReflexive called on non-key type " + t.String())
	}
}

// needKeyUpdate reports whether map overwrites require the key to be copied.
func needKeyUpdate(t *rtype) bool {
	switch t.Kind() {
	case Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr, Chan, Pointer, UnsafePointer:
		return false
	case Float32, Float64, Complex64, Complex128, Interface, String:
		// Float keys can be updated from +0 to -0.
		// String keys can be updated to use a smaller backing store.
		// Interfaces might have floats of strings in them.
		return true
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return needKeyUpdate(tt.elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.fields {
			if needKeyUpdate(f.typ) {
				return true
			}
		}
		return false
	default:
		// Func, Map, Slice, Invalid
		panic("needKeyUpdate called on non-key type " + t.String())
	}
}

// hashMightPanic reports whether the hash of a map key of type t might panic.
func hashMightPanic(t *rtype) bool {
	switch t.Kind() {
	case Interface:
		return true
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return hashMightPanic(tt.elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.fields {
			if hashMightPanic(f.typ) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// Make sure these routines stay in sync with ../../runtime/map.go!
// These types exist only for GC, so we only fill out GC relevant info.
// Currently, that's just size and the GC program. We also fill in string
// for possible debugging use.
const (
	bucketSize uintptr = 8
	maxKeySize uintptr = 128
	maxValSize uintptr = 128
)

func bucketOf(ktyp, etyp *rtype) *rtype {
	if ktyp.size > maxKeySize {
		ktyp = PointerTo(ktyp).(*rtype)
	}
	if etyp.size > maxValSize {
		etyp = PointerTo(etyp).(*rtype)
	}

	// Prepare GC data if any.
	// A bucket is at most bucketSize*(1+maxKeySize+maxValSize)+2*ptrSize bytes,
	// or 2072 bytes, or 259 pointer-size words, or 33 bytes of pointer bitmap.
	// Note that since the key and value are known to be <= 128 bytes,
	// they're guaranteed to have bitmaps instead of GC programs.
	var gcdata *byte
	var ptrdata uintptr
	var overflowPad uintptr

	size := bucketSize*(1+ktyp.size+etyp.size) + overflowPad + goarch.PtrSize
	if size&uintptr(ktyp.align-1) != 0 || size&uintptr(etyp.align-1) != 0 {
		panic("reflect: bad size computation in MapOf")
	}

	if ktyp.ptrdata != 0 || etyp.ptrdata != 0 {
		nptr := (bucketSize*(1+ktyp.size+etyp.size) + goarch.PtrSize) / goarch.PtrSize
		mask := make([]byte, (nptr+7)/8)
		base := bucketSize / goarch.PtrSize

		if ktyp.ptrdata != 0 {
			emitGCMask(mask, base, ktyp, bucketSize)
		}
		base += bucketSize * ktyp.size / goarch.PtrSize

		if etyp.ptrdata != 0 {
			emitGCMask(mask, base, etyp, bucketSize)
		}
		base += bucketSize * etyp.size / goarch.PtrSize
		base += overflowPad / goarch.PtrSize

		word := base
		mask[word/8] |= 1 << (word % 8)
		gcdata = &mask[0]
		ptrdata = (word + 1) * goarch.PtrSize

		// overflow word must be last
		if ptrdata != size {
			panic("reflect: bad layout computation in MapOf")
		}
	}

	b := &rtype{
		align:   goarch.PtrSize,
		size:    size,
		kind:    uint8(Struct),
		ptrdata: ptrdata,
		gcdata:  gcdata,
	}
	if overflowPad > 0 {
		b.align = 8
	}
	s := "bucket(" + ktyp.String() + "," + etyp.String() + ")"
	b.str = resolveReflectName(newName(s, "", false))
	return b
}

func (t *rtype) gcSlice(begin, end uintptr) []byte {
	return (*[1 << 30]byte)(unsafe.Pointer(t.gcdata))[begin:end:end]
}

// emitGCMask writes the GC mask for [n]typ into out, starting at bit
// offset base.
func emitGCMask(out []byte, base uintptr, typ *rtype, n uintptr) {
	if typ.kind&kindGCProg != 0 {
		panic("reflect: unexpected GC program")
	}
	ptrs := typ.ptrdata / goarch.PtrSize
	words := typ.size / goarch.PtrSize
	mask := typ.gcSlice(0, (ptrs+7)/8)
	for j := uintptr(0); j < ptrs; j++ {
		if (mask[j/8]>>(j%8))&1 != 0 {
			for i := uintptr(0); i < n; i++ {
				k := base + i*words + j
				out[k/8] |= 1 << (k % 8)
			}
		}
	}
}

// appendGCProg appends the GC program for the first ptrdata bytes of
// typ to dst and returns the extended slice.
func appendGCProg(dst []byte, typ *rtype) []byte {
	if typ.kind&kindGCProg != 0 {
		// Element has GC program; emit one element.
		n := uintptr(*(*uint32)(unsafe.Pointer(typ.gcdata)))
		prog := typ.gcSlice(4, 4+n-1)
		return append(dst, prog...)
	}

	// Element is small with pointer mask; use as literal bits.
	ptrs := typ.ptrdata / goarch.PtrSize
	mask := typ.gcSlice(0, (ptrs+7)/8)

	// Emit 120-bit chunks of full bytes (max is 127 but we avoid using partial bytes).
	for ; ptrs > 120; ptrs -= 120 {
		dst = append(dst, 120)
		dst = append(dst, mask[:15]...)
		mask = mask[15:]
	}

	dst = append(dst, byte(ptrs))
	dst = append(dst, mask...)
	return dst
}

// SliceOf returns the slice type with element type t.
// For example, if t represents int, SliceOf(t) represents []int.
func SliceOf(t Type) Type {
	typ := t.(*rtype)

	// Look in cache.
	ckey := cacheKey{Slice, typ, nil, 0}
	if slice, ok := lookupCache.Load(ckey); ok {
		return slice.(Type)
	}

	// Look in known types.
	s := "[]" + typ.String()
	for _, tt := range typesByString(s) {
		slice := (*sliceType)(unsafe.Pointer(tt))
		if slice.elem == typ {
			ti, _ := lookupCache.LoadOrStore(ckey, tt)
			return ti.(Type)
		}
	}

	// Make a slice type.
	var islice any = ([]unsafe.Pointer)(nil)
	prototype := *(**sliceType)(unsafe.Pointer(&islice))
	slice := *prototype
	slice.tflag = 0
	slice.str = resolveReflectName(newName(s, "", false))
	slice.hash = fnv1(typ.hash, '[')
	slice.elem = typ
	slice.ptrToThis = 0

	ti, _ := lookupCache.LoadOrStore(ckey, &slice.rtype)
	return ti.(Type)
}

// The structLookupCache caches StructOf lookups.
// StructOf does not share the common lookupCache since we need to pin
// the memory associated with *structTypeFixedN.
var structLookupCache struct {
	sync.Mutex // Guards stores (but not loads) on m.

	// m is a map[uint32][]Type keyed by the hash calculated in StructOf.
	// Elements in m are append-only and thus safe for concurrent reading.
	m sync.Map
}

type structTypeUncommon struct {
	structType
	u uncommonType
}

// isLetter reports whether a given 'rune' is classified as a Letter.
func isLetter(ch rune) bool {
	return 'a' <= ch && ch <= 'z' || 'A' <= ch && ch <= 'Z' || ch == '_' || ch >= utf8.RuneSelf && unicode.IsLetter(ch)
}

// isValidFieldName checks if a string is a valid (struct) field name or not.
//
// According to the language spec, a field name should be an identifier.
//
// identifier = letter { letter | unicode_digit } .
// letter = unicode_letter | "_" .
func isValidFieldName(fieldName string) bool {
	for i, c := range fieldName {
		if i == 0 && !isLetter(c) {
			return false
		}

		if !(isLetter(c) || unicode.IsDigit(c)) {
			return false
		}
	}

	return len(fieldName) > 0
}

// StructOf returns the struct type containing fields.
// The Offset and Index fields are ignored and computed as they would be
// by the compiler.
//
// StructOf currently does not generate wrapper methods for embedded
// fields and panics if passed unexported StructFields.
// These limitations may be lifted in a future version.
func StructOf(fields []StructField) Type {
	var (
		hash       = fnv1(0, []byte("struct {")...)
		size       uintptr
		typalign   uint8
		comparable = true
		methods    []method

		fs   = make([]structField, len(fields))
		repr = make([]byte, 0, 64)
		fset = map[string]struct{}{} // fields' names

		hasGCProg = false // records whether a struct-field type has a GCProg
	)

	lastzero := uintptr(0)
	repr = append(repr, "struct {"...)
	pkgpath := ""
	for i, field := range fields {
		if field.Name == "" {
			panic("reflect.StructOf: field " + strconv.Itoa(i) + " has no name")
		}
		if !isValidFieldName(field.Name) {
			panic("reflect.StructOf: field " + strconv.Itoa(i) + " has invalid name")
		}
		if field.Type == nil {
			panic("reflect.StructOf: field " + strconv.Itoa(i) + " has no type")
		}
		f, fpkgpath := runtimeStructField(field)
		ft := f.typ
		if ft.kind&kindGCProg != 0 {
			hasGCProg = true
		}
		if fpkgpath != "" {
			if pkgpath == "" {
				pkgpath = fpkgpath
			} else if pkgpath != fpkgpath {
				panic("reflect.Struct: fields with different PkgPath " + pkgpath + " and " + fpkgpath)
			}
		}

		// Update string and hash
		name := f.name.name()
		hash = fnv1(hash, []byte(name)...)
		repr = append(repr, (" " + name)...)
		if f.embedded() {
			// Embedded field
			if f.typ.Kind() == Pointer {
				// Embedded ** and *interface{} are illegal
				elem := ft.Elem()
				if k := elem.Kind(); k == Pointer || k == Interface {
					panic("reflect.StructOf: illegal embedded field type " + ft.String())
				}
			}

			switch f.typ.Kind() {
			case Interface:
				ift := (*interfaceType)(unsafe.Pointer(ft))
				for im, m := range ift.methods {
					if ift.nameOff(m.name).pkgPath() != "" {
						// TODO(sbinet).  Issue 15924.
						panic("reflect: embedded interface with unexported method(s) not implemented")
					}

					var (
						mtyp    = ift.typeOff(m.typ)
						ifield  = i
						imethod = im
						ifn     Value
						tfn     Value
					)

					if ft.kind&kindDirectIface != 0 {
						tfn = MakeFunc(mtyp, func(in []Value) []Value {
							var args []Value
							var recv = in[0]
							if len(in) > 1 {
								args = in[1:]
							}
							return recv.Field(ifield).Method(imethod).Call(args)
						})
						ifn = MakeFunc(mtyp, func(in []Value) []Value {
							var args []Value
							var recv = in[0]
							if len(in) > 1 {
								args = in[1:]
							}
							return recv.Field(ifield).Method(imethod).Call(args)
						})
					} else {
						tfn = MakeFunc(mtyp, func(in []Value) []Value {
							var args []Value
							var recv = in[0]
							if len(in) > 1 {
								args = in[1:]
							}
							return recv.Field(ifield).Method(imethod).Call(args)
						})
						ifn = MakeFunc(mtyp, func(in []Value) []Value {
							var args []Value
							var recv = Indirect(in[0])
							if len(in) > 1 {
								args = in[1:]
							}
							return recv.Field(ifield).Method(imethod).Call(args)
						})
					}

					methods = append(methods, method{
						name: resolveReflectName(ift.nameOff(m.name)),
						mtyp: resolveReflectType(mtyp),
						ifn:  resolveReflectText(unsafe.Pointer(&ifn)),
						tfn:  resolveReflectText(unsafe.Pointer(&tfn)),
					})
				}
			case Pointer:
				ptr := (*ptrType)(unsafe.Pointer(ft))
				if unt := ptr.uncommon(); unt != nil {
					if i > 0 && unt.mcount > 0 {
						// Issue 15924.
						panic("reflect: embedded type with methods not implemented if type is not first field")
					}
					if len(fields) > 1 {
						panic("reflect: embedded type with methods not implemented if there is more than one field")
					}
					for _, m := range unt.methods() {
						mname := ptr.nameOff(m.name)
						if mname.pkgPath() != "" {
							// TODO(sbinet).
							// Issue 15924.
							panic("reflect: embedded interface with unexported method(s) not implemented")
						}
						methods = append(methods, method{
							name: resolveReflectName(mname),
							mtyp: resolveReflectType(ptr.typeOff(m.mtyp)),
							ifn:  resolveReflectText(ptr.textOff(m.ifn)),
							tfn:  resolveReflectText(ptr.textOff(m.tfn)),
						})
					}
				}
				if unt := ptr.elem.uncommon(); unt != nil {
					for _, m := range unt.methods() {
						mname := ptr.nameOff(m.name)
						if mname.pkgPath() != "" {
							// TODO(sbinet)
							// Issue 15924.
							panic("reflect: embedded interface with unexported method(s) not implemented")
						}
						methods = append(methods, method{
							name: resolveReflectName(mname),
							mtyp: resolveReflectType(ptr.elem.typeOff(m.mtyp)),
							ifn:  resolveReflectText(ptr.elem.textOff(m.ifn)),
							tfn:  resolveReflectText(ptr.elem.textOff(m.tfn)),
						})
					}
				}
			default:
				if unt := ft.uncommon(); unt != nil {
					if i > 0 && unt.mcount > 0 {
						// Issue 15924.
						panic("reflect: embedded type with methods not implemented if type is not first field")
					}
					if len(fields) > 1 && ft.kind&kindDirectIface != 0 {
						panic("reflect: embedded type with methods not implemented for non-pointer type")
					}
					for _, m := range unt.methods() {
						mname := ft.nameOff(m.name)
						if mname.pkgPath() != "" {
							// TODO(sbinet)
							// Issue 15924.
							panic("reflect: embedded interface with unexported method(s) not implemented")
						}
						methods = append(methods, method{
							name: resolveReflectName(mname),
							mtyp: resolveReflectType(ft.typeOff(m.mtyp)),
							ifn:  resolveReflectText(ft.textOff(m.ifn)),
							tfn:  resolveReflectText(ft.textOff(m.tfn)),
						})

					}
				}
			}
		}
		if _, dup := fset[name]; dup && name != "_" {
			panic("reflect.StructOf: duplicate field " + name)
		}
		fset[name] = struct{}{}

		hash = fnv1(hash, byte(ft.hash>>24), byte(ft.hash>>16), byte(ft.hash>>8), byte(ft.hash))

		repr = append(repr, (" " + ft.String())...)
		if f.name.hasTag() {
			hash = fnv1(hash, []byte(f.name.tag())...)
			repr = append(repr, (" " + strconv.Quote(f.name.tag()))...)
		}
		if i < len(fields)-1 {
			repr = append(repr, ';')
		}

		comparable = comparable && (ft.equal != nil)

		offset := align(size, uintptr(ft.align))
		if ft.align > typalign {
			typalign = ft.align
		}
		size = offset + ft.size
		f.offsetEmbed |= offset << 1

		if ft.size == 0 {
			lastzero = size
		}

		fs[i] = f
	}

	if size > 0 && lastzero == size {
		// This is a non-zero sized struct that ends in a
		// zero-sized field. We add an extra byte of padding,
		// to ensure that taking the address of the final
		// zero-sized field can't manufacture a pointer to the
		// next object in the heap. See issue 9401.
		size++
	}

	var typ *structType
	var ut *uncommonType

	if len(methods) == 0 {
		t := new(structTypeUncommon)
		typ = &t.structType
		ut = &t.u
	} else {
		// A *rtype representing a struct is followed directly in memory by an
		// array of method objects representing the methods attached to the
		// struct. To get the same layout for a run time generated type, we
		// need an array directly following the uncommonType memory.
		// A similar strategy is used for funcTypeFixed4, ...funcTypeFixedN.
		tt := New(StructOf([]StructField{
			{Name: "S", Type: TypeOf(structType{})},
			{Name: "U", Type: TypeOf(uncommonType{})},
			{Name: "M", Type: ArrayOf(len(methods), TypeOf(methods[0]))},
		}))

		typ = (*structType)(tt.Elem().Field(0).Addr().UnsafePointer())
		ut = (*uncommonType)(tt.Elem().Field(1).Addr().UnsafePointer())

		copy(tt.Elem().Field(2).Slice(0, len(methods)).Interface().([]method), methods)
	}
	// TODO(sbinet): Once we allow embedding multiple types,
	// methods will need to be sorted like the compiler does.
	// TODO(sbinet): Once we allow non-exported methods, we will
	// need to compute xcount as the number of exported methods.
	ut.mcount = uint16(len(methods))
	ut.xcount = ut.mcount
	ut.moff = uint32(unsafe.Sizeof(uncommonType{}))

	if len(fs) > 0 {
		repr = append(repr, ' ')
	}
	repr = append(repr, '}')
	hash = fnv1(hash, '}')
	str := string(repr)

	// Round the size up to be a multiple of the alignment.
	size = align(size, uintptr(typalign))

	// Make the struct type.
	var istruct any = struct{}{}
	prototype := *(**structType)(unsafe.Pointer(&istruct))
	*typ = *prototype
	typ.fields = fs
	if pkgpath != "" {
		typ.pkgPath = newName(pkgpath, "", false)
	}

	// Look in cache.
	if ts, ok := structLookupCache.m.Load(hash); ok {
		for _, st := range ts.([]Type) {
			t := st.common()
			if haveIdenticalUnderlyingType(&typ.rtype, t, true) {
				return t
			}
		}
	}

	// Not in cache, lock and retry.
	structLookupCache.Lock()
	defer structLookupCache.Unlock()
	if ts, ok := structLookupCache.m.Load(hash); ok {
		for _, st := range ts.([]Type) {
			t := st.common()
			if haveIdenticalUnderlyingType(&typ.rtype, t, true) {
				return t
			}
		}
	}

	addToCache := func(t Type) Type {
		var ts []Type
		if ti, ok := structLookupCache.m.Load(hash); ok {
			ts = ti.([]Type)
		}
		structLookupCache.m.Store(hash, append(ts, t))
		return t
	}

	// Look in known types.
	for _, t := range typesByString(str) {
		if haveIdenticalUnderlyingType(&typ.rtype, t, true) {
			// even if 't' wasn't a structType with methods, we should be ok
			// as the 'u uncommonType' field won't be accessed except when
			// tflag&tflagUncommon is set.
			return addToCache(t)
		}
	}

	typ.str = resolveReflectName(newName(str, "", false))
	typ.tflag = 0 // TODO: set tflagRegularMemory
	typ.hash = hash
	typ.size = size
	typ.ptrdata = typeptrdata(typ.common())
	typ.align = typalign
	typ.fieldAlign = typalign
	typ.ptrToThis = 0
	if len(methods) > 0 {
		typ.tflag |= tflagUncommon
	}

	if hasGCProg {
		lastPtrField := 0
		for i, ft := range fs {
			if ft.typ.pointers() {
				lastPtrField = i
			}
		}
		prog := []byte{0, 0, 0, 0} // will be length of prog
		var off uintptr
		for i, ft := range fs {
			if i > lastPtrField {
				// gcprog should not include anything for any field after
				// the last field that contains pointer data
				break
			}
			if !ft.typ.pointers() {
				// Ignore pointerless fields.
				continue
			}
			// Pad to start of this field with zeros.
			if ft.offset() > off {
				n := (ft.offset() - off) / goarch.PtrSize
				prog = append(prog, 0x01, 0x00) // emit a 0 bit
				if n > 1 {
					prog = append(prog, 0x81)      // repeat previous bit
					prog = appendVarint(prog, n-1) // n-1 times
				}
				off = ft.offset()
			}

			prog = appendGCProg(prog, ft.typ)
			off += ft.typ.ptrdata
		}
		prog = append(prog, 0)
		*(*uint32)(unsafe.Pointer(&prog[0])) = uint32(len(prog) - 4)
		typ.kind |= kindGCProg
		typ.gcdata = &prog[0]
	} else {
		typ.kind &^= kindGCProg
		bv := new(bitVector)
		addTypeBits(bv, 0, typ.common())
		if len(bv.data) > 0 {
			typ.gcdata = &bv.data[0]
		}
	}
	typ.equal = nil
	if comparable {
		typ.equal = func(p, q unsafe.Pointer) bool {
			for _, ft := range typ.fields {
				pi := add(p, ft.offset(), "&x.field safe")
				qi := add(q, ft.offset(), "&x.field safe")
				if !ft.typ.equal(pi, qi) {
					return false
				}
			}
			return true
		}
	}

	switch {
	case len(fs) == 1 && !ifaceIndir(fs[0].typ):
		// structs of 1 direct iface type can be direct
		typ.kind |= kindDirectIface
	default:
		typ.kind &^= kindDirectIface
	}

	return addToCache(&typ.rtype)
}

// runtimeStructField takes a StructField value passed to StructOf and
// returns both the corresponding internal representation, of type
// structField, and the pkgpath value to use for this field.
func runtimeStructField(field StructField) (structField, string) {
	if field.Anonymous && field.PkgPath != "" {
		panic("reflect.StructOf: field \"" + field.Name + "\" is anonymous but has PkgPath set")
	}

	if field.IsExported() {
		// Best-effort check for misuse.
		// Since this field will be treated as exported, not much harm done if Unicode lowercase slips through.
		c := field.Name[0]
		if 'a' <= c && c <= 'z' || c == '_' {
			panic("reflect.StructOf: field \"" + field.Name + "\" is unexported but missing PkgPath")
		}
	}

	offsetEmbed := uintptr(0)
	if field.Anonymous {
		offsetEmbed |= 1
	}

	resolveReflectType(field.Type.common()) // install in runtime
	f := structField{
		name:        newName(field.Name, string(field.Tag), field.IsExported()),
		typ:         field.Type.common(),
		offsetEmbed: offsetEmbed,
	}
	return f, field.PkgPath
}

// typeptrdata returns the length in bytes of the prefix of t
// containing pointer data. Anything after this offset is scalar data.
// keep in sync with ../cmd/compile/internal/reflectdata/reflect.go
func typeptrdata(t *rtype) uintptr {
	switch t.Kind() {
	case Struct:
		st := (*structType)(unsafe.Pointer(t))
		// find the last field that has pointers.
		field := -1
		for i := range st.fields {
			ft := st.fields[i].typ
			if ft.pointers() {
				field = i
			}
		}
		if field == -1 {
			return 0
		}
		f := st.fields[field]
		return f.offset() + f.typ.ptrdata

	default:
		panic("reflect.typeptrdata: unexpected type, " + t.String())
	}
}

// See cmd/compile/internal/reflectdata/reflect.go for derivation of constant.
const maxPtrmaskBytes = 2048

// ArrayOf returns the array type with the given length and element type.
// For example, if t represents int, ArrayOf(5, t) represents [5]int.
//
// If the resulting type would be larger than the available address space,
// ArrayOf panics.
func ArrayOf(length int, elem Type) Type {
	if length < 0 {
		panic("reflect: negative length passed to ArrayOf")
	}

	typ := elem.(*rtype)

	// Look in cache.
	ckey := cacheKey{Array, typ, nil, uintptr(length)}
	if array, ok := lookupCache.Load(ckey); ok {
		return array.(Type)
	}

	// Look in known types.
	s := "[" + strconv.Itoa(length) + "]" + typ.String()
	for _, tt := range typesByString(s) {
		array := (*arrayType)(unsafe.Pointer(tt))
		if array.elem == typ {
			ti, _ := lookupCache.LoadOrStore(ckey, tt)
			return ti.(Type)
		}
	}

	// Make an array type.
	var iarray any = [1]unsafe.Pointer{}
	prototype := *(**arrayType)(unsafe.Pointer(&iarray))
	array := *prototype
	array.tflag = typ.tflag & tflagRegularMemory
	array.str = resolveReflectName(newName(s, "", false))
	array.hash = fnv1(typ.hash, '[')
	for n := uint32(length); n > 0; n >>= 8 {
		array.hash = fnv1(array.hash, byte(n))
	}
	array.hash = fnv1(array.hash, ']')
	array.elem = typ
	array.ptrToThis = 0
	if typ.size > 0 {
		max := ^uintptr(0) / typ.size
		if uintptr(length) > max {
			panic("reflect.ArrayOf: array size would exceed virtual address space")
		}
	}
	array.size = typ.size * uintptr(length)
	if length > 0 && typ.ptrdata != 0 {
		array.ptrdata = typ.size*uintptr(length-1) + typ.ptrdata
	}
	array.align = typ.align
	array.fieldAlign = typ.fieldAlign
	array.len = uintptr(length)
	array.slice = SliceOf(elem).(*rtype)

	switch {
	case typ.ptrdata == 0 || array.size == 0:
		// No pointers.
		array.gcdata = nil
		array.ptrdata = 0

	case length == 1:
		// In memory, 1-element array looks just like the element.
		array.kind |= typ.kind & kindGCProg
		array.gcdata = typ.gcdata
		array.ptrdata = typ.ptrdata

	case typ.kind&kindGCProg == 0 && array.size <= maxPtrmaskBytes*8*goarch.PtrSize:
		// Element is small with pointer mask; array is still small.
		// Create direct pointer mask by turning each 1 bit in elem
		// into length 1 bits in larger mask.
		mask := make([]byte, (array.ptrdata/goarch.PtrSize+7)/8)
		emitGCMask(mask, 0, typ, array.len)
		array.gcdata = &mask[0]

	default:
		// Create program that emits one element
		// and then repeats to make the array.
		prog := []byte{0, 0, 0, 0} // will be length of prog
		prog = appendGCProg(prog, typ)
		// Pad from ptrdata to size.
		elemPtrs := typ.ptrdata / goarch.PtrSize
		elemWords := typ.size / goarch.PtrSize
		if elemPtrs < elemWords {
			// Emit literal 0 bit, then repeat as needed.
			prog = append(prog, 0x01, 0x00)
			if elemPtrs+1 < elemWords {
				prog = append(prog, 0x81)
				prog = appendVarint(prog, elemWords-elemPtrs-1)
			}
		}
		// Repeat length-1 times.
		if elemWords < 0x80 {
			prog = append(prog, byte(elemWords|0x80))
		} else {
			prog = append(prog, 0x80)
			prog = appendVarint(prog, elemWords)
		}
		prog = appendVarint(prog, uintptr(length)-1)
		prog = append(prog, 0)
		*(*uint32)(unsafe.Pointer(&prog[0])) = uint32(len(prog) - 4)
		array.kind |= kindGCProg
		array.gcdata = &prog[0]
		array.ptrdata = array.size // overestimate but ok; must match program
	}

	etyp := typ.common()
	esize := etyp.Size()

	array.equal = nil
	if eequal := etyp.equal; eequal != nil {
		array.equal = func(p, q unsafe.Pointer) bool {
			for i := 0; i < length; i++ {
				pi := arrayAt(p, i, esize, "i < length")
				qi := arrayAt(q, i, esize, "i < length")
				if !eequal(pi, qi) {
					return false
				}

			}
			return true
		}
	}

	switch {
	case length == 1 && !ifaceIndir(typ):
		// array of 1 direct iface type can be direct
		array.kind |= kindDirectIface
	default:
		array.kind &^= kindDirectIface
	}

	ti, _ := lookupCache.LoadOrStore(ckey, &array.rtype)
	return ti.(Type)
}

func appendVarint(x []byte, v uintptr) []byte {
	for ; v >= 0x80; v >>= 7 {
		x = append(x, byte(v|0x80))
	}
	x = append(x, byte(v))
	return x
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

type layoutKey struct {
	ftyp *funcType // function signature
	rcvr *rtype    // receiver type, or nil if none
}

type layoutType struct {
	t         *rtype
	framePool *sync.Pool
	abi       abiDesc
}

var layoutCache sync.Map // map[layoutKey]layoutType

// funcLayout computes a struct type representing the layout of the
// stack-assigned function arguments and return values for the function
// type t.
// If rcvr != nil, rcvr specifies the type of the receiver.
// The returned type exists only for GC, so we only fill out GC relevant info.
// Currently, that's just size and the GC program. We also fill in
// the name for possible debugging use.
func funcLayout(t *funcType, rcvr *rtype) (frametype *rtype, framePool *sync.Pool, abi abiDesc) {
	if t.Kind() != Func {
		panic("reflect: funcLayout of non-func type " + t.String())
	}
	if rcvr != nil && rcvr.Kind() == Interface {
		panic("reflect: funcLayout with interface receiver " + rcvr.String())
	}
	k := layoutKey{t, rcvr}
	if lti, ok := layoutCache.Load(k); ok {
		lt := lti.(layoutType)
		return lt.t, lt.framePool, lt.abi
	}

	// Compute the ABI layout.
	abi = newAbiDesc(t, rcvr)

	// build dummy rtype holding gc program
	x := &rtype{
		align: goarch.PtrSize,
		// Don't add spill space here; it's only necessary in
		// reflectcall's frame, not in the allocated frame.
		// TODO(mknyszek): Remove this comment when register
		// spill space in the frame is no longer required.
		size:    align(abi.retOffset+abi.ret.stackBytes, goarch.PtrSize),
		ptrdata: uintptr(abi.stackPtrs.n) * goarch.PtrSize,
	}
	if abi.stackPtrs.n > 0 {
		x.gcdata = &abi.stackPtrs.data[0]
	}

	var s string
	if rcvr != nil {
		s = "methodargs(" + rcvr.String() + ")(" + t.String() + ")"
	} else {
		s = "funcargs(" + t.String() + ")"
	}
	x.str = resolveReflectName(newName(s, "", false))

	// cache result for future callers
	framePool = &sync.Pool{New: func() any {
		return unsafe_New(x)
	}}
	lti, _ := layoutCache.LoadOrStore(k, layoutType{
		t:         x,
		framePool: framePool,
		abi:       abi,
	})
	lt := lti.(layoutType)
	return lt.t, lt.framePool, lt.abi
}

// ifaceIndir reports whether t is stored indirectly in an interface value.
func ifaceIndir(t *rtype) bool {
	return t.kind&kindDirectIface == 0
}

// Note: this type must agree with runtime.bitvector.
type bitVector struct {
	n    uint32 // number of bits
	data []byte
}

// append a bit to the bitmap.
func (bv *bitVector) append(bit uint8) {
	if bv.n%8 == 0 {
		bv.data = append(bv.data, 0)
	}
	bv.data[bv.n/8] |= bit << (bv.n % 8)
	bv.n++
}

func addTypeBits(bv *bitVector, offset uintptr, t *rtype) {
	if t.ptrdata == 0 {
		return
	}

	switch Kind(t.kind & kindMask) {
	case Chan, Func, Map, Pointer, Slice, String, UnsafePointer:
		// 1 pointer at start of representation
		for bv.n < uint32(offset/uintptr(goarch.PtrSize)) {
			bv.append(0)
		}
		bv.append(1)

	case Interface:
		// 2 pointers
		for bv.n < uint32(offset/uintptr(goarch.PtrSize)) {
			bv.append(0)
		}
		bv.append(1)
		bv.append(1)

	case Array:
		// repeat inner type
		tt := (*arrayType)(unsafe.Pointer(t))
		for i := 0; i < int(tt.len); i++ {
			addTypeBits(bv, offset+uintptr(i)*tt.elem.size, tt.elem)
		}

	case Struct:
		// apply fields
		tt := (*structType)(unsafe.Pointer(t))
		for i := range tt.fields {
			f := &tt.fields[i]
			addTypeBits(bv, offset+f.offset(), f.typ)
		}
	}
}
