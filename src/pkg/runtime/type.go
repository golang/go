// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Runtime type representation.
 *
 * The following files know the exact layout of these
 * data structures and must be kept in sync with this file:
 *
 *	../../cmd/gc/reflect.c
 *	../../cmd/ld/dwarf.c decodetype_*
 *	../reflect/type.go
 *	type.h
 */

package runtime

import "unsafe"

// The compiler can only construct empty interface values at
// compile time; non-empty interface values get created
// during initialization.  Type is an empty interface
// so that the compiler can lay out references as data.
type Type interface{}

// All types begin with a few common fields needed for
// the interface runtime.
type commonType struct {
	size          uintptr  // size in bytes
	hash          uint32   // hash of type; avoids computation in hash tables
	_             uint8    // unused
	align         uint8    // alignment of variable with this type
	fieldAlign    uint8    // alignment of struct field with this type
	kind          uint8    // enumeration for C
	alg           *uintptr // algorithm table (../runtime/runtime.h:/Alg)
	string        *string  // string form; unnecessary  but undeniably useful
	*uncommonType          // (relatively) uncommon fields
	ptrToThis     *Type    // pointer to this type, if used in binary or has methods
}

// Values for commonType.kind.
const (
	kindBool = 1 + iota
	kindInt
	kindInt8
	kindInt16
	kindInt32
	kindInt64
	kindUint
	kindUint8
	kindUint16
	kindUint32
	kindUint64
	kindUintptr
	kindFloat32
	kindFloat64
	kindComplex64
	kindComplex128
	kindArray
	kindChan
	kindFunc
	kindInterface
	kindMap
	kindPtr
	kindSlice
	kindString
	kindStruct
	kindUnsafePointer

	kindNoPointers = 1 << 7 // OR'ed into kind
)

// Method on non-interface type
type _method struct { // underscore is to avoid collision with C
	name    *string        // name of method
	pkgPath *string        // nil for exported Names; otherwise import path
	mtyp    *Type          // method type (without receiver)
	typ     *Type          // .(*FuncType) underneath (with receiver)
	ifn     unsafe.Pointer // fn used in interface call (one-word receiver)
	tfn     unsafe.Pointer // fn used for normal method call
}

// uncommonType is present only for types with names or methods
// (if T is a named type, the uncommonTypes for T and *T have methods).
// Using a pointer to this struct reduces the overall size required
// to describe an unnamed type with no methods.
type uncommonType struct {
	name    *string   // name of type
	pkgPath *string   // import path; nil for built-in types like int, string
	methods []_method // methods associated with type
}

// BoolType represents a boolean type.
type BoolType commonType

// FloatType represents a float type.
type FloatType commonType

// ComplexType represents a complex type.
type ComplexType commonType

// IntType represents an int type.
type IntType commonType

// UintType represents a uint type.
type UintType commonType

// StringType represents a string type.
type StringType commonType

// UintptrType represents a uintptr type.
type UintptrType commonType

// UnsafePointerType represents an unsafe.Pointer type.
type UnsafePointerType commonType

// ArrayType represents a fixed array type.
type ArrayType struct {
	commonType
	elem  *Type // array element type
	slice *Type // slice type
	len   uintptr
}

// SliceType represents a slice type.
type SliceType struct {
	commonType
	elem *Type // slice element type
}

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir ChanDir             = 1 << iota // <-chan
	SendDir                                 // chan<-
	BothDir = RecvDir | SendDir             // chan
)

// ChanType represents a channel type.
type ChanType struct {
	commonType
	elem *Type   // channel element type
	dir  uintptr // channel direction (ChanDir)
}

// FuncType represents a function type.
type FuncType struct {
	commonType
	dotdotdot bool    // last input parameter is ...
	in        []*Type // input parameter types
	out       []*Type // output parameter types
}

// Method on interface type
type _imethod struct { // underscore is to avoid collision with C
	name    *string // name of method
	pkgPath *string // nil for exported Names; otherwise import path
	typ     *Type   // .(*FuncType) underneath
}

// InterfaceType represents an interface type.
type InterfaceType struct {
	commonType
	methods []_imethod // sorted by hash
}

// MapType represents a map type.
type MapType struct {
	commonType
	key  *Type // map key type
	elem *Type // map element (value) type
}

// PtrType represents a pointer type.
type PtrType struct {
	commonType
	elem *Type // pointer element (pointed at) type
}

// Struct field
type structField struct {
	name    *string // nil for embedded fields
	pkgPath *string // nil for exported Names; otherwise import path
	typ     *Type   // type of field
	tag     *string // nil if no tag
	offset  uintptr // byte offset of field within struct
}

// StructType represents a struct type.
type StructType struct {
	commonType
	fields []structField // sorted by offset
}

/*
 * Must match iface.c:/Itab and compilers.
 * NOTE: this is the version used by the reflection code, there is another
 * one in iface_defs.go that is closer to the original C version.
 */
type Itable struct {
	Itype  *Type // (*tab.inter).(*InterfaceType) is the interface type
	Type   *Type
	link   *Itable
	bad    int32
	unused int32
	Fn     [100000]uintptr // bigger than we'll ever see
}
