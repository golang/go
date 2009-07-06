// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"runtime";
	"strconv";
	"strings";
	"unsafe";
)

/*
 * Copy of data structures from ../runtime/type.go.
 * For comments, see the ones in that file.
 *
 * These data structures are known to the compiler and the runtime.
 *
 * Putting these types in runtime instead of reflect means that
 * reflect doesn't need to be autolinked into every binary, which
 * simplifies bootstrapping and package dependencies.
 * Unfortunately, it also means that reflect needs its own
 * copy in order to access the private fields.
 */

type uncommonType struct

type commonType struct {
	size uintptr;
	hash uint32;
	alg uint8;
	align uint8;
	fieldAlign uint8;
	string *string;
	*uncommonType;
}

type method struct {
	hash uint32;
	name *string;
	PkgPath *string;
	typ *runtime.Type;
	ifn unsafe.Pointer;
	tfn unsafe.Pointer;
}

type uncommonType struct {
	name *string;
	pkgPath *string;
	methods []method;
}

// BoolType represents a boolean type.
type BoolType struct {
	commonType
}

// Float32Type represents a float32 type.
type Float32Type struct {
	commonType
}

// Float64Type represents a float64 type.
type Float64Type struct {
	commonType
}

// FloatType represents a float type.
type FloatType struct {
	commonType
}

// Int16Type represents an int16 type.
type Int16Type struct {
	commonType
}

// Int32Type represents an int32 type.
type Int32Type struct {
	commonType
}

// Int64Type represents an int64 type.
type Int64Type struct {
	commonType
}

// Int8Type represents an int8 type.
type Int8Type struct {
	commonType
}

// IntType represents an int type.
type IntType struct {
	commonType
}

// Uint16Type represents a uint16 type.
type Uint16Type struct {
	commonType
}

// Uint32Type represents a uint32 type.
type Uint32Type struct {
	commonType
}

// Uint64Type represents a uint64 type.
type Uint64Type struct {
	commonType
}

// Uint8Type represents a uint8 type.
type Uint8Type struct {
	commonType
}

// UintType represents a uint type.
type UintType struct {
	commonType
}

// StringType represents a string type.
type StringType struct {
	commonType
}

// UintptrType represents a uintptr type.
type UintptrType struct {
	commonType
}

// DotDotDotType represents the ... that can
// be used as the type of the final function parameter.
type DotDotDotType struct {
	commonType
}

// UnsafePointerType represents an unsafe.Pointer type.
type UnsafePointerType struct {
	commonType
}

// ArrayType represents a fixed array type.
type ArrayType struct {
	commonType;
	elem *runtime.Type;
	len uintptr;
}

// SliceType represents a slice type.
type SliceType struct {
	commonType;
	elem *runtime.Type;
}

// ChanDir represents a channel type's direction.
type ChanDir int
const (
	RecvDir ChanDir = 1<<iota;
	SendDir;
	BothDir = RecvDir | SendDir;
)

// ChanType represents a channel type.
type ChanType struct {
	commonType;
	elem *runtime.Type;
	dir uintptr;
}

// FuncType represents a function type.
type FuncType struct {
	commonType;
	in []*runtime.Type;
	out []*runtime.Type;
}

// Method on interface type
type imethod struct {
	hash uint32;
	perm uint32;
	name *string;
	pkgPath *string;
	typ *runtime.Type;
}

// InterfaceType represents an interface type.
type InterfaceType struct {
	commonType;
	methods []imethod;
}

// MapType represents a map type.
type MapType struct {
	commonType;
	key *runtime.Type;
	elem *runtime.Type;
}

// PtrType represents a pointer type.
type PtrType struct {
	commonType;
	elem *runtime.Type;
}

// Struct field
type structField struct {
	name *string;
	pkgPath *string;
	typ *runtime.Type;
	tag *string;
	offset uintptr;
}

// StructType represents a struct type.
type StructType struct {
	commonType;
	fields []structField;
}


/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

type Type interface
type addr unsafe.Pointer
type FuncValue struct
func newFuncValue(typ Type, addr addr) *FuncValue

// Method represents a single method.
type Method struct {
	PkgPath string;		// empty for uppercase Name
	Name string;
	Type *FuncType;
	Func *FuncValue;
}

// Type is the runtime representation of a Go type.
// Every type implements the methods listed here.
// Some types implement additional interfaces;
// use a type switch to find out what kind of type a Type is.
// Each type in a program has a unique Type, so == on Types
// corresponds to Go's type equality.
type Type interface {
	// Name returns the type's package and name.
	// The package is a full package import path like "container/vector".
	Name()	(pkgPath string, name string);

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., vector instead of "container/vector") and is not
	// guaranteed to be unique among types.  To test for equality,
	// compare the Types directly.
	String()	string;

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size()	uintptr;

	// Align returns the alignment of a value of this type
	// when allocated in memory.
	Align()	int;

	// FieldAlign returns the alignment of a value of this type
	// when used as a field in a struct.
	FieldAlign()	int;

	// For non-interface types, Method returns the i'th method with receiver T.
	// For interface types, Method returns the i'th method in the interface.
	// NumMethod returns the number of such methods.
	Method(int)	Method;
	NumMethod()	int;
}

func toType(i interface{}) Type

func (t *uncommonType) Name() (pkgPath string, name string) {
}

func (t *commonType) String() string {
}

func (t *commonType) Size() uintptr {
}

func (t *commonType) Align() int {
}

func (t *commonType) FieldAlign() int {
}

func (t *uncommonType) Method(i int) (m Method) {
}

func (t *uncommonType) NumMethod() int {
}

// Len returns the number of elements in the array.
func (t *ArrayType) Len() int {
}

// Elem returns the type of the array's elements.
func (t *ArrayType) Elem() Type {
}

// Dir returns the channel direction.
func (t *ChanType) Dir() ChanDir {
}

// Elem returns the channel's element type.
func (t *ChanType) Elem() Type {
}

func (d ChanDir) String() string {
}

// In returns the type of the i'th function input parameter.
func (t *FuncType) In(i int) Type {
}

// NumIn returns the number of input parameters.
func (t *FuncType) NumIn() int {
}

// Out returns the type of the i'th function output parameter.
func (t *FuncType) Out(i int) Type {
}

// NumOut returns the number of function output parameters.
func (t *FuncType) NumOut() int {
}

// Method returns the i'th interface method.
func (t *InterfaceType) Method(i int) (m Method) {
}

// NumMethod returns the number of interface methods.
func (t *InterfaceType) NumMethod() int {
}

// Key returns the map key type.
func (t *MapType) Key() Type {
}

// Elem returns the map element type.
func (t *MapType) Elem() Type {
}

// Elem returns the pointer element type.
func (t *PtrType) Elem() Type {
}

// Elem returns the type of the slice's elements.
func (t *SliceType) Elem() Type {
}

type StructField struct {
	PkgPath string;		// empty for uppercase Name
	Name string;
	Type Type;
	Tag string;
	Offset uintptr;
	Anonymous bool;
}

// Field returns the i'th struct field.
func (t *StructType) Field(i int) (f StructField) {
}

// NumField returns the number of struct fields.
func (t *StructType) NumField() int {
}

// ArrayOrSliceType is the common interface implemented
// by both ArrayType and SliceType.
type ArrayOrSliceType interface {
	Type;
	Elem() Type;
}


