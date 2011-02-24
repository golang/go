// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The reflect package implements run-time reflection, allowing a program to
// manipulate objects with arbitrary types.  The typical use is to take a
// value with static type interface{} and extract its dynamic type
// information by calling Typeof, which returns an object with interface
// type Type.  That contains a pointer to a struct of type *StructType,
// *IntType, etc. representing the details of the underlying type.  A type
// switch or type assertion can reveal which.
//
// A call to NewValue creates a Value representing the run-time data; it
// contains a *StructValue, *IntValue, etc.  MakeZero takes a Type and
// returns a Value representing a zero value for that type.
package reflect

import (
	"runtime"
	"strconv"
	"unsafe"
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

// commonType is the common implementation of most values.
// It is embedded in other, public struct types, but always
// with a unique tag like "uint" or "float" so that the client cannot
// convert from, say, *UintType to *FloatType.

type commonType struct {
	size       uintptr
	hash       uint32
	alg        uint8
	align      uint8
	fieldAlign uint8
	kind       uint8
	string     *string
	*uncommonType
	ptrToThis *runtime.Type
}

type method struct {
	name    *string
	pkgPath *string
	mtyp    *runtime.Type
	typ     *runtime.Type
	ifn     unsafe.Pointer
	tfn     unsafe.Pointer
}

type uncommonType struct {
	name    *string
	pkgPath *string
	methods []method
}

// BoolType represents a boolean type.
type BoolType struct {
	commonType "bool"
}

// FloatType represents a float type.
type FloatType struct {
	commonType "float"
}

// ComplexType represents a complex type.
type ComplexType struct {
	commonType "complex"
}

// IntType represents a signed integer type.
type IntType struct {
	commonType "int"
}

// UintType represents a uint type.
type UintType struct {
	commonType "uint"
}

// StringType represents a string type.
type StringType struct {
	commonType "string"
}

// UnsafePointerType represents an unsafe.Pointer type.
type UnsafePointerType struct {
	commonType "unsafe.Pointer"
}

// ArrayType represents a fixed array type.
type ArrayType struct {
	commonType "array"
	elem       *runtime.Type
	len        uintptr
}

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir ChanDir = 1 << iota
	SendDir
	BothDir = RecvDir | SendDir
)

// ChanType represents a channel type.
type ChanType struct {
	commonType "chan"
	elem       *runtime.Type
	dir        uintptr
}

// FuncType represents a function type.
type FuncType struct {
	commonType "func"
	dotdotdot  bool
	in         []*runtime.Type
	out        []*runtime.Type
}

// Method on interface type
type imethod struct {
	name    *string
	pkgPath *string
	typ     *runtime.Type
}

// InterfaceType represents an interface type.
type InterfaceType struct {
	commonType "interface"
	methods    []imethod
}

// MapType represents a map type.
type MapType struct {
	commonType "map"
	key        *runtime.Type
	elem       *runtime.Type
}

// PtrType represents a pointer type.
type PtrType struct {
	commonType "ptr"
	elem       *runtime.Type
}

// SliceType represents a slice type.
type SliceType struct {
	commonType "slice"
	elem       *runtime.Type
}

// arrayOrSliceType is an unexported method that guarantees only
// arrays and slices implement ArrayOrSliceType.
func (*SliceType) arrayOrSliceType() {}

// Struct field
type structField struct {
	name    *string
	pkgPath *string
	typ     *runtime.Type
	tag     *string
	offset  uintptr
}

// StructType represents a struct type.
type StructType struct {
	commonType "struct"
	fields     []structField
}


/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

// Method represents a single method.
type Method struct {
	PkgPath string // empty for uppercase Name
	Name    string
	Type    *FuncType
	Func    *FuncValue
}

// Type is the runtime representation of a Go type.
// Every type implements the methods listed here.
// Some types implement additional interfaces;
// use a type switch to find out what kind of type a Type is.
// Each type in a program has a unique Type, so == on Types
// corresponds to Go's type equality.
type Type interface {
	// PkgPath returns the type's package path.
	// The package path is a full package import path like "container/vector".
	// PkgPath returns an empty string for unnamed types.
	PkgPath() string

	// Name returns the type's name within its package.
	// Name returns an empty string for unnamed types.
	Name() string

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., vector instead of "container/vector") and is not
	// guaranteed to be unique among types.  To test for equality,
	// compare the Types directly.
	String() string

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size() uintptr

	// Bits returns the size of the type in bits.
	// It is intended for use with numeric types and may overflow
	// when used for composite types.
	Bits() int

	// Align returns the alignment of a value of this type
	// when allocated in memory.
	Align() int

	// FieldAlign returns the alignment of a value of this type
	// when used as a field in a struct.
	FieldAlign() int

	// Kind returns the specific kind of this type.
	Kind() Kind

	// Method returns the i'th method in the type's method set.
	//
	// For a non-interface type T or *T, the returned Method's Type and Func
	// fields describe a function whose first argument is the receiver.
	//
	// For an interface type, the returned Method's Type field gives the
	// method signature, without a receiver, and the Func field is nil.
	Method(int) Method

	// NumMethods returns the number of methods in the type's method set.
	NumMethod() int
	uncommon() *uncommonType
}

// A Kind represents the specific kind of type that a Type represents.
// For numeric types, the Kind gives more information than the Type's
// dynamic type.  For example, the Type of a float32 is FloatType, but
// the Kind is Float32.
//
// The zero Kind is not a valid kind.
type Kind uint8

const (
	Bool Kind = 1 + iota
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
	Ptr
	Slice
	String
	Struct
	UnsafePointer
)

// High bit says whether type has
// embedded pointers,to help garbage collector.
const kindMask = 0x7f

func (k Kind) String() string {
	if int(k) < len(kindNames) {
		return kindNames[k]
	}
	return "kind" + strconv.Itoa(int(k))
}

var kindNames = []string{
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

func (t *uncommonType) uncommon() *uncommonType {
	return t
}

func (t *uncommonType) PkgPath() string {
	if t == nil || t.pkgPath == nil {
		return ""
	}
	return *t.pkgPath
}

func (t *uncommonType) Name() string {
	if t == nil || t.name == nil {
		return ""
	}
	return *t.name
}

func (t *commonType) String() string { return *t.string }

func (t *commonType) Size() uintptr { return t.size }

func (t *commonType) Bits() int { return int(t.size * 8) }

func (t *commonType) Align() int { return int(t.align) }

func (t *commonType) FieldAlign() int { return int(t.fieldAlign) }

func (t *commonType) Kind() Kind { return Kind(t.kind & kindMask) }

func (t *uncommonType) Method(i int) (m Method) {
	if t == nil || i < 0 || i >= len(t.methods) {
		return
	}
	p := &t.methods[i]
	if p.name != nil {
		m.Name = *p.name
	}
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
	}
	m.Type = toType(*p.typ).(*FuncType)
	fn := p.tfn
	m.Func = &FuncValue{value: value{m.Type, addr(&fn), true}}
	return
}

func (t *uncommonType) NumMethod() int {
	if t == nil {
		return 0
	}
	return len(t.methods)
}

// TODO(rsc): 6g supplies these, but they are not
// as efficient as they could be: they have commonType
// as the receiver instead of *commonType.
func (t *commonType) NumMethod() int { return t.uncommonType.NumMethod() }

func (t *commonType) Method(i int) (m Method) { return t.uncommonType.Method(i) }

func (t *commonType) PkgPath() string { return t.uncommonType.PkgPath() }

func (t *commonType) Name() string { return t.uncommonType.Name() }

// Len returns the number of elements in the array.
func (t *ArrayType) Len() int { return int(t.len) }

// Elem returns the type of the array's elements.
func (t *ArrayType) Elem() Type { return toType(*t.elem) }

// arrayOrSliceType is an unexported method that guarantees only
// arrays and slices implement ArrayOrSliceType.
func (*ArrayType) arrayOrSliceType() {}

// Dir returns the channel direction.
func (t *ChanType) Dir() ChanDir { return ChanDir(t.dir) }

// Elem returns the channel's element type.
func (t *ChanType) Elem() Type { return toType(*t.elem) }

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

// In returns the type of the i'th function input parameter.
func (t *FuncType) In(i int) Type {
	if i < 0 || i >= len(t.in) {
		return nil
	}
	return toType(*t.in[i])
}

// DotDotDot returns true if the final function input parameter
// is a "..." parameter.  If so, t.In(t.NumIn() - 1) returns the
// parameter's underlying static type []T.
//
// For concreteness, if t is func(x int, y ... float), then
//
//	t.NumIn() == 2
//	t.In(0) is the reflect.Type for "int"
//	t.In(1) is the reflect.Type for "[]float"
//	t.DotDotDot() == true
//
func (t *FuncType) DotDotDot() bool { return t.dotdotdot }

// NumIn returns the number of input parameters.
func (t *FuncType) NumIn() int { return len(t.in) }

// Out returns the type of the i'th function output parameter.
func (t *FuncType) Out(i int) Type {
	if i < 0 || i >= len(t.out) {
		return nil
	}
	return toType(*t.out[i])
}

// NumOut returns the number of function output parameters.
func (t *FuncType) NumOut() int { return len(t.out) }

// Method returns the i'th method in the type's method set.
func (t *InterfaceType) Method(i int) (m Method) {
	if i < 0 || i >= len(t.methods) {
		return
	}
	p := &t.methods[i]
	m.Name = *p.name
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
	}
	m.Type = toType(*p.typ).(*FuncType)
	return
}

// NumMethod returns the number of interface methods in the type's method set.
func (t *InterfaceType) NumMethod() int { return len(t.methods) }

// Key returns the map key type.
func (t *MapType) Key() Type { return toType(*t.key) }

// Elem returns the map element type.
func (t *MapType) Elem() Type { return toType(*t.elem) }

// Elem returns the pointer element type.
func (t *PtrType) Elem() Type { return toType(*t.elem) }

// Elem returns the type of the slice's elements.
func (t *SliceType) Elem() Type { return toType(*t.elem) }

type StructField struct {
	PkgPath   string // empty for uppercase Name
	Name      string
	Type      Type
	Tag       string
	Offset    uintptr
	Index     []int
	Anonymous bool
}

// Field returns the i'th struct field.
func (t *StructType) Field(i int) (f StructField) {
	if i < 0 || i >= len(t.fields) {
		return
	}
	p := t.fields[i]
	f.Type = toType(*p.typ)
	if p.name != nil {
		f.Name = *p.name
	} else {
		t := f.Type
		if pt, ok := t.(*PtrType); ok {
			t = pt.Elem()
		}
		f.Name = t.Name()
		f.Anonymous = true
	}
	if p.pkgPath != nil {
		f.PkgPath = *p.pkgPath
	}
	if p.tag != nil {
		f.Tag = *p.tag
	}
	f.Offset = p.offset
	f.Index = []int{i}
	return
}

// TODO(gri): Should there be an error/bool indicator if the index
//            is wrong for FieldByIndex?

// FieldByIndex returns the nested field corresponding to index.
func (t *StructType) FieldByIndex(index []int) (f StructField) {
	for i, x := range index {
		if i > 0 {
			ft := f.Type
			if pt, ok := ft.(*PtrType); ok {
				ft = pt.Elem()
			}
			if st, ok := ft.(*StructType); ok {
				t = st
			} else {
				var f0 StructField
				f = f0
				return
			}
		}
		f = t.Field(x)
	}
	return
}

const inf = 1 << 30 // infinity - no struct has that many nesting levels

func (t *StructType) fieldByNameFunc(match func(string) bool, mark map[*StructType]bool, depth int) (ff StructField, fd int) {
	fd = inf // field depth

	if mark[t] {
		// Struct already seen.
		return
	}
	mark[t] = true

	var fi int // field index
	n := 0     // number of matching fields at depth fd
L:
	for i := range t.fields {
		f := t.Field(i)
		d := inf
		switch {
		case match(f.Name):
			// Matching top-level field.
			d = depth
		case f.Anonymous:
			ft := f.Type
			if pt, ok := ft.(*PtrType); ok {
				ft = pt.Elem()
			}
			switch {
			case match(ft.Name()):
				// Matching anonymous top-level field.
				d = depth
			case fd > depth:
				// No top-level field yet; look inside nested structs.
				if st, ok := ft.(*StructType); ok {
					f, d = st.fieldByNameFunc(match, mark, depth+1)
				}
			}
		}

		switch {
		case d < fd:
			// Found field at shallower depth.
			ff, fi, fd = f, i, d
			n = 1
		case d == fd:
			// More than one matching field at the same depth (or d, fd == inf).
			// Same as no field found at this depth.
			n++
			if d == depth {
				// Impossible to find a field at lower depth.
				break L
			}
		}
	}

	if n == 1 {
		// Found matching field.
		if len(ff.Index) <= depth {
			ff.Index = make([]int, depth+1)
		}
		ff.Index[depth] = fi
	} else {
		// None or more than one matching field found.
		fd = inf
	}

	mark[t] = false, false
	return
}

// FieldByName returns the struct field with the given name
// and a boolean to indicate if the field was found.
func (t *StructType) FieldByName(name string) (f StructField, present bool) {
	return t.FieldByNameFunc(func(s string) bool { return s == name })
}

// FieldByNameFunc returns the struct field with a name that satisfies the
// match function and a boolean to indicate if the field was found.
func (t *StructType) FieldByNameFunc(match func(string) bool) (f StructField, present bool) {
	if ff, fd := t.fieldByNameFunc(match, make(map[*StructType]bool), 0); fd < inf {
		ff.Index = ff.Index[0 : fd+1]
		f, present = ff, true
	}
	return
}

// NumField returns the number of struct fields.
func (t *StructType) NumField() int { return len(t.fields) }

// Convert runtime type to reflect type.
// Same memory layouts, different method sets.
func toType(i interface{}) Type {
	switch v := i.(type) {
	case nil:
		return nil
	case *runtime.BoolType:
		return (*BoolType)(unsafe.Pointer(v))
	case *runtime.FloatType:
		return (*FloatType)(unsafe.Pointer(v))
	case *runtime.ComplexType:
		return (*ComplexType)(unsafe.Pointer(v))
	case *runtime.IntType:
		return (*IntType)(unsafe.Pointer(v))
	case *runtime.StringType:
		return (*StringType)(unsafe.Pointer(v))
	case *runtime.UintType:
		return (*UintType)(unsafe.Pointer(v))
	case *runtime.UnsafePointerType:
		return (*UnsafePointerType)(unsafe.Pointer(v))
	case *runtime.ArrayType:
		return (*ArrayType)(unsafe.Pointer(v))
	case *runtime.ChanType:
		return (*ChanType)(unsafe.Pointer(v))
	case *runtime.FuncType:
		return (*FuncType)(unsafe.Pointer(v))
	case *runtime.InterfaceType:
		return (*InterfaceType)(unsafe.Pointer(v))
	case *runtime.MapType:
		return (*MapType)(unsafe.Pointer(v))
	case *runtime.PtrType:
		return (*PtrType)(unsafe.Pointer(v))
	case *runtime.SliceType:
		return (*SliceType)(unsafe.Pointer(v))
	case *runtime.StructType:
		return (*StructType)(unsafe.Pointer(v))
	}
	println(i)
	panic("toType")
}

// ArrayOrSliceType is the common interface implemented
// by both ArrayType and SliceType.
type ArrayOrSliceType interface {
	Type
	Elem() Type
	arrayOrSliceType() // Guarantees only Array and Slice implement this interface.
}

// Typeof returns the reflection Type of the value in the interface{}.
func Typeof(i interface{}) Type { return toType(unsafe.Typeof(i)) }
