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
	"internal/abi"
	"internal/goarch"
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
	// For a non-interface type, it returns the number of exported methods.
	//
	// For an interface type, it returns the number of exported and unexported methods.
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
	// If the returned field is promoted from an embedded struct,
	// then Offset in the returned StructField is the offset in
	// the embedded struct.
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
	//
	// If the returned field is promoted from an embedded struct,
	// then Offset in the returned StructField is the offset in
	// the embedded struct.
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

	// OverflowComplex reports whether the complex128 x cannot be represented by type t.
	// It panics if t's Kind is not Complex64 or Complex128.
	OverflowComplex(x complex128) bool

	// OverflowFloat reports whether the float64 x cannot be represented by type t.
	// It panics if t's Kind is not Float32 or Float64.
	OverflowFloat(x float64) bool

	// OverflowInt reports whether the int64 x cannot be represented by type t.
	// It panics if t's Kind is not Int, Int8, Int16, Int32, or Int64.
	OverflowInt(x int64) bool

	// OverflowUint reports whether the uint64 x cannot be represented by type t.
	// It panics if t's Kind is not Uint, Uintptr, Uint8, Uint16, Uint32, or Uint64.
	OverflowUint(x uint64) bool

	// CanSeq reports whether a [Value] with this type can be iterated over using [Value.Seq].
	CanSeq() bool

	// CanSeq2 reports whether a [Value] with this type can be iterated over using [Value.Seq2].
	CanSeq2() bool

	common() *abi.Type
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
 * They are also known to ../internal/abi/type.go.
 */

// A Kind represents the specific kind of type that a [Type] represents.
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

// Ptr is the old name for the [Pointer] kind.
const Ptr = Pointer

// uncommonType is present only for defined types or types with methods
// (if T is a defined type, the uncommonTypes for T and *T have methods).
// Using a pointer to this struct reduces the overall size required
// to describe a non-defined type with no methods.
type uncommonType = abi.UncommonType

// Embed this type to get common/uncommon
type common struct {
	abi.Type
}

// rtype is the common implementation of most values.
// It is embedded in other struct types.
type rtype struct {
	t abi.Type
}

func (t *rtype) common() *abi.Type {
	return &t.t
}

func (t *rtype) uncommon() *abi.UncommonType {
	return t.t.Uncommon()
}

type aNameOff = abi.NameOff
type aTypeOff = abi.TypeOff
type aTextOff = abi.TextOff

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir ChanDir             = 1 << iota // <-chan
	SendDir                                 // chan<-
	BothDir = RecvDir | SendDir             // chan
)

// arrayType represents a fixed array type.
type arrayType = abi.ArrayType

// chanType represents a channel type.
type chanType = abi.ChanType

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
type funcType = abi.FuncType

// interfaceType represents an interface type.
type interfaceType struct {
	abi.InterfaceType // can embed directly because not a public type.
}

func (t *interfaceType) nameOff(off aNameOff) abi.Name {
	return toRType(&t.Type).nameOff(off)
}

func nameOffFor(t *abi.Type, off aNameOff) abi.Name {
	return toRType(t).nameOff(off)
}

func typeOffFor(t *abi.Type, off aTypeOff) *abi.Type {
	return toRType(t).typeOff(off)
}

func (t *interfaceType) typeOff(off aTypeOff) *abi.Type {
	return toRType(&t.Type).typeOff(off)
}

func (t *interfaceType) common() *abi.Type {
	return &t.Type
}

func (t *interfaceType) uncommon() *abi.UncommonType {
	return t.Uncommon()
}

// ptrType represents a pointer type.
type ptrType struct {
	abi.PtrType
}

// sliceType represents a slice type.
type sliceType struct {
	abi.SliceType
}

// Struct field
type structField = abi.StructField

// structType represents a struct type.
type structType struct {
	abi.StructType
}

func pkgPath(n abi.Name) string {
	if n.Bytes == nil || *n.DataChecked(0, "name flag field")&(1<<2) == 0 {
		return ""
	}
	i, l := n.ReadVarint(1)
	off := 1 + i + l
	if n.HasTag() {
		i2, l2 := n.ReadVarint(off)
		off += i2 + l2
	}
	var nameOff int32
	// Note that this field may not be aligned in memory,
	// so we cannot use a direct int32 assignment here.
	copy((*[4]byte)(unsafe.Pointer(&nameOff))[:], (*[4]byte)(unsafe.Pointer(n.DataChecked(off, "name offset field")))[:])
	pkgPathName := abi.Name{Bytes: (*byte)(resolveTypeOff(unsafe.Pointer(n.Bytes), nameOff))}
	return pkgPathName.Name()
}

func newName(n, tag string, exported, embedded bool) abi.Name {
	return abi.NewName(n, tag, exported, embedded)
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

// String returns the name of k.
func (k Kind) String() string {
	if uint(k) < uint(len(kindNames)) {
		return kindNames[uint(k)]
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

// resolveNameOff resolves a name offset from a base pointer.
// The (*rtype).nameOff method is a convenience wrapper for this function.
// Implemented in the runtime package.
//
//go:noescape
func resolveNameOff(ptrInModule unsafe.Pointer, off int32) unsafe.Pointer

// resolveTypeOff resolves an *rtype offset from a base type.
// The (*rtype).typeOff method is a convenience wrapper for this function.
// Implemented in the runtime package.
//
//go:noescape
func resolveTypeOff(rtype unsafe.Pointer, off int32) unsafe.Pointer

// resolveTextOff resolves a function pointer offset from a base type.
// The (*rtype).textOff method is a convenience wrapper for this function.
// Implemented in the runtime package.
//
//go:noescape
func resolveTextOff(rtype unsafe.Pointer, off int32) unsafe.Pointer

// addReflectOff adds a pointer to the reflection lookup map in the runtime.
// It returns a new ID that can be used as a typeOff or textOff, and will
// be resolved correctly. Implemented in the runtime package.
//
// addReflectOff should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/goplus/reflectx
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname addReflectOff
//go:noescape
func addReflectOff(ptr unsafe.Pointer) int32

// resolveReflectName adds a name to the reflection lookup map in the runtime.
// It returns a new nameOff that can be used to refer to the pointer.
func resolveReflectName(n abi.Name) aNameOff {
	return aNameOff(addReflectOff(unsafe.Pointer(n.Bytes)))
}

// resolveReflectType adds a *rtype to the reflection lookup map in the runtime.
// It returns a new typeOff that can be used to refer to the pointer.
func resolveReflectType(t *abi.Type) aTypeOff {
	return aTypeOff(addReflectOff(unsafe.Pointer(t)))
}

// resolveReflectText adds a function pointer to the reflection lookup map in
// the runtime. It returns a new textOff that can be used to refer to the
// pointer.
func resolveReflectText(ptr unsafe.Pointer) aTextOff {
	return aTextOff(addReflectOff(ptr))
}

func (t *rtype) nameOff(off aNameOff) abi.Name {
	return abi.Name{Bytes: (*byte)(resolveNameOff(unsafe.Pointer(t), int32(off)))}
}

func (t *rtype) typeOff(off aTypeOff) *abi.Type {
	return (*abi.Type)(resolveTypeOff(unsafe.Pointer(t), int32(off)))
}

func (t *rtype) textOff(off aTextOff) unsafe.Pointer {
	return resolveTextOff(unsafe.Pointer(t), int32(off))
}

func textOffFor(t *abi.Type, off aTextOff) unsafe.Pointer {
	return toRType(t).textOff(off)
}

func (t *rtype) String() string {
	s := t.nameOff(t.t.Str).Name()
	if t.t.TFlag&abi.TFlagExtraStar != 0 {
		return s[1:]
	}
	return s
}

func (t *rtype) Size() uintptr { return t.t.Size() }

func (t *rtype) Bits() int {
	if t == nil {
		panic("reflect: Bits of nil Type")
	}
	k := t.Kind()
	if k < Int || k > Complex128 {
		panic("reflect: Bits of non-arithmetic Type " + t.String())
	}
	return int(t.t.Size_) * 8
}

func (t *rtype) Align() int { return t.t.Align() }

func (t *rtype) FieldAlign() int { return t.t.FieldAlign() }

func (t *rtype) Kind() Kind { return Kind(t.t.Kind()) }

func (t *rtype) exportedMethods() []abi.Method {
	ut := t.uncommon()
	if ut == nil {
		return nil
	}
	return ut.ExportedMethods()
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
	pname := t.nameOff(p.Name)
	m.Name = pname.Name()
	fl := flag(Func)
	mtyp := t.typeOff(p.Mtyp)
	ft := (*funcType)(unsafe.Pointer(mtyp))
	in := make([]Type, 0, 1+ft.NumIn())
	in = append(in, t)
	for _, arg := range ft.InSlice() {
		in = append(in, toRType(arg))
	}
	out := make([]Type, 0, ft.NumOut())
	for _, ret := range ft.OutSlice() {
		out = append(out, toRType(ret))
	}
	mt := FuncOf(in, out, ft.IsVariadic())
	m.Type = mt
	tfn := t.textOff(p.Tfn)
	fn := unsafe.Pointer(&tfn)
	m.Func = Value{&mt.(*rtype).t, fn, fl}

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

	methods := ut.ExportedMethods()

	// We are looking for the first index i where the string becomes >= s.
	// This is a copy of sort.Search, with f(h) replaced by (t.nameOff(methods[h].name).name() >= name).
	i, j := 0, len(methods)
	for i < j {
		h := int(uint(i+j) >> 1) // avoid overflow when computing h
		// i ≤ h < j
		if !(t.nameOff(methods[h].Name).Name() >= name) {
			i = h + 1 // preserves f(i-1) == false
		} else {
			j = h // preserves f(j) == true
		}
	}
	// i == j, f(i-1) == false, and f(j) (= f(i)) == true  =>  answer is i.
	if i < len(methods) && name == t.nameOff(methods[i].Name).Name() {
		return t.Method(i), true
	}

	return Method{}, false
}

func (t *rtype) PkgPath() string {
	if t.t.TFlag&abi.TFlagNamed == 0 {
		return ""
	}
	ut := t.uncommon()
	if ut == nil {
		return ""
	}
	return t.nameOff(ut.PkgPath).Name()
}

func pkgPathFor(t *abi.Type) string {
	return toRType(t).PkgPath()
}

func (t *rtype) Name() string {
	if !t.t.HasName() {
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

func nameFor(t *abi.Type) string {
	return toRType(t).Name()
}

func (t *rtype) ChanDir() ChanDir {
	if t.Kind() != Chan {
		panic("reflect: ChanDir of non-chan type " + t.String())
	}
	tt := (*abi.ChanType)(unsafe.Pointer(t))
	return ChanDir(tt.Dir)
}

func toRType(t *abi.Type) *rtype {
	return (*rtype)(unsafe.Pointer(t))
}

func elem(t *abi.Type) *abi.Type {
	et := t.Elem()
	if et != nil {
		return et
	}
	panic("reflect: Elem of invalid type " + stringFor(t))
}

func (t *rtype) Elem() Type {
	return toType(elem(t.common()))
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

func (t *rtype) Len() int {
	if t.Kind() != Array {
		panic("reflect: Len of non-array type " + t.String())
	}
	tt := (*arrayType)(unsafe.Pointer(t))
	return int(tt.Len)
}

func (t *rtype) NumField() int {
	if t.Kind() != Struct {
		panic("reflect: NumField of non-struct type " + t.String())
	}
	tt := (*structType)(unsafe.Pointer(t))
	return len(tt.Fields)
}

func (t *rtype) In(i int) Type {
	if t.Kind() != Func {
		panic("reflect: In of non-func type " + t.String())
	}
	tt := (*abi.FuncType)(unsafe.Pointer(t))
	return toType(tt.InSlice()[i])
}

func (t *rtype) NumIn() int {
	if t.Kind() != Func {
		panic("reflect: NumIn of non-func type " + t.String())
	}
	tt := (*abi.FuncType)(unsafe.Pointer(t))
	return tt.NumIn()
}

func (t *rtype) NumOut() int {
	if t.Kind() != Func {
		panic("reflect: NumOut of non-func type " + t.String())
	}
	tt := (*abi.FuncType)(unsafe.Pointer(t))
	return tt.NumOut()
}

func (t *rtype) Out(i int) Type {
	if t.Kind() != Func {
		panic("reflect: Out of non-func type " + t.String())
	}
	tt := (*abi.FuncType)(unsafe.Pointer(t))
	return toType(tt.OutSlice()[i])
}

func (t *rtype) IsVariadic() bool {
	if t.Kind() != Func {
		panic("reflect: IsVariadic of non-func type " + t.String())
	}
	tt := (*abi.FuncType)(unsafe.Pointer(t))
	return tt.IsVariadic()
}

func (t *rtype) OverflowComplex(x complex128) bool {
	k := t.Kind()
	switch k {
	case Complex64:
		return overflowFloat32(real(x)) || overflowFloat32(imag(x))
	case Complex128:
		return false
	}
	panic("reflect: OverflowComplex of non-complex type " + t.String())
}

func (t *rtype) OverflowFloat(x float64) bool {
	k := t.Kind()
	switch k {
	case Float32:
		return overflowFloat32(x)
	case Float64:
		return false
	}
	panic("reflect: OverflowFloat of non-float type " + t.String())
}

func (t *rtype) OverflowInt(x int64) bool {
	k := t.Kind()
	switch k {
	case Int, Int8, Int16, Int32, Int64:
		bitSize := t.Size() * 8
		trunc := (x << (64 - bitSize)) >> (64 - bitSize)
		return x != trunc
	}
	panic("reflect: OverflowInt of non-int type " + t.String())
}

func (t *rtype) OverflowUint(x uint64) bool {
	k := t.Kind()
	switch k {
	case Uint, Uintptr, Uint8, Uint16, Uint32, Uint64:
		bitSize := t.Size() * 8
		trunc := (x << (64 - bitSize)) >> (64 - bitSize)
		return x != trunc
	}
	panic("reflect: OverflowUint of non-uint type " + t.String())
}

func (t *rtype) CanSeq() bool {
	switch t.Kind() {
	case Int8, Int16, Int32, Int64, Int, Uint8, Uint16, Uint32, Uint64, Uint, Uintptr, Array, Slice, Chan, String, Map:
		return true
	case Func:
		return canRangeFunc(&t.t)
	case Pointer:
		return t.Elem().Kind() == Array
	}
	return false
}

func canRangeFunc(t *abi.Type) bool {
	if t.Kind() != abi.Func {
		return false
	}
	f := t.FuncType()
	if f.InCount != 1 || f.OutCount != 0 {
		return false
	}
	y := f.In(0)
	if y.Kind() != abi.Func {
		return false
	}
	yield := y.FuncType()
	return yield.InCount == 1 && yield.OutCount == 1 && yield.Out(0).Kind() == abi.Bool
}

func (t *rtype) CanSeq2() bool {
	switch t.Kind() {
	case Array, Slice, String, Map:
		return true
	case Func:
		return canRangeFunc2(&t.t)
	case Pointer:
		return t.Elem().Kind() == Array
	}
	return false
}

func canRangeFunc2(t *abi.Type) bool {
	if t.Kind() != abi.Func {
		return false
	}
	f := t.FuncType()
	if f.InCount != 1 || f.OutCount != 0 {
		return false
	}
	y := f.In(0)
	if y.Kind() != abi.Func {
		return false
	}
	yield := y.FuncType()
	return yield.InCount == 2 && yield.OutCount == 1 && yield.Out(0).Kind() == abi.Bool
}

// add returns p+x.
//
// The whySafe string is ignored, so that the function still inlines
// as efficiently as p+x, but all call sites should use the string to
// record why the addition is safe, which is to say why the addition
// does not cause x to advance to the very end of p's allocation
// and therefore point incorrectly at the next block in memory.
//
// add should be an internal detail (and is trivially copyable),
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/pinpoint-apm/pinpoint-go-agent
//   - github.com/vmware/govmomi
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname add
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
	if i < 0 || i >= len(t.Methods) {
		return
	}
	p := &t.Methods[i]
	pname := t.nameOff(p.Name)
	m.Name = pname.Name()
	if !pname.IsExported() {
		m.PkgPath = pkgPath(pname)
		if m.PkgPath == "" {
			m.PkgPath = t.PkgPath.Name()
		}
	}
	m.Type = toType(t.typeOff(p.Typ))
	m.Index = i
	return
}

// NumMethod returns the number of interface methods in the type's method set.
func (t *interfaceType) NumMethod() int { return len(t.Methods) }

// MethodByName method with the given name in the type's method set.
func (t *interfaceType) MethodByName(name string) (m Method, ok bool) {
	if t == nil {
		return
	}
	var p *abi.Imethod
	for i := range t.Methods {
		p = &t.Methods[i]
		if t.nameOff(p.Name).Name() == name {
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
// explicitly set to the empty string, use [StructTag.Lookup].
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
	if i < 0 || i >= len(t.Fields) {
		panic("reflect: Field index out of bounds")
	}
	p := &t.Fields[i]
	f.Type = toType(p.Typ)
	f.Name = p.Name.Name()
	f.Anonymous = p.Embedded()
	if !p.Name.IsExported() {
		f.PkgPath = t.PkgPath.Name()
	}
	if tag := p.Name.Tag(); tag != "" {
		f.Tag = StructTag(tag)
	}
	f.Offset = p.Offset

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
// is wrong for FieldByIndex?

// FieldByIndex returns the nested field corresponding to index.
func (t *structType) FieldByIndex(index []int) (f StructField) {
	f.Type = toType(&t.Type)
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
			for i := range t.Fields {
				f := &t.Fields[i]
				// Find name and (for embedded field) type for field f.
				fname := f.Name.Name()
				var ntyp *abi.Type
				if f.Embedded() {
					// Embedded field of type T or *T.
					ntyp = f.Typ
					if ntyp.Kind() == abi.Pointer {
						ntyp = ntyp.Elem()
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
				if ok || ntyp == nil || ntyp.Kind() != abi.Struct {
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
		for i := range t.Fields {
			tf := &t.Fields[i]
			if tf.Name.Name() == name {
				return t.Field(i), true
			}
			if tf.Embedded() {
				hasEmbeds = true
			}
		}
	}
	if !hasEmbeds {
		return
	}
	return t.FieldByNameFunc(func(s string) bool { return s == name })
}

// TypeOf returns the reflection [Type] that represents the dynamic type of i.
// If i is a nil interface value, TypeOf returns nil.
func TypeOf(i any) Type {
	return toType(abi.TypeOf(i))
}

// rtypeOf directly extracts the *rtype of the provided value.
func rtypeOf(i any) *abi.Type {
	return abi.TypeOf(i)
}

// ptrMap is the cache for PointerTo.
var ptrMap sync.Map // map[*rtype]*ptrType

// PtrTo returns the pointer type with element t.
// For example, if t represents type Foo, PtrTo(t) represents *Foo.
//
// PtrTo is the old spelling of [PointerTo].
// The two functions behave identically.
//
// Deprecated: Superseded by [PointerTo].
func PtrTo(t Type) Type { return PointerTo(t) }

// PointerTo returns the pointer type with element t.
// For example, if t represents type Foo, PointerTo(t) represents *Foo.
func PointerTo(t Type) Type {
	return toRType(t.(*rtype).ptrTo())
}

func (t *rtype) ptrTo() *abi.Type {
	at := &t.t
	if at.PtrToThis != 0 {
		return t.typeOff(at.PtrToThis)
	}

	// Check the cache.
	if pi, ok := ptrMap.Load(t); ok {
		return &pi.(*ptrType).Type
	}

	// Look in known types.
	s := "*" + t.String()
	for _, tt := range typesByString(s) {
		p := (*ptrType)(unsafe.Pointer(tt))
		if p.Elem != &t.t {
			continue
		}
		pi, _ := ptrMap.LoadOrStore(t, p)
		return &pi.(*ptrType).Type
	}

	// Create a new ptrType starting with the description
	// of an *unsafe.Pointer.
	var iptr any = (*unsafe.Pointer)(nil)
	prototype := *(**ptrType)(unsafe.Pointer(&iptr))
	pp := *prototype

	pp.Str = resolveReflectName(newName(s, "", false, false))
	pp.PtrToThis = 0

	// For the type structures linked into the binary, the
	// compiler provides a good hash of the string.
	// Create a good hash for the new string by using
	// the FNV-1 hash's mixing function to combine the
	// old hash and the new "*".
	pp.Hash = fnv1(t.t.Hash, '*')

	pp.Elem = at

	pi, _ := ptrMap.LoadOrStore(t, &pp)
	return &pi.(*ptrType).Type
}

func ptrTo(t *abi.Type) *abi.Type {
	return toRType(t).ptrTo()
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
	return implements(u.common(), t.common())
}

func (t *rtype) AssignableTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.AssignableTo")
	}
	uu := u.common()
	return directlyAssignable(uu, t.common()) || implements(uu, t.common())
}

func (t *rtype) ConvertibleTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.ConvertibleTo")
	}
	return convertOp(u.common(), t.common()) != nil
}

func (t *rtype) Comparable() bool {
	return t.t.Equal != nil
}

// implements reports whether the type V implements the interface type T.
func implements(T, V *abi.Type) bool {
	if T.Kind() != abi.Interface {
		return false
	}
	t := (*interfaceType)(unsafe.Pointer(T))
	if len(t.Methods) == 0 {
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
	if V.Kind() == abi.Interface {
		v := (*interfaceType)(unsafe.Pointer(V))
		i := 0
		for j := 0; j < len(v.Methods); j++ {
			tm := &t.Methods[i]
			tmName := t.nameOff(tm.Name)
			vm := &v.Methods[j]
			vmName := nameOffFor(V, vm.Name)
			if vmName.Name() == tmName.Name() && typeOffFor(V, vm.Typ) == t.typeOff(tm.Typ) {
				if !tmName.IsExported() {
					tmPkgPath := pkgPath(tmName)
					if tmPkgPath == "" {
						tmPkgPath = t.PkgPath.Name()
					}
					vmPkgPath := pkgPath(vmName)
					if vmPkgPath == "" {
						vmPkgPath = v.PkgPath.Name()
					}
					if tmPkgPath != vmPkgPath {
						continue
					}
				}
				if i++; i >= len(t.Methods) {
					return true
				}
			}
		}
		return false
	}

	v := V.Uncommon()
	if v == nil {
		return false
	}
	i := 0
	vmethods := v.Methods()
	for j := 0; j < int(v.Mcount); j++ {
		tm := &t.Methods[i]
		tmName := t.nameOff(tm.Name)
		vm := vmethods[j]
		vmName := nameOffFor(V, vm.Name)
		if vmName.Name() == tmName.Name() && typeOffFor(V, vm.Mtyp) == t.typeOff(tm.Typ) {
			if !tmName.IsExported() {
				tmPkgPath := pkgPath(tmName)
				if tmPkgPath == "" {
					tmPkgPath = t.PkgPath.Name()
				}
				vmPkgPath := pkgPath(vmName)
				if vmPkgPath == "" {
					vmPkgPath = nameOffFor(V, v.PkgPath).Name()
				}
				if tmPkgPath != vmPkgPath {
					continue
				}
			}
			if i++; i >= len(t.Methods) {
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
func specialChannelAssignability(T, V *abi.Type) bool {
	// Special case:
	// x is a bidirectional channel value, T is a channel type,
	// x's type V and T have identical element types,
	// and at least one of V or T is not a defined type.
	return V.ChanDir() == abi.BothDir && (nameFor(T) == "" || nameFor(V) == "") && haveIdenticalType(T.Elem(), V.Elem(), true)
}

// directlyAssignable reports whether a value x of type V can be directly
// assigned (using memmove) to a value of type T.
// https://golang.org/doc/go_spec.html#Assignability
// Ignoring the interface rules (implemented elsewhere)
// and the ideal constant rules (no ideal constants at run time).
func directlyAssignable(T, V *abi.Type) bool {
	// x's type V is identical to T?
	if T == V {
		return true
	}

	// Otherwise at least one of T and V must not be defined
	// and they must have the same kind.
	if T.HasName() && V.HasName() || T.Kind() != V.Kind() {
		return false
	}

	if T.Kind() == abi.Chan && specialChannelAssignability(T, V) {
		return true
	}

	// x's type T and V must have identical underlying types.
	return haveIdenticalUnderlyingType(T, V, true)
}

func haveIdenticalType(T, V *abi.Type, cmpTags bool) bool {
	if cmpTags {
		return T == V
	}

	if nameFor(T) != nameFor(V) || T.Kind() != V.Kind() || pkgPathFor(T) != pkgPathFor(V) {
		return false
	}

	return haveIdenticalUnderlyingType(T, V, false)
}

func haveIdenticalUnderlyingType(T, V *abi.Type, cmpTags bool) bool {
	if T == V {
		return true
	}

	kind := Kind(T.Kind())
	if kind != Kind(V.Kind()) {
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
		if t.OutCount != v.OutCount || t.InCount != v.InCount {
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
		if len(t.Methods) == 0 && len(v.Methods) == 0 {
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
		if len(t.Fields) != len(v.Fields) {
			return false
		}
		if t.PkgPath.Name() != v.PkgPath.Name() {
			return false
		}
		for i := range t.Fields {
			tf := &t.Fields[i]
			vf := &v.Fields[i]
			if tf.Name.Name() != vf.Name.Name() {
				return false
			}
			if !haveIdenticalType(tf.Typ, vf.Typ, cmpTags) {
				return false
			}
			if cmpTags && tf.Name.Tag() != vf.Name.Tag() {
				return false
			}
			if tf.Offset != vf.Offset {
				return false
			}
			if tf.Embedded() != vf.Embedded() {
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

// rtypeOff should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/goccy/go-json
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname rtypeOff
func rtypeOff(section unsafe.Pointer, off int32) *abi.Type {
	return (*abi.Type)(add(section, uintptr(off), "sizeof(rtype) > 0"))
}

// typesByString returns the subslice of typelinks() whose elements have
// the given string representation.
// It may be empty (no known types with that string) or may have
// multiple elements (multiple types with that string).
//
// typesByString should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/aristanetworks/goarista
//   - fortio.org/log
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname typesByString
func typesByString(s string) []*abi.Type {
	sections, offset := typelinks()
	var ret []*abi.Type

	for offsI, offs := range offset {
		section := sections[offsI]

		// We are looking for the first index i where the string becomes >= s.
		// This is a copy of sort.Search, with f(h) replaced by (*typ[h].String() >= s).
		i, j := 0, len(offs)
		for i < j {
			h := int(uint(i+j) >> 1) // avoid overflow when computing h
			// i ≤ h < j
			if !(stringFor(rtypeOff(section, offs[h])) >= s) {
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
			if stringFor(typ) != s {
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
	t1    *abi.Type
	t2    *abi.Type
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
	typ := t.common()

	// Look in cache.
	ckey := cacheKey{Chan, typ, nil, uintptr(dir)}
	if ch, ok := lookupCache.Load(ckey); ok {
		return ch.(*rtype)
	}

	// This restriction is imposed by the gc compiler and the runtime.
	if typ.Size_ >= 1<<16 {
		panic("reflect.ChanOf: element size too large")
	}

	// Look in known types.
	var s string
	switch dir {
	default:
		panic("reflect.ChanOf: invalid dir")
	case SendDir:
		s = "chan<- " + stringFor(typ)
	case RecvDir:
		s = "<-chan " + stringFor(typ)
	case BothDir:
		typeStr := stringFor(typ)
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
		if ch.Elem == typ && ch.Dir == abi.ChanDir(dir) {
			ti, _ := lookupCache.LoadOrStore(ckey, toRType(tt))
			return ti.(Type)
		}
	}

	// Make a channel type.
	var ichan any = (chan unsafe.Pointer)(nil)
	prototype := *(**chanType)(unsafe.Pointer(&ichan))
	ch := *prototype
	ch.TFlag = abi.TFlagRegularMemory
	ch.Dir = abi.ChanDir(dir)
	ch.Str = resolveReflectName(newName(s, "", false, false))
	ch.Hash = fnv1(typ.Hash, 'c', byte(dir))
	ch.Elem = typ

	ti, _ := lookupCache.LoadOrStore(ckey, toRType(&ch.Type))
	return ti.(Type)
}

var funcTypes []Type
var funcTypesMutex sync.Mutex

func initFuncTypes(n int) Type {
	funcTypesMutex.Lock()
	defer funcTypesMutex.Unlock()
	if n >= len(funcTypes) {
		newFuncTypes := make([]Type, n+1)
		copy(newFuncTypes, funcTypes)
		funcTypes = newFuncTypes
	}
	if funcTypes[n] != nil {
		return funcTypes[n]
	}

	funcTypes[n] = StructOf([]StructField{
		{
			Name: "FuncType",
			Type: TypeOf(funcType{}),
		},
		{
			Name: "Args",
			Type: ArrayOf(n, TypeOf(&rtype{})),
		},
	})
	return funcTypes[n]
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

	if n > 128 {
		panic("reflect.FuncOf: too many arguments")
	}

	o := New(initFuncTypes(n)).Elem()
	ft := (*funcType)(unsafe.Pointer(o.Field(0).Addr().Pointer()))
	args := unsafe.Slice((**rtype)(unsafe.Pointer(o.Field(1).Addr().Pointer())), n)[0:0:n]
	*ft = *prototype

	// Build a hash and minimally populate ft.
	var hash uint32
	for _, in := range in {
		t := in.(*rtype)
		args = append(args, t)
		hash = fnv1(hash, byte(t.t.Hash>>24), byte(t.t.Hash>>16), byte(t.t.Hash>>8), byte(t.t.Hash))
	}
	if variadic {
		hash = fnv1(hash, 'v')
	}
	hash = fnv1(hash, '.')
	for _, out := range out {
		t := out.(*rtype)
		args = append(args, t)
		hash = fnv1(hash, byte(t.t.Hash>>24), byte(t.t.Hash>>16), byte(t.t.Hash>>8), byte(t.t.Hash))
	}

	ft.TFlag = 0
	ft.Hash = hash
	ft.InCount = uint16(len(in))
	ft.OutCount = uint16(len(out))
	if variadic {
		ft.OutCount |= 1 << 15
	}

	// Look in cache.
	if ts, ok := funcLookupCache.m.Load(hash); ok {
		for _, t := range ts.([]*abi.Type) {
			if haveIdenticalUnderlyingType(&ft.Type, t, true) {
				return toRType(t)
			}
		}
	}

	// Not in cache, lock and retry.
	funcLookupCache.Lock()
	defer funcLookupCache.Unlock()
	if ts, ok := funcLookupCache.m.Load(hash); ok {
		for _, t := range ts.([]*abi.Type) {
			if haveIdenticalUnderlyingType(&ft.Type, t, true) {
				return toRType(t)
			}
		}
	}

	addToCache := func(tt *abi.Type) Type {
		var rts []*abi.Type
		if rti, ok := funcLookupCache.m.Load(hash); ok {
			rts = rti.([]*abi.Type)
		}
		funcLookupCache.m.Store(hash, append(rts, tt))
		return toType(tt)
	}

	// Look in known types for the same string representation.
	str := funcStr(ft)
	for _, tt := range typesByString(str) {
		if haveIdenticalUnderlyingType(&ft.Type, tt, true) {
			return addToCache(tt)
		}
	}

	// Populate the remaining fields of ft and store in cache.
	ft.Str = resolveReflectName(newName(str, "", false, false))
	ft.PtrToThis = 0
	return addToCache(&ft.Type)
}
func stringFor(t *abi.Type) string {
	return toRType(t).String()
}

// funcStr builds a string representation of a funcType.
func funcStr(ft *funcType) string {
	repr := make([]byte, 0, 64)
	repr = append(repr, "func("...)
	for i, t := range ft.InSlice() {
		if i > 0 {
			repr = append(repr, ", "...)
		}
		if ft.IsVariadic() && i == int(ft.InCount)-1 {
			repr = append(repr, "..."...)
			repr = append(repr, stringFor((*sliceType)(unsafe.Pointer(t)).Elem)...)
		} else {
			repr = append(repr, stringFor(t)...)
		}
	}
	repr = append(repr, ')')
	out := ft.OutSlice()
	if len(out) == 1 {
		repr = append(repr, ' ')
	} else if len(out) > 1 {
		repr = append(repr, " ("...)
	}
	for i, t := range out {
		if i > 0 {
			repr = append(repr, ", "...)
		}
		repr = append(repr, stringFor(t)...)
	}
	if len(out) > 1 {
		repr = append(repr, ')')
	}
	return string(repr)
}

// isReflexive reports whether the == operation on the type is reflexive.
// That is, x == x for all values x of type t.
func isReflexive(t *abi.Type) bool {
	switch Kind(t.Kind()) {
	case Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr, Chan, Pointer, String, UnsafePointer:
		return true
	case Float32, Float64, Complex64, Complex128, Interface:
		return false
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return isReflexive(tt.Elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.Fields {
			if !isReflexive(f.Typ) {
				return false
			}
		}
		return true
	default:
		// Func, Map, Slice, Invalid
		panic("isReflexive called on non-key type " + stringFor(t))
	}
}

// needKeyUpdate reports whether map overwrites require the key to be copied.
func needKeyUpdate(t *abi.Type) bool {
	switch Kind(t.Kind()) {
	case Bool, Int, Int8, Int16, Int32, Int64, Uint, Uint8, Uint16, Uint32, Uint64, Uintptr, Chan, Pointer, UnsafePointer:
		return false
	case Float32, Float64, Complex64, Complex128, Interface, String:
		// Float keys can be updated from +0 to -0.
		// String keys can be updated to use a smaller backing store.
		// Interfaces might have floats or strings in them.
		return true
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return needKeyUpdate(tt.Elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.Fields {
			if needKeyUpdate(f.Typ) {
				return true
			}
		}
		return false
	default:
		// Func, Map, Slice, Invalid
		panic("needKeyUpdate called on non-key type " + stringFor(t))
	}
}

// hashMightPanic reports whether the hash of a map key of type t might panic.
func hashMightPanic(t *abi.Type) bool {
	switch Kind(t.Kind()) {
	case Interface:
		return true
	case Array:
		tt := (*arrayType)(unsafe.Pointer(t))
		return hashMightPanic(tt.Elem)
	case Struct:
		tt := (*structType)(unsafe.Pointer(t))
		for _, f := range tt.Fields {
			if hashMightPanic(f.Typ) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

func (t *rtype) gcSlice(begin, end uintptr) []byte {
	return (*[1 << 30]byte)(unsafe.Pointer(t.t.GCData))[begin:end:end]
}

// emitGCMask writes the GC mask for [n]typ into out, starting at bit
// offset base.
func emitGCMask(out []byte, base uintptr, typ *abi.Type, n uintptr) {
	if typ.Kind_&abi.KindGCProg != 0 {
		panic("reflect: unexpected GC program")
	}
	ptrs := typ.PtrBytes / goarch.PtrSize
	words := typ.Size_ / goarch.PtrSize
	mask := typ.GcSlice(0, (ptrs+7)/8)
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
func appendGCProg(dst []byte, typ *abi.Type) []byte {
	if typ.Kind_&abi.KindGCProg != 0 {
		// Element has GC program; emit one element.
		n := uintptr(*(*uint32)(unsafe.Pointer(typ.GCData)))
		prog := typ.GcSlice(4, 4+n-1)
		return append(dst, prog...)
	}

	// Element is small with pointer mask; use as literal bits.
	ptrs := typ.PtrBytes / goarch.PtrSize
	mask := typ.GcSlice(0, (ptrs+7)/8)

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
	typ := t.common()

	// Look in cache.
	ckey := cacheKey{Slice, typ, nil, 0}
	if slice, ok := lookupCache.Load(ckey); ok {
		return slice.(Type)
	}

	// Look in known types.
	s := "[]" + stringFor(typ)
	for _, tt := range typesByString(s) {
		slice := (*sliceType)(unsafe.Pointer(tt))
		if slice.Elem == typ {
			ti, _ := lookupCache.LoadOrStore(ckey, toRType(tt))
			return ti.(Type)
		}
	}

	// Make a slice type.
	var islice any = ([]unsafe.Pointer)(nil)
	prototype := *(**sliceType)(unsafe.Pointer(&islice))
	slice := *prototype
	slice.TFlag = 0
	slice.Str = resolveReflectName(newName(s, "", false, false))
	slice.Hash = fnv1(typ.Hash, '[')
	slice.Elem = typ
	slice.PtrToThis = 0

	ti, _ := lookupCache.LoadOrStore(ckey, toRType(&slice.Type))
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

// This must match cmd/compile/internal/compare.IsRegularMemory
func isRegularMemory(t Type) bool {
	switch t.Kind() {
	case Array:
		elem := t.Elem()
		if isRegularMemory(elem) {
			return true
		}
		return elem.Comparable() && t.Len() == 0
	case Int8, Int16, Int32, Int64, Int, Uint8, Uint16, Uint32, Uint64, Uint, Uintptr, Chan, Pointer, Bool, UnsafePointer:
		return true
	case Struct:
		num := t.NumField()
		switch num {
		case 0:
			return true
		case 1:
			field := t.Field(0)
			if field.Name == "_" {
				return false
			}
			return isRegularMemory(field.Type)
		default:
			for i := range num {
				field := t.Field(i)
				if field.Name == "_" || !isRegularMemory(field.Type) || isPaddedField(t, i) {
					return false
				}
			}
			return true
		}
	}
	return false
}

// isPaddedField reports whether the i'th field of struct type t is followed
// by padding.
func isPaddedField(t Type, i int) bool {
	field := t.Field(i)
	if i+1 < t.NumField() {
		return field.Offset+field.Type.Size() != t.Field(i+1).Offset
	}
	return field.Offset+field.Type.Size() != t.Size()
}

// StructOf returns the struct type containing fields.
// The Offset and Index fields are ignored and computed as they would be
// by the compiler.
//
// StructOf currently does not support promoted methods of embedded fields
// and panics if passed unexported StructFields.
func StructOf(fields []StructField) Type {
	var (
		hash       = fnv1(0, []byte("struct {")...)
		size       uintptr
		typalign   uint8
		comparable = true
		methods    []abi.Method

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
		ft := f.Typ
		if ft.Kind_&abi.KindGCProg != 0 {
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
		name := f.Name.Name()
		hash = fnv1(hash, []byte(name)...)
		if !f.Embedded() {
			repr = append(repr, (" " + name)...)
		} else {
			// Embedded field
			if f.Typ.Kind() == abi.Pointer {
				// Embedded ** and *interface{} are illegal
				elem := ft.Elem()
				if k := elem.Kind(); k == abi.Pointer || k == abi.Interface {
					panic("reflect.StructOf: illegal embedded field type " + stringFor(ft))
				}
			}

			switch Kind(f.Typ.Kind()) {
			case Interface:
				ift := (*interfaceType)(unsafe.Pointer(ft))
				for _, m := range ift.Methods {
					if pkgPath(ift.nameOff(m.Name)) != "" {
						// TODO(sbinet).  Issue 15924.
						panic("reflect: embedded interface with unexported method(s) not implemented")
					}

					fnStub := resolveReflectText(unsafe.Pointer(abi.FuncPCABIInternal(embeddedIfaceMethStub)))
					methods = append(methods, abi.Method{
						Name: resolveReflectName(ift.nameOff(m.Name)),
						Mtyp: resolveReflectType(ift.typeOff(m.Typ)),
						Ifn:  fnStub,
						Tfn:  fnStub,
					})
				}
			case Pointer:
				ptr := (*ptrType)(unsafe.Pointer(ft))
				if unt := ptr.Uncommon(); unt != nil {
					if i > 0 && unt.Mcount > 0 {
						// Issue 15924.
						panic("reflect: embedded type with methods not implemented if type is not first field")
					}
					if len(fields) > 1 {
						panic("reflect: embedded type with methods not implemented if there is more than one field")
					}
					for _, m := range unt.Methods() {
						mname := nameOffFor(ft, m.Name)
						if pkgPath(mname) != "" {
							// TODO(sbinet).
							// Issue 15924.
							panic("reflect: embedded interface with unexported method(s) not implemented")
						}
						methods = append(methods, abi.Method{
							Name: resolveReflectName(mname),
							Mtyp: resolveReflectType(typeOffFor(ft, m.Mtyp)),
							Ifn:  resolveReflectText(textOffFor(ft, m.Ifn)),
							Tfn:  resolveReflectText(textOffFor(ft, m.Tfn)),
						})
					}
				}
				if unt := ptr.Elem.Uncommon(); unt != nil {
					for _, m := range unt.Methods() {
						mname := nameOffFor(ft, m.Name)
						if pkgPath(mname) != "" {
							// TODO(sbinet)
							// Issue 15924.
							panic("reflect: embedded interface with unexported method(s) not implemented")
						}
						methods = append(methods, abi.Method{
							Name: resolveReflectName(mname),
							Mtyp: resolveReflectType(typeOffFor(ptr.Elem, m.Mtyp)),
							Ifn:  resolveReflectText(textOffFor(ptr.Elem, m.Ifn)),
							Tfn:  resolveReflectText(textOffFor(ptr.Elem, m.Tfn)),
						})
					}
				}
			default:
				if unt := ft.Uncommon(); unt != nil {
					if i > 0 && unt.Mcount > 0 {
						// Issue 15924.
						panic("reflect: embedded type with methods not implemented if type is not first field")
					}
					if len(fields) > 1 && ft.Kind_&abi.KindDirectIface != 0 {
						panic("reflect: embedded type with methods not implemented for non-pointer type")
					}
					for _, m := range unt.Methods() {
						mname := nameOffFor(ft, m.Name)
						if pkgPath(mname) != "" {
							// TODO(sbinet)
							// Issue 15924.
							panic("reflect: embedded interface with unexported method(s) not implemented")
						}
						methods = append(methods, abi.Method{
							Name: resolveReflectName(mname),
							Mtyp: resolveReflectType(typeOffFor(ft, m.Mtyp)),
							Ifn:  resolveReflectText(textOffFor(ft, m.Ifn)),
							Tfn:  resolveReflectText(textOffFor(ft, m.Tfn)),
						})

					}
				}
			}
		}
		if _, dup := fset[name]; dup && name != "_" {
			panic("reflect.StructOf: duplicate field " + name)
		}
		fset[name] = struct{}{}

		hash = fnv1(hash, byte(ft.Hash>>24), byte(ft.Hash>>16), byte(ft.Hash>>8), byte(ft.Hash))

		repr = append(repr, (" " + stringFor(ft))...)
		if f.Name.HasTag() {
			hash = fnv1(hash, []byte(f.Name.Tag())...)
			repr = append(repr, (" " + strconv.Quote(f.Name.Tag()))...)
		}
		if i < len(fields)-1 {
			repr = append(repr, ';')
		}

		comparable = comparable && (ft.Equal != nil)

		offset := align(size, uintptr(ft.Align_))
		if offset < size {
			panic("reflect.StructOf: struct size would exceed virtual address space")
		}
		if ft.Align_ > typalign {
			typalign = ft.Align_
		}
		size = offset + ft.Size_
		if size < offset {
			panic("reflect.StructOf: struct size would exceed virtual address space")
		}
		f.Offset = offset

		if ft.Size_ == 0 {
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
		if size == 0 {
			panic("reflect.StructOf: struct size would exceed virtual address space")
		}
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

		copy(tt.Elem().Field(2).Slice(0, len(methods)).Interface().([]abi.Method), methods)
	}
	// TODO(sbinet): Once we allow embedding multiple types,
	// methods will need to be sorted like the compiler does.
	// TODO(sbinet): Once we allow non-exported methods, we will
	// need to compute xcount as the number of exported methods.
	ut.Mcount = uint16(len(methods))
	ut.Xcount = ut.Mcount
	ut.Moff = uint32(unsafe.Sizeof(uncommonType{}))

	if len(fs) > 0 {
		repr = append(repr, ' ')
	}
	repr = append(repr, '}')
	hash = fnv1(hash, '}')
	str := string(repr)

	// Round the size up to be a multiple of the alignment.
	s := align(size, uintptr(typalign))
	if s < size {
		panic("reflect.StructOf: struct size would exceed virtual address space")
	}
	size = s

	// Make the struct type.
	var istruct any = struct{}{}
	prototype := *(**structType)(unsafe.Pointer(&istruct))
	*typ = *prototype
	typ.Fields = fs
	if pkgpath != "" {
		typ.PkgPath = newName(pkgpath, "", false, false)
	}

	// Look in cache.
	if ts, ok := structLookupCache.m.Load(hash); ok {
		for _, st := range ts.([]Type) {
			t := st.common()
			if haveIdenticalUnderlyingType(&typ.Type, t, true) {
				return toType(t)
			}
		}
	}

	// Not in cache, lock and retry.
	structLookupCache.Lock()
	defer structLookupCache.Unlock()
	if ts, ok := structLookupCache.m.Load(hash); ok {
		for _, st := range ts.([]Type) {
			t := st.common()
			if haveIdenticalUnderlyingType(&typ.Type, t, true) {
				return toType(t)
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
		if haveIdenticalUnderlyingType(&typ.Type, t, true) {
			// even if 't' wasn't a structType with methods, we should be ok
			// as the 'u uncommonType' field won't be accessed except when
			// tflag&abi.TFlagUncommon is set.
			return addToCache(toType(t))
		}
	}

	typ.Str = resolveReflectName(newName(str, "", false, false))
	if isRegularMemory(toType(&typ.Type)) {
		typ.TFlag = abi.TFlagRegularMemory
	} else {
		typ.TFlag = 0
	}
	typ.Hash = hash
	typ.Size_ = size
	typ.PtrBytes = typeptrdata(&typ.Type)
	typ.Align_ = typalign
	typ.FieldAlign_ = typalign
	typ.PtrToThis = 0
	if len(methods) > 0 {
		typ.TFlag |= abi.TFlagUncommon
	}

	if hasGCProg {
		lastPtrField := 0
		for i, ft := range fs {
			if ft.Typ.Pointers() {
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
			if !ft.Typ.Pointers() {
				// Ignore pointerless fields.
				continue
			}
			// Pad to start of this field with zeros.
			if ft.Offset > off {
				n := (ft.Offset - off) / goarch.PtrSize
				prog = append(prog, 0x01, 0x00) // emit a 0 bit
				if n > 1 {
					prog = append(prog, 0x81)      // repeat previous bit
					prog = appendVarint(prog, n-1) // n-1 times
				}
				off = ft.Offset
			}

			prog = appendGCProg(prog, ft.Typ)
			off += ft.Typ.PtrBytes
		}
		prog = append(prog, 0)
		*(*uint32)(unsafe.Pointer(&prog[0])) = uint32(len(prog) - 4)
		typ.Kind_ |= abi.KindGCProg
		typ.GCData = &prog[0]
	} else {
		typ.Kind_ &^= abi.KindGCProg
		bv := new(bitVector)
		addTypeBits(bv, 0, &typ.Type)
		if len(bv.data) > 0 {
			typ.GCData = &bv.data[0]
		}
	}
	typ.Equal = nil
	if comparable {
		typ.Equal = func(p, q unsafe.Pointer) bool {
			for _, ft := range typ.Fields {
				pi := add(p, ft.Offset, "&x.field safe")
				qi := add(q, ft.Offset, "&x.field safe")
				if !ft.Typ.Equal(pi, qi) {
					return false
				}
			}
			return true
		}
	}

	switch {
	case len(fs) == 1 && !fs[0].Typ.IfaceIndir():
		// structs of 1 direct iface type can be direct
		typ.Kind_ |= abi.KindDirectIface
	default:
		typ.Kind_ &^= abi.KindDirectIface
	}

	return addToCache(toType(&typ.Type))
}

func embeddedIfaceMethStub() {
	panic("reflect: StructOf does not support methods of embedded interfaces")
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

	resolveReflectType(field.Type.common()) // install in runtime
	f := structField{
		Name:   newName(field.Name, string(field.Tag), field.IsExported(), field.Anonymous),
		Typ:    field.Type.common(),
		Offset: 0,
	}
	return f, field.PkgPath
}

// typeptrdata returns the length in bytes of the prefix of t
// containing pointer data. Anything after this offset is scalar data.
// keep in sync with ../cmd/compile/internal/reflectdata/reflect.go
func typeptrdata(t *abi.Type) uintptr {
	switch t.Kind() {
	case abi.Struct:
		st := (*structType)(unsafe.Pointer(t))
		// find the last field that has pointers.
		field := -1
		for i := range st.Fields {
			ft := st.Fields[i].Typ
			if ft.Pointers() {
				field = i
			}
		}
		if field == -1 {
			return 0
		}
		f := st.Fields[field]
		return f.Offset + f.Typ.PtrBytes

	default:
		panic("reflect.typeptrdata: unexpected type, " + stringFor(t))
	}
}

// ArrayOf returns the array type with the given length and element type.
// For example, if t represents int, ArrayOf(5, t) represents [5]int.
//
// If the resulting type would be larger than the available address space,
// ArrayOf panics.
func ArrayOf(length int, elem Type) Type {
	if length < 0 {
		panic("reflect: negative length passed to ArrayOf")
	}

	typ := elem.common()

	// Look in cache.
	ckey := cacheKey{Array, typ, nil, uintptr(length)}
	if array, ok := lookupCache.Load(ckey); ok {
		return array.(Type)
	}

	// Look in known types.
	s := "[" + strconv.Itoa(length) + "]" + stringFor(typ)
	for _, tt := range typesByString(s) {
		array := (*arrayType)(unsafe.Pointer(tt))
		if array.Elem == typ {
			ti, _ := lookupCache.LoadOrStore(ckey, toRType(tt))
			return ti.(Type)
		}
	}

	// Make an array type.
	var iarray any = [1]unsafe.Pointer{}
	prototype := *(**arrayType)(unsafe.Pointer(&iarray))
	array := *prototype
	array.TFlag = typ.TFlag & abi.TFlagRegularMemory
	array.Str = resolveReflectName(newName(s, "", false, false))
	array.Hash = fnv1(typ.Hash, '[')
	for n := uint32(length); n > 0; n >>= 8 {
		array.Hash = fnv1(array.Hash, byte(n))
	}
	array.Hash = fnv1(array.Hash, ']')
	array.Elem = typ
	array.PtrToThis = 0
	if typ.Size_ > 0 {
		max := ^uintptr(0) / typ.Size_
		if uintptr(length) > max {
			panic("reflect.ArrayOf: array size would exceed virtual address space")
		}
	}
	array.Size_ = typ.Size_ * uintptr(length)
	if length > 0 && typ.Pointers() {
		array.PtrBytes = typ.Size_*uintptr(length-1) + typ.PtrBytes
	}
	array.Align_ = typ.Align_
	array.FieldAlign_ = typ.FieldAlign_
	array.Len = uintptr(length)
	array.Slice = &(SliceOf(elem).(*rtype).t)

	switch {
	case !typ.Pointers() || array.Size_ == 0:
		// No pointers.
		array.GCData = nil
		array.PtrBytes = 0

	case length == 1:
		// In memory, 1-element array looks just like the element.
		array.Kind_ |= typ.Kind_ & abi.KindGCProg
		array.GCData = typ.GCData
		array.PtrBytes = typ.PtrBytes

	case typ.Kind_&abi.KindGCProg == 0 && array.Size_ <= abi.MaxPtrmaskBytes*8*goarch.PtrSize:
		// Element is small with pointer mask; array is still small.
		// Create direct pointer mask by turning each 1 bit in elem
		// into length 1 bits in larger mask.
		n := (array.PtrBytes/goarch.PtrSize + 7) / 8
		// Runtime needs pointer masks to be a multiple of uintptr in size.
		n = (n + goarch.PtrSize - 1) &^ (goarch.PtrSize - 1)
		mask := make([]byte, n)
		emitGCMask(mask, 0, typ, array.Len)
		array.GCData = &mask[0]

	default:
		// Create program that emits one element
		// and then repeats to make the array.
		prog := []byte{0, 0, 0, 0} // will be length of prog
		prog = appendGCProg(prog, typ)
		// Pad from ptrdata to size.
		elemPtrs := typ.PtrBytes / goarch.PtrSize
		elemWords := typ.Size_ / goarch.PtrSize
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
		array.Kind_ |= abi.KindGCProg
		array.GCData = &prog[0]
		array.PtrBytes = array.Size_ // overestimate but ok; must match program
	}

	etyp := typ
	esize := etyp.Size()

	array.Equal = nil
	if eequal := etyp.Equal; eequal != nil {
		array.Equal = func(p, q unsafe.Pointer) bool {
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
	case length == 1 && !typ.IfaceIndir():
		// array of 1 direct iface type can be direct
		array.Kind_ |= abi.KindDirectIface
	default:
		array.Kind_ &^= abi.KindDirectIface
	}

	ti, _ := lookupCache.LoadOrStore(ckey, toRType(&array.Type))
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
//
// toType should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - fortio.org/log
//   - github.com/goccy/go-json
//   - github.com/goccy/go-reflect
//   - github.com/sohaha/zlsgo
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname toType
func toType(t *abi.Type) Type {
	if t == nil {
		return nil
	}
	return toRType(t)
}

type layoutKey struct {
	ftyp *funcType // function signature
	rcvr *abi.Type // receiver type, or nil if none
}

type layoutType struct {
	t         *abi.Type
	framePool *sync.Pool
	abid      abiDesc
}

var layoutCache sync.Map // map[layoutKey]layoutType

// funcLayout computes a struct type representing the layout of the
// stack-assigned function arguments and return values for the function
// type t.
// If rcvr != nil, rcvr specifies the type of the receiver.
// The returned type exists only for GC, so we only fill out GC relevant info.
// Currently, that's just size and the GC program. We also fill in
// the name for possible debugging use.
func funcLayout(t *funcType, rcvr *abi.Type) (frametype *abi.Type, framePool *sync.Pool, abid abiDesc) {
	if t.Kind() != abi.Func {
		panic("reflect: funcLayout of non-func type " + stringFor(&t.Type))
	}
	if rcvr != nil && rcvr.Kind() == abi.Interface {
		panic("reflect: funcLayout with interface receiver " + stringFor(rcvr))
	}
	k := layoutKey{t, rcvr}
	if lti, ok := layoutCache.Load(k); ok {
		lt := lti.(layoutType)
		return lt.t, lt.framePool, lt.abid
	}

	// Compute the ABI layout.
	abid = newAbiDesc(t, rcvr)

	// build dummy rtype holding gc program
	x := &abi.Type{
		Align_: goarch.PtrSize,
		// Don't add spill space here; it's only necessary in
		// reflectcall's frame, not in the allocated frame.
		// TODO(mknyszek): Remove this comment when register
		// spill space in the frame is no longer required.
		Size_:    align(abid.retOffset+abid.ret.stackBytes, goarch.PtrSize),
		PtrBytes: uintptr(abid.stackPtrs.n) * goarch.PtrSize,
	}
	if abid.stackPtrs.n > 0 {
		x.GCData = &abid.stackPtrs.data[0]
	}

	var s string
	if rcvr != nil {
		s = "methodargs(" + stringFor(rcvr) + ")(" + stringFor(&t.Type) + ")"
	} else {
		s = "funcargs(" + stringFor(&t.Type) + ")"
	}
	x.Str = resolveReflectName(newName(s, "", false, false))

	// cache result for future callers
	framePool = &sync.Pool{New: func() any {
		return unsafe_New(x)
	}}
	lti, _ := layoutCache.LoadOrStore(k, layoutType{
		t:         x,
		framePool: framePool,
		abid:      abid,
	})
	lt := lti.(layoutType)
	return lt.t, lt.framePool, lt.abid
}

// Note: this type must agree with runtime.bitvector.
type bitVector struct {
	n    uint32 // number of bits
	data []byte
}

// append a bit to the bitmap.
func (bv *bitVector) append(bit uint8) {
	if bv.n%(8*goarch.PtrSize) == 0 {
		// Runtime needs pointer masks to be a multiple of uintptr in size.
		// Since reflect passes bv.data directly to the runtime as a pointer mask,
		// we append a full uintptr of zeros at a time.
		for i := 0; i < goarch.PtrSize; i++ {
			bv.data = append(bv.data, 0)
		}
	}
	bv.data[bv.n/8] |= bit << (bv.n % 8)
	bv.n++
}

func addTypeBits(bv *bitVector, offset uintptr, t *abi.Type) {
	if !t.Pointers() {
		return
	}

	switch Kind(t.Kind_ & abi.KindMask) {
	case Chan, Func, Map, Pointer, Slice, String, UnsafePointer:
		// 1 pointer at start of representation
		for bv.n < uint32(offset/goarch.PtrSize) {
			bv.append(0)
		}
		bv.append(1)

	case Interface:
		// 2 pointers
		for bv.n < uint32(offset/goarch.PtrSize) {
			bv.append(0)
		}
		bv.append(1)
		bv.append(1)

	case Array:
		// repeat inner type
		tt := (*arrayType)(unsafe.Pointer(t))
		for i := 0; i < int(tt.Len); i++ {
			addTypeBits(bv, offset+uintptr(i)*tt.Elem.Size_, tt.Elem)
		}

	case Struct:
		// apply fields
		tt := (*structType)(unsafe.Pointer(t))
		for i := range tt.Fields {
			f := &tt.Fields[i]
			addTypeBits(bv, offset+f.Offset, f.Typ)
		}
	}
}

// TypeFor returns the [Type] that represents the type argument T.
func TypeFor[T any]() Type {
	var v T
	if t := TypeOf(v); t != nil {
		return t // optimize for T being a non-interface kind
	}
	return TypeOf((*T)(nil)).Elem() // only for an interface kind
}
