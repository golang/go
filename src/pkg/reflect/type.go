// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package reflect implements run-time reflection, allowing a program to
// manipulate objects with arbitrary types.  The typical use is to take a value
// with static type interface{} and extract its dynamic type information by
// calling TypeOf, which returns a Type.
//
// A call to ValueOf returns a Value representing the run-time data.
// Zero takes a Type and returns a Value representing a zero value
// for that type.
//
// See "The Laws of Reflection" for an introduction to reflection in Go:
// http://blog.golang.org/2011/09/laws-of-reflection.html
package reflect

import (
	"strconv"
	"sync"
	"unsafe"
)

// Type is the representation of a Go type.
//
// Not all methods apply to all kinds of types.  Restrictions,
// if any, are noted in the documentation for each method.
// Use the Kind method to find out the kind of type before
// calling kind-specific methods.  Calling a method
// inappropriate to the kind of type causes a run-time panic.
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
	// fields describe a function whose first argument is the receiver.
	//
	// For an interface type, the returned Method's Type field gives the
	// method signature, without a receiver, and the Func field is nil.
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

	// NumMethod returns the number of methods in the type's method set.
	NumMethod() int

	// Name returns the type's name within its package.
	// It returns an empty string for unnamed types.
	Name() string

	// PkgPath returns the type's package path.
	// The package path is a full package import path like "encoding/base64".
	// PkgPath returns an empty string for unnamed or predeclared types.
	PkgPath() string

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size() uintptr

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., base64 instead of "encoding/base64") and is not
	// guaranteed to be unique among types.  To test for equality,
	// compare the Types directly.
	String() string

	// Kind returns the specific kind of this type.
	Kind() Kind

	// Implements returns true if the type implements the interface type u.
	Implements(u Type) bool

	// AssignableTo returns true if a value of the type is assignable to type u.
	AssignableTo(u Type) bool

	// Methods applicable only to some types, depending on Kind.
	// The methods allowed for each kind are:
	//
	//	Int*, Uint*, Float*, Complex*: Bits
	//	Array: Elem, Len
	//	Chan: ChanDir, Elem
	//	Func: In, NumIn, Out, NumOut, IsVariadic.
	//	Map: Key, Elem
	//	Ptr: Elem
	//	Slice: Elem
	//	Struct: Field, FieldByIndex, FieldByName, FieldByNameFunc, NumField

	// Bits returns the size of the type in bits.
	// It panics if the type's Kind is not one of the
	// sized or unsized Int, Uint, Float, or Complex kinds.
	Bits() int

	// ChanDir returns a channel type's direction.
	// It panics if the type's Kind is not Chan.
	ChanDir() ChanDir

	// IsVariadic returns true if a function type's final input parameter
	// is a "..." parameter.  If so, t.In(t.NumIn() - 1) returns the parameter's
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
	// It panics if the type's Kind is not Array, Chan, Map, Ptr, or Slice.
	Elem() Type

	// Field returns a struct type's i'th field.
	// It panics if the type's Kind is not Struct.
	// It panics if i is not in the range [0, NumField()).
	Field(i int) StructField

	// FieldByIndex returns the nested field corresponding
	// to the index sequence.  It is equivalent to calling Field
	// successively for each index i.
	// It panics if the type's Kind is not Struct.
	FieldByIndex(index []int) StructField

	// FieldByName returns the struct field with the given name
	// and a boolean indicating if the field was found.
	FieldByName(name string) (StructField, bool)

	// FieldByNameFunc returns the first struct field with a name
	// that satisfies the match function and a boolean indicating if
	// the field was found.
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

	runtimeType() *runtimeType
	common() *commonType
	uncommon() *uncommonType
}

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
	Ptr
	Slice
	String
	Struct
	UnsafePointer
)

/*
 * These data structures are known to the compiler (../../cmd/gc/reflect.c).
 * A few are known to ../runtime/type.go to convey to debuggers.
 */

// The compiler can only construct empty interface values at
// compile time; non-empty interface values get created
// during initialization.  Type is an empty interface
// so that the compiler can lay out references as data.
// The underlying type is *reflect.ArrayType and so on.
type runtimeType interface{}

// commonType is the common implementation of most values.
// It is embedded in other, public struct types, but always
// with a unique tag like `reflect:"array"` or `reflect:"ptr"`
// so that code cannot convert from, say, *arrayType to *ptrType.
type commonType struct {
	size          uintptr      // size in bytes
	hash          uint32       // hash of type; avoids computation in hash tables
	_             uint8        // unused/padding
	align         uint8        // alignment of variable with this type
	fieldAlign    uint8        // alignment of struct field with this type
	kind          uint8        // enumeration for C
	alg           *uintptr     // algorithm table (../runtime/runtime.h:/Alg)
	string        *string      // string form; unnecessary  but undeniably useful
	*uncommonType              // (relatively) uncommon fields
	ptrToThis     *runtimeType // pointer to this type, if used in binary or has methods
}

// Method on non-interface type
type method struct {
	name    *string        // name of method
	pkgPath *string        // nil for exported Names; otherwise import path
	mtyp    *runtimeType   // method type (without receiver)
	typ     *runtimeType   // .(*FuncType) underneath (with receiver)
	ifn     unsafe.Pointer // fn used in interface call (one-word receiver)
	tfn     unsafe.Pointer // fn used for normal method call
}

// uncommonType is present only for types with names or methods
// (if T is a named type, the uncommonTypes for T and *T have methods).
// Using a pointer to this struct reduces the overall size required
// to describe an unnamed type with no methods.
type uncommonType struct {
	name    *string  // name of type
	pkgPath *string  // import path; nil for built-in types like int, string
	methods []method // methods associated with type
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
	commonType `reflect:"array"`
	elem       *runtimeType // array element type
	slice      *runtimeType // slice type
	len        uintptr
}

// chanType represents a channel type.
type chanType struct {
	commonType `reflect:"chan"`
	elem       *runtimeType // channel element type
	dir        uintptr      // channel direction (ChanDir)
}

// funcType represents a function type.
type funcType struct {
	commonType `reflect:"func"`
	dotdotdot  bool           // last input parameter is ...
	in         []*runtimeType // input parameter types
	out        []*runtimeType // output parameter types
}

// imethod represents a method on an interface type
type imethod struct {
	name    *string      // name of method
	pkgPath *string      // nil for exported Names; otherwise import path
	typ     *runtimeType // .(*FuncType) underneath
}

// interfaceType represents an interface type.
type interfaceType struct {
	commonType `reflect:"interface"`
	methods    []imethod // sorted by hash
}

// mapType represents a map type.
type mapType struct {
	commonType `reflect:"map"`
	key        *runtimeType // map key type
	elem       *runtimeType // map element (value) type
}

// ptrType represents a pointer type.
type ptrType struct {
	commonType `reflect:"ptr"`
	elem       *runtimeType // pointer element (pointed at) type
}

// sliceType represents a slice type.
type sliceType struct {
	commonType `reflect:"slice"`
	elem       *runtimeType // slice element type
}

// Struct field
type structField struct {
	name    *string      // nil for embedded fields
	pkgPath *string      // nil for exported Names; otherwise import path
	typ     *runtimeType // type of field
	tag     *string      // nil if no tag
	offset  uintptr      // byte offset of field within struct
}

// structType represents a struct type.
type structType struct {
	commonType `reflect:"struct"`
	fields     []structField // sorted by offset
}

/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

// Method represents a single method.
type Method struct {
	PkgPath string // empty for uppercase Name
	Name    string
	Type    Type
	Func    Value
	Index   int
}

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

func (t *commonType) toType() Type {
	if t == nil {
		return nil
	}
	return t
}

func (t *commonType) String() string { return *t.string }

func (t *commonType) Size() uintptr { return t.size }

func (t *commonType) Bits() int {
	if t == nil {
		panic("reflect: Bits of nil Type")
	}
	k := t.Kind()
	if k < Int || k > Complex128 {
		panic("reflect: Bits of non-arithmetic Type " + t.String())
	}
	return int(t.size) * 8
}

func (t *commonType) Align() int { return int(t.align) }

func (t *commonType) FieldAlign() int { return int(t.fieldAlign) }

func (t *commonType) Kind() Kind { return Kind(t.kind & kindMask) }

func (t *commonType) common() *commonType { return t }

func (t *uncommonType) Method(i int) (m Method) {
	if t == nil || i < 0 || i >= len(t.methods) {
		panic("reflect: Method index out of range")
	}
	p := &t.methods[i]
	if p.name != nil {
		m.Name = *p.name
	}
	fl := flag(Func) << flagKindShift
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
		fl |= flagRO
	}
	mt := toCommonType(p.typ)
	m.Type = mt
	fn := p.tfn
	m.Func = Value{mt, fn, fl}
	m.Index = i
	return
}

func (t *uncommonType) NumMethod() int {
	if t == nil {
		return 0
	}
	return len(t.methods)
}

func (t *uncommonType) MethodByName(name string) (m Method, ok bool) {
	if t == nil {
		return
	}
	var p *method
	for i := range t.methods {
		p = &t.methods[i]
		if p.name != nil && *p.name == name {
			return t.Method(i), true
		}
	}
	return
}

// TODO(rsc): 6g supplies these, but they are not
// as efficient as they could be: they have commonType
// as the receiver instead of *commonType.
func (t *commonType) NumMethod() int {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.NumMethod()
	}
	return t.uncommonType.NumMethod()
}

func (t *commonType) Method(i int) (m Method) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.Method(i)
	}
	return t.uncommonType.Method(i)
}

func (t *commonType) MethodByName(name string) (m Method, ok bool) {
	if t.Kind() == Interface {
		tt := (*interfaceType)(unsafe.Pointer(t))
		return tt.MethodByName(name)
	}
	return t.uncommonType.MethodByName(name)
}

func (t *commonType) PkgPath() string {
	return t.uncommonType.PkgPath()
}

func (t *commonType) Name() string {
	return t.uncommonType.Name()
}

func (t *commonType) ChanDir() ChanDir {
	if t.Kind() != Chan {
		panic("reflect: ChanDir of non-chan type")
	}
	tt := (*chanType)(unsafe.Pointer(t))
	return ChanDir(tt.dir)
}

func (t *commonType) IsVariadic() bool {
	if t.Kind() != Func {
		panic("reflect: IsVariadic of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return tt.dotdotdot
}

func (t *commonType) Elem() Type {
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
	panic("reflect; Elem of invalid type")
}

func (t *commonType) Field(i int) StructField {
	if t.Kind() != Struct {
		panic("reflect: Field of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.Field(i)
}

func (t *commonType) FieldByIndex(index []int) StructField {
	if t.Kind() != Struct {
		panic("reflect: FieldByIndex of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByIndex(index)
}

func (t *commonType) FieldByName(name string) (StructField, bool) {
	if t.Kind() != Struct {
		panic("reflect: FieldByName of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByName(name)
}

func (t *commonType) FieldByNameFunc(match func(string) bool) (StructField, bool) {
	if t.Kind() != Struct {
		panic("reflect: FieldByNameFunc of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return tt.FieldByNameFunc(match)
}

func (t *commonType) In(i int) Type {
	if t.Kind() != Func {
		panic("reflect: In of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.in[i])
}

func (t *commonType) Key() Type {
	if t.Kind() != Map {
		panic("reflect: Key of non-map type")
	}
	tt := (*mapType)(unsafe.Pointer(t))
	return toType(tt.key)
}

func (t *commonType) Len() int {
	if t.Kind() != Array {
		panic("reflect: Len of non-array type")
	}
	tt := (*arrayType)(unsafe.Pointer(t))
	return int(tt.len)
}

func (t *commonType) NumField() int {
	if t.Kind() != Struct {
		panic("reflect: NumField of non-struct type")
	}
	tt := (*structType)(unsafe.Pointer(t))
	return len(tt.fields)
}

func (t *commonType) NumIn() int {
	if t.Kind() != Func {
		panic("reflect; NumIn of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return len(tt.in)
}

func (t *commonType) NumOut() int {
	if t.Kind() != Func {
		panic("reflect; NumOut of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return len(tt.out)
}

func (t *commonType) Out(i int) Type {
	if t.Kind() != Func {
		panic("reflect: Out of non-func type")
	}
	tt := (*funcType)(unsafe.Pointer(t))
	return toType(tt.out[i])
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
	m.Name = *p.name
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
	}
	m.Type = toType(p.typ)
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
		if *p.name == name {
			return t.Method(i), true
		}
	}
	return
}

type StructField struct {
	PkgPath   string // empty for uppercase Name
	Name      string
	Type      Type
	Tag       StructTag
	Offset    uintptr
	Index     []int
	Anonymous bool
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
// returned by Get is unspecified.
func (tag StructTag) Get(key string) string {
	for tag != "" {
		// skip leading space
		i := 0
		for i < len(tag) && tag[i] == ' ' {
			i++
		}
		tag = tag[i:]
		if tag == "" {
			break
		}

		// scan to colon.
		// a space or a quote is a syntax error
		i = 0
		for i < len(tag) && tag[i] != ' ' && tag[i] != ':' && tag[i] != '"' {
			i++
		}
		if i+1 >= len(tag) || tag[i] != ':' || tag[i+1] != '"' {
			break
		}
		name := string(tag[:i])
		tag = tag[i+1:]

		// scan quoted string to find value
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
			value, _ := strconv.Unquote(qvalue)
			return value
		}
	}
	return ""
}

// Field returns the i'th struct field.
func (t *structType) Field(i int) (f StructField) {
	if i < 0 || i >= len(t.fields) {
		return
	}
	p := &t.fields[i]
	f.Type = toType(p.typ)
	if p.name != nil {
		f.Name = *p.name
	} else {
		t := f.Type
		if t.Kind() == Ptr {
			t = t.Elem()
		}
		f.Name = t.Name()
		f.Anonymous = true
	}
	if p.pkgPath != nil {
		f.PkgPath = *p.pkgPath
	}
	if p.tag != nil {
		f.Tag = StructTag(*p.tag)
	}
	f.Offset = p.offset

	// NOTE(rsc): This is the only allocation in the interface
	// presented by a reflect.Type.  It would be nice to avoid,
	// at least in the common cases, but we need to make sure
	// that misbehaving clients of reflect cannot affect other
	// uses of reflect.  One possibility is CL 5371098, but we
	// postponed that ugliness until there is a demonstrated
	// need for the performance.  This is issue 2320.
	f.Index = []int{i}
	return
}

// TODO(gri): Should there be an error/bool indicator if the index
//            is wrong for FieldByIndex?

// FieldByIndex returns the nested field corresponding to index.
func (t *structType) FieldByIndex(index []int) (f StructField) {
	f.Type = Type(t.toType())
	for i, x := range index {
		if i > 0 {
			ft := f.Type
			if ft.Kind() == Ptr && ft.Elem().Kind() == Struct {
				ft = ft.Elem()
			}
			f.Type = ft
		}
		f = f.Type.Field(x)
	}
	return
}

const inf = 1 << 30 // infinity - no struct has that many nesting levels

func (t *structType) fieldByNameFunc(match func(string) bool, mark map[*structType]bool, depth int) (ff StructField, fd int) {
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
			if ft.Kind() == Ptr {
				ft = ft.Elem()
			}
			switch {
			case match(ft.Name()):
				// Matching anonymous top-level field.
				d = depth
			case fd > depth:
				// No top-level field yet; look inside nested structs.
				if ft.Kind() == Struct {
					st := (*structType)(unsafe.Pointer(ft.(*commonType)))
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
		if depth >= len(ff.Index) {
			ff.Index = make([]int, depth+1)
		}
		if len(ff.Index) > 1 {
			ff.Index[depth] = fi
		}
	} else {
		// None or more than one matching field found.
		fd = inf
	}

	delete(mark, t)
	return
}

// FieldByName returns the struct field with the given name
// and a boolean to indicate if the field was found.
func (t *structType) FieldByName(name string) (f StructField, present bool) {
	return t.FieldByNameFunc(func(s string) bool { return s == name })
}

// FieldByNameFunc returns the struct field with a name that satisfies the
// match function and a boolean to indicate if the field was found.
func (t *structType) FieldByNameFunc(match func(string) bool) (f StructField, present bool) {
	if ff, fd := t.fieldByNameFunc(match, make(map[*structType]bool), 0); fd < inf {
		ff.Index = ff.Index[0 : fd+1]
		f, present = ff, true
	}
	return
}

// Convert runtime type to reflect type.
func toCommonType(p *runtimeType) *commonType {
	if p == nil {
		return nil
	}
	return (*p).(*commonType)
}

func toType(p *runtimeType) Type {
	if p == nil {
		return nil
	}
	return (*p).(*commonType)
}

// TypeOf returns the reflection Type of the value in the interface{}.
func TypeOf(i interface{}) Type {
	eface := *(*emptyInterface)(unsafe.Pointer(&i))
	return toType(eface.typ)
}

// ptrMap is the cache for PtrTo.
var ptrMap struct {
	sync.RWMutex
	m map[*commonType]*ptrType
}

func (t *commonType) runtimeType() *runtimeType {
	// The runtimeType always precedes the commonType in memory.
	// Adjust pointer to find it.
	var rt struct {
		i  runtimeType
		ct commonType
	}
	return (*runtimeType)(unsafe.Pointer(uintptr(unsafe.Pointer(t)) - unsafe.Offsetof(rt.ct)))
}

// PtrTo returns the pointer type with element t.
// For example, if t represents type Foo, PtrTo(t) represents *Foo.
func PtrTo(t Type) Type {
	return t.(*commonType).ptrTo()
}

func (ct *commonType) ptrTo() *commonType {
	if p := ct.ptrToThis; p != nil {
		return toCommonType(p)
	}

	// Otherwise, synthesize one.
	// This only happens for pointers with no methods.
	// We keep the mapping in a map on the side, because
	// this operation is rare and a separate map lets us keep
	// the type structures in read-only memory.
	ptrMap.RLock()
	if m := ptrMap.m; m != nil {
		if p := m[ct]; p != nil {
			ptrMap.RUnlock()
			return &p.commonType
		}
	}
	ptrMap.RUnlock()
	ptrMap.Lock()
	if ptrMap.m == nil {
		ptrMap.m = make(map[*commonType]*ptrType)
	}
	p := ptrMap.m[ct]
	if p != nil {
		// some other goroutine won the race and created it
		ptrMap.Unlock()
		return &p.commonType
	}

	var rt struct {
		i runtimeType
		ptrType
	}
	rt.i = &rt.commonType

	// initialize p using *byte's ptrType as a prototype.
	p = &rt.ptrType
	var ibyte interface{} = (*byte)(nil)
	bp := (*ptrType)(unsafe.Pointer((**(**runtimeType)(unsafe.Pointer(&ibyte))).(*commonType)))
	*p = *bp

	s := "*" + *ct.string
	p.string = &s

	// For the type structures linked into the binary, the
	// compiler provides a good hash of the string.
	// Create a good hash for the new string by using
	// the FNV-1 hash's mixing function to combine the
	// old hash and the new "*".
	p.hash = ct.hash*16777619 ^ '*'

	p.uncommonType = nil
	p.ptrToThis = nil
	p.elem = (*runtimeType)(unsafe.Pointer(uintptr(unsafe.Pointer(ct)) - unsafe.Offsetof(rt.ptrType)))

	ptrMap.m[ct] = p
	ptrMap.Unlock()
	return &p.commonType
}

func (t *commonType) Implements(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.Implements")
	}
	if u.Kind() != Interface {
		panic("reflect: non-interface type passed to Type.Implements")
	}
	return implements(u.(*commonType), t)
}

func (t *commonType) AssignableTo(u Type) bool {
	if u == nil {
		panic("reflect: nil type passed to Type.AssignableTo")
	}
	uu := u.(*commonType)
	return directlyAssignable(uu, t) || implements(uu, t)
}

// implements returns true if the type V implements the interface type T.
func implements(T, V *commonType) bool {
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
	// See also ../runtime/iface.c.
	if V.Kind() == Interface {
		v := (*interfaceType)(unsafe.Pointer(V))
		i := 0
		for j := 0; j < len(v.methods); j++ {
			tm := &t.methods[i]
			vm := &v.methods[j]
			if vm.name == tm.name && vm.pkgPath == tm.pkgPath && vm.typ == tm.typ {
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
	for j := 0; j < len(v.methods); j++ {
		tm := &t.methods[i]
		vm := &v.methods[j]
		if vm.name == tm.name && vm.pkgPath == tm.pkgPath && vm.mtyp == tm.typ {
			if i++; i >= len(t.methods) {
				return true
			}
		}
	}
	return false
}

// directlyAssignable returns true if a value x of type V can be directly
// assigned (using memmove) to a value of type T.
// http://golang.org/doc/go_spec.html#Assignability
// Ignoring the interface rules (implemented elsewhere)
// and the ideal constant rules (no ideal constants at run time).
func directlyAssignable(T, V *commonType) bool {
	// x's type V is identical to T?
	if T == V {
		return true
	}

	// Otherwise at least one of T and V must be unnamed
	// and they must have the same kind.
	if T.Name() != "" && V.Name() != "" || T.Kind() != V.Kind() {
		return false
	}

	// x's type T and V have identical underlying types.
	// Since at least one is unnamed, only the composite types
	// need to be considered.
	switch T.Kind() {
	case Array:
		return T.Elem() == V.Elem() && T.Len() == V.Len()

	case Chan:
		// Special case:
		// x is a bidirectional channel value, T is a channel type,
		// and x's type V and T have identical element types.
		if V.ChanDir() == BothDir && T.Elem() == V.Elem() {
			return true
		}

		// Otherwise continue test for identical underlying type.
		return V.ChanDir() == T.ChanDir() && T.Elem() == V.Elem()

	case Func:
		t := (*funcType)(unsafe.Pointer(T))
		v := (*funcType)(unsafe.Pointer(V))
		if t.dotdotdot != v.dotdotdot || len(t.in) != len(v.in) || len(t.out) != len(v.out) {
			return false
		}
		for i, typ := range t.in {
			if typ != v.in[i] {
				return false
			}
		}
		for i, typ := range t.out {
			if typ != v.out[i] {
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
		return T.Key() == V.Key() && T.Elem() == V.Elem()

	case Ptr, Slice:
		return T.Elem() == V.Elem()

	case Struct:
		t := (*structType)(unsafe.Pointer(T))
		v := (*structType)(unsafe.Pointer(V))
		if len(t.fields) != len(v.fields) {
			return false
		}
		for i := range t.fields {
			tf := &t.fields[i]
			vf := &v.fields[i]
			if tf.name != vf.name || tf.pkgPath != vf.pkgPath ||
				tf.typ != vf.typ || tf.tag != vf.tag || tf.offset != vf.offset {
				return false
			}
		}
		return true
	}

	return false
}
