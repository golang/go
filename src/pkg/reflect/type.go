// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The reflect package implements run-time reflection, allowing a program to
// manipulate objects with arbitrary types.  The typical use is to take a
// value with static type interface{} and extract its dynamic type
// information by calling Typeof, which returns a Type.
//
// A call to NewValue returns a Value representing the run-time data.
// Zero takes a Type and returns a Value representing a zero value
// for that type.
package reflect

import (
	"runtime"
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

	// NumMethod returns the number of methods in the type's method set.
	NumMethod() int

	// Name returns the type's name within its package.
	// It returns an empty string for unnamed types.
	Name() string

	// PkgPath returns the type's package path.
	// The package path is a full package import path like "container/vector".
	// PkgPath returns an empty string for unnamed types.
	PkgPath() string

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size() uintptr

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., vector instead of "container/vector") and is not
	// guaranteed to be unique among types.  To test for equality,
	// compare the Types directly.
	String() string

	// Kind returns the specific kind of this type.
	Kind() Kind

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
	// For concreteness, if t represents func(x int, y ... float), then
	//
	//	t.NumIn() == 2
	//	t.In(0) is the reflect.Type for "int"
	//	t.In(1) is the reflect.Type for "[]float"
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

	runtimeType() *runtime.Type
	common() *commonType
	uncommon() *uncommonType
}

// A Kind represents the specific kind of type that a Type represents.
// The zero Kind is not a valid kind.
type Kind uint8

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

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir ChanDir = 1 << iota
	SendDir
	BothDir = RecvDir | SendDir
)


// arrayType represents a fixed array type.
type arrayType struct {
	commonType "array"
	elem       *runtime.Type
	slice      *runtime.Type
	len        uintptr
}

// chanType represents a channel type.
type chanType struct {
	commonType "chan"
	elem       *runtime.Type
	dir        uintptr
}

// funcType represents a function type.
type funcType struct {
	commonType "func"
	dotdotdot  bool
	in         []*runtime.Type
	out        []*runtime.Type
}

// imethod represents a method on an interface type
type imethod struct {
	name    *string
	pkgPath *string
	typ     *runtime.Type
}

// interfaceType represents an interface type.
type interfaceType struct {
	commonType "interface"
	methods    []imethod
}

// mapType represents a map type.
type mapType struct {
	commonType "map"
	key        *runtime.Type
	elem       *runtime.Type
}

// ptrType represents a pointer type.
type ptrType struct {
	commonType "ptr"
	elem       *runtime.Type
}

// sliceType represents a slice type.
type sliceType struct {
	commonType "slice"
	elem       *runtime.Type
}

// Struct field
type structField struct {
	name    *string
	pkgPath *string
	typ     *runtime.Type
	tag     *string
	offset  uintptr
}

// structType represents a struct type.
type structType struct {
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
	Type    Type
	Func    Value
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
		return
	}
	p := &t.methods[i]
	if p.name != nil {
		m.Name = *p.name
	}
	flag := uint32(0)
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
		flag |= flagRO
	}
	m.Type = toType(p.typ)
	fn := p.tfn
	m.Func = valueFromIword(flag, m.Type, iword(fn))
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
	return
}

// NumMethod returns the number of interface methods in the type's method set.
func (t *interfaceType) NumMethod() int { return len(t.methods) }

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
func (t *structType) Field(i int) (f StructField) {
	if i < 0 || i >= len(t.fields) {
		return
	}
	p := t.fields[i]
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
		f.Tag = *p.tag
	}
	f.Offset = p.offset
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
func toCommonType(p *runtime.Type) *commonType {
	if p == nil {
		return nil
	}
	type hdr struct {
		x interface{}
		t commonType
	}
	x := unsafe.Pointer(p)
	if uintptr(x)&reflectFlags != 0 {
		panic("invalid interface value")
	}
	return &(*hdr)(x).t
}

func toType(p *runtime.Type) Type {
	if p == nil {
		return nil
	}
	return toCommonType(p).toType()
}

// Typeof returns the reflection Type of the value in the interface{}.
func Typeof(i interface{}) Type {
	eface := *(*emptyInterface)(unsafe.Pointer(&i))
	return toType(eface.typ)
}

// ptrMap is the cache for PtrTo.
var ptrMap struct {
	sync.RWMutex
	m map[*commonType]*ptrType
}

func (t *commonType) runtimeType() *runtime.Type {
	// The runtime.Type always precedes the commonType in memory.
	// Adjust pointer to find it.
	var rt struct {
		i  runtime.Type
		ct commonType
	}
	return (*runtime.Type)(unsafe.Pointer(uintptr(unsafe.Pointer(t)) - uintptr(unsafe.Offsetof(rt.ct))))
}

// PtrTo returns the pointer type with element t.
// For example, if t represents type Foo, PtrTo(t) represents *Foo.
func PtrTo(t Type) Type {
	// If t records its pointer-to type, use it.
	ct := t.(*commonType)
	if p := ct.ptrToThis; p != nil {
		return toType(p)
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
			return p.commonType.toType()
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
		return p
	}

	var rt struct {
		i runtime.Type
		ptrType
	}
	rt.i = (*runtime.PtrType)(unsafe.Pointer(&rt.ptrType))

	// initialize p using *byte's PtrType as a prototype.
	// have to do assignment as PtrType, not runtime.PtrType,
	// in order to write to unexported fields.
	p = &rt.ptrType
	bp := (*ptrType)(unsafe.Pointer(unsafe.Typeof((*byte)(nil)).(*runtime.PtrType)))
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
	p.elem = (*runtime.Type)(unsafe.Pointer(uintptr(unsafe.Pointer(ct)) - uintptr(unsafe.Offsetof(rt.ptrType))))

	ptrMap.m[ct] = p
	ptrMap.Unlock()
	return p.commonType.toType()
}
