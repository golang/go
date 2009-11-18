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
	"runtime";
	"strconv";
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

type commonType struct {
	size		uintptr;
	hash		uint32;
	alg		uint8;
	align		uint8;
	fieldAlign	uint8;
	string		*string;
	*uncommonType;
}

type method struct {
	hash	uint32;
	name	*string;
	pkgPath	*string;
	typ	*runtime.Type;
	ifn	unsafe.Pointer;
	tfn	unsafe.Pointer;
}

type uncommonType struct {
	name	*string;
	pkgPath	*string;
	methods	[]method;
}

// BoolType represents a boolean type.
type BoolType struct {
	commonType;
}

// Float32Type represents a float32 type.
type Float32Type struct {
	commonType;
}

// Float64Type represents a float64 type.
type Float64Type struct {
	commonType;
}

// FloatType represents a float type.
type FloatType struct {
	commonType;
}

// Int16Type represents an int16 type.
type Int16Type struct {
	commonType;
}

// Int32Type represents an int32 type.
type Int32Type struct {
	commonType;
}

// Int64Type represents an int64 type.
type Int64Type struct {
	commonType;
}

// Int8Type represents an int8 type.
type Int8Type struct {
	commonType;
}

// IntType represents an int type.
type IntType struct {
	commonType;
}

// Uint16Type represents a uint16 type.
type Uint16Type struct {
	commonType;
}

// Uint32Type represents a uint32 type.
type Uint32Type struct {
	commonType;
}

// Uint64Type represents a uint64 type.
type Uint64Type struct {
	commonType;
}

// Uint8Type represents a uint8 type.
type Uint8Type struct {
	commonType;
}

// UintType represents a uint type.
type UintType struct {
	commonType;
}

// StringType represents a string type.
type StringType struct {
	commonType;
}

// UintptrType represents a uintptr type.
type UintptrType struct {
	commonType;
}

// DotDotDotType represents the ... that can
// be used as the type of the final function parameter.
type DotDotDotType struct {
	commonType;
}

// UnsafePointerType represents an unsafe.Pointer type.
type UnsafePointerType struct {
	commonType;
}

// ArrayType represents a fixed array type.
type ArrayType struct {
	commonType;
	elem	*runtime.Type;
	len	uintptr;
}

// ChanDir represents a channel type's direction.
type ChanDir int

const (
	RecvDir	ChanDir	= 1 << iota;
	SendDir;
	BothDir	= RecvDir | SendDir;
)

// ChanType represents a channel type.
type ChanType struct {
	commonType;
	elem	*runtime.Type;
	dir	uintptr;
}

// FuncType represents a function type.
type FuncType struct {
	commonType;
	in	[]*runtime.Type;
	out	[]*runtime.Type;
}

// Method on interface type
type imethod struct {
	hash	uint32;
	perm	uint32;
	name	*string;
	pkgPath	*string;
	typ	*runtime.Type;
}

// InterfaceType represents an interface type.
type InterfaceType struct {
	commonType;
	methods	[]imethod;
}

// MapType represents a map type.
type MapType struct {
	commonType;
	key	*runtime.Type;
	elem	*runtime.Type;
}

// PtrType represents a pointer type.
type PtrType struct {
	commonType;
	elem	*runtime.Type;
}

// SliceType represents a slice type.
type SliceType struct {
	commonType;
	elem	*runtime.Type;
}

// Struct field
type structField struct {
	name	*string;
	pkgPath	*string;
	typ	*runtime.Type;
	tag	*string;
	offset	uintptr;
}

// StructType represents a struct type.
type StructType struct {
	commonType;
	fields	[]structField;
}


/*
 * The compiler knows the exact layout of all the data structures above.
 * The compiler does not know about the data structures and methods below.
 */

// Method represents a single method.
type Method struct {
	PkgPath	string;	// empty for uppercase Name
	Name	string;
	Type	*FuncType;
	Func	*FuncValue;
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
	PkgPath() string;

	// Name returns the type's name within its package.
	// Name returns an empty string for unnamed types.
	Name() string;

	// String returns a string representation of the type.
	// The string representation may use shortened package names
	// (e.g., vector instead of "container/vector") and is not
	// guaranteed to be unique among types.  To test for equality,
	// compare the Types directly.
	String() string;

	// Size returns the number of bytes needed to store
	// a value of the given type; it is analogous to unsafe.Sizeof.
	Size() uintptr;

	// Align returns the alignment of a value of this type
	// when allocated in memory.
	Align() int;

	// FieldAlign returns the alignment of a value of this type
	// when used as a field in a struct.
	FieldAlign() int;

	// For non-interface types, Method returns the i'th method with receiver T.
	// For interface types, Method returns the i'th method in the interface.
	// NumMethod returns the number of such methods.
	Method(int) Method;
	NumMethod() int;
	uncommon() *uncommonType;
}

func (t *uncommonType) uncommon() *uncommonType {
	return t
}

func (t *uncommonType) PkgPath() string {
	if t == nil || t.pkgPath == nil {
		return ""
	}
	return *t.pkgPath;
}

func (t *uncommonType) Name() string {
	if t == nil || t.name == nil {
		return ""
	}
	return *t.name;
}

func (t *commonType) String() string	{ return *t.string }

func (t *commonType) Size() uintptr	{ return t.size }

func (t *commonType) Align() int	{ return int(t.align) }

func (t *commonType) FieldAlign() int	{ return int(t.fieldAlign) }

func (t *uncommonType) Method(i int) (m Method) {
	if t == nil || i < 0 || i >= len(t.methods) {
		return
	}
	p := &t.methods[i];
	if p.name != nil {
		m.Name = *p.name
	}
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
	}
	m.Type = toType(*p.typ).(*FuncType);
	fn := p.tfn;
	m.Func = newFuncValue(m.Type, addr(&fn), true);
	return;
}

func (t *uncommonType) NumMethod() int {
	if t == nil {
		return 0
	}
	return len(t.methods);
}

// TODO(rsc): 6g supplies these, but they are not
// as efficient as they could be: they have commonType
// as the receiver instead of *commonType.
func (t *commonType) NumMethod() int	{ return t.uncommonType.NumMethod() }

func (t *commonType) Method(i int) (m Method)	{ return t.uncommonType.Method(i) }

func (t *commonType) PkgPath() string	{ return t.uncommonType.PkgPath() }

func (t *commonType) Name() string	{ return t.uncommonType.Name() }

// Len returns the number of elements in the array.
func (t *ArrayType) Len() int	{ return int(t.len) }

// Elem returns the type of the array's elements.
func (t *ArrayType) Elem() Type	{ return toType(*t.elem) }

// Dir returns the channel direction.
func (t *ChanType) Dir() ChanDir	{ return ChanDir(t.dir) }

// Elem returns the channel's element type.
func (t *ChanType) Elem() Type	{ return toType(*t.elem) }

func (d ChanDir) String() string {
	switch d {
	case SendDir:
		return "chan<-"
	case RecvDir:
		return "<-chan"
	case BothDir:
		return "chan"
	}
	return "ChanDir" + strconv.Itoa(int(d));
}

// In returns the type of the i'th function input parameter.
func (t *FuncType) In(i int) Type {
	if i < 0 || i >= len(t.in) {
		return nil
	}
	return toType(*t.in[i]);
}

// NumIn returns the number of input parameters.
func (t *FuncType) NumIn() int	{ return len(t.in) }

// Out returns the type of the i'th function output parameter.
func (t *FuncType) Out(i int) Type {
	if i < 0 || i >= len(t.out) {
		return nil
	}
	return toType(*t.out[i]);
}

// NumOut returns the number of function output parameters.
func (t *FuncType) NumOut() int	{ return len(t.out) }

// Method returns the i'th interface method.
func (t *InterfaceType) Method(i int) (m Method) {
	if i < 0 || i >= len(t.methods) {
		return
	}
	p := &t.methods[i];
	m.Name = *p.name;
	if p.pkgPath != nil {
		m.PkgPath = *p.pkgPath
	}
	m.Type = toType(*p.typ).(*FuncType);
	return;
}

// NumMethod returns the number of interface methods.
func (t *InterfaceType) NumMethod() int	{ return len(t.methods) }

// Key returns the map key type.
func (t *MapType) Key() Type	{ return toType(*t.key) }

// Elem returns the map element type.
func (t *MapType) Elem() Type	{ return toType(*t.elem) }

// Elem returns the pointer element type.
func (t *PtrType) Elem() Type	{ return toType(*t.elem) }

// Elem returns the type of the slice's elements.
func (t *SliceType) Elem() Type	{ return toType(*t.elem) }

type StructField struct {
	PkgPath		string;	// empty for uppercase Name
	Name		string;
	Type		Type;
	Tag		string;
	Offset		uintptr;
	Index		[]int;
	Anonymous	bool;
}

// Field returns the i'th struct field.
func (t *StructType) Field(i int) (f StructField) {
	if i < 0 || i >= len(t.fields) {
		return
	}
	p := t.fields[i];
	f.Type = toType(*p.typ);
	if p.name != nil {
		f.Name = *p.name
	} else {
		t := f.Type;
		if pt, ok := t.(*PtrType); ok {
			t = pt.Elem()
		}
		f.Name = t.Name();
		f.Anonymous = true;
	}
	if p.pkgPath != nil {
		f.PkgPath = *p.pkgPath
	}
	if p.tag != nil {
		f.Tag = *p.tag
	}
	f.Offset = p.offset;
	f.Index = []int{i};
	return;
}

// TODO(gri): Should there be an error/bool indicator if the index
//            is wrong for FieldByIndex?

// FieldByIndex returns the nested field corresponding to index.
func (t *StructType) FieldByIndex(index []int) (f StructField) {
	for i, x := range index {
		if i > 0 {
			ft := f.Type;
			if pt, ok := ft.(*PtrType); ok {
				ft = pt.Elem()
			}
			if st, ok := ft.(*StructType); ok {
				t = st
			} else {
				var f0 StructField;
				f = f0;
				return;
			}
		}
		f = t.Field(x);
	}
	return;
}

const inf = 1 << 30	// infinity - no struct has that many nesting levels

func (t *StructType) fieldByName(name string, mark map[*StructType]bool, depth int) (ff StructField, fd int) {
	fd = inf;	// field depth

	if _, marked := mark[t]; marked {
		// Struct already seen.
		return
	}
	mark[t] = true;

	var fi int;	// field index
	n := 0;		// number of matching fields at depth fd
L:	for i, _ := range t.fields {
		f := t.Field(i);
		d := inf;
		switch {
		case f.Name == name:
			// Matching top-level field.
			d = depth
		case f.Anonymous:
			ft := f.Type;
			if pt, ok := ft.(*PtrType); ok {
				ft = pt.Elem()
			}
			switch {
			case ft.Name() == name:
				// Matching anonymous top-level field.
				d = depth
			case fd > depth:
				// No top-level field yet; look inside nested structs.
				if st, ok := ft.(*StructType); ok {
					f, d = st.fieldByName(name, mark, depth+1)
				}
			}
		}

		switch {
		case d < fd:
			// Found field at shallower depth.
			ff, fi, fd = f, i, d;
			n = 1;
		case d == fd:
			// More than one matching field at the same depth (or d, fd == inf).
			// Same as no field found at this depth.
			n++;
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
		ff.Index[depth] = fi;
	} else {
		// None or more than one matching field found.
		fd = inf
	}

	mark[t] = false, false;
	return;
}

// FieldByName returns the struct field with the given name
// and a boolean to indicate if the field was found.
func (t *StructType) FieldByName(name string) (f StructField, present bool) {
	if ff, fd := t.fieldByName(name, make(map[*StructType]bool), 0); fd < inf {
		ff.Index = ff.Index[0 : fd+1];
		f, present = ff, true;
	}
	return;
}

// NumField returns the number of struct fields.
func (t *StructType) NumField() int	{ return len(t.fields) }

// Convert runtime type to reflect type.
// Same memory layouts, different method sets.
func toType(i interface{}) Type {
	switch v := i.(type) {
	case *runtime.BoolType:
		return (*BoolType)(unsafe.Pointer(v))
	case *runtime.DotDotDotType:
		return (*DotDotDotType)(unsafe.Pointer(v))
	case *runtime.FloatType:
		return (*FloatType)(unsafe.Pointer(v))
	case *runtime.Float32Type:
		return (*Float32Type)(unsafe.Pointer(v))
	case *runtime.Float64Type:
		return (*Float64Type)(unsafe.Pointer(v))
	case *runtime.IntType:
		return (*IntType)(unsafe.Pointer(v))
	case *runtime.Int8Type:
		return (*Int8Type)(unsafe.Pointer(v))
	case *runtime.Int16Type:
		return (*Int16Type)(unsafe.Pointer(v))
	case *runtime.Int32Type:
		return (*Int32Type)(unsafe.Pointer(v))
	case *runtime.Int64Type:
		return (*Int64Type)(unsafe.Pointer(v))
	case *runtime.StringType:
		return (*StringType)(unsafe.Pointer(v))
	case *runtime.UintType:
		return (*UintType)(unsafe.Pointer(v))
	case *runtime.Uint8Type:
		return (*Uint8Type)(unsafe.Pointer(v))
	case *runtime.Uint16Type:
		return (*Uint16Type)(unsafe.Pointer(v))
	case *runtime.Uint32Type:
		return (*Uint32Type)(unsafe.Pointer(v))
	case *runtime.Uint64Type:
		return (*Uint64Type)(unsafe.Pointer(v))
	case *runtime.UintptrType:
		return (*UintptrType)(unsafe.Pointer(v))
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
	panicln("toType", i);
}

// ArrayOrSliceType is the common interface implemented
// by both ArrayType and SliceType.
type ArrayOrSliceType interface {
	Type;
	Elem() Type;
}

// Typeof returns the reflection Type of the value in the interface{}.
func Typeof(i interface{}) Type	{ return toType(unsafe.Typeof(i)) }
