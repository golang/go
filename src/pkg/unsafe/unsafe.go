// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
	Package unsafe contains operations that step around the type safety of Go programs.
*/
package unsafe

// ArbitraryType is here for the purposes of documentation only and is not actually
// part of the unsafe package.  It represents the type of an arbitrary Go expression.
type ArbitraryType int

// Pointer represents a pointer to an arbitrary type.  There are three special operations
// available for type Pointer that are not available for other types.
//	1) A pointer value of any type can be converted to a Pointer.
//	2) A Pointer can be converted to a pointer value of any type.
//	3) A uintptr can be converted to a Pointer.
//	4) A Pointer can be converted to a uintptr.
// Pointer therefore allows a program to defeat the type system and read and write
// arbitrary memory. It should be used with extreme care.
type Pointer *ArbitraryType

// Sizeof returns the size in bytes occupied by the value v.  The size is that of the
// "top level" of the value only.  For instance, if v is a slice, it returns the size of
// the slice descriptor, not the size of the memory referenced by the slice.
func Sizeof(v ArbitraryType) uintptr

// Offsetof returns the offset within the struct of the field represented by v,
// which must be of the form structValue.field.  In other words, it returns the
// number of bytes between the start of the struct and the start of the field.
func Offsetof(v ArbitraryType) uintptr

// Alignof returns the alignment of the value v.  It is the maximum value m such
// that the address of a variable with the type of v will always always be zero mod m.
// If v is of the form structValue.field, it returns the alignment of field f within struct object obj.
func Alignof(v ArbitraryType) uintptr

// Typeof returns the type of an interface value, a runtime.Type.
func Typeof(i interface{}) (typ interface{})

// Reflect unpacks an interface value into its type and the address of a copy of the
// internal value.
func Reflect(i interface{}) (typ interface{}, addr Pointer)

// Unreflect inverts Reflect: Given a type and a pointer to a value, it returns an
// empty interface value with contents the type and the value (not the pointer to
// the value).  The typ is assumed to contain a pointer to a runtime type; the type
// information in the interface{} is ignored, so that, for example, both
// *reflect.structType and *runtime.StructType can be passed for typ.
func Unreflect(typ interface{}, addr Pointer) (ret interface{})

// New allocates and returns a pointer to memory for a new value of the given type.
// The typ is assumed to hold a pointer to a runtime type.
// Callers should use reflect.New or reflect.Zero instead of invoking unsafe.New directly.
func New(typ interface{}) Pointer

// NewArray allocates and returns a pointer to an array of n elements of the given type.
// The typ is assumed to hold a pointer to a runtime type.
// Callers should use reflect.MakeSlice instead of invoking unsafe.NewArray directly.
func NewArray(typ interface{}, n int) Pointer
