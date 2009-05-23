// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Doc comments

package runtime

import "unsafe"

// The Type interface stands for any of the run-time type structures
// in this package.
type Type interface { }

// All types begin with a few common fields needed for
// the interface runtime.
type CommonType struct {
	Size uintptr;
	Hash uint32;
	Alg uint8;
	Align uint8;
	FieldAlign uint8;
}

// Basic types; should these be one struct with an enum kind?
// The benefit of splitting them up into many types is that
// one can use a single type switch instead of needing an
// enum switch inside a type switch.
type BoolType CommonType
type Float32Type CommonType
type Float64Type CommonType
type FloatType CommonType
type Int16Type CommonType
type Int32Type CommonType
type Int64Type CommonType
type Int8Type CommonType
type IntType CommonType
type Uint16Type CommonType
type Uint32Type CommonType
type Uint64Type CommonType
type Uint8Type CommonType
type UintType CommonType
type StringType CommonType
type UintptrType CommonType
type UnsafePointerType CommonType

type ArrayType struct {
	CommonType;
	Elem *Type;
	Bound int32;	// -1 means slice
}

type ChanDir int
const (
	SendDir ChanDir = 1<<iota;
	RecvDir;
	BothDir = SendDir | RecvDir;
)

type ChanType struct {
	CommonType;
	Elem *Type;
	Dir ChanDir;
}

type FuncType struct {
	CommonType;
	In []*Type;
	Out []*Type;
}

type IMethod struct {
	Name *string;
	Package *string;
	Type *Type;
}

type InterfaceType struct {
	CommonType;
	Methods []*IMethod;
}

type MapType struct {
	CommonType;
	Key *Type;
	Elem *Type;
}

type Method struct {
	Name *string;
	Package *string;
	Type *Type;
	Func unsafe.Pointer;
}

type NamedType struct {
	CommonType;
	Name *string;
	Package *string;
	Type *Type;
	ValueMethods []*Method;
	PtrMethods []*Method;
}

type PtrType struct {
	CommonType;
	Sub *Type;
}

type StructField struct {
	Name *string;
	Type *Type;
	Tag *string;
	Offset uintptr;
}

type StructType struct {
	CommonType;
	Fields []*StructField;
}

