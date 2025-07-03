// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "go/constant"

// A Type represents a type of Go.
// All types implement the Type interface.
// (This type originally lived in types2. We moved it here
// so we could depend on it from other packages without
// introducing an import cycle.)
type Type interface {
	// Underlying returns the underlying type of a type.
	// Underlying types are never Named, TypeParam, or Alias types.
	//
	// See https://go.dev/ref/spec#Underlying_types.
	Underlying() Type

	// String returns a string representation of a type.
	String() string
}

// Expressions in the syntax package provide storage for
// the typechecker to record its results. This interface
// is the mechanism the typechecker uses to record results,
// and clients use to retrieve those results.
type typeInfo interface {
	SetTypeInfo(TypeAndValue)
	GetTypeInfo() TypeAndValue
}

// A TypeAndValue records the type information, constant
// value if known, and various other flags associated with
// an expression.
// This type is similar to types2.TypeAndValue, but exposes
// none of types2's internals.
type TypeAndValue struct {
	Type  Type
	Value constant.Value
	exprFlags
}

type exprFlags uint16

func (f exprFlags) IsVoid() bool          { return f&1 != 0 }
func (f exprFlags) IsType() bool          { return f&2 != 0 }
func (f exprFlags) IsBuiltin() bool       { return f&4 != 0 } // a language builtin that resembles a function call, e.g., "make, append, new"
func (f exprFlags) IsValue() bool         { return f&8 != 0 }
func (f exprFlags) IsNil() bool           { return f&16 != 0 }
func (f exprFlags) Addressable() bool     { return f&32 != 0 }
func (f exprFlags) Assignable() bool      { return f&64 != 0 }
func (f exprFlags) HasOk() bool           { return f&128 != 0 }
func (f exprFlags) IsRuntimeHelper() bool { return f&256 != 0 } // a runtime function called from transformed syntax

func (f *exprFlags) SetIsVoid()          { *f |= 1 }
func (f *exprFlags) SetIsType()          { *f |= 2 }
func (f *exprFlags) SetIsBuiltin()       { *f |= 4 }
func (f *exprFlags) SetIsValue()         { *f |= 8 }
func (f *exprFlags) SetIsNil()           { *f |= 16 }
func (f *exprFlags) SetAddressable()     { *f |= 32 }
func (f *exprFlags) SetAssignable()      { *f |= 64 }
func (f *exprFlags) SetHasOk()           { *f |= 128 }
func (f *exprFlags) SetIsRuntimeHelper() { *f |= 256 }

// a typeAndValue contains the results of typechecking an expression.
// It is embedded in expression nodes.
type typeAndValue struct {
	tv TypeAndValue
}

func (x *typeAndValue) SetTypeInfo(tv TypeAndValue) {
	x.tv = tv
}
func (x *typeAndValue) GetTypeInfo() TypeAndValue {
	return x.tv
}
