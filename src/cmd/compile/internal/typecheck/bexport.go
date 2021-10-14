// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package typecheck

import "cmd/compile/internal/types"

// ----------------------------------------------------------------------------
// Export format

// Tags. Must be < 0.
const (
	// Objects
	packageTag = -(iota + 1)
	constTag
	typeTag
	varTag
	funcTag
	endTag

	// Types
	namedTag
	arrayTag
	sliceTag
	dddTag
	structTag
	pointerTag
	signatureTag
	interfaceTag
	mapTag
	chanTag

	// Values
	falseTag
	trueTag
	int64Tag
	floatTag
	fractionTag // not used by gc
	complexTag
	stringTag
	nilTag
	unknownTag // not used by gc (only appears in packages with errors)

	// Type aliases
	aliasTag
)

var predecl []*types.Type // initialized lazily

func predeclared() []*types.Type {
	if predecl == nil {
		// initialize lazily to be sure that all
		// elements have been initialized before
		predecl = []*types.Type{
			// basic types
			types.Types[types.TBOOL],
			types.Types[types.TINT],
			types.Types[types.TINT8],
			types.Types[types.TINT16],
			types.Types[types.TINT32],
			types.Types[types.TINT64],
			types.Types[types.TUINT],
			types.Types[types.TUINT8],
			types.Types[types.TUINT16],
			types.Types[types.TUINT32],
			types.Types[types.TUINT64],
			types.Types[types.TUINTPTR],
			types.Types[types.TFLOAT32],
			types.Types[types.TFLOAT64],
			types.Types[types.TCOMPLEX64],
			types.Types[types.TCOMPLEX128],
			types.Types[types.TSTRING],

			// basic type aliases
			types.ByteType,
			types.RuneType,

			// error
			types.ErrorType,

			// untyped types
			types.UntypedBool,
			types.UntypedInt,
			types.UntypedRune,
			types.UntypedFloat,
			types.UntypedComplex,
			types.UntypedString,
			types.Types[types.TNIL],

			// package unsafe
			types.Types[types.TUNSAFEPTR],

			// invalid type (package contains errors)
			types.Types[types.Txxx],

			// any type, for builtin export data
			types.Types[types.TANY],

			// comparable
			types.ComparableType,

			// any
			types.AnyType,
		}
	}
	return predecl
}
