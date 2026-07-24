// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gcimporter

import (
	"go/types"
	"sync"
)

// predecl is a cache for the predeclared types in types.Universe.
//
// Cache a distinct result based on the runtime value of any.
// The pointer value of the any type varies based on GODEBUG settings.
var predeclMu sync.Mutex
var predecl map[types.Type][]types.Type

func predeclared() []types.Type {
	anyt := types.Universe.Lookup("any").Type()

	predeclMu.Lock()
	defer predeclMu.Unlock()

	if pre, ok := predecl[anyt]; ok {
		return pre
	}

	if predecl == nil {
		predecl = make(map[types.Type][]types.Type)
	}

	decls := []types.Type{ // basic types
		types.Typ[types.Bool],
		types.Typ[types.Int],
		types.Typ[types.Int8],
		types.Typ[types.Int16],
		types.Typ[types.Int32],
		types.Typ[types.Int64],
		types.Typ[types.Uint],
		types.Typ[types.Uint8],
		types.Typ[types.Uint16],
		types.Typ[types.Uint32],
		types.Typ[types.Uint64],
		types.Typ[types.Uintptr],
		types.Typ[types.Float32],
		types.Typ[types.Float64],
		types.Typ[types.Complex64],
		types.Typ[types.Complex128],
		types.Typ[types.String],

		// basic type aliases
		types.Universe.Lookup("byte").Type(),
		types.Universe.Lookup("rune").Type(),

		// error
		types.Universe.Lookup("error").Type(),

		// untyped types
		types.Typ[types.UntypedBool],
		types.Typ[types.UntypedInt],
		types.Typ[types.UntypedRune],
		types.Typ[types.UntypedFloat],
		types.Typ[types.UntypedComplex],
		types.Typ[types.UntypedString],
		types.Typ[types.UntypedNil],

		// package unsafe
		types.Typ[types.UnsafePointer],

		// invalid type
		types.Typ[types.Invalid], // only appears in packages with errors

		// used internally by gc; never used by this package or in .a files
		anyType{},

		// comparable
		types.Universe.Lookup("comparable").Type(),

		// any
		anyt,
	}

	predecl[anyt] = decls
	return decls
}

type anyType struct{}

func (t anyType) Underlying() types.Type { return t }
func (t anyType) String() string         { return "any" }
