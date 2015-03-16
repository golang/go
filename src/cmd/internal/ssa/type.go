// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/ssa/types" // TODO: use golang.org/x/tools/go/types instead
)

// We just inherit types from go/types
type Type types.Type

var (
	// shortcuts for commonly used basic types
	TypeInt     = types.Typ[types.Int]
	TypeUint    = types.Typ[types.Uint]
	TypeInt8    = types.Typ[types.Int8]
	TypeInt16   = types.Typ[types.Int16]
	TypeInt32   = types.Typ[types.Int32]
	TypeInt64   = types.Typ[types.Int64]
	TypeUint8   = types.Typ[types.Uint8]
	TypeUint16  = types.Typ[types.Uint16]
	TypeUint32  = types.Typ[types.Uint32]
	TypeUint64  = types.Typ[types.Uint64]
	TypeUintptr = types.Typ[types.Uintptr]
	TypeBool    = types.Typ[types.Bool]
	TypeString  = types.Typ[types.String]

	TypeInvalid = types.Typ[types.Invalid]

	// Additional compiler-only types go here.
	TypeMem   = &Memory{}
	TypeFlags = &Flags{}
)

// typeIdentical reports whether its two arguments are the same type.
func typeIdentical(t, u Type) bool {
	if t == TypeMem {
		return u == TypeMem
	}
	if t == TypeFlags {
		return u == TypeFlags
	}
	return types.Identical(t, u)
}

// A type representing all of memory
type Memory struct {
}

func (t *Memory) Underlying() types.Type { panic("Underlying of Memory") }
func (t *Memory) String() string         { return "mem" }

// A type representing the unknown type
type Unknown struct {
}

func (t *Unknown) Underlying() types.Type { panic("Underlying of Unknown") }
func (t *Unknown) String() string         { return "unk" }

// A type representing the void type.  Used during building, should always
// be eliminated by the first deadcode pass.
type Void struct {
}

func (t *Void) Underlying() types.Type { panic("Underlying of Void") }
func (t *Void) String() string         { return "void" }

// A type representing the results of a nil check or bounds check.
// TODO: or type check?
// TODO: just use bool?
type Check struct {
}

func (t *Check) Underlying() types.Type { panic("Underlying of Check") }
func (t *Check) String() string         { return "check" }

// x86 flags type
type Flags struct {
}

func (t *Flags) Underlying() types.Type { panic("Underlying of Flags") }
func (t *Flags) String() string         { return "flags" }
