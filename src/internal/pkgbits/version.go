// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

// Version indicates a version of a unified IR bitstream.
// Each Version indicates the addition, removal, or change of
// new data in the bitstream.
//
// These are serialized to disk and the interpretation remains fixed.
type Version uint32

const (
	// V0: initial prototype.
	//
	// All data that is not assigned a Field is in version V0
	// and has not been deprecated.
	V0 Version = iota

	// V1: adds the Flags uint32 word
	V1

	// V2: removes unused legacy fields and supports type parameters for aliases.
	// - remove the legacy "has init" bool from the public root
	// - remove obj's "derived func instance" bool
	// - add a TypeParamNames field to ObjAlias
	// - remove derived info "needed" bool
	V2

	numVersions = iota
)

// Field denotes a unit of data in the serialized unified IR bitstream.
// It is conceptually a like field in a structure.
//
// We only really need Fields when the data may or may not be present
// in a stream based on the Version of the bitstream.
//
// Unlike much of pkgbits, Fields are not serialized and
// can change values as needed.
type Field int

const (
	// Flags in a uint32 in the header of a bitstream
	// that is used to indicate whether optional features are enabled.
	Flags Field = iota

	// Deprecated: HasInit was a bool indicating whether a package
	// has any init functions.
	HasInit

	// Deprecated: DerivedFuncInstance was a bool indicating
	// whether an object was a function instance.
	DerivedFuncInstance

	// ObjAlias has a list of TypeParamNames.
	AliasTypeParamNames

	// Deprecated: DerivedInfoNeeded was a bool indicating
	// whether a type was a derived type.
	DerivedInfoNeeded

	numFields = iota
)

// introduced is the version a field was added.
var introduced = [numFields]Version{
	Flags:               V1,
	AliasTypeParamNames: V2,
}

// removed is the version a field was removed in or 0 for fields
// that have not yet been deprecated.
// (So removed[f]-1 is the last version it is included in.)
var removed = [numFields]Version{
	HasInit:             V2,
	DerivedFuncInstance: V2,
	DerivedInfoNeeded:   V2,
}

// Has reports whether field f is present in a bitstream at version v.
func (v Version) Has(f Field) bool {
	return introduced[f] <= v && (v < removed[f] || removed[f] == V0)
}
