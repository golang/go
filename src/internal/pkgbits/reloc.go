// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

// A RelocKind indicates a particular section within a unified IR export.
type RelocKind int

// An Index represents a bitstream element index within a particular
// section.
type Index int

// A relocEnt (relocation entry) is an entry in an element's local
// reference table.
//
// TODO(mdempsky): Rename this too.
type RelocEnt struct {
	Kind RelocKind
	Idx  Index
}

// Reserved indices within the meta relocation section.
const (
	PublicRootIdx  Index = 0
	PrivateRootIdx Index = 1
)

const (
	RelocString RelocKind = iota
	RelocMeta
	RelocPosBase
	RelocPkg
	RelocName
	RelocType
	RelocObj
	RelocObjExt
	RelocObjDict
	RelocBody

	numRelocs = iota
)
