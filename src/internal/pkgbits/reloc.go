// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

// A RelocKind indicates a section, as well as the ordering of sections within
// unified export data. Any object given a dedicated section can be referred to
// via a section / index pair (and thus dereferenced) in other sections.
type RelocKind int32 // TODO(markfreeman): Replace with uint8.

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

// An Index represents a bitstream element index *within* (i.e., relative to) a
// particular section.
type Index int32

// A RelocEnt, or relocation entry, is an entry in an element's reference
// table. All elements are preceded by a reference table which provides
// locations for all dereferences that the element may use.
type RelocEnt struct {
	Kind RelocKind
	Idx  Index
}

// Reserved indices within the [RelocMeta] section.
const (
	PublicRootIdx  Index = 0
	PrivateRootIdx Index = 1
)
