// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pkgbits

// A SectionKind indicates a section, as well as the ordering of sections within
// unified export data. Any object given a dedicated section can be referred to
// via a section / index pair (and thus dereferenced) in other sections.
type SectionKind int32 // TODO(markfreeman): Replace with uint8.

const (
	SectionString SectionKind = iota
	SectionMeta
	SectionPosBase
	SectionPkg
	SectionName
	SectionType
	SectionObj
	SectionObjExt
	SectionObjDict
	SectionBody

	numRelocs = iota
)

// An Index represents a bitstream element index *within* (i.e., relative to) a
// particular section.
type Index int32

// TODO(markfreeman): Make RelIndex its own named type once we point external
// references from Index to RelIndex.
type RelIndex = Index

// A RelocEnt, or relocation entry, is an entry in an element's reference
// table. All elements are preceded by a reference table which provides
// locations for all dereferences that the element may use.
type RelocEnt struct {
	Kind SectionKind
	Idx  RelIndex
}

// Reserved indices within the [SectionMeta] section.
const (
	PublicRootIdx  RelIndex = 0
	PrivateRootIdx RelIndex = 1
)
