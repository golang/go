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

// An AbsElemIdx, or absolute element index, is an index into the elements
// that is not relative to some other index.
type AbsElemIdx = uint32

// TODO(markfreeman): Make this its own type.
// A RelElemIdx, or relative element index, is an index into the elements
// relative to some other index, such as the start of a section.
type RelElemIdx = Index

// TODO(markfreeman): Isn't this strictly less efficient than an AbsElemIdx?
// A RefTableEntry is an entry in an element's reference table. All
// elements are preceded by a reference table which provides locations
// for referenced elements.
type RefTableEntry struct {
	Kind SectionKind
	Idx  RelElemIdx
}

// Reserved indices within the [SectionMeta] section.
const (
	PublicRootIdx  RelElemIdx = 0
	PrivateRootIdx RelElemIdx = 1
)
