// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The Unified IR (UIR) format is implicitly defined by the package noder.

At the highest level, a package encoded in UIR follows the grammar below.

File        = Header Payload fingerprint .
Header      = version [ flags ] sectionEnds elementEnds .

version     = uint32 .     // used for backward compatibility
flags       = uint32 .     // feature flags used across versions
sectionEnds = [10]uint32 . // defines section boundaries
elementEnds = []uint32 .   // defines element boundaries
fingerprint = [8]byte .    // sha256 fingerprint

The payload is a series of sections. Each section has a kind which determines
its index in the series.

SectionKind = Uint64 .
TODO(markfreeman): Update when we rename RelocFoo to SectionFoo.
Payload     = RelocString  // TODO(markfreeman) Define.
              RelocMeta
              RelocPosBase // TODO(markfreeman) Define.
              RelocPkg     // TODO(markfreeman) Define.
              RelocName    // TODO(markfreeman) Define.
              RelocType    // TODO(markfreeman) Define.
              RelocObj     // TODO(markfreeman) Define.
              RelocObjExt  // TODO(markfreeman) Define.
              RelocObjDict // TODO(markfreeman) Define.
              RelocBody    // TODO(markfreeman) Define.
              .

# Sections
A section is a series of elements of a type determined by the section's kind.
Go constructs are mapped onto (potentially multiple) elements. Elements are
accessed using an index relative to the start of the section.

// TODO(markfreeman): Rename to SectionIndex.
RelIndex = Uint64 .

## Meta Section
The meta section provides fundamental information for a package. It contains
exactly two elements — a public root and a private root.

RelocMeta  = PublicRoot
             PrivateRoot     // TODO(markfreeman): Define.
             .

The public root element identifies the package and provides references for all
exported objects it contains.

PublicRoot = Relocs
             [ SyncPublic ] // TODO(markfreeman): Define.
             PackageRef     // TODO(markfreeman): Define.
             [ HasInit ]
             ObjectRefCount // TODO(markfreeman): Define.
             { ObjectRef }  // TODO(markfreeman): Define.
             .
HasInit    = Bool .         // Whether the package uses any initialization
                            // functions.

# References
A reference table precedes every element. Each entry in the table contains a
section / index pair denoting the location of the referenced element.

// TODO(markfreeman): Rename to RefTable.
Relocs     = [ SyncRelocs ]   // TODO(markfreeman): Define.
             RelocCount
             { Reloc }
             .
// TODO(markfreeman): Rename to RefTableEntryCount.
RelocCount = Uint64 .
// TODO(markfreeman): Rename to RefTableEntry.
Reloc      = [ SyncReloc ]    // TODO(markfreeman): Define.
             SectionKind
             RelIndex
             .

Elements encode references to other elements as an index in the reference
table — not the location of the referenced element directly.

// TODO(markfreeman): Rename to RefUse.
UseReloc   = [ SyncUseReloc ] // TODO(markfreeman): Define.
             RelIndex
             .

# Primitives
Primitive encoding is handled separately by the pkgbits package. Check there
for definitions of the below productions.

    * Bool
    * Int64
    * Uint64
    * String
*/

package noder
