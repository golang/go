// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
The Unified IR (UIR) format is implicitly defined by the package noder.

At the highest level, a package encoded in UIR follows the grammar
below.

    File        = Header Payload fingerprint .
    Header      = version [ flags ] sectionEnds elementEnds .

    version     = uint32 .     // used for backward compatibility
    flags       = uint32 .     // feature flags used across versions
    sectionEnds = [10]uint32 . // defines section boundaries
    elementEnds = []uint32 .   // defines element boundaries
    fingerprint = [8]byte .    // sha256 fingerprint

The payload is a series of sections. Each section has a kind which
determines its index in the series.

    SectionKind = Uint64 .
    Payload     = SectionString
                  SectionMeta
                  SectionPosBase
                  SectionPkg
                  SectionName    // TODO(markfreeman) Define.
                  SectionType    // TODO(markfreeman) Define.
                  SectionObj     // TODO(markfreeman) Define.
                  SectionObjExt  // TODO(markfreeman) Define.
                  SectionObjDict // TODO(markfreeman) Define.
                  SectionBody    // TODO(markfreeman) Define.
                  .

# Sections
A section is a series of elements of a type determined by the section's
kind. Go constructs are mapped onto (potentially multiple) elements.
Elements are accessed using an index relative to the start of the
section.

    // TODO(markfreeman): Rename to SectionIndex.
    RelIndex = Uint64 .

## String Section
String values are stored as elements in the string section. Elements
outside the string section access string values by reference.

    SectionString = { String } .

## Meta Section
The meta section provides fundamental information for a package. It
contains exactly two elements — a public root and a private root.

    SectionMeta = PublicRoot
                  PrivateRoot     // TODO(markfreeman): Define.
                  .

The public root element identifies the package and provides references
for all exported objects it contains.

    PublicRoot  = Relocs
                  [ Sync ]
                  PkgRef
                  [ HasInit ]
                  ObjectRefCount // TODO(markfreeman): Define.
                  { ObjectRef }  // TODO(markfreeman): Define.
                  .
    HasInit     = Bool .         // Whether the package uses any
                                 // initialization functions.

## PosBase Section
This section provides position information. It is a series of PosBase
elements.

    SectionPosBase = { PosBase } .

A base is either a file base or line base (produced by a line
directive). Every base has a position, line, and column; these are
constant for file bases and hence not encoded.

    PosBase = Relocs
              [ Sync ]
              StringRef       // the (absolute) file name for the base
              Bool            // true if a file base, else a line base
              // The below is ommitted for file bases.
              [ Pos
                Uint64        // line
                Uint64 ]      // column
              .

A source position Pos represents a file-absolute (line, column) pair
and a PosBase indicating the position Pos is relative to. Positions
without a PosBase have no line or column.

    Pos     = [ Sync ]
              Bool             // true if the position has a base
              // The below is ommitted if the position has no base.
              [ Ref[PosBase]
                Uint64         // line
                Uint64 ]       // column
              .

## Package Section
The package section holds package information. It is a series of Pkg
elements.

    SectionPkg = { Pkg } .

A Pkg element contains a (path, name) pair and a series of imported
packages. The below package paths have special meaning.

    +--------------+-----------------------------------+
    | package path |             indicates             |
    +--------------+-----------------------------------+
    | ""           | the current package               |
    | "builtin"    | the fake builtin package          |
    | "unsafe"     | the compiler-known unsafe package |
    +--------------+-----------------------------------+

    Pkg        = Relocs
                 [ Sync ]
                 StringRef      // path
                 // The below is ommitted for the special package paths
                 // "builtin" and "unsafe".
                 [ StringRef    // name
                   Imports ]
                 .
    Imports    = Uint64         // the number of declared imports
                 { PkgRef }     // references to declared imports
                 .

Note, a PkgRef is *not* equivalent to Ref[Pkg] due to an extra marker.

    PkgRef     = [ Sync ]
                 Ref[Pkg]
                 .

# References
A reference table precedes every element. Each entry in the table
contains a section / index pair denoting the location of the referenced
element.

    // TODO(markfreeman): Rename to RefTable.
    Relocs   = [ Sync ]
               RelocCount
               { Reloc }
               .
    // TODO(markfreeman): Rename to RefTableEntryCount.
    RelocCount = Uint64 .
    // TODO(markfreeman): Rename to RefTableEntry.
    Reloc    = [ Sync ]
               SectionKind
               RelIndex
               .

Elements encode references to other elements as an index in the
reference table — not the location of the referenced element directly.

    // TODO(markfreeman): Rename to RefUse.
    UseReloc = [ Sync ]
               RelIndex
               .

# Primitives
Primitive encoding is handled separately by the pkgbits package. Check
there for definitions of the below productions.

    * Bool
    * Int64
    * Uint64
    * String
    * Sync
*/

package noder
