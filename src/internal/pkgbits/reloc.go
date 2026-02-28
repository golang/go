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

/*
All elements are preceded by a reference table. Reference tables provide an
additional indirection layer for element references. That is, for element A to
reference element B, A encodes the reference table index pointing to B, rather
than the table entry itself.

# Functional Considerations
Reference table layout is important primarily to the UIR linker. After noding,
the UIR linker sees a UIR file for each package with imported objects
represented as stubs. In a simple sense, the job of the UIR linker is to merge
these "stubbed" UIR files into a single "linked" UIR file for the target package
with stubs replaced by object definitions.

To do this, the UIR linker walks each stubbed UIR file and pulls in elements in
dependency order; that is, if A references B, then B must be placed into the
linked UIR file first. This depth-first traversal is done by recursing through
each element's reference table.

When placing A in the linked UIR file, the reference table entry for B must be
updated, since B is unlikely to be at the same relative element index as it was
in the stubbed UIR file.

Without reference tables, the UIR linker would need to read in the element to
discover its references. Note that the UIR linker cannot jump directly to the
reference locations after discovering merely the type of the element;
variable-width primitives prevent this.

After updating the reference table, the rest of the element may be copied
directly into the linked UIR file. Note that the UIR linker may decide to read
in the element anyway (for unrelated reasons).

In short, reference tables provide an efficient mechanism for traversing,
discovering, and updating element references during UIR linking.

# Storage Considerations
Reference tables also have compactness benefits:
  - If A refers to B multiple times, the entry is deduplicated and referred to
    more compactly by the index.
  - Relative (to a section) element indices are typically smaller than absolute
    element indices, and thus fit into smaller varints.
  - Most elements do not reference many elements; thus table size indicators and
    table indices are typically a byte each.

Thus, the storage performance is as follows:
+-----------------------------+-----------+--------------+
|          Scenario           | Best Case | Typical Case |
+-----------------------------+-----------+--------------+
| First reference from A to B | 3 Bytes   | 4 Bytes      |
| Other reference from A to B | 1 Byte    | 1 Byte       |
+-----------------------------+-----------+--------------+

The typical case for the first scenario changes because many sections have more
than 127 (range of a 1-byte uvarint) elements and thus the relative index is
typically 2 bytes, though this depends on the distribution of referenced indices
within the section.

The second does not because most elements do not reference more than 127
elements and the table index can thus keep to 1 byte.

Typically, A will only reference B once, so most references are 4 bytes.
*/

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
