// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

// Writing and reading of Go object files.
//
// Originally, Go object files were Plan 9 object files, but no longer.
// Now they are more like standard object files, in that each symbol is defined
// by an associated memory image (bytes) and a list of relocations to apply
// during linking. We do not (yet?) use a standard file format, however.
// For now, the format is chosen to be as simple as possible to read and write.
// It may change for reasons of efficiency, or we may even switch to a
// standard file format if there are compelling benefits to doing so.
// See golang.org/s/go13linker for more background.
//
// The file format is:
//
//	- magic header: "\x00\x00go13ld"
//	- byte 1 - version number
//	- sequence of strings giving dependencies (imported packages)
//	- empty string (marks end of sequence)
//	- sequence of defined symbols
//	- byte 0xff (marks end of sequence)
//	- magic footer: "\xff\xffgo13ld"
//
// All integers are stored in a zigzag varint format.
// See golang.org/s/go12symtab for a definition.
//
// Data blocks and strings are both stored as an integer
// followed by that many bytes.
//
// A symbol reference is a string name followed by a version.
// An empty name corresponds to a nil LSym* pointer.
//
// Each symbol is laid out as the following fields (taken from LSym*):
//
//	- byte 0xfe (sanity check for synchronization)
//	- type [int]
//	- name [string]
//	- version [int]
//	- flags [int]
//		1 dupok
//	- size [int]
//	- gotype [symbol reference]
//	- p [data block]
//	- nr [int]
//	- r [nr relocations, sorted by off]
//
// If type == STEXT, there are a few more fields:
//
//	- args [int]
//	- locals [int]
//	- nosplit [int]
//	- flags [int]
//		1 leaf
//		2 C function
//	- nlocal [int]
//	- local [nlocal automatics]
//	- pcln [pcln table]
//
// Each relocation has the encoding:
//
//	- off [int]
//	- siz [int]
//	- type [int]
//	- add [int]
//	- xadd [int]
//	- sym [symbol reference]
//	- xsym [symbol reference]
//
// Each local has the encoding:
//
//	- asym [symbol reference]
//	- offset [int]
//	- type [int]
//	- gotype [symbol reference]
//
// The pcln table has the encoding:
//
//	- pcsp [data block]
//	- pcfile [data block]
//	- pcline [data block]
//	- npcdata [int]
//	- pcdata [npcdata data blocks]
//	- nfuncdata [int]
//	- funcdata [nfuncdata symbol references]
//	- funcdatasym [nfuncdata ints]
//	- nfile [int]
//	- file [nfile symbol references]
//
// The file layout and meaning of type integers are architecture-independent.
//
// TODO(rsc): The file format is good for a first pass but needs work.
//	- There are SymID in the object file that should really just be strings.
//	- The actual symbol memory images are interlaced with the symbol
//	  metadata. They should be separated, to reduce the I/O required to
//	  load just the metadata.
//	- The symbol references should be shortened, either with a symbol
//	  table or by using a simple backward index to an earlier mentioned symbol.

// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file defines flags attached to various functions
// and data objects.  The compilers, assemblers, and linker must
// all agree on these values.

// Don't profile the marked routine.  This flag is deprecated.

// It is ok for the linker to get multiple of these symbols.  It will
// pick one of the duplicates to use.

// Don't insert stack check preamble.

// Put this data in a read-only section.

// This data contains no pointers.

// This is a wrapper function and should not count as disabling 'recover'.

// This function uses its incoming context register.
const (
	NOPROF   = 1
	DUPOK    = 2
	NOSPLIT  = 4
	RODATA   = 8
	NOPTR    = 16
	WRAPPER  = 32
	NEEDCTXT = 64
)
