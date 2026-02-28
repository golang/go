// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import "cmd/internal/dwarf"

// LoaderSym holds a loader.Sym value. We can't refer to this
// type from the sym package since loader imports sym.
type LoaderSym int

// A CompilationUnit represents a set of source files that are compiled
// together. Since all Go sources in a Go package are compiled together,
// there's one CompilationUnit per package that represents all Go sources in
// that package, plus one for each assembly file.
//
// Equivalently, there's one CompilationUnit per object file in each Library
// loaded by the linker.
//
// These are used for both DWARF and pclntab generation.
type CompilationUnit struct {
	Lib       *Library      // Our library
	PclnIndex int           // Index of this CU in pclntab
	PCs       []dwarf.Range // PC ranges, relative to Textp[0]
	DWInfo    *dwarf.DWDie  // CU root DIE
	FileTable []string      // The file table used in this compilation unit.

	Consts    LoaderSym   // Package constants DIEs
	FuncDIEs  []LoaderSym // Function DIE subtrees
	VarDIEs   []LoaderSym // Global variable DIEs
	AbsFnDIEs []LoaderSym // Abstract function DIE subtrees
	RangeSyms []LoaderSym // Symbols for debug_range
	Textp     []LoaderSym // Text symbols in this CU
}
