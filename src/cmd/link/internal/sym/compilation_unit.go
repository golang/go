// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import "cmd/internal/dwarf"

// LoaderSym holds a loader.Sym value. We can't refer to this
// type from the sym package since loader imports sym.
type LoaderSym int

// CompilationUnit is an abstraction used by DWARF to represent a chunk of
// debug-related data. We create a CompilationUnit per Object file in a
// library (so, one for all the Go code, one for each assembly file, etc.).
type CompilationUnit struct {
	Pkg            string        // The package name, eg ("fmt", or "runtime")
	Lib            *Library      // Our library
	PCs            []dwarf.Range // PC ranges, relative to Textp[0]
	DWInfo         *dwarf.DWDie  // CU root DIE
	DWARFFileTable []string      // The file table used to generate the .debug_lines

	Consts    LoaderSym   // Package constants DIEs
	FuncDIEs  []LoaderSym // Function DIE subtrees
	AbsFnDIEs []LoaderSym // Abstract function DIE subtrees
	RangeSyms []LoaderSym // Symbols for debug_range
	Textp     []LoaderSym // Text symbols in this CU
}
