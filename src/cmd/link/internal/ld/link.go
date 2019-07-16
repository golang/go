// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package ld

import (
	"bufio"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
)

type Shlib struct {
	Path            string
	Hash            []byte
	Deps            []string
	File            *elf.File
	gcdataAddresses map[*sym.Symbol]uint64
}

// Link holds the context for writing object code from a compiler
// or for reading that input into the linker.
type Link struct {
	Out *OutBuf

	Syms *sym.Symbols

	Arch      *sys.Arch
	Debugvlog int
	Bso       *bufio.Writer

	Loaded bool // set after all inputs have been loaded as symbols

	IsELF    bool
	HeadType objabi.HeadType

	linkShared    bool // link against installed Go shared libraries
	LinkMode      LinkMode
	BuildMode     BuildMode
	compressDWARF bool

	Tlsg         *sym.Symbol
	Libdir       []string
	Library      []*sym.Library
	LibraryByPkg map[string]*sym.Library
	Shlibs       []Shlib
	Tlsoffset    int
	Textp        []*sym.Symbol
	Filesyms     []*sym.Symbol
	Moduledata   *sym.Symbol

	PackageFile  map[string]string
	PackageShlib map[string]string

	tramps []*sym.Symbol // trampolines

	// unresolvedSymSet is a set of erroneous unresolved references.
	// Used to avoid duplicated error messages.
	unresolvedSymSet map[unresolvedSymKey]bool

	// Used to implement field tracking.
	Reachparent map[*sym.Symbol]*sym.Symbol

	compUnits         []*compilationUnit // DWARF compilation units
	compUnitByPackage map[*sym.Library]*compilationUnit

	relocbuf []byte // temporary buffer for applying relocations
}

type unresolvedSymKey struct {
	from *sym.Symbol // Symbol that referenced unresolved "to"
	to   *sym.Symbol // Unresolved symbol referenced by "from"
}

// ErrorUnresolved prints unresolved symbol error for r.Sym that is referenced from s.
func (ctxt *Link) ErrorUnresolved(s *sym.Symbol, r *sym.Reloc) {
	if ctxt.unresolvedSymSet == nil {
		ctxt.unresolvedSymSet = make(map[unresolvedSymKey]bool)
	}

	k := unresolvedSymKey{from: s, to: r.Sym}
	if !ctxt.unresolvedSymSet[k] {
		ctxt.unresolvedSymSet[k] = true

		// Try to find symbol under another ABI.
		var reqABI, haveABI obj.ABI
		haveABI = ^obj.ABI(0)
		reqABI, ok := sym.VersionToABI(int(r.Sym.Version))
		if ok {
			for abi := obj.ABI(0); abi < obj.ABICount; abi++ {
				v := sym.ABIToVersion(abi)
				if v == -1 {
					continue
				}
				if rs := ctxt.Syms.ROLookup(r.Sym.Name, v); rs != nil && rs.Type != sym.Sxxx {
					haveABI = abi
				}
			}
		}

		// Give a special error message for main symbol (see #24809).
		if r.Sym.Name == "main.main" {
			Errorf(s, "function main is undeclared in the main package")
		} else if haveABI != ^obj.ABI(0) {
			Errorf(s, "relocation target %s not defined for %s (but is defined for %s)", r.Sym.Name, reqABI, haveABI)
		} else {
			Errorf(s, "relocation target %s not defined", r.Sym.Name)
		}
	}
}

// The smallest possible offset from the hardware stack pointer to a local
// variable on the stack. Architectures that use a link register save its value
// on the stack in the function prologue and so always have a pointer between
// the hardware stack pointer and the local variable area.
func (ctxt *Link) FixedFrameSize() int64 {
	switch ctxt.Arch.Family {
	case sys.AMD64, sys.I386:
		return 0
	case sys.PPC64:
		// PIC code on ppc64le requires 32 bytes of stack, and it's easier to
		// just use that much stack always on ppc64x.
		return int64(4 * ctxt.Arch.PtrSize)
	default:
		return int64(ctxt.Arch.PtrSize)
	}
}

func (ctxt *Link) Logf(format string, args ...interface{}) {
	fmt.Fprintf(ctxt.Bso, format, args...)
	ctxt.Bso.Flush()
}

func addImports(ctxt *Link, l *sym.Library, pn string) {
	pkg := objabi.PathToPrefix(l.Pkg)
	for _, importStr := range l.ImportStrings {
		lib := addlib(ctxt, pkg, pn, importStr)
		if lib != nil {
			l.Imports = append(l.Imports, lib)
		}
	}
	l.ImportStrings = nil
}
