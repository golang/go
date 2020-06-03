// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/l.h
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
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
)

type Shlib struct {
	Path string
	Hash []byte
	Deps []string
	File *elf.File
}

// Link holds the context for writing object code from a compiler
// or for reading that input into the linker.
type Link struct {
	Target
	ErrorReporter
	ArchSyms

	outSem chan int // limits the number of output writers
	Out    *OutBuf

	Syms *sym.Symbols

	Debugvlog int
	Bso       *bufio.Writer

	Loaded bool // set after all inputs have been loaded as symbols

	compressDWARF bool

	Libdir       []string
	Library      []*sym.Library
	LibraryByPkg map[string]*sym.Library
	Shlibs       []Shlib
	Textp        []*sym.Symbol
	Textp2       []loader.Sym
	NumFilesyms  int
	Moduledata   *sym.Symbol
	Moduledata2  loader.Sym

	PackageFile  map[string]string
	PackageShlib map[string]string

	tramps []loader.Sym // trampolines

	compUnits []*sym.CompilationUnit // DWARF compilation units
	runtimeCU *sym.CompilationUnit   // One of the runtime CUs, the last one seen.

	loader  *loader.Loader
	cgodata []cgodata // cgo directives to load, three strings are args for loadcgo

	cgo_export_static  map[string]bool
	cgo_export_dynamic map[string]bool

	datap   []*sym.Symbol
	datap2  []loader.Sym
	dynexp2 []loader.Sym

	// Elf symtab variables.
	numelfsym int // starts at 0, 1 is reserved
	elfbind   int
}

type cgodata struct {
	file       string
	pkg        string
	directives [][]string
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
	for _, imp := range l.Autolib {
		lib := addlib(ctxt, pkg, pn, imp.Pkg, imp.Fingerprint)
		if lib != nil {
			l.Imports = append(l.Imports, lib)
		}
	}
	l.Autolib = nil
}
