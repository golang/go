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
	// For every symbol defined in the shared library, record its address
	// in the original shared library address space.
	symAddr map[string]uint64
	// For relocations in the shared library, map from the address
	// (in the shared library address space) at which that
	// relocation applies to the target symbol.  We only keep
	// track of a single kind of relocation: a standard absolute
	// address relocation with no addend. These were R_ADDR
	// relocations when the shared library was built.
	relocTarget map[uint64]string
}

// A relocation that applies to part of the shared library.
type shlibReloc struct {
	// Address (in the shared library address space) the relocation applies to.
	addr uint64
	// Target symbol name.
	target string
}

type shlibRelocs []shlibReloc

func (s shlibRelocs) Len() int           { return len(s) }
func (s shlibRelocs) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s shlibRelocs) Less(i, j int) bool { return s[i].addr < s[j].addr }

// Link holds the context for writing object code from a compiler
// or for reading that input into the linker.
type Link struct {
	Target
	ErrorReporter
	ArchSyms

	outSem chan int // limits the number of output writers
	Out    *OutBuf

	version int // current version number for static/file-local symbols

	Debugvlog int
	Bso       *bufio.Writer

	Loaded bool // set after all inputs have been loaded as symbols

	compressDWARF bool

	Libdir       []string
	Library      []*sym.Library
	LibraryByPkg map[string]*sym.Library
	Shlibs       []Shlib
	Textp        []loader.Sym
	Moduledata   loader.Sym

	PackageFile  map[string]string
	PackageShlib map[string]string

	tramps []loader.Sym // trampolines

	compUnits []*sym.CompilationUnit // DWARF compilation units
	runtimeCU *sym.CompilationUnit   // One of the runtime CUs, the last one seen.

	loader  *loader.Loader
	cgodata []cgodata // cgo directives to load, three strings are args for loadcgo

	datap  []loader.Sym
	dynexp []loader.Sym

	// Elf symtab variables.
	numelfsym int // starts at 0, 1 is reserved

	// These are symbols that created and written by the linker.
	// Rather than creating a symbol, and writing all its data into the heap,
	// you can create a symbol, and just a generation function will be called
	// after the symbol's been created in the output mmap.
	generatorSyms map[loader.Sym]generatorFunc
}

type cgodata struct {
	file       string
	pkg        string
	directives [][]string
}

func (ctxt *Link) Logf(format string, args ...any) {
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

// Allocate a new version (i.e. symbol namespace).
func (ctxt *Link) IncVersion() int {
	ctxt.version++
	return ctxt.version - 1
}

// returns the maximum version number
func (ctxt *Link) MaxVersion() int {
	return ctxt.version
}

// generatorFunc is a convenience type.
// Some linker-created Symbols are large and shouldn't really live in the heap.
// Such Symbols can define a generator function. Their bytes can be generated
// directly in the output mmap.
//
// Relocations are applied prior to emitting generator Symbol contents.
// Generator Symbols that require relocations can be written in two passes.
// The first pass, at Symbol creation time, adds only relocations.
// The second pass, at content generation time, adds the rest.
// See generateFunctab for an example.
//
// Generator functions shouldn't grow the Symbol size.
// Generator functions must be safe for concurrent use.
//
// Generator Symbols have their Data set to the mmapped area when the
// generator is called.
type generatorFunc func(*Link, loader.Sym)

// createGeneratorSymbol is a convenience method for creating a generator
// symbol.
func (ctxt *Link) createGeneratorSymbol(name string, version int, t sym.SymKind, size int64, gen generatorFunc) loader.Sym {
	ldr := ctxt.loader
	s := ldr.LookupOrCreateSym(name, version)
	ldr.SetIsGeneratedSym(s, true)
	sb := ldr.MakeSymbolUpdater(s)
	sb.SetType(t)
	sb.SetSize(size)
	ctxt.generatorSyms[s] = gen
	return s
}
