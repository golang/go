// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"debug/goobj"
	"encoding/binary"
	"fmt"
	"go/build"
	"io"
	"os"
	"runtime"
)

// A Prog holds state for constructing an executable (program) image.
//
// The usual sequence of operations on a Prog is:
//
//	p.init()
//	p.scan(file)
//	p.dead()
//	p.runtime()
//	p.layout()
//	p.load()
//	p.debug()
//	p.write(w)
//
// p.init is in this file. The rest of the methods are in files
// named for the method. The convenience method p.link runs
// this sequence.
//
type Prog struct {
	// Context
	GOOS     string       // target operating system
	GOARCH   string       // target architecture
	Format   string       // desired file format ("elf", "macho", ...)
	Error    func(string) // called to report an error (if set)
	NumError int          // number of errors printed
	StartSym string

	// Derived context
	arch
	formatter   formatter
	startSym    goobj.SymID
	pkgdir      string
	omitRuntime bool // do not load runtime package

	// Input
	Packages   map[string]*Package  // loaded packages, by import path
	Syms       map[goobj.SymID]*Sym // defined symbols, by symbol ID
	Missing    map[goobj.SymID]bool // missing symbols
	Dead       map[goobj.SymID]bool // symbols removed as dead
	SymOrder   []*Sym               // order syms were scanned
	MaxVersion int                  // max SymID.Version, for generating fresh symbol IDs

	// Output
	UnmappedSize Addr       // size of unmapped region at address 0
	HeaderSize   Addr       // size of object file header
	Entry        Addr       // virtual address where execution begins
	Segments     []*Segment // loaded memory segments
}

// An arch describes architecture-dependent settings.
type arch struct {
	byteorder binary.ByteOrder
	ptrsize   int
	pcquantum int
}

// A formatter takes care of the details of generating a particular
// kind of executable file.
type formatter interface {
	// headerSize returns the footprint of the header for p
	// in both virtual address space and file bytes.
	// The footprint does not include any bytes stored at the
	// end of the file.
	headerSize(p *Prog) (virt, file Addr)

	// write writes the executable file for p to w.
	write(w io.Writer, p *Prog)
}

// An Addr represents a virtual memory address, a file address, or a size.
// It must be a uint64, not a uintptr, so that a 32-bit linker can still generate a 64-bit binary.
// It must be unsigned in order to link programs placed at very large start addresses.
// Math involving Addrs must be checked carefully not to require negative numbers.
type Addr uint64

// A Package is a Go package loaded from a file.
type Package struct {
	*goobj.Package        // table of contents
	File           string // file name for reopening
	Syms           []*Sym // symbols defined by this package
}

// A Sym is a symbol defined in a loaded package.
type Sym struct {
	*goobj.Sym          // symbol metadata from package file
	Package    *Package // package defining symbol
	Section    *Section // section where symbol is placed in output program
	Addr       Addr     // virtual address of symbol in output program
	Bytes      []byte   // symbol data, for internally defined symbols
}

// A Segment is a loaded memory segment.
// A Prog is expected to have segments named "text" and optionally "data",
// in that order, before any other segments.
type Segment struct {
	Name       string     // name of segment: "text", "data", ...
	VirtAddr   Addr       // virtual memory address of segment base
	VirtSize   Addr       // size of segment in memory
	FileOffset Addr       // file offset of segment base
	FileSize   Addr       // size of segment in file; can be less than VirtSize
	Sections   []*Section // sections inside segment
	Data       []byte     // raw data of segment image
}

// A Section is part of a loaded memory segment.
type Section struct {
	Name     string   // name of section: "text", "rodata", "noptrbss", and so on
	VirtAddr Addr     // virtual memory address of section base
	Size     Addr     // size of section in memory
	Align    Addr     // required alignment
	InFile   bool     // section has image data in file (like data, unlike bss)
	Syms     []*Sym   // symbols stored in section
	Segment  *Segment // segment containing section
}

func (p *Prog) errorf(format string, args ...interface{}) {
	if p.Error != nil {
		p.Error(fmt.Sprintf(format, args...))
	} else {
		fmt.Fprintf(os.Stderr, format+"\n", args...)
	}
	p.NumError++
}

// link is the one-stop convenience method for running a link.
// It writes to w the object file generated from using mainFile as the main package.
func (p *Prog) link(w io.Writer, mainFile string) {
	p.init()
	p.scan(mainFile)
	if p.NumError > 0 {
		return
	}
	p.dead()
	p.runtime()
	p.autoData()
	p.layout()
	p.autoConst()
	if p.NumError > 0 {
		return
	}
	p.load()
	if p.NumError > 0 {
		return
	}
	p.debug()
	if p.NumError > 0 {
		return
	}
	p.write(w)
}

// init initializes p for use by the other methods.
func (p *Prog) init() {
	// Set default context if not overridden.
	if p.GOOS == "" {
		p.GOOS = build.Default.GOOS
	}
	if p.GOARCH == "" {
		p.GOARCH = build.Default.GOARCH
	}
	if p.Format == "" {
		p.Format = goosFormat[p.GOOS]
		if p.Format == "" {
			p.errorf("no default file format for GOOS %q", p.GOOS)
			return
		}
	}
	if p.StartSym == "" {
		p.StartSym = fmt.Sprintf("_rt0_%s_%s", p.GOARCH, p.GOOS)
	}

	// Derive internal context.
	p.formatter = formatters[p.Format]
	if p.formatter == nil {
		p.errorf("unknown output file format %q", p.Format)
		return
	}
	p.startSym = goobj.SymID{Name: p.StartSym}
	arch, ok := arches[p.GOARCH]
	if !ok {
		p.errorf("unknown GOOS %q", p.GOOS)
		return
	}
	p.arch = arch

	p.pkgdir = fmt.Sprintf("%s/pkg/%s_%s", runtime.GOROOT(), p.GOOS, p.GOARCH)
}

// goosFormat records the default format for each known GOOS value.
var goosFormat = map[string]string{
	"darwin": "darwin",
}

// formatters records the format implementation for each known format value.
var formatters = map[string]formatter{
	"darwin": machoFormat{},
}

var arches = map[string]arch{
	"amd64": {
		byteorder: binary.LittleEndian,
		ptrsize:   8,
		pcquantum: 1,
	},
}
