// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package objfile implements portable access to OS-specific executable files.
package objfile

import (
	"debug/dwarf"
	"debug/gosym"
	"fmt"
	"io"
	"os"
	"sort"
)

type rawFile interface {
	symbols() (syms []Sym, err error)
	pcln() (textStart uint64, symtab, pclntab []byte, err error)
	text() (textStart uint64, text []byte, err error)
	goarch() string
	loadAddress() (uint64, error)
	dwarf() (*dwarf.Data, error)
}

// A File is an opened executable file.
type File struct {
	r       *os.File
	entries []*Entry
}

type Entry struct {
	name string
	raw  rawFile
}

// A Sym is a symbol defined in an executable file.
type Sym struct {
	Name   string  // symbol name
	Addr   uint64  // virtual address of symbol
	Size   int64   // size in bytes
	Code   rune    // nm code (T for text, D for data, and so on)
	Type   string  // XXX?
	Relocs []Reloc // in increasing Addr order
}

type Reloc struct {
	Addr     uint64 // Address of first byte that reloc applies to.
	Size     uint64 // Number of bytes
	Stringer RelocStringer
}

type RelocStringer interface {
	// insnOffset is the offset of the instruction containing the relocation
	// from the start of the symbol containing the relocation.
	String(insnOffset uint64) string
}

var openers = []func(io.ReaderAt) (rawFile, error){
	openElf,
	openMacho,
	openPE,
	openPlan9,
	openXcoff,
}

// Open opens the named file.
// The caller must call f.Close when the file is no longer needed.
func Open(name string) (*File, error) {
	r, err := os.Open(name)
	if err != nil {
		return nil, err
	}
	if f, err := openGoFile(r); err == nil {
		return f, nil
	}
	for _, try := range openers {
		if raw, err := try(r); err == nil {
			return &File{r, []*Entry{&Entry{raw: raw}}}, nil
		}
	}
	r.Close()
	return nil, fmt.Errorf("open %s: unrecognized object file", name)
}

func (f *File) Close() error {
	return f.r.Close()
}

func (f *File) Entries() []*Entry {
	return f.entries
}

func (f *File) Symbols() ([]Sym, error) {
	return f.entries[0].Symbols()
}

func (f *File) PCLineTable() (Liner, error) {
	return f.entries[0].PCLineTable()
}

func (f *File) Text() (uint64, []byte, error) {
	return f.entries[0].Text()
}

func (f *File) GOARCH() string {
	return f.entries[0].GOARCH()
}

func (f *File) LoadAddress() (uint64, error) {
	return f.entries[0].LoadAddress()
}

func (f *File) DWARF() (*dwarf.Data, error) {
	return f.entries[0].DWARF()
}

func (f *File) Disasm() (*Disasm, error) {
	return f.entries[0].Disasm()
}

func (e *Entry) Name() string {
	return e.name
}

func (e *Entry) Symbols() ([]Sym, error) {
	syms, err := e.raw.symbols()
	if err != nil {
		return nil, err
	}
	sort.Sort(byAddr(syms))
	return syms, nil
}

type byAddr []Sym

func (x byAddr) Less(i, j int) bool { return x[i].Addr < x[j].Addr }
func (x byAddr) Len() int           { return len(x) }
func (x byAddr) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }

func (e *Entry) PCLineTable() (Liner, error) {
	// If the raw file implements Liner directly, use that.
	// Currently, only Go intermediate objects and archives (goobj) use this path.
	if pcln, ok := e.raw.(Liner); ok {
		return pcln, nil
	}
	// Otherwise, read the pcln tables and build a Liner out of that.
	textStart, symtab, pclntab, err := e.raw.pcln()
	if err != nil {
		return nil, err
	}
	return gosym.NewTable(symtab, gosym.NewLineTable(pclntab, textStart))
}

func (e *Entry) Text() (uint64, []byte, error) {
	return e.raw.text()
}

func (e *Entry) GOARCH() string {
	return e.raw.goarch()
}

// LoadAddress returns the expected load address of the file.
// This differs from the actual load address for a position-independent
// executable.
func (e *Entry) LoadAddress() (uint64, error) {
	return e.raw.loadAddress()
}

// DWARF returns DWARF debug data for the file, if any.
// This is for cmd/pprof to locate cgo functions.
func (e *Entry) DWARF() (*dwarf.Data, error) {
	return e.raw.dwarf()
}
