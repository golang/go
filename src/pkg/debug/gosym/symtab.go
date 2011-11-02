// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gosym implements access to the Go symbol
// and line number tables embedded in Go binaries generated
// by the gc compilers.
package gosym

// The table format is a variant of the format used in Plan 9's a.out
// format, documented at http://plan9.bell-labs.com/magic/man2html/6/a.out.
// The best reference for the differences between the Plan 9 format
// and the Go format is the runtime source, specifically ../../runtime/symtab.c.

import (
	"encoding/binary"
	"fmt"
	"strconv"
	"strings"
)

/*
 * Symbols
 */

// A Sym represents a single symbol table entry.
type Sym struct {
	Value  uint64
	Type   byte
	Name   string
	GoType uint64
	// If this symbol if a function symbol, the corresponding Func
	Func *Func
}

// Static returns whether this symbol is static (not visible outside its file).
func (s *Sym) Static() bool { return s.Type >= 'a' }

// PackageName returns the package part of the symbol name,
// or the empty string if there is none.
func (s *Sym) PackageName() string {
	if i := strings.Index(s.Name, "."); i != -1 {
		return s.Name[0:i]
	}
	return ""
}

// ReceiverName returns the receiver type name of this symbol,
// or the empty string if there is none.
func (s *Sym) ReceiverName() string {
	l := strings.Index(s.Name, ".")
	r := strings.LastIndex(s.Name, ".")
	if l == -1 || r == -1 || l == r {
		return ""
	}
	return s.Name[l+1 : r]
}

// BaseName returns the symbol name without the package or receiver name.
func (s *Sym) BaseName() string {
	if i := strings.LastIndex(s.Name, "."); i != -1 {
		return s.Name[i+1:]
	}
	return s.Name
}

// A Func collects information about a single function.
type Func struct {
	Entry uint64
	*Sym
	End       uint64
	Params    []*Sym
	Locals    []*Sym
	FrameSize int
	LineTable *LineTable
	Obj       *Obj
}

// An Obj represents a single object file.
type Obj struct {
	Funcs []Func
	Paths []Sym
}

/*
 * Symbol tables
 */

// Table represents a Go symbol table.  It stores all of the
// symbols decoded from the program and provides methods to translate
// between symbols, names, and addresses.
type Table struct {
	Syms  []Sym
	Funcs []Func
	Files map[string]*Obj
	Objs  []Obj
	//	textEnd uint64;
}

type sym struct {
	value  uint32
	gotype uint32
	typ    byte
	name   []byte
}

func walksymtab(data []byte, fn func(sym) error) error {
	var s sym
	p := data
	for len(p) >= 6 {
		s.value = binary.BigEndian.Uint32(p[0:4])
		typ := p[4]
		if typ&0x80 == 0 {
			return &DecodingError{len(data) - len(p) + 4, "bad symbol type", typ}
		}
		typ &^= 0x80
		s.typ = typ
		p = p[5:]
		var i int
		var nnul int
		for i = 0; i < len(p); i++ {
			if p[i] == 0 {
				nnul = 1
				break
			}
		}
		switch typ {
		case 'z', 'Z':
			p = p[i+nnul:]
			for i = 0; i+2 <= len(p); i += 2 {
				if p[i] == 0 && p[i+1] == 0 {
					nnul = 2
					break
				}
			}
		}
		if i+nnul+4 > len(p) {
			return &DecodingError{len(data), "unexpected EOF", nil}
		}
		s.name = p[0:i]
		i += nnul
		s.gotype = binary.BigEndian.Uint32(p[i : i+4])
		p = p[i+4:]
		fn(s)
	}
	return nil
}

// NewTable decodes the Go symbol table in data,
// returning an in-memory representation.
func NewTable(symtab []byte, pcln *LineTable) (*Table, error) {
	var n int
	err := walksymtab(symtab, func(s sym) error {
		n++
		return nil
	})
	if err != nil {
		return nil, err
	}

	var t Table
	fname := make(map[uint16]string)
	t.Syms = make([]Sym, 0, n)
	nf := 0
	nz := 0
	lasttyp := uint8(0)
	err = walksymtab(symtab, func(s sym) error {
		n := len(t.Syms)
		t.Syms = t.Syms[0 : n+1]
		ts := &t.Syms[n]
		ts.Type = s.typ
		ts.Value = uint64(s.value)
		ts.GoType = uint64(s.gotype)
		switch s.typ {
		default:
			// rewrite name to use . instead of · (c2 b7)
			w := 0
			b := s.name
			for i := 0; i < len(b); i++ {
				if b[i] == 0xc2 && i+1 < len(b) && b[i+1] == 0xb7 {
					i++
					b[i] = '.'
				}
				b[w] = b[i]
				w++
			}
			ts.Name = string(s.name[0:w])
		case 'z', 'Z':
			if lasttyp != 'z' && lasttyp != 'Z' {
				nz++
			}
			for i := 0; i < len(s.name); i += 2 {
				eltIdx := binary.BigEndian.Uint16(s.name[i : i+2])
				elt, ok := fname[eltIdx]
				if !ok {
					return &DecodingError{-1, "bad filename code", eltIdx}
				}
				if n := len(ts.Name); n > 0 && ts.Name[n-1] != '/' {
					ts.Name += "/"
				}
				ts.Name += elt
			}
		}
		switch s.typ {
		case 'T', 't', 'L', 'l':
			nf++
		case 'f':
			fname[uint16(s.value)] = ts.Name
		}
		lasttyp = s.typ
		return nil
	})
	if err != nil {
		return nil, err
	}

	t.Funcs = make([]Func, 0, nf)
	t.Objs = make([]Obj, 0, nz)
	t.Files = make(map[string]*Obj)

	// Count text symbols and attach frame sizes, parameters, and
	// locals to them.  Also, find object file boundaries.
	var obj *Obj
	lastf := 0
	for i := 0; i < len(t.Syms); i++ {
		sym := &t.Syms[i]
		switch sym.Type {
		case 'Z', 'z': // path symbol
			// Finish the current object
			if obj != nil {
				obj.Funcs = t.Funcs[lastf:]
			}
			lastf = len(t.Funcs)

			// Start new object
			n := len(t.Objs)
			t.Objs = t.Objs[0 : n+1]
			obj = &t.Objs[n]

			// Count & copy path symbols
			var end int
			for end = i + 1; end < len(t.Syms); end++ {
				if c := t.Syms[end].Type; c != 'Z' && c != 'z' {
					break
				}
			}
			obj.Paths = t.Syms[i:end]
			i = end - 1 // loop will i++

			// Record file names
			depth := 0
			for j := range obj.Paths {
				s := &obj.Paths[j]
				if s.Name == "" {
					depth--
				} else {
					if depth == 0 {
						t.Files[s.Name] = obj
					}
					depth++
				}
			}

		case 'T', 't', 'L', 'l': // text symbol
			if n := len(t.Funcs); n > 0 {
				t.Funcs[n-1].End = sym.Value
			}
			if sym.Name == "etext" {
				continue
			}

			// Count parameter and local (auto) syms
			var np, na int
			var end int
		countloop:
			for end = i + 1; end < len(t.Syms); end++ {
				switch t.Syms[end].Type {
				case 'T', 't', 'L', 'l', 'Z', 'z':
					break countloop
				case 'p':
					np++
				case 'a':
					na++
				}
			}

			// Fill in the function symbol
			n := len(t.Funcs)
			t.Funcs = t.Funcs[0 : n+1]
			fn := &t.Funcs[n]
			sym.Func = fn
			fn.Params = make([]*Sym, 0, np)
			fn.Locals = make([]*Sym, 0, na)
			fn.Sym = sym
			fn.Entry = sym.Value
			fn.Obj = obj
			if pcln != nil {
				fn.LineTable = pcln.slice(fn.Entry)
				pcln = fn.LineTable
			}
			for j := i; j < end; j++ {
				s := &t.Syms[j]
				switch s.Type {
				case 'm':
					fn.FrameSize = int(s.Value)
				case 'p':
					n := len(fn.Params)
					fn.Params = fn.Params[0 : n+1]
					fn.Params[n] = s
				case 'a':
					n := len(fn.Locals)
					fn.Locals = fn.Locals[0 : n+1]
					fn.Locals[n] = s
				}
			}
			i = end - 1 // loop will i++
		}
	}
	if obj != nil {
		obj.Funcs = t.Funcs[lastf:]
	}
	return &t, nil
}

// PCToFunc returns the function containing the program counter pc,
// or nil if there is no such function.
func (t *Table) PCToFunc(pc uint64) *Func {
	funcs := t.Funcs
	for len(funcs) > 0 {
		m := len(funcs) / 2
		fn := &funcs[m]
		switch {
		case pc < fn.Entry:
			funcs = funcs[0:m]
		case fn.Entry <= pc && pc < fn.End:
			return fn
		default:
			funcs = funcs[m+1:]
		}
	}
	return nil
}

// PCToLine looks up line number information for a program counter.
// If there is no information, it returns fn == nil.
func (t *Table) PCToLine(pc uint64) (file string, line int, fn *Func) {
	if fn = t.PCToFunc(pc); fn == nil {
		return
	}
	file, line = fn.Obj.lineFromAline(fn.LineTable.PCToLine(pc))
	return
}

// LineToPC looks up the first program counter on the given line in
// the named file.  Returns UnknownPathError or UnknownLineError if
// there is an error looking up this line.
func (t *Table) LineToPC(file string, line int) (pc uint64, fn *Func, err error) {
	obj, ok := t.Files[file]
	if !ok {
		return 0, nil, UnknownFileError(file)
	}
	abs, err := obj.alineFromLine(file, line)
	if err != nil {
		return
	}
	for i := range obj.Funcs {
		f := &obj.Funcs[i]
		pc := f.LineTable.LineToPC(abs, f.End)
		if pc != 0 {
			return pc, f, nil
		}
	}
	return 0, nil, &UnknownLineError{file, line}
}

// LookupSym returns the text, data, or bss symbol with the given name,
// or nil if no such symbol is found.
func (t *Table) LookupSym(name string) *Sym {
	// TODO(austin) Maybe make a map
	for i := range t.Syms {
		s := &t.Syms[i]
		switch s.Type {
		case 'T', 't', 'L', 'l', 'D', 'd', 'B', 'b':
			if s.Name == name {
				return s
			}
		}
	}
	return nil
}

// LookupFunc returns the text, data, or bss symbol with the given name,
// or nil if no such symbol is found.
func (t *Table) LookupFunc(name string) *Func {
	for i := range t.Funcs {
		f := &t.Funcs[i]
		if f.Sym.Name == name {
			return f
		}
	}
	return nil
}

// SymByAddr returns the text, data, or bss symbol starting at the given address.
// TODO(rsc): Allow lookup by any address within the symbol.
func (t *Table) SymByAddr(addr uint64) *Sym {
	// TODO(austin) Maybe make a map
	for i := range t.Syms {
		s := &t.Syms[i]
		switch s.Type {
		case 'T', 't', 'L', 'l', 'D', 'd', 'B', 'b':
			if s.Value == addr {
				return s
			}
		}
	}
	return nil
}

/*
 * Object files
 */

func (o *Obj) lineFromAline(aline int) (string, int) {
	type stackEnt struct {
		path   string
		start  int
		offset int
		prev   *stackEnt
	}

	noPath := &stackEnt{"", 0, 0, nil}
	tos := noPath

	// TODO(austin) I have no idea how 'Z' symbols work, except
	// that they pop the stack.
pathloop:
	for _, s := range o.Paths {
		val := int(s.Value)
		switch {
		case val > aline:
			break pathloop

		case val == 1:
			// Start a new stack
			tos = &stackEnt{s.Name, val, 0, noPath}

		case s.Name == "":
			// Pop
			if tos == noPath {
				return "<malformed symbol table>", 0
			}
			tos.prev.offset += val - tos.start
			tos = tos.prev

		default:
			// Push
			tos = &stackEnt{s.Name, val, 0, tos}
		}
	}

	if tos == noPath {
		return "", 0
	}
	return tos.path, aline - tos.start - tos.offset + 1
}

func (o *Obj) alineFromLine(path string, line int) (int, error) {
	if line < 1 {
		return 0, &UnknownLineError{path, line}
	}

	for i, s := range o.Paths {
		// Find this path
		if s.Name != path {
			continue
		}

		// Find this line at this stack level
		depth := 0
		var incstart int
		line += int(s.Value)
	pathloop:
		for _, s := range o.Paths[i:] {
			val := int(s.Value)
			switch {
			case depth == 1 && val >= line:
				return line - 1, nil

			case s.Name == "":
				depth--
				if depth == 0 {
					break pathloop
				} else if depth == 1 {
					line += val - incstart
				}

			default:
				if depth == 1 {
					incstart = val
				}
				depth++
			}
		}
		return 0, &UnknownLineError{path, line}
	}
	return 0, UnknownFileError(path)
}

/*
 * Errors
 */

// UnknownFileError represents a failure to find the specific file in
// the symbol table.
type UnknownFileError string

func (e UnknownFileError) Error() string { return "unknown file: " + string(e) }

// UnknownLineError represents a failure to map a line to a program
// counter, either because the line is beyond the bounds of the file
// or because there is no code on the given line.
type UnknownLineError struct {
	File string
	Line int
}

func (e *UnknownLineError) Error() string {
	return "no code at " + e.File + ":" + strconv.Itoa(e.Line)
}

// DecodingError represents an error during the decoding of
// the symbol table.
type DecodingError struct {
	off int
	msg string
	val interface{}
}

func (e *DecodingError) Error() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v'", e.val)
	}
	msg += fmt.Sprintf(" at byte %#x", e.off)
	return msg
}
