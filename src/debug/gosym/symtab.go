// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package gosym implements access to the Go symbol
// and line number tables embedded in Go binaries generated
// by the gc compilers.
package gosym

import (
	"bytes"
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
	// If this symbol is a function symbol, the corresponding Func
	Func *Func

	goVersion version
}

// Static reports whether this symbol is static (not visible outside its file).
func (s *Sym) Static() bool { return s.Type >= 'a' }

// nameWithoutInst returns s.Name if s.Name has no brackets (does not reference an
// instantiated type, function, or method). If s.Name contains brackets, then it
// returns s.Name with all the contents between (and including) the outermost left
// and right bracket removed. This is useful to ignore any extra slashes or dots
// inside the brackets from the string searches below, where needed.
func (s *Sym) nameWithoutInst() string {
	start := strings.Index(s.Name, "[")
	if start < 0 {
		return s.Name
	}
	end := strings.LastIndex(s.Name, "]")
	if end < 0 {
		// Malformed name, should contain closing bracket too.
		return s.Name
	}
	return s.Name[0:start] + s.Name[end+1:]
}

// PackageName returns the package part of the symbol name,
// or the empty string if there is none.
func (s *Sym) PackageName() string {
	name := s.nameWithoutInst()

	// Since go1.20, a prefix of "type:" and "go:" is a compiler-generated symbol,
	// they do not belong to any package.
	//
	// See cmd/compile/internal/base/link.go:ReservedImports variable.
	if s.goVersion >= ver120 && (strings.HasPrefix(name, "go:") || strings.HasPrefix(name, "type:")) {
		return ""
	}

	// For go1.18 and below, the prefix are "type." and "go." instead.
	if s.goVersion <= ver118 && (strings.HasPrefix(name, "go.") || strings.HasPrefix(name, "type.")) {
		return ""
	}

	pathend := strings.LastIndex(name, "/")
	if pathend < 0 {
		pathend = 0
	}

	if i := strings.Index(name[pathend:], "."); i != -1 {
		return name[:pathend+i]
	}
	return ""
}

// ReceiverName returns the receiver type name of this symbol,
// or the empty string if there is none.  A receiver name is only detected in
// the case that s.Name is fully-specified with a package name.
func (s *Sym) ReceiverName() string {
	name := s.nameWithoutInst()
	// If we find a slash in name, it should precede any bracketed expression
	// that was removed, so pathend will apply correctly to name and s.Name.
	pathend := strings.LastIndex(name, "/")
	if pathend < 0 {
		pathend = 0
	}
	// Find the first dot after pathend (or from the beginning, if there was
	// no slash in name).
	l := strings.Index(name[pathend:], ".")
	// Find the last dot after pathend (or the beginning).
	r := strings.LastIndex(name[pathend:], ".")
	if l == -1 || r == -1 || l == r {
		// There is no receiver if we didn't find two distinct dots after pathend.
		return ""
	}
	// Given there is a trailing '.' that is in name, find it now in s.Name.
	// pathend+l should apply to s.Name, because it should be the dot in the
	// package name.
	r = strings.LastIndex(s.Name[pathend:], ".")
	return s.Name[pathend+l+1 : pathend+r]
}

// BaseName returns the symbol name without the package or receiver name.
func (s *Sym) BaseName() string {
	name := s.nameWithoutInst()
	if i := strings.LastIndex(name, "."); i != -1 {
		if s.Name != name {
			brack := strings.Index(s.Name, "[")
			if i > brack {
				// BaseName is a method name after the brackets, so
				// recalculate for s.Name. Otherwise, i applies
				// correctly to s.Name, since it is before the
				// brackets.
				i = strings.LastIndex(s.Name, ".")
			}
		}
		return s.Name[i+1:]
	}
	return s.Name
}

// A Func collects information about a single function.
type Func struct {
	Entry uint64
	*Sym
	End       uint64
	Params    []*Sym // nil for Go 1.3 and later binaries
	Locals    []*Sym // nil for Go 1.3 and later binaries
	FrameSize int
	LineTable *LineTable
	Obj       *Obj
}

// An Obj represents a collection of functions in a symbol table.
//
// The exact method of division of a binary into separate Objs is an internal detail
// of the symbol table format.
//
// In early versions of Go each source file became a different Obj.
//
// In Go 1 and Go 1.1, each package produced one Obj for all Go sources
// and one Obj per C source file.
//
// In Go 1.2, there is a single Obj for the entire program.
type Obj struct {
	// Funcs is a list of functions in the Obj.
	Funcs []Func

	// In Go 1.1 and earlier, Paths is a list of symbols corresponding
	// to the source file names that produced the Obj.
	// In Go 1.2, Paths is nil.
	// Use the keys of Table.Files to obtain a list of source files.
	Paths []Sym // meta
}

/*
 * Symbol tables
 */

// Table represents a Go symbol table. It stores all of the
// symbols decoded from the program and provides methods to translate
// between symbols, names, and addresses.
type Table struct {
	Syms  []Sym // nil for Go 1.3 and later binaries
	Funcs []Func
	Files map[string]*Obj // for Go 1.2 and later all files map to one Obj
	Objs  []Obj           // for Go 1.2 and later only one Obj in slice

	go12line *LineTable // Go 1.2 line number table
}

type sym struct {
	value  uint64
	gotype uint64
	typ    byte
	name   []byte
}

var (
	littleEndianSymtab    = []byte{0xFD, 0xFF, 0xFF, 0xFF, 0x00, 0x00, 0x00}
	bigEndianSymtab       = []byte{0xFF, 0xFF, 0xFF, 0xFD, 0x00, 0x00, 0x00}
	oldLittleEndianSymtab = []byte{0xFE, 0xFF, 0xFF, 0xFF, 0x00, 0x00}
)

func walksymtab(data []byte, fn func(sym) error) error {
	if len(data) == 0 { // missing symtab is okay
		return nil
	}
	var order binary.ByteOrder = binary.BigEndian
	newTable := false
	switch {
	case bytes.HasPrefix(data, oldLittleEndianSymtab):
		// Same as Go 1.0, but little endian.
		// Format was used during interim development between Go 1.0 and Go 1.1.
		// Should not be widespread, but easy to support.
		data = data[6:]
		order = binary.LittleEndian
	case bytes.HasPrefix(data, bigEndianSymtab):
		newTable = true
	case bytes.HasPrefix(data, littleEndianSymtab):
		newTable = true
		order = binary.LittleEndian
	}
	var ptrsz int
	if newTable {
		if len(data) < 8 {
			return &DecodingError{len(data), "unexpected EOF", nil}
		}
		ptrsz = int(data[7])
		if ptrsz != 4 && ptrsz != 8 {
			return &DecodingError{7, "invalid pointer size", ptrsz}
		}
		data = data[8:]
	}
	var s sym
	p := data
	for len(p) >= 4 {
		var typ byte
		if newTable {
			// Symbol type, value, Go type.
			typ = p[0] & 0x3F
			wideValue := p[0]&0x40 != 0
			goType := p[0]&0x80 != 0
			if typ < 26 {
				typ += 'A'
			} else {
				typ += 'a' - 26
			}
			s.typ = typ
			p = p[1:]
			if wideValue {
				if len(p) < ptrsz {
					return &DecodingError{len(data), "unexpected EOF", nil}
				}
				// fixed-width value
				if ptrsz == 8 {
					s.value = order.Uint64(p[0:8])
					p = p[8:]
				} else {
					s.value = uint64(order.Uint32(p[0:4]))
					p = p[4:]
				}
			} else {
				// varint value
				s.value = 0
				shift := uint(0)
				for len(p) > 0 && p[0]&0x80 != 0 {
					s.value |= uint64(p[0]&0x7F) << shift
					shift += 7
					p = p[1:]
				}
				if len(p) == 0 {
					return &DecodingError{len(data), "unexpected EOF", nil}
				}
				s.value |= uint64(p[0]) << shift
				p = p[1:]
			}
			if goType {
				if len(p) < ptrsz {
					return &DecodingError{len(data), "unexpected EOF", nil}
				}
				// fixed-width go type
				if ptrsz == 8 {
					s.gotype = order.Uint64(p[0:8])
					p = p[8:]
				} else {
					s.gotype = uint64(order.Uint32(p[0:4]))
					p = p[4:]
				}
			}
		} else {
			// Value, symbol type.
			s.value = uint64(order.Uint32(p[0:4]))
			if len(p) < 5 {
				return &DecodingError{len(data), "unexpected EOF", nil}
			}
			typ = p[4]
			if typ&0x80 == 0 {
				return &DecodingError{len(data) - len(p) + 4, "bad symbol type", typ}
			}
			typ &^= 0x80
			s.typ = typ
			p = p[5:]
		}

		// Name.
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
		if len(p) < i+nnul {
			return &DecodingError{len(data), "unexpected EOF", nil}
		}
		s.name = p[0:i]
		i += nnul
		p = p[i:]

		if !newTable {
			if len(p) < 4 {
				return &DecodingError{len(data), "unexpected EOF", nil}
			}
			// Go type.
			s.gotype = uint64(order.Uint32(p[:4]))
			p = p[4:]
		}
		fn(s)
	}
	return nil
}

// NewTable decodes the Go symbol table (the ".gosymtab" section in ELF),
// returning an in-memory representation.
// Starting with Go 1.3, the Go symbol table no longer includes symbol data.
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
	if pcln.isGo12() {
		t.go12line = pcln
	}
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
		ts.Value = s.value
		ts.GoType = s.gotype
		ts.goVersion = pcln.version
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
	t.Files = make(map[string]*Obj)

	var obj *Obj
	if t.go12line != nil {
		// Put all functions into one Obj.
		t.Objs = make([]Obj, 1)
		obj = &t.Objs[0]
		t.go12line.go12MapFiles(t.Files, obj)
	} else {
		t.Objs = make([]Obj, 0, nz)
	}

	// Count text symbols and attach frame sizes, parameters, and
	// locals to them. Also, find object file boundaries.
	lastf := 0
	for i := 0; i < len(t.Syms); i++ {
		sym := &t.Syms[i]
		switch sym.Type {
		case 'Z', 'z': // path symbol
			if t.go12line != nil {
				// Go 1.2 binaries have the file information elsewhere. Ignore.
				break
			}
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
			if sym.Name == "runtime.etext" || sym.Name == "etext" {
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
			if t.go12line != nil {
				// All functions share the same line table.
				// It knows how to narrow down to a specific
				// function quickly.
				fn.LineTable = t.go12line
			} else if pcln != nil {
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

	if t.go12line != nil && nf == 0 {
		t.Funcs = t.go12line.go12Funcs()
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
	if t.go12line != nil {
		file = t.go12line.go12PCToFile(pc)
		line = t.go12line.go12PCToLine(pc)
	} else {
		file, line = fn.Obj.lineFromAline(fn.LineTable.PCToLine(pc))
	}
	return
}

// LineToPC looks up the first program counter on the given line in
// the named file. It returns [UnknownFileError] or [UnknownLineError] if
// there is an error looking up this line.
func (t *Table) LineToPC(file string, line int) (pc uint64, fn *Func, err error) {
	obj, ok := t.Files[file]
	if !ok {
		return 0, nil, UnknownFileError(file)
	}

	if t.go12line != nil {
		pc := t.go12line.go12LineToPC(file, line)
		if pc == 0 {
			return 0, nil, &UnknownLineError{file, line}
		}
		return pc, t.PCToFunc(pc), nil
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
func (t *Table) SymByAddr(addr uint64) *Sym {
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

// This is legacy code for Go 1.1 and earlier, which used the
// Plan 9 format for pc-line tables. This code was never quite
// correct. It's probably very close, and it's usually correct, but
// we never quite found all the corner cases.
//
// Go 1.2 and later use a simpler format, documented at golang.org/s/go12symtab.

func (o *Obj) lineFromAline(aline int) (string, int) {
	type stackEnt struct {
		path   string
		start  int
		offset int
		prev   *stackEnt
	}

	noPath := &stackEnt{"", 0, 0, nil}
	tos := noPath

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
	val any
}

func (e *DecodingError) Error() string {
	msg := e.msg
	if e.val != nil {
		msg += fmt.Sprintf(" '%v'", e.val)
	}
	msg += fmt.Sprintf(" at byte %#x", e.off)
	return msg
}
