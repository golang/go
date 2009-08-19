// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

// The Go symbol table and line number table formats are based on
// the Plan9 a.out format, which is documented here:
//   http://plan9.bell-labs.com/magic/man2html/6/a.out
// The best reference for the differences between the Plan9 format and
// the Go format is the runtime source, particularly:
//   src/pkg/runtime/symtab.c

import (
	"io";
	"os";
	"sort";
	"strings";
)

/*
 * Symbols
 */

type GoSym interface {
	Common() *CommonSym;
}

// CommonSym represents information that all symbols have in common.
// The meaning of the symbol value differs between symbol types.
type CommonSym struct {
	Value uint64;
	Type byte;
	Name string;
	GoType uint64;
}

func (c *CommonSym) Common() *CommonSym {
	return c;
}

// Static returns whether this symbol is static (not visible outside its file).
func (c *CommonSym) Static() bool {
	switch c.Type {
	case 't', 'l', 'd', 'b':
		return true;
	}
	return false;
}

// PackageName returns the package part of the symbol name, or empty
// string if there is none.
func (c *CommonSym) PackageName() string {
	if i := strings.Index(c.Name, "·"); i != -1 {
		return c.Name[0:i];
	}
	return "";
}

// ReceiverName returns the receiver type name of this symbol, or
// empty string if there is none.
func (c *CommonSym) ReceiverName() string {
	l := strings.Index(c.Name, "·");
	r := strings.LastIndex(c.Name, "·");
	if l == -1 || r == -1 {
		return "";
	}
	return c.Name[l+len("·"):r];
}

// BaseName returns the symbol name without the package or receiver name.
func (c *CommonSym) BaseName() string {
	if i := strings.LastIndex(c.Name, "·"); i != -1 {
		return c.Name[i+len("·"):len(c.Name)];
	}
	return c.Name;
}

// TextSym represents a function symbol.  In addition to the common
// symbol fields, it has a frame size, parameters, and local variables.
type TextSym struct {
	CommonSym;
	obj *object;
	lt *lineTable;
	// Ths size of this function's frame.
	FrameSize int;
	// The value of each parameter symbol is its positive offset
	// from the stack base pointer.  This includes out parameters,
	// even if they are unnamed.
	Params []*ParamSym;
	// The value of each local symbol is its negative offset from
	// the stack base pointer.
	Locals []*LocalSym;
}

func (s *TextSym) Entry() uint64 {
	return s.Value;
}

type LeafSym struct {
	CommonSym;
}

type DataSym struct {
	CommonSym;
}

type BSSSym struct {
	CommonSym;
}

type FrameSym struct {
	CommonSym;
}

type LocalSym struct {
	CommonSym;
}

type ParamSym struct {
	CommonSym;
}

type PathSym struct {
	CommonSym;
}

/*
 * Symbol tables
 */

type object struct {
	paths []*PathSym;
	funcs []*TextSym;
}

type lineTable struct {
	blob []byte;
	pc uint64;
	line int;
}

// GoSymTable represents a Go symbol table.  It stores all of the
// symbols decoded from the program and provides methods to translate
// between symbols, names, and addresses.
type GoSymTable struct {
	textEnd uint64;
	Syms []GoSym;
	funcs []*TextSym;
}

func growGoSyms(s *[]GoSym) (*GoSym) {
	n := len(*s);
	if n == cap(*s) {
		n := make([]GoSym, n, n * 2);
		for i := range *s {
			n[i] = (*s)[i];
		}
		*s = n;
	}
	*s = (*s)[0:n+1];
	return &(*s)[n];
}

func (t *GoSymTable) readGoSymTab(r io.Reader) os.Error {
	t.Syms = make([]GoSym, 0, 16);
	filenames := make(map[uint32] string);

	br := newBinaryReader(r, msb);
	off := int64(0);
	for {
		// Read symbol
		value := br.ReadUint32();
		if br.Error() == os.EOF {
			break;
		}
		typ := br.ReadUint8();
		if br.Error() == nil && typ & 0x80 == 0 {
			return &FormatError{off, "bad symbol type code", typ};
		}
		typ &^= 0x80;
		name := br.ReadCString();
		extraOff := int64(0);
		if typ == 'z' || typ == 'Z' {
			if name != "" {
				return &FormatError{off, "path symbol has non-empty name", name};
			}
			// Decode path entry
			for i := 0; ; i++ {
				eltIdx := uint32(br.ReadUint16());
				extraOff += 2;
				if eltIdx == 0 {
					break;
				}
				elt, ok := filenames[eltIdx];
				if !ok {
					return &FormatError{off, "bad filename code", eltIdx};
				}
				if name != "" && name[len(name)-1] != '/' {
					name += "/";
				}
				name += elt;
			}
		}
		gotype := br.ReadUint32();
		if err := br.Error(); err != nil {
			if err == os.EOF {
				err = io.ErrUnexpectedEOF;
			}
			return err;
		}

		off += 4 + 1 + int64(len(name)) + 1 + extraOff + 4;

		// Handle file name components
		if typ == 'f' {
			filenames[value] = name;
		}

		// Create the GoSym
		sym := growGoSyms(&t.Syms);

		switch typ {
		case 'T', 't':
			*sym = &TextSym{};
		case 'L', 'l':
			*sym = &LeafSym{};
		case 'D', 'd':
			*sym = &DataSym{};
		case 'B', 'b':
			*sym = &BSSSym{};
		case 'm':
			*sym = &FrameSym{};
		case 'a':
			*sym = &LocalSym{};
		case 'p':
			*sym = &ParamSym{};
		case 'z', 'Z':
			*sym = &PathSym{};
		default:
			*sym = &CommonSym{};
		}

		common := sym.Common();
		common.Value = uint64(value);
		common.Type = typ;
		common.Name = name;
		common.GoType = uint64(gotype);
	}

	return nil;
}

// byValue is a []*TextSym sorter.
type byValue []*TextSym

func (s byValue) Len() int {
	return len(s);
}

func (s byValue) Less(i, j int) bool {
	return s[i].Value < s[j].Value;
}

func (s byValue) Swap(i, j int) {
	t := s[i];
	s[i] = s[j];
	s[j] = t;
}

func (t *GoSymTable) processTextSyms() {
	// Count text symbols and attach frame sizes, parameters, and
	// locals to them.  Also, find object file boundaries.
	count := 0;
	var obj *object;
	var objCount int;
	for i := 0; i < len(t.Syms); i++ {
		switch sym := t.Syms[i].(type) {
		case *PathSym:
			// Finish the current object
			if obj != nil {
				obj.funcs = make([]*TextSym, 0, objCount);
			}

			// Count path symbols
			end := i+1;
			for ; end < len(t.Syms); end++ {
				_, ok := t.Syms[end].(*PathSym);
				if !ok {
					break;
				}
			}

			// Copy path symbols
			obj = &object{make([]*PathSym, end - i), nil};
			for j, s := range t.Syms[i:end] {
				obj.paths[j] = s.(*PathSym);
			}

			objCount = 0;
			i = end-1;

		case *TextSym:
			if sym.Name == "etext" {
				continue;
			}

			// Count parameter and local syms
			var np, nl int;
			end := i+1;
		countloop:
			for ; end < len(t.Syms); end++ {
				switch _ := t.Syms[end].(type) {
				// TODO(austin) Use type switch list
				case *TextSym:
					break countloop;
				case *PathSym:
					break countloop;
				case *ParamSym:
					np++;
				case *LocalSym:
					nl++;
				}
			}

			// Fill in the function symbol
			var ip, ia int;
			sym.obj = obj;
			sym.Params = make([]*ParamSym, np);
			sym.Locals = make([]*LocalSym, nl);
			for _, s := range t.Syms[i:end] {
				switch s := s.(type) {
				case *FrameSym:
					sym.FrameSize = int(s.Value);
				case *ParamSym:
					sym.Params[ip] = s;
					ip++;
				case *LocalSym:
					sym.Locals[ia] = s;
					ia++;
				}
			}

			count++;
			objCount++;
			i = end-1;
		}
	}

	if obj != nil {
		obj.funcs = make([]*TextSym, 0, objCount);
	}

	// Extract text symbols into function array and individual
	// object function arrys.
	t.funcs = make([]*TextSym, 0, count);
	for _, sym := range t.Syms {
		sym, ok := sym.(*TextSym);
		if !ok || sym.Name == "etext" {
			continue;
		}

		t.funcs = t.funcs[0:len(t.funcs)+1];
		t.funcs[len(t.funcs)-1] = sym;
		sym.obj.funcs = sym.obj.funcs[0:len(sym.obj.funcs)+1];
		sym.obj.funcs[len(sym.obj.funcs)-1] = sym;
	}

	// Sort text symbols
	sort.Sort(byValue(t.funcs));
}

func (t *GoSymTable) sliceLineTable(lt *lineTable) {
	for _, fn := range t.funcs {
		fn.lt = lt.slice(fn.Entry());
		lt = fn.lt;;
	}
}

// SymFromPC looks up a text symbol given a program counter within
// some function.  Returns nil if no function contains this PC.
func (t *GoSymTable) SymFromPC(pc uint64) *TextSym {
	syms := t.funcs;
	if pc > t.textEnd {
		return nil;
	}

	if len(syms) == 0 || pc < syms[0].Value {
		return nil;
	}
	if pc >= syms[len(syms)-1].Value {
		return syms[len(syms)-1];
	}

	l := 0;
	n := len(syms);
	for n > 0 {
		m := n/2;
		s := syms[l+m];
		switch {
		case s.Value <= pc && pc < syms[l+m+1].Value:
			return s;
		case pc < s.Value:
			n = m;
		default:
			l += m+1;
			n -= m+1;
		}
	}
	panic("not reached, pc=", pc);
}

// LineFromPC looks up line number information for a program counter.
// Returns a file path, a line number within that file, and the
// TextSym at pc.
func (t *GoSymTable) LineFromPC(pc uint64) (string, int, *TextSym) {
	sym := t.SymFromPC(pc);
	if sym == nil {
		return "", 0, nil;
	}

	aline := sym.lt.alineFromPC(pc);

	path, line := sym.obj.lineFromAline(aline);

	return path, line, sym;
}

// SymFromName looks up a symbol by name.  The name must refer to a
// global text, data, or BSS symbol.
func (t *GoSymTable) SymFromName(name string) GoSym {
	// TODO(austin) Maybe make a map
	for _, v := range t.Syms {
		c := v.Common();
		switch c.Type {
		case 'T', 't', 'L', 'l', 'D', 'd', 'B', 'b':
			if c.Name == name {
				return v;
			}
		}
	}
	return nil;
}

// SymFromAddr looks up a symbol by address.  The symbol will be a
// text, data, or BSS symbol.  addr must be the exact address of the
// symbol, unlike for SymFromPC.
func (t *GoSymTable) SymFromAddr(addr uint64) GoSym {
	// TODO(austin) Maybe make a map
	for _, v := range t.Syms {
		c := v.Common();
		switch c.Type {
		case 'T', 't', 'L', 'l', 'D', 'd', 'B', 'b':
			if c.Value == addr {
				return v;
			}
		}
	}
	return nil;
}

// TODO(austin) Implement PCFromLine.  This is more difficult because
// we first have to figure out which object file PC is in, and which
// segment of the line table that corresponds to.
//
// For each place path appears (either from push or pop),
// 1. Turn line into an absolute line number using the history stack
// 2. minpc = Entry of the first text sym in the object
// 3. maxpc = Entry of the first text sym in the next object
// 4. lt = lt.slice(minpc);
// 5. Find PC of first occurrence of absolute line number between minpc and maxpc
//
// I'm not sure if this guarantees a PC at the begining of an
// instruction.

/*
 * Object files
 */

func (o *object) lineFromAline(aline int) (string, int) {
	type stackEnt struct {
		path string;
		start int;
		offset int;
		prev *stackEnt;
	};

	noPath := &stackEnt{"<malformed absolute line>", 0, 0, nil};
	tos := noPath;

	// TODO(austin) I have no idea how 'Z' symbols work, except
	// that they pop the stack.
	for _, s := range o.paths {
		val := int(s.Value);
		switch {
		case val > aline:
			break;

		case val == 1:
			// Start a new stack
			tos = &stackEnt{s.Name, val, 0, noPath};

		case s.Name == "":
			// Pop
			if tos == noPath {
				return "<malformed symbol table>", 0;
			}
			tos.prev.offset += val - tos.start;
			tos = tos.prev;

		default:
			// Push
			tos = &stackEnt{s.Name, val, 0, tos};
		}
	}

	return tos.path, aline - tos.start - tos.offset + 1;
}

/*
 * Line tables
 */

func (lt *lineTable) parse(targetPC uint64) ([]byte, uint64, int) {
	// The PC/line table can be thought of as a sequence of
	//  <pc update>* <line update>
	// batches.  Each update batch results in a (pc, line) pair,
	// where line applies to every PC from pc up to but not
	// including the pc of the next pair.
	//
	// Here we process each update individually, which simplifies
	// the code, but makes the corner cases more confusing.

	const quantum = 1;
	b, pc, line := lt.blob, lt.pc, lt.line;
	for pc <= targetPC && len(b) != 0 {
		code := b[0];
		b = b[1:len(b)];
		switch {
		case code == 0:
			if len(b) < 4 {
				b = b[0:1];
				break;
			}
			val := msb.Uint32(b);
			b = b[4:len(b)];
			line += int(val);
		case code <= 64:
			line += int(code);
		case code <= 128:
			line -= int(code - 64);
		default:
			pc += quantum*uint64(code - 128);
			continue;
		}
		pc += quantum;
	}
	return b, pc, line;
}

func (lt *lineTable) slice(pc uint64) *lineTable {
	blob, pc, line := lt.parse(pc);
	return &lineTable{blob, pc, line};
}

func (lt *lineTable) alineFromPC(targetPC uint64) int {
	_1, _2, aline := lt.parse(targetPC);
	return aline;
}

/*
 * ELF
 */

func ElfGoSyms(elf *Elf) (*GoSymTable, os.Error) {
	text := elf.Section(".text");
	if text == nil {
		return nil, nil;
	}

	tab := &GoSymTable{textEnd: text.Addr + text.Size};

	// Symbol table
	sec := elf.Section(".gosymtab");
	if sec == nil {
		return nil, nil;
	}
	sr, err := sec.Open();
	if err != nil {
		return nil, err;
	}
	err = tab.readGoSymTab(sr);
	if err != nil {
		return nil, err;
	}

	// Line table
	sec = elf.Section(".gopclntab");
	if sec == nil {
		return nil, nil;
	}
	sr, err = sec.Open();
	if err != nil {
		return nil, err;
	}
	blob, err := io.ReadAll(sr);
	if err != nil {
		return nil, err;
	}
	lt := &lineTable{blob, text.Addr, 0};

	tab.processTextSyms();
	tab.sliceLineTable(lt);
	return tab, nil;
}
