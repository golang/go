// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing of Go object files.
//
// Originally, Go object files were Plan 9 object files, but no longer.
// Now they are more like standard object files, in that each symbol is defined
// by an associated memory image (bytes) and a list of relocations to apply
// during linking. We do not (yet?) use a standard file format, however.
// For now, the format is chosen to be as simple as possible to read and write.
// It may change for reasons of efficiency, or we may even switch to a
// standard file format if there are compelling benefits to doing so.
// See golang.org/s/go13linker for more background.
//
// The file format is:
//
//	- magic header: "\x00\x00go17ld"
//	- byte 1 - version number
//	- sequence of strings giving dependencies (imported packages)
//	- empty string (marks end of sequence)
//	- sequence of symbol references used by the defined symbols
//	- byte 0xff (marks end of sequence)
//	- sequence of integer lengths:
//		- total data length
//		- total number of relocations
//		- total number of pcdata
//		- total number of automatics
//		- total number of funcdata
//		- total number of files
//	- data, the content of the defined symbols
//	- sequence of defined symbols
//	- byte 0xff (marks end of sequence)
//	- magic footer: "\xff\xffgo17ld"
//
// All integers are stored in a zigzag varint format.
// See golang.org/s/go12symtab for a definition.
//
// Data blocks and strings are both stored as an integer
// followed by that many bytes.
//
// A symbol reference is a string name followed by a version.
//
// A symbol points to other symbols using an index into the symbol
// reference sequence. Index 0 corresponds to a nil LSym* pointer.
// In the symbol layout described below "symref index" stands for this
// index.
//
// Each symbol is laid out as the following fields (taken from LSym*):
//
//	- byte 0xfe (sanity check for synchronization)
//	- type [int]
//	- name & version [symref index]
//	- flags [int]
//		1 dupok
//	- size [int]
//	- gotype [symref index]
//	- p [data block]
//	- nr [int]
//	- r [nr relocations, sorted by off]
//
// If type == STEXT, there are a few more fields:
//
//	- args [int]
//	- locals [int]
//	- nosplit [int]
//	- flags [int]
//		1<<0 leaf
//		1<<1 C function
//		1<<2 function may call reflect.Type.Method
//	- nlocal [int]
//	- local [nlocal automatics]
//	- pcln [pcln table]
//
// Each relocation has the encoding:
//
//	- off [int]
//	- siz [int]
//	- type [int]
//	- add [int]
//	- sym [symref index]
//
// Each local has the encoding:
//
//	- asym [symref index]
//	- offset [int]
//	- type [int]
//	- gotype [symref index]
//
// The pcln table has the encoding:
//
//	- pcsp [data block]
//	- pcfile [data block]
//	- pcline [data block]
//	- npcdata [int]
//	- pcdata [npcdata data blocks]
//	- nfuncdata [int]
//	- funcdata [nfuncdata symref index]
//	- funcdatasym [nfuncdata ints]
//	- nfile [int]
//	- file [nfile symref index]
//
// The file layout and meaning of type integers are architecture-independent.
//
// TODO(rsc): The file format is good for a first pass but needs work.
//	- There are SymID in the object file that should really just be strings.

package obj

import (
	"bufio"
	"cmd/internal/sys"
	"fmt"
	"log"
	"path/filepath"
	"sort"
)

// The Go and C compilers, and the assembler, call writeobj to write
// out a Go object file. The linker does not call this; the linker
// does not write out object files.
func Writeobjdirect(ctxt *Link, b *bufio.Writer) {
	Flushplist(ctxt)
	WriteObjFile(ctxt, b)
}

// objWriter writes Go object files.
type objWriter struct {
	wr   *bufio.Writer
	ctxt *Link
	// Temporary buffer for zigzag int writing.
	varintbuf [10]uint8

	// Provide the the index of a symbol reference by symbol name.
	// One map for versioned symbols and one for unversioned symbols.
	// Used for deduplicating the symbol reference list.
	refIdx  map[string]int
	vrefIdx map[string]int

	// Number of objects written of each type.
	nRefs     int
	nData     int
	nReloc    int
	nPcdata   int
	nAutom    int
	nFuncdata int
	nFile     int
}

func (w *objWriter) addLengths(s *LSym) {
	w.nData += len(s.P)
	w.nReloc += len(s.R)

	if s.Type != STEXT {
		return
	}

	pc := s.Pcln

	data := 0
	data += len(pc.Pcsp.P)
	data += len(pc.Pcfile.P)
	data += len(pc.Pcline.P)
	for i := 0; i < len(pc.Pcdata); i++ {
		data += len(pc.Pcdata[i].P)
	}

	w.nData += data
	w.nPcdata += len(pc.Pcdata)

	autom := 0
	for a := s.Autom; a != nil; a = a.Link {
		autom++
	}
	w.nAutom += autom
	w.nFuncdata += len(pc.Funcdataoff)
	w.nFile += len(pc.File)
}

func (w *objWriter) writeLengths() {
	w.writeInt(int64(w.nData))
	w.writeInt(int64(w.nReloc))
	w.writeInt(int64(w.nPcdata))
	w.writeInt(int64(w.nAutom))
	w.writeInt(int64(w.nFuncdata))
	w.writeInt(int64(w.nFile))
}

func newObjWriter(ctxt *Link, b *bufio.Writer) *objWriter {
	return &objWriter{
		ctxt:    ctxt,
		wr:      b,
		vrefIdx: make(map[string]int),
		refIdx:  make(map[string]int),
	}
}

func WriteObjFile(ctxt *Link, b *bufio.Writer) {
	w := newObjWriter(ctxt, b)

	// Magic header
	w.wr.WriteString("\x00\x00go17ld")

	// Version
	w.wr.WriteByte(1)

	// Autolib
	for _, pkg := range ctxt.Imports {
		w.writeString(pkg)
	}
	w.writeString("")

	// Symbol references
	for _, s := range ctxt.Text {
		w.writeRefs(s)
		w.addLengths(s)
	}
	for _, s := range ctxt.Data {
		w.writeRefs(s)
		w.addLengths(s)
	}
	// End symbol references
	w.wr.WriteByte(0xff)

	// Lengths
	w.writeLengths()

	// Data block
	for _, s := range ctxt.Text {
		w.wr.Write(s.P)
		pc := s.Pcln
		w.wr.Write(pc.Pcsp.P)
		w.wr.Write(pc.Pcfile.P)
		w.wr.Write(pc.Pcline.P)
		for i := 0; i < len(pc.Pcdata); i++ {
			w.wr.Write(pc.Pcdata[i].P)
		}
	}
	for _, s := range ctxt.Data {
		w.wr.Write(s.P)
	}

	// Symbols
	for _, s := range ctxt.Text {
		w.writeSym(s)
	}
	for _, s := range ctxt.Data {
		w.writeSym(s)
	}

	// Magic footer
	w.wr.WriteString("\xff\xffgo17ld")
}

// Symbols are prefixed so their content doesn't get confused with the magic footer.
const symPrefix = 0xfe

func (w *objWriter) writeRef(s *LSym, isPath bool) {
	if s == nil || s.RefIdx != 0 {
		return
	}
	var m map[string]int
	switch s.Version {
	case 0:
		m = w.refIdx
	case 1:
		m = w.vrefIdx
	default:
		log.Fatalf("%s: invalid version number %d", s.Name, s.Version)
	}

	idx := m[s.Name]
	if idx != 0 {
		s.RefIdx = idx
		return
	}
	w.wr.WriteByte(symPrefix)
	if isPath {
		w.writeString(filepath.ToSlash(s.Name))
	} else {
		w.writeString(s.Name)
	}
	w.writeInt(int64(s.Version))
	w.nRefs++
	s.RefIdx = w.nRefs
	m[s.Name] = w.nRefs
}

func (w *objWriter) writeRefs(s *LSym) {
	w.writeRef(s, false)
	w.writeRef(s.Gotype, false)
	for i := range s.R {
		w.writeRef(s.R[i].Sym, false)
	}

	if s.Type == STEXT {
		for a := s.Autom; a != nil; a = a.Link {
			w.writeRef(a.Asym, false)
			w.writeRef(a.Gotype, false)
		}
		pc := s.Pcln
		for _, d := range pc.Funcdata {
			w.writeRef(d, false)
		}
		for _, f := range pc.File {
			w.writeRef(f, true)
		}
	}
}

func (w *objWriter) writeSymDebug(s *LSym) {
	ctxt := w.ctxt
	fmt.Fprintf(ctxt.Bso, "%s ", s.Name)
	if s.Version != 0 {
		fmt.Fprintf(ctxt.Bso, "v=%d ", s.Version)
	}
	if s.Type != 0 {
		fmt.Fprintf(ctxt.Bso, "t=%d ", s.Type)
	}
	if s.Dupok {
		fmt.Fprintf(ctxt.Bso, "dupok ")
	}
	if s.Cfunc {
		fmt.Fprintf(ctxt.Bso, "cfunc ")
	}
	if s.Nosplit {
		fmt.Fprintf(ctxt.Bso, "nosplit ")
	}
	fmt.Fprintf(ctxt.Bso, "size=%d", s.Size)
	if s.Type == STEXT {
		fmt.Fprintf(ctxt.Bso, " args=%#x locals=%#x", uint64(s.Args), uint64(s.Locals))
		if s.Leaf {
			fmt.Fprintf(ctxt.Bso, " leaf")
		}
	}

	fmt.Fprintf(ctxt.Bso, "\n")
	for p := s.Text; p != nil; p = p.Link {
		fmt.Fprintf(ctxt.Bso, "\t%#04x %v\n", uint(int(p.Pc)), p)
	}
	var c int
	var j int
	for i := 0; i < len(s.P); {
		fmt.Fprintf(ctxt.Bso, "\t%#04x", uint(i))
		for j = i; j < i+16 && j < len(s.P); j++ {
			fmt.Fprintf(ctxt.Bso, " %02x", s.P[j])
		}
		for ; j < i+16; j++ {
			fmt.Fprintf(ctxt.Bso, "   ")
		}
		fmt.Fprintf(ctxt.Bso, "  ")
		for j = i; j < i+16 && j < len(s.P); j++ {
			c = int(s.P[j])
			if ' ' <= c && c <= 0x7e {
				fmt.Fprintf(ctxt.Bso, "%c", c)
			} else {
				fmt.Fprintf(ctxt.Bso, ".")
			}
		}

		fmt.Fprintf(ctxt.Bso, "\n")
		i += 16
	}

	sort.Sort(relocByOff(s.R)) // generate stable output
	for _, r := range s.R {
		name := ""
		if r.Sym != nil {
			name = r.Sym.Name
		} else if r.Type == R_TLS_LE {
			name = "TLS"
		}
		if ctxt.Arch.InFamily(sys.ARM, sys.PPC64) {
			fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s+%x\n", int(r.Off), r.Siz, r.Type, name, uint64(r.Add))
		} else {
			fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s+%d\n", int(r.Off), r.Siz, r.Type, name, r.Add)
		}
	}
}

func (w *objWriter) writeSym(s *LSym) {
	ctxt := w.ctxt
	if ctxt.Debugasm != 0 {
		w.writeSymDebug(s)
	}

	w.wr.WriteByte(symPrefix)
	w.writeInt(int64(s.Type))
	w.writeRefIndex(s)
	flags := int64(0)
	if s.Dupok {
		flags |= 1
	}
	if s.Local {
		flags |= 1 << 1
	}
	w.writeInt(flags)
	w.writeInt(s.Size)
	w.writeRefIndex(s.Gotype)
	w.writeInt(int64(len(s.P)))

	w.writeInt(int64(len(s.R)))
	var r *Reloc
	for i := 0; i < len(s.R); i++ {
		r = &s.R[i]
		w.writeInt(int64(r.Off))
		w.writeInt(int64(r.Siz))
		w.writeInt(int64(r.Type))
		w.writeInt(r.Add)
		w.writeRefIndex(r.Sym)
	}

	if s.Type != STEXT {
		return
	}

	w.writeInt(int64(s.Args))
	w.writeInt(int64(s.Locals))
	if s.Nosplit {
		w.writeInt(1)
	} else {
		w.writeInt(0)
	}
	flags = int64(0)
	if s.Leaf {
		flags |= 1
	}
	if s.Cfunc {
		flags |= 1 << 1
	}
	if s.ReflectMethod {
		flags |= 1 << 2
	}
	w.writeInt(flags)
	n := 0
	for a := s.Autom; a != nil; a = a.Link {
		n++
	}
	w.writeInt(int64(n))
	for a := s.Autom; a != nil; a = a.Link {
		w.writeRefIndex(a.Asym)
		w.writeInt(int64(a.Aoffset))
		if a.Name == NAME_AUTO {
			w.writeInt(A_AUTO)
		} else if a.Name == NAME_PARAM {
			w.writeInt(A_PARAM)
		} else {
			log.Fatalf("%s: invalid local variable type %d", s.Name, a.Name)
		}
		w.writeRefIndex(a.Gotype)
	}

	pc := s.Pcln
	w.writeInt(int64(len(pc.Pcsp.P)))
	w.writeInt(int64(len(pc.Pcfile.P)))
	w.writeInt(int64(len(pc.Pcline.P)))
	w.writeInt(int64(len(pc.Pcdata)))
	for i := 0; i < len(pc.Pcdata); i++ {
		w.writeInt(int64(len(pc.Pcdata[i].P)))
	}
	w.writeInt(int64(len(pc.Funcdataoff)))
	for i := 0; i < len(pc.Funcdataoff); i++ {
		w.writeRefIndex(pc.Funcdata[i])
	}
	for i := 0; i < len(pc.Funcdataoff); i++ {
		w.writeInt(pc.Funcdataoff[i])
	}
	w.writeInt(int64(len(pc.File)))
	for _, f := range pc.File {
		w.writeRefIndex(f)
	}
}

func (w *objWriter) writeInt(sval int64) {
	var v uint64
	uv := (uint64(sval) << 1) ^ uint64(sval>>63)
	p := w.varintbuf[:]
	for v = uv; v >= 0x80; v >>= 7 {
		p[0] = uint8(v | 0x80)
		p = p[1:]
	}
	p[0] = uint8(v)
	p = p[1:]
	w.wr.Write(w.varintbuf[:len(w.varintbuf)-len(p)])
}

func (w *objWriter) writeString(s string) {
	w.writeInt(int64(len(s)))
	w.wr.WriteString(s)
}

func (w *objWriter) writeRefIndex(s *LSym) {
	if s == nil {
		w.writeInt(0)
		return
	}
	if s.RefIdx == 0 {
		log.Fatalln("writing an unreferenced symbol", s.Name)
	}
	w.writeInt(int64(s.RefIdx))
}

// relocByOff sorts relocations by their offsets.
type relocByOff []Reloc

func (x relocByOff) Len() int           { return len(x) }
func (x relocByOff) Less(i, j int) bool { return x[i].Off < x[j].Off }
func (x relocByOff) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
