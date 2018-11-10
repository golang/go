// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing of Go object files.

package obj

import (
	"bufio"
	"cmd/internal/dwarf"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"log"
	"path/filepath"
	"sort"
)

// objWriter writes Go object files.
type objWriter struct {
	wr   *bufio.Writer
	ctxt *Link
	// Temporary buffer for zigzag int writing.
	varintbuf [10]uint8

	// Provide the index of a symbol reference by symbol name.
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

	if s.Type != objabi.STEXT {
		return
	}

	pc := &s.Func.Pcln

	data := 0
	data += len(pc.Pcsp.P)
	data += len(pc.Pcfile.P)
	data += len(pc.Pcline.P)
	data += len(pc.Pcinline.P)
	for i := 0; i < len(pc.Pcdata); i++ {
		data += len(pc.Pcdata[i].P)
	}

	w.nData += data
	w.nPcdata += len(pc.Pcdata)

	w.nAutom += len(s.Func.Autom)
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
	w.wr.WriteString("\x00\x00go19ld")

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
		pc := &s.Func.Pcln
		w.wr.Write(pc.Pcsp.P)
		w.wr.Write(pc.Pcfile.P)
		w.wr.Write(pc.Pcline.P)
		w.wr.Write(pc.Pcinline.P)
		for i := 0; i < len(pc.Pcdata); i++ {
			w.wr.Write(pc.Pcdata[i].P)
		}
	}
	for _, s := range ctxt.Data {
		if len(s.P) > 0 {
			switch s.Type {
			case objabi.SBSS, objabi.SNOPTRBSS, objabi.STLSBSS:
				ctxt.Diag("cannot provide data for %v sym %v", s.Type, s.Name)
			}
		}
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
	w.wr.WriteString("\xff\xffgo19ld")
}

// Symbols are prefixed so their content doesn't get confused with the magic footer.
const symPrefix = 0xfe

func (w *objWriter) writeRef(s *LSym, isPath bool) {
	if s == nil || s.RefIdx != 0 {
		return
	}
	var m map[string]int
	if !s.Static() {
		m = w.refIdx
	} else {
		m = w.vrefIdx
	}

	if idx := m[s.Name]; idx != 0 {
		s.RefIdx = idx
		return
	}
	w.wr.WriteByte(symPrefix)
	if isPath {
		w.writeString(filepath.ToSlash(s.Name))
	} else {
		w.writeString(s.Name)
	}
	// Write "version".
	if s.Static() {
		w.writeInt(1)
	} else {
		w.writeInt(0)
	}
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

	if s.Type == objabi.STEXT {
		for _, a := range s.Func.Autom {
			w.writeRef(a.Asym, false)
			w.writeRef(a.Gotype, false)
		}
		pc := &s.Func.Pcln
		for _, d := range pc.Funcdata {
			w.writeRef(d, false)
		}
		for _, f := range pc.File {
			fsym := w.ctxt.Lookup(f)
			w.writeRef(fsym, true)
		}
		for _, call := range pc.InlTree.nodes {
			w.writeRef(call.Func, false)
			f, _ := linkgetlineFromPos(w.ctxt, call.Pos)
			fsym := w.ctxt.Lookup(f)
			w.writeRef(fsym, true)
		}
	}
}

func (w *objWriter) writeSymDebug(s *LSym) {
	ctxt := w.ctxt
	fmt.Fprintf(ctxt.Bso, "%s ", s.Name)
	if s.Type != 0 {
		fmt.Fprintf(ctxt.Bso, "%v ", s.Type)
	}
	if s.Static() {
		fmt.Fprint(ctxt.Bso, "static ")
	}
	if s.DuplicateOK() {
		fmt.Fprintf(ctxt.Bso, "dupok ")
	}
	if s.CFunc() {
		fmt.Fprintf(ctxt.Bso, "cfunc ")
	}
	if s.NoSplit() {
		fmt.Fprintf(ctxt.Bso, "nosplit ")
	}
	fmt.Fprintf(ctxt.Bso, "size=%d", s.Size)
	if s.Type == objabi.STEXT {
		fmt.Fprintf(ctxt.Bso, " args=%#x locals=%#x", uint64(s.Func.Args), uint64(s.Func.Locals))
		if s.Leaf() {
			fmt.Fprintf(ctxt.Bso, " leaf")
		}
	}
	fmt.Fprintf(ctxt.Bso, "\n")
	if s.Type == objabi.STEXT {
		for p := s.Func.Text; p != nil; p = p.Link {
			fmt.Fprintf(ctxt.Bso, "\t%#04x %v\n", uint(int(p.Pc)), p)
		}
	}
	for i := 0; i < len(s.P); i += 16 {
		fmt.Fprintf(ctxt.Bso, "\t%#04x", uint(i))
		j := i
		for j = i; j < i+16 && j < len(s.P); j++ {
			fmt.Fprintf(ctxt.Bso, " %02x", s.P[j])
		}
		for ; j < i+16; j++ {
			fmt.Fprintf(ctxt.Bso, "   ")
		}
		fmt.Fprintf(ctxt.Bso, "  ")
		for j = i; j < i+16 && j < len(s.P); j++ {
			c := int(s.P[j])
			if ' ' <= c && c <= 0x7e {
				fmt.Fprintf(ctxt.Bso, "%c", c)
			} else {
				fmt.Fprintf(ctxt.Bso, ".")
			}
		}

		fmt.Fprintf(ctxt.Bso, "\n")
	}

	sort.Sort(relocByOff(s.R)) // generate stable output
	for _, r := range s.R {
		name := ""
		if r.Sym != nil {
			name = r.Sym.Name
		} else if r.Type == objabi.R_TLS_LE {
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
	if ctxt.Debugasm {
		w.writeSymDebug(s)
	}

	w.wr.WriteByte(symPrefix)
	w.wr.WriteByte(byte(s.Type))
	w.writeRefIndex(s)
	flags := int64(0)
	if s.DuplicateOK() {
		flags |= 1
	}
	if s.Local() {
		flags |= 1 << 1
	}
	if s.MakeTypelink() {
		flags |= 1 << 2
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

	if s.Type != objabi.STEXT {
		return
	}

	w.writeInt(int64(s.Func.Args))
	w.writeInt(int64(s.Func.Locals))
	if s.NoSplit() {
		w.writeInt(1)
	} else {
		w.writeInt(0)
	}
	flags = int64(0)
	if s.Leaf() {
		flags |= 1
	}
	if s.CFunc() {
		flags |= 1 << 1
	}
	if s.ReflectMethod() {
		flags |= 1 << 2
	}
	if ctxt.Flag_shared {
		flags |= 1 << 3
	}
	w.writeInt(flags)
	w.writeInt(int64(len(s.Func.Autom)))
	for _, a := range s.Func.Autom {
		w.writeRefIndex(a.Asym)
		w.writeInt(int64(a.Aoffset))
		if a.Name == NAME_AUTO {
			w.writeInt(objabi.A_AUTO)
		} else if a.Name == NAME_PARAM {
			w.writeInt(objabi.A_PARAM)
		} else {
			log.Fatalf("%s: invalid local variable type %d", s.Name, a.Name)
		}
		w.writeRefIndex(a.Gotype)
	}

	pc := &s.Func.Pcln
	w.writeInt(int64(len(pc.Pcsp.P)))
	w.writeInt(int64(len(pc.Pcfile.P)))
	w.writeInt(int64(len(pc.Pcline.P)))
	w.writeInt(int64(len(pc.Pcinline.P)))
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
		fsym := ctxt.Lookup(f)
		w.writeRefIndex(fsym)
	}
	w.writeInt(int64(len(pc.InlTree.nodes)))
	for _, call := range pc.InlTree.nodes {
		w.writeInt(int64(call.Parent))
		f, l := linkgetlineFromPos(w.ctxt, call.Pos)
		fsym := ctxt.Lookup(f)
		w.writeRefIndex(fsym)
		w.writeInt(int64(l))
		w.writeRefIndex(call.Func)
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

// implement dwarf.Context
type dwCtxt struct{ *Link }

func (c dwCtxt) PtrSize() int {
	return c.Arch.PtrSize
}
func (c dwCtxt) AddInt(s dwarf.Sym, size int, i int64) {
	ls := s.(*LSym)
	ls.WriteInt(c.Link, ls.Size, size, i)
}
func (c dwCtxt) AddBytes(s dwarf.Sym, b []byte) {
	ls := s.(*LSym)
	ls.WriteBytes(c.Link, ls.Size, b)
}
func (c dwCtxt) AddString(s dwarf.Sym, v string) {
	ls := s.(*LSym)
	ls.WriteString(c.Link, ls.Size, len(v), v)
	ls.WriteInt(c.Link, ls.Size, 1, 0)
}
func (c dwCtxt) SymValue(s dwarf.Sym) int64 {
	return 0
}
func (c dwCtxt) AddAddress(s dwarf.Sym, data interface{}, value int64) {
	ls := s.(*LSym)
	size := c.PtrSize()
	if data != nil {
		rsym := data.(*LSym)
		ls.WriteAddr(c.Link, ls.Size, size, rsym, value)
	} else {
		ls.WriteInt(c.Link, ls.Size, size, value)
	}
}
func (c dwCtxt) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
	ls := s.(*LSym)
	rsym := t.(*LSym)
	ls.WriteAddr(c.Link, ls.Size, size, rsym, ofs)
	r := &ls.R[len(ls.R)-1]
	r.Type = objabi.R_DWARFREF
}

// dwarfSym returns the DWARF symbols for TEXT symbol.
func (ctxt *Link) dwarfSym(s *LSym) (dwarfInfoSym, dwarfRangesSym *LSym) {
	if s.Type != objabi.STEXT {
		ctxt.Diag("dwarfSym of non-TEXT %v", s)
	}
	if s.Func.dwarfSym == nil {
		s.Func.dwarfSym = ctxt.LookupDerived(s, dwarf.InfoPrefix+s.Name)
		s.Func.dwarfRangesSym = ctxt.LookupDerived(s, dwarf.RangePrefix+s.Name)
	}
	return s.Func.dwarfSym, s.Func.dwarfRangesSym
}

func (s *LSym) Len() int64 {
	return s.Size
}

// populateDWARF fills in the DWARF Debugging Information Entries for TEXT symbol s.
// The DWARFs symbol must already have been initialized in InitTextSym.
func (ctxt *Link) populateDWARF(curfn interface{}, s *LSym) {
	dsym, drsym := ctxt.dwarfSym(s)
	if dsym.Size != 0 {
		ctxt.Diag("makeFuncDebugEntry double process %v", s)
	}
	var scopes []dwarf.Scope
	if ctxt.DebugInfo != nil {
		scopes = ctxt.DebugInfo(s, curfn)
	}
	err := dwarf.PutFunc(dwCtxt{ctxt}, dsym, drsym, s.Name, !s.Static(), s, s.Size, scopes)
	if err != nil {
		ctxt.Diag("emitting DWARF for %s failed: %v", s.Name, err)
	}
}
