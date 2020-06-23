// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing of Go object files.

package obj

import (
	"bufio"
	"cmd/internal/bio"
	"cmd/internal/dwarf"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"fmt"
	"io"
	"log"
	"path/filepath"
	"sort"
	"strings"
	"sync"
)

// objWriter writes Go object files.
type objWriter struct {
	wr   *bufio.Writer
	ctxt *Link
	// Temporary buffer for zigzag int writing.
	varintbuf [10]uint8

	// Number of objects written of each type.
	nRefs     int
	nData     int
	nReloc    int
	nPcdata   int
	nFuncdata int
	nFile     int

	pkgpath string // the package import path (escaped), "" if unknown
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
	for _, pcd := range pc.Pcdata {
		data += len(pcd.P)
	}

	w.nData += data
	w.nPcdata += len(pc.Pcdata)

	w.nFuncdata += len(pc.Funcdataoff)
	w.nFile += len(pc.File)
}

func (w *objWriter) writeLengths() {
	w.writeInt(int64(w.nData))
	w.writeInt(int64(w.nReloc))
	w.writeInt(int64(w.nPcdata))
	w.writeInt(int64(0)) // TODO: remove at next object file rev
	w.writeInt(int64(w.nFuncdata))
	w.writeInt(int64(w.nFile))
}

func newObjWriter(ctxt *Link, b *bufio.Writer, pkgpath string) *objWriter {
	return &objWriter{
		ctxt:    ctxt,
		wr:      b,
		pkgpath: objabi.PathToPrefix(pkgpath),
	}
}

func WriteObjFile(ctxt *Link, bout *bio.Writer, pkgpath string) {
	if ctxt.Flag_go115newobj {
		WriteObjFile2(ctxt, bout, pkgpath)
		return
	}

	b := bout.Writer
	w := newObjWriter(ctxt, b, pkgpath)

	// Magic header
	w.wr.WriteString("\x00go114ld")

	// Version
	w.wr.WriteByte(1)

	// Autolib
	for _, p := range ctxt.Imports {
		w.writeString(p.Pkg)
		// This object format ignores p.Fingerprint.
	}
	w.writeString("")

	// DWARF File Table
	fileTable := ctxt.PosTable.DebugLinesFileTable()
	w.writeInt(int64(len(fileTable)))
	for _, str := range fileTable {
		w.writeString(filepath.ToSlash(str))
	}

	// Symbol references
	for _, s := range ctxt.Text {
		w.writeRefs(s)
		w.addLengths(s)
	}

	if ctxt.Headtype == objabi.Haix {
		// Data must be sorted to keep a constant order in TOC symbols.
		// As they are created during Progedit, two symbols can be switched between
		// two different compilations. Therefore, BuildID will be different.
		// TODO: find a better place and optimize to only sort TOC symbols
		sort.Slice(ctxt.Data, func(i, j int) bool {
			return ctxt.Data[i].Name < ctxt.Data[j].Name
		})
	}

	for _, s := range ctxt.Data {
		w.writeRefs(s)
		w.addLengths(s)
	}
	for _, s := range ctxt.ABIAliases {
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
		for _, pcd := range pc.Pcdata {
			w.wr.Write(pcd.P)
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
	for _, s := range ctxt.ABIAliases {
		w.writeSym(s)
	}

	// Magic footer
	w.wr.WriteString("\xffgo114ld")
}

// Symbols are prefixed so their content doesn't get confused with the magic footer.
const symPrefix = 0xfe

func (w *objWriter) writeRef(s *LSym, isPath bool) {
	if s == nil || s.RefIdx != 0 {
		return
	}
	w.wr.WriteByte(symPrefix)
	if isPath {
		w.writeString(filepath.ToSlash(s.Name))
	} else if w.pkgpath != "" {
		// w.pkgpath is already escaped.
		n := strings.Replace(s.Name, "\"\".", w.pkgpath+".", -1)
		w.writeString(n)
	} else {
		w.writeString(s.Name)
	}
	// Write ABI/static information.
	abi := int64(s.ABI())
	if s.Static() {
		abi = -1
	}
	w.writeInt(abi)
	w.nRefs++
	s.RefIdx = w.nRefs
}

func (w *objWriter) writeRefs(s *LSym) {
	w.writeRef(s, false)
	w.writeRef(s.Gotype, false)
	for _, r := range s.R {
		w.writeRef(r.Sym, false)
	}

	if s.Type == objabi.STEXT {
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

func (ctxt *Link) writeSymDebug(s *LSym) {
	ctxt.writeSymDebugNamed(s, s.Name)
}

func (ctxt *Link) writeSymDebugNamed(s *LSym, name string) {
	fmt.Fprintf(ctxt.Bso, "%s ", name)
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
	if s.TopFrame() {
		fmt.Fprintf(ctxt.Bso, "topframe ")
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
			fmt.Fprintf(ctxt.Bso, "\t%#04x ", uint(int(p.Pc)))
			if ctxt.Debugasm > 1 {
				io.WriteString(ctxt.Bso, p.String())
			} else {
				p.InnermostString(ctxt.Bso)
			}
			fmt.Fprintln(ctxt.Bso)
		}
	}
	for i := 0; i < len(s.P); i += 16 {
		fmt.Fprintf(ctxt.Bso, "\t%#04x", uint(i))
		j := i
		for ; j < i+16 && j < len(s.P); j++ {
			fmt.Fprintf(ctxt.Bso, " %02x", s.P[j])
		}
		for ; j < i+16; j++ {
			fmt.Fprintf(ctxt.Bso, "   ")
		}
		fmt.Fprintf(ctxt.Bso, "  ")
		for j = i; j < i+16 && j < len(s.P); j++ {
			c := int(s.P[j])
			b := byte('.')
			if ' ' <= c && c <= 0x7e {
				b = byte(c)
			}
			ctxt.Bso.WriteByte(b)
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
	if ctxt.Debugasm > 0 {
		w.ctxt.writeSymDebug(s)
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
	for i := range s.R {
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
	w.writeInt(int64(s.Func.Align))
	w.writeBool(s.NoSplit())
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
	if s.TopFrame() {
		flags |= 1 << 4
	}
	w.writeInt(flags)
	w.writeInt(int64(0)) // TODO: remove at next object file rev

	pc := &s.Func.Pcln
	w.writeInt(int64(len(pc.Pcsp.P)))
	w.writeInt(int64(len(pc.Pcfile.P)))
	w.writeInt(int64(len(pc.Pcline.P)))
	w.writeInt(int64(len(pc.Pcinline.P)))
	w.writeInt(int64(len(pc.Pcdata)))
	for _, pcd := range pc.Pcdata {
		w.writeInt(int64(len(pcd.P)))
	}
	w.writeInt(int64(len(pc.Funcdataoff)))
	for i := range pc.Funcdataoff {
		w.writeRefIndex(pc.Funcdata[i])
	}
	for i := range pc.Funcdataoff {
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
		w.writeInt(int64(call.ParentPC))
	}
}

func (w *objWriter) writeBool(b bool) {
	if b {
		w.writeInt(1)
	} else {
		w.writeInt(0)
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
func (c dwCtxt) AddUint16(s dwarf.Sym, i uint16) {
	c.AddInt(s, 2, int64(i))
}
func (c dwCtxt) AddUint8(s dwarf.Sym, i uint8) {
	b := []byte{byte(i)}
	c.AddBytes(s, b)
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
func (c dwCtxt) AddCURelativeAddress(s dwarf.Sym, data interface{}, value int64) {
	ls := s.(*LSym)
	rsym := data.(*LSym)
	ls.WriteCURelativeAddr(c.Link, ls.Size, rsym, value)
}
func (c dwCtxt) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
	panic("should be used only in the linker")
}
func (c dwCtxt) AddDWARFAddrSectionOffset(s dwarf.Sym, t interface{}, ofs int64) {
	size := 4
	if isDwarf64(c.Link) {
		size = 8
	}

	ls := s.(*LSym)
	rsym := t.(*LSym)
	ls.WriteAddr(c.Link, ls.Size, size, rsym, ofs)
	r := &ls.R[len(ls.R)-1]
	r.Type = objabi.R_DWARFSECREF
}

func (c dwCtxt) AddFileRef(s dwarf.Sym, f interface{}) {
	ls := s.(*LSym)
	rsym := f.(*LSym)
	if c.Link.Flag_go115newobj {
		fidx := c.Link.PosTable.FileIndex(rsym.Name)
		// Note the +1 here -- the value we're writing is going to be an
		// index into the DWARF line table file section, whose entries
		// are numbered starting at 1, not 0.
		ls.WriteInt(c.Link, ls.Size, 4, int64(fidx+1))
	} else {
		ls.WriteAddr(c.Link, ls.Size, 4, rsym, 0)
		r := &ls.R[len(ls.R)-1]
		r.Type = objabi.R_DWARFFILEREF
	}
}

func (c dwCtxt) CurrentOffset(s dwarf.Sym) int64 {
	ls := s.(*LSym)
	return ls.Size
}

// Here "from" is a symbol corresponding to an inlined or concrete
// function, "to" is the symbol for the corresponding abstract
// function, and "dclIdx" is the index of the symbol of interest with
// respect to the Dcl slice of the original pre-optimization version
// of the inlined function.
func (c dwCtxt) RecordDclReference(from dwarf.Sym, to dwarf.Sym, dclIdx int, inlIndex int) {
	ls := from.(*LSym)
	tls := to.(*LSym)
	ridx := len(ls.R) - 1
	c.Link.DwFixups.ReferenceChildDIE(ls, ridx, tls, dclIdx, inlIndex)
}

func (c dwCtxt) RecordChildDieOffsets(s dwarf.Sym, vars []*dwarf.Var, offsets []int32) {
	ls := s.(*LSym)
	c.Link.DwFixups.RegisterChildDIEOffsets(ls, vars, offsets)
}

func (c dwCtxt) Logf(format string, args ...interface{}) {
	c.Link.Logf(format, args...)
}

func isDwarf64(ctxt *Link) bool {
	return ctxt.Headtype == objabi.Haix
}

func (ctxt *Link) dwarfSym(s *LSym) (dwarfInfoSym, dwarfLocSym, dwarfRangesSym, dwarfAbsFnSym, dwarfDebugLines *LSym) {
	if s.Type != objabi.STEXT {
		ctxt.Diag("dwarfSym of non-TEXT %v", s)
	}
	if s.Func.dwarfInfoSym == nil {
		if ctxt.Flag_go115newobj {
			s.Func.dwarfInfoSym = &LSym{
				Type: objabi.SDWARFINFO,
			}
			if ctxt.Flag_locationlists {
				s.Func.dwarfLocSym = &LSym{
					Type: objabi.SDWARFLOC,
				}
			}
			s.Func.dwarfRangesSym = &LSym{
				Type: objabi.SDWARFRANGE,
			}
			s.Func.dwarfDebugLinesSym = &LSym{
				Type: objabi.SDWARFLINES,
			}
		} else {
			s.Func.dwarfInfoSym = ctxt.LookupDerived(s, dwarf.InfoPrefix+s.Name)
			if ctxt.Flag_locationlists {
				s.Func.dwarfLocSym = ctxt.LookupDerived(s, dwarf.LocPrefix+s.Name)
			}
			s.Func.dwarfRangesSym = ctxt.LookupDerived(s, dwarf.RangePrefix+s.Name)
			s.Func.dwarfDebugLinesSym = ctxt.LookupDerived(s, dwarf.DebugLinesPrefix+s.Name)
		}
		if s.WasInlined() {
			s.Func.dwarfAbsFnSym = ctxt.DwFixups.AbsFuncDwarfSym(s)
		}
	}
	return s.Func.dwarfInfoSym, s.Func.dwarfLocSym, s.Func.dwarfRangesSym, s.Func.dwarfAbsFnSym, s.Func.dwarfDebugLinesSym
}

func (s *LSym) Length(dwarfContext interface{}) int64 {
	return s.Size
}

// fileSymbol returns a symbol corresponding to the source file of the
// first instruction (prog) of the specified function. This will
// presumably be the file in which the function is defined.
func (ctxt *Link) fileSymbol(fn *LSym) *LSym {
	p := fn.Func.Text
	if p != nil {
		f, _ := linkgetlineFromPos(ctxt, p.Pos)
		fsym := ctxt.Lookup(f)
		return fsym
	}
	return nil
}

// populateDWARF fills in the DWARF Debugging Information Entries for
// TEXT symbol 's'. The various DWARF symbols must already have been
// initialized in InitTextSym.
func (ctxt *Link) populateDWARF(curfn interface{}, s *LSym, myimportpath string) {
	info, loc, ranges, absfunc, lines := ctxt.dwarfSym(s)
	if info.Size != 0 {
		ctxt.Diag("makeFuncDebugEntry double process %v", s)
	}
	var scopes []dwarf.Scope
	var inlcalls dwarf.InlCalls
	if ctxt.DebugInfo != nil {
		scopes, inlcalls = ctxt.DebugInfo(s, info, curfn)
	}
	var err error
	dwctxt := dwCtxt{ctxt}
	filesym := ctxt.fileSymbol(s)
	fnstate := &dwarf.FnState{
		Name:          s.Name,
		Importpath:    myimportpath,
		Info:          info,
		Filesym:       filesym,
		Loc:           loc,
		Ranges:        ranges,
		Absfn:         absfunc,
		StartPC:       s,
		Size:          s.Size,
		External:      !s.Static(),
		Scopes:        scopes,
		InlCalls:      inlcalls,
		UseBASEntries: ctxt.UseBASEntries,
	}
	if absfunc != nil {
		err = dwarf.PutAbstractFunc(dwctxt, fnstate)
		if err != nil {
			ctxt.Diag("emitting DWARF for %s failed: %v", s.Name, err)
		}
		err = dwarf.PutConcreteFunc(dwctxt, fnstate)
	} else {
		err = dwarf.PutDefaultFunc(dwctxt, fnstate)
	}
	if err != nil {
		ctxt.Diag("emitting DWARF for %s failed: %v", s.Name, err)
	}
	// Fill in the debug lines symbol.
	ctxt.generateDebugLinesSymbol(s, lines)
}

// DwarfIntConst creates a link symbol for an integer constant with the
// given name, type and value.
func (ctxt *Link) DwarfIntConst(myimportpath, name, typename string, val int64) {
	if myimportpath == "" {
		return
	}
	s := ctxt.LookupInit(dwarf.ConstInfoPrefix+myimportpath, func(s *LSym) {
		s.Type = objabi.SDWARFINFO
		ctxt.Data = append(ctxt.Data, s)
	})
	dwarf.PutIntConst(dwCtxt{ctxt}, s, ctxt.Lookup(dwarf.InfoPrefix+typename), myimportpath+"."+name, val)
}

func (ctxt *Link) DwarfAbstractFunc(curfn interface{}, s *LSym, myimportpath string) {
	absfn := ctxt.DwFixups.AbsFuncDwarfSym(s)
	if absfn.Size != 0 {
		ctxt.Diag("internal error: DwarfAbstractFunc double process %v", s)
	}
	if s.Func == nil {
		s.Func = new(FuncInfo)
	}
	scopes, _ := ctxt.DebugInfo(s, absfn, curfn)
	dwctxt := dwCtxt{ctxt}
	filesym := ctxt.fileSymbol(s)
	fnstate := dwarf.FnState{
		Name:          s.Name,
		Importpath:    myimportpath,
		Info:          absfn,
		Filesym:       filesym,
		Absfn:         absfn,
		External:      !s.Static(),
		Scopes:        scopes,
		UseBASEntries: ctxt.UseBASEntries,
	}
	if err := dwarf.PutAbstractFunc(dwctxt, &fnstate); err != nil {
		ctxt.Diag("emitting DWARF for %s failed: %v", s.Name, err)
	}
}

// This table is designed to aid in the creation of references between
// DWARF subprogram DIEs.
//
// In most cases when one DWARF DIE has to refer to another DWARF DIE,
// the target of the reference has an LSym, which makes it easy to use
// the existing relocation mechanism. For DWARF inlined routine DIEs,
// however, the subprogram DIE has to refer to a child
// parameter/variable DIE of the abstract subprogram. This child DIE
// doesn't have an LSym, and also of interest is the fact that when
// DWARF generation is happening for inlined function F within caller
// G, it's possible that DWARF generation hasn't happened yet for F,
// so there is no way to know the offset of a child DIE within F's
// abstract function. Making matters more complex, each inlined
// instance of F may refer to a subset of the original F's variables
// (depending on what happens with optimization, some vars may be
// eliminated).
//
// The fixup table below helps overcome this hurdle. At the point
// where a parameter/variable reference is made (via a call to
// "ReferenceChildDIE"), a fixup record is generate that records
// the relocation that is targeting that child variable. At a later
// point when the abstract function DIE is emitted, there will be
// a call to "RegisterChildDIEOffsets", at which point the offsets
// needed to apply fixups are captured. Finally, once the parallel
// portion of the compilation is done, fixups can actually be applied
// during the "Finalize" method (this can't be done during the
// parallel portion of the compile due to the possibility of data
// races).
//
// This table is also used to record the "precursor" function node for
// each function that is the target of an inline -- child DIE references
// have to be made with respect to the original pre-optimization
// version of the function (to allow for the fact that each inlined
// body may be optimized differently).
type DwarfFixupTable struct {
	ctxt      *Link
	mu        sync.Mutex
	symtab    map[*LSym]int // maps abstract fn LSYM to index in svec
	svec      []symFixups
	precursor map[*LSym]fnState // maps fn Lsym to precursor Node, absfn sym
}

type symFixups struct {
	fixups   []relFixup
	doffsets []declOffset
	inlIndex int32
	defseen  bool
}

type declOffset struct {
	// Index of variable within DCL list of pre-optimization function
	dclIdx int32
	// Offset of var's child DIE with respect to containing subprogram DIE
	offset int32
}

type relFixup struct {
	refsym *LSym
	relidx int32
	dclidx int32
}

type fnState struct {
	// precursor function (really *gc.Node)
	precursor interface{}
	// abstract function symbol
	absfn *LSym
}

func NewDwarfFixupTable(ctxt *Link) *DwarfFixupTable {
	return &DwarfFixupTable{
		ctxt:      ctxt,
		symtab:    make(map[*LSym]int),
		precursor: make(map[*LSym]fnState),
	}
}

func (ft *DwarfFixupTable) GetPrecursorFunc(s *LSym) interface{} {
	if fnstate, found := ft.precursor[s]; found {
		return fnstate.precursor
	}
	return nil
}

func (ft *DwarfFixupTable) SetPrecursorFunc(s *LSym, fn interface{}) {
	if _, found := ft.precursor[s]; found {
		ft.ctxt.Diag("internal error: DwarfFixupTable.SetPrecursorFunc double call on %v", s)
	}

	// initialize abstract function symbol now. This is done here so
	// as to avoid data races later on during the parallel portion of
	// the back end.
	absfn := ft.ctxt.LookupDerived(s, dwarf.InfoPrefix+s.Name+dwarf.AbstractFuncSuffix)
	absfn.Set(AttrDuplicateOK, true)
	absfn.Type = objabi.SDWARFINFO
	ft.ctxt.Data = append(ft.ctxt.Data, absfn)

	// In the case of "late" inlining (inlines that happen during
	// wrapper generation as opposed to the main inlining phase) it's
	// possible that we didn't cache the abstract function sym for the
	// text symbol -- do so now if needed. See issue 38068.
	if s.Func != nil && s.Func.dwarfAbsFnSym == nil {
		s.Func.dwarfAbsFnSym = absfn
	}

	ft.precursor[s] = fnState{precursor: fn, absfn: absfn}
}

// Make a note of a child DIE reference: relocation 'ridx' within symbol 's'
// is targeting child 'c' of DIE with symbol 'tgt'.
func (ft *DwarfFixupTable) ReferenceChildDIE(s *LSym, ridx int, tgt *LSym, dclidx int, inlIndex int) {
	// Protect against concurrent access if multiple backend workers
	ft.mu.Lock()
	defer ft.mu.Unlock()

	// Create entry for symbol if not already present.
	idx, found := ft.symtab[tgt]
	if !found {
		ft.svec = append(ft.svec, symFixups{inlIndex: int32(inlIndex)})
		idx = len(ft.svec) - 1
		ft.symtab[tgt] = idx
	}

	// Do we have child DIE offsets available? If so, then apply them,
	// otherwise create a fixup record.
	sf := &ft.svec[idx]
	if len(sf.doffsets) > 0 {
		found := false
		for _, do := range sf.doffsets {
			if do.dclIdx == int32(dclidx) {
				off := do.offset
				s.R[ridx].Add += int64(off)
				found = true
				break
			}
		}
		if !found {
			ft.ctxt.Diag("internal error: DwarfFixupTable.ReferenceChildDIE unable to locate child DIE offset for dclIdx=%d src=%v tgt=%v", dclidx, s, tgt)
		}
	} else {
		sf.fixups = append(sf.fixups, relFixup{s, int32(ridx), int32(dclidx)})
	}
}

// Called once DWARF generation is complete for a given abstract function,
// whose children might have been referenced via a call above. Stores
// the offsets for any child DIEs (vars, params) so that they can be
// consumed later in on DwarfFixupTable.Finalize, which applies any
// outstanding fixups.
func (ft *DwarfFixupTable) RegisterChildDIEOffsets(s *LSym, vars []*dwarf.Var, coffsets []int32) {
	// Length of these two slices should agree
	if len(vars) != len(coffsets) {
		ft.ctxt.Diag("internal error: RegisterChildDIEOffsets vars/offsets length mismatch")
		return
	}

	// Generate the slice of declOffset's based in vars/coffsets
	doffsets := make([]declOffset, len(coffsets))
	for i := range coffsets {
		doffsets[i].dclIdx = vars[i].ChildIndex
		doffsets[i].offset = coffsets[i]
	}

	ft.mu.Lock()
	defer ft.mu.Unlock()

	// Store offsets for this symbol.
	idx, found := ft.symtab[s]
	if !found {
		sf := symFixups{inlIndex: -1, defseen: true, doffsets: doffsets}
		ft.svec = append(ft.svec, sf)
		ft.symtab[s] = len(ft.svec) - 1
	} else {
		sf := &ft.svec[idx]
		sf.doffsets = doffsets
		sf.defseen = true
	}
}

func (ft *DwarfFixupTable) processFixups(slot int, s *LSym) {
	sf := &ft.svec[slot]
	for _, f := range sf.fixups {
		dfound := false
		for _, doffset := range sf.doffsets {
			if doffset.dclIdx == f.dclidx {
				f.refsym.R[f.relidx].Add += int64(doffset.offset)
				dfound = true
				break
			}
		}
		if !dfound {
			ft.ctxt.Diag("internal error: DwarfFixupTable has orphaned fixup on %v targeting %v relidx=%d dclidx=%d", f.refsym, s, f.relidx, f.dclidx)
		}
	}
}

// return the LSym corresponding to the 'abstract subprogram' DWARF
// info entry for a function.
func (ft *DwarfFixupTable) AbsFuncDwarfSym(fnsym *LSym) *LSym {
	// Protect against concurrent access if multiple backend workers
	ft.mu.Lock()
	defer ft.mu.Unlock()

	if fnstate, found := ft.precursor[fnsym]; found {
		return fnstate.absfn
	}
	ft.ctxt.Diag("internal error: AbsFuncDwarfSym requested for %v, not seen during inlining", fnsym)
	return nil
}

// Called after all functions have been compiled; the main job of this
// function is to identify cases where there are outstanding fixups.
// This scenario crops up when we have references to variables of an
// inlined routine, but that routine is defined in some other package.
// This helper walks through and locate these fixups, then invokes a
// helper to create an abstract subprogram DIE for each one.
func (ft *DwarfFixupTable) Finalize(myimportpath string, trace bool) {
	if trace {
		ft.ctxt.Logf("DwarfFixupTable.Finalize invoked for %s\n", myimportpath)
	}

	// Collect up the keys from the precursor map, then sort the
	// resulting list (don't want to rely on map ordering here).
	fns := make([]*LSym, len(ft.precursor))
	idx := 0
	for fn := range ft.precursor {
		fns[idx] = fn
		idx++
	}
	sort.Sort(BySymName(fns))

	// Should not be called during parallel portion of compilation.
	if ft.ctxt.InParallel {
		ft.ctxt.Diag("internal error: DwarfFixupTable.Finalize call during parallel backend")
	}

	// Generate any missing abstract functions.
	for _, s := range fns {
		absfn := ft.AbsFuncDwarfSym(s)
		slot, found := ft.symtab[absfn]
		if !found || !ft.svec[slot].defseen {
			ft.ctxt.GenAbstractFunc(s)
		}
	}

	// Apply fixups.
	for _, s := range fns {
		absfn := ft.AbsFuncDwarfSym(s)
		slot, found := ft.symtab[absfn]
		if !found {
			ft.ctxt.Diag("internal error: DwarfFixupTable.Finalize orphan abstract function for %v", s)
		} else {
			ft.processFixups(slot, s)
		}
	}
}

type BySymName []*LSym

func (s BySymName) Len() int           { return len(s) }
func (s BySymName) Less(i, j int) bool { return s[i].Name < s[j].Name }
func (s BySymName) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
