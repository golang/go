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
//	- magic header: "\x00\x00go13ld"
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
//	- magic footer: "\xff\xffgo13ld"
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
	"fmt"
	"log"
	"path/filepath"
	"sort"
	"strings"
)

// The Go and C compilers, and the assembler, call writeobj to write
// out a Go object file. The linker does not call this; the linker
// does not write out object files.
func Writeobjdirect(ctxt *Link, b *Biobuf) {
	Flushplist(ctxt)
	Writeobjfile(ctxt, b)
}

func Flushplist(ctxt *Link) {
	flushplist(ctxt, ctxt.Debugasm == 0)
}
func FlushplistNoFree(ctxt *Link) {
	flushplist(ctxt, false)
}
func flushplist(ctxt *Link, freeProgs bool) {
	// Build list of symbols, and assign instructions to lists.
	// Ignore ctxt->plist boundaries. There are no guarantees there,
	// and the assemblers just use one big list.
	var curtext *LSym
	var etext *Prog
	var text []*LSym

	for pl := ctxt.Plist; pl != nil; pl = pl.Link {
		var plink *Prog
		for p := pl.Firstpc; p != nil; p = plink {
			if ctxt.Debugasm != 0 && ctxt.Debugvlog != 0 {
				fmt.Printf("obj: %v\n", p)
			}
			plink = p.Link
			p.Link = nil

			switch p.As {
			case AEND:
				continue

			case ATYPE:
				// Assume each TYPE instruction describes
				// a different local variable or parameter,
				// so no dedup.
				// Using only the TYPE instructions means
				// that we discard location information about local variables
				// in C and assembly functions; that information is inferred
				// from ordinary references, because there are no TYPE
				// instructions there. Without the type information, gdb can't
				// use the locations, so we don't bother to save them.
				// If something else could use them, we could arrange to
				// preserve them.
				if curtext == nil {
					continue
				}
				a := new(Auto)
				a.Asym = p.From.Sym
				a.Aoffset = int32(p.From.Offset)
				a.Name = int16(p.From.Name)
				a.Gotype = p.From.Gotype
				a.Link = curtext.Autom
				curtext.Autom = a
				continue

			case AGLOBL:
				s := p.From.Sym
				if s.Seenglobl {
					fmt.Printf("duplicate %v\n", p)
				}
				s.Seenglobl = true
				if s.Onlist {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Onlist = true
				ctxt.Data = append(ctxt.Data, s)
				s.Size = p.To.Offset
				if s.Type == 0 || s.Type == SXREF {
					s.Type = SBSS
				}
				flag := int(p.From3.Offset)
				if flag&DUPOK != 0 {
					s.Dupok = true
				}
				if flag&RODATA != 0 {
					s.Type = SRODATA
				} else if flag&NOPTR != 0 {
					s.Type = SNOPTRBSS
				} else if flag&TLSBSS != 0 {
					s.Type = STLSBSS
				}
				continue

			case ATEXT:
				s := p.From.Sym
				if s == nil {
					// func _() { }
					curtext = nil

					continue
				}

				if s.Text != nil {
					log.Fatalf("duplicate TEXT for %s", s.Name)
				}
				if s.Onlist {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Onlist = true
				text = append(text, s)
				flag := int(p.From3Offset())
				if flag&DUPOK != 0 {
					s.Dupok = true
				}
				if flag&NOSPLIT != 0 {
					s.Nosplit = true
				}
				if flag&REFLECTMETHOD != 0 {
					s.ReflectMethod = true
				}
				s.Type = STEXT
				s.Text = p
				etext = p
				curtext = s
				continue

			case AFUNCDATA:
				// Rewrite reference to go_args_stackmap(SB) to the Go-provided declaration information.
				if curtext == nil { // func _() {}
					continue
				}
				if p.To.Sym.Name == "go_args_stackmap" {
					if p.From.Type != TYPE_CONST || p.From.Offset != FUNCDATA_ArgsPointerMaps {
						ctxt.Diag("FUNCDATA use of go_args_stackmap(SB) without FUNCDATA_ArgsPointerMaps")
					}
					p.To.Sym = Linklookup(ctxt, fmt.Sprintf("%s.args_stackmap", curtext.Name), int(curtext.Version))
				}

			}

			if curtext == nil {
				etext = nil
				continue
			}
			etext.Link = p
			etext = p
		}
	}

	// Add reference to Go arguments for C or assembly functions without them.
	for _, s := range text {
		if !strings.HasPrefix(s.Name, "\"\".") {
			continue
		}
		found := false
		var p *Prog
		for p = s.Text; p != nil; p = p.Link {
			if p.As == AFUNCDATA && p.From.Type == TYPE_CONST && p.From.Offset == FUNCDATA_ArgsPointerMaps {
				found = true
				break
			}
		}

		if !found {
			p = Appendp(ctxt, s.Text)
			p.As = AFUNCDATA
			p.From.Type = TYPE_CONST
			p.From.Offset = FUNCDATA_ArgsPointerMaps
			p.To.Type = TYPE_MEM
			p.To.Name = NAME_EXTERN
			p.To.Sym = Linklookup(ctxt, fmt.Sprintf("%s.args_stackmap", s.Name), int(s.Version))
		}
	}

	// Turn functions into machine code images.
	for _, s := range text {
		mkfwd(s)
		linkpatch(ctxt, s)
		if ctxt.Flag_optimize {
			ctxt.Arch.Follow(ctxt, s)
		}
		ctxt.Arch.Preprocess(ctxt, s)
		ctxt.Arch.Assemble(ctxt, s)
		fieldtrack(ctxt, s)
		linkpcln(ctxt, s)
		if freeProgs {
			s.Text = nil
		}
	}

	// Add to running list in ctxt.
	ctxt.Text = append(ctxt.Text, text...)
	ctxt.Plist = nil
	ctxt.Plast = nil
	ctxt.Curp = nil
	if freeProgs {
		ctxt.freeProgs()
	}
}

type sectionLengths struct {
	data     int
	reloc    int
	pcdata   int
	autom    int
	funcdata int
	file     int
}

func (l *sectionLengths) add(s *LSym) {
	l.data += len(s.P)
	l.reloc += len(s.R)

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

	l.data += data
	l.pcdata += len(pc.Pcdata)

	autom := 0
	for a := s.Autom; a != nil; a = a.Link {
		autom++
	}
	l.autom += autom
	l.funcdata += len(pc.Funcdataoff)
	l.file += len(pc.File)
}

func wrlengths(b *Biobuf, sl sectionLengths) {
	wrint(b, int64(sl.data))
	wrint(b, int64(sl.reloc))
	wrint(b, int64(sl.pcdata))
	wrint(b, int64(sl.autom))
	wrint(b, int64(sl.funcdata))
	wrint(b, int64(sl.file))
}

func Writeobjfile(ctxt *Link, b *Biobuf) {
	// Emit header.
	Bputc(b, 0)

	Bputc(b, 0)
	fmt.Fprintf(b, "go13ld")
	Bputc(b, 1) // version

	// Emit autolib.
	for _, pkg := range ctxt.Imports {
		wrstring(b, pkg)
	}
	wrstring(b, "")

	var lengths sectionLengths

	// Emit symbol references.
	for _, s := range ctxt.Text {
		writerefs(ctxt, b, s)
		lengths.add(s)
	}
	for _, s := range ctxt.Data {
		writerefs(ctxt, b, s)
		lengths.add(s)
	}
	Bputc(b, 0xff)

	wrlengths(b, lengths)

	// Write data block
	for _, s := range ctxt.Text {
		b.w.Write(s.P)
		pc := s.Pcln
		b.w.Write(pc.Pcsp.P)
		b.w.Write(pc.Pcfile.P)
		b.w.Write(pc.Pcline.P)
		for i := 0; i < len(pc.Pcdata); i++ {
			b.w.Write(pc.Pcdata[i].P)
		}
	}
	for _, s := range ctxt.Data {
		b.w.Write(s.P)
	}

	// Emit symbols.
	for _, s := range ctxt.Text {
		writesym(ctxt, b, s)
	}
	for _, s := range ctxt.Data {
		writesym(ctxt, b, s)
	}

	// Emit footer.
	Bputc(b, 0xff)

	Bputc(b, 0xff)
	fmt.Fprintf(b, "go13ld")
}

// Provide the the index of a symbol reference by symbol name.
// One map for versioned symbols and one for unversioned symbols.
// Used for deduplicating the symbol reference list.
var refIdx = make(map[string]int)
var vrefIdx = make(map[string]int)

func wrref(ctxt *Link, b *Biobuf, s *LSym, isPath bool) {
	if s == nil || s.RefIdx != 0 {
		return
	}
	var m map[string]int
	switch s.Version {
	case 0:
		m = refIdx
	case 1:
		m = vrefIdx
	default:
		log.Fatalf("%s: invalid version number %d", s.Name, s.Version)
	}

	idx := m[s.Name]
	if idx != 0 {
		s.RefIdx = idx
		return
	}
	Bputc(b, 0xfe)
	if isPath {
		wrstring(b, filepath.ToSlash(s.Name))
	} else {
		wrstring(b, s.Name)
	}
	wrint(b, int64(s.Version))
	ctxt.RefsWritten++
	s.RefIdx = ctxt.RefsWritten
	m[s.Name] = ctxt.RefsWritten
}

func writerefs(ctxt *Link, b *Biobuf, s *LSym) {
	wrref(ctxt, b, s, false)
	wrref(ctxt, b, s.Gotype, false)
	for i := range s.R {
		wrref(ctxt, b, s.R[i].Sym, false)
	}

	if s.Type == STEXT {
		for a := s.Autom; a != nil; a = a.Link {
			wrref(ctxt, b, a.Asym, false)
			wrref(ctxt, b, a.Gotype, false)
		}
		pc := s.Pcln
		for _, d := range pc.Funcdata {
			wrref(ctxt, b, d, false)
		}
		for _, f := range pc.File {
			wrref(ctxt, b, f, true)
		}
	}
}

func writesym(ctxt *Link, b *Biobuf, s *LSym) {
	if ctxt.Debugasm != 0 {
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
			}
			if ctxt.Arch.Thechar == '5' || ctxt.Arch.Thechar == '9' {
				fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s+%x\n", int(r.Off), r.Siz, r.Type, name, uint64(int64(r.Add)))
			} else {
				fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s+%d\n", int(r.Off), r.Siz, r.Type, name, int64(r.Add))
			}
		}
	}

	Bputc(b, 0xfe)
	wrint(b, int64(s.Type))
	wrsym(b, s)
	flags := int64(0)
	if s.Dupok {
		flags |= 1
	}
	if s.Local {
		flags |= 1 << 1
	}
	wrint(b, flags)
	wrint(b, s.Size)
	wrsym(b, s.Gotype)
	wrint(b, int64(len(s.P)))

	wrint(b, int64(len(s.R)))
	var r *Reloc
	for i := 0; i < len(s.R); i++ {
		r = &s.R[i]
		wrint(b, int64(r.Off))
		wrint(b, int64(r.Siz))
		wrint(b, int64(r.Type))
		wrint(b, r.Add)
		wrsym(b, r.Sym)
	}

	if s.Type == STEXT {
		wrint(b, int64(s.Args))
		wrint(b, int64(s.Locals))
		if s.Nosplit {
			wrint(b, 1)
		} else {
			wrint(b, 0)
		}
		flags := int64(0)
		if s.Leaf {
			flags |= 1
		}
		if s.Cfunc {
			flags |= 1 << 1
		}
		if s.ReflectMethod {
			flags |= 1 << 2
		}
		wrint(b, flags)
		n := 0
		for a := s.Autom; a != nil; a = a.Link {
			n++
		}
		wrint(b, int64(n))
		for a := s.Autom; a != nil; a = a.Link {
			wrsym(b, a.Asym)
			wrint(b, int64(a.Aoffset))
			if a.Name == NAME_AUTO {
				wrint(b, A_AUTO)
			} else if a.Name == NAME_PARAM {
				wrint(b, A_PARAM)
			} else {
				log.Fatalf("%s: invalid local variable type %d", s.Name, a.Name)
			}
			wrsym(b, a.Gotype)
		}

		pc := s.Pcln
		wrint(b, int64(len(pc.Pcsp.P)))
		wrint(b, int64(len(pc.Pcfile.P)))
		wrint(b, int64(len(pc.Pcline.P)))
		wrint(b, int64(len(pc.Pcdata)))
		for i := 0; i < len(pc.Pcdata); i++ {
			wrint(b, int64(len(pc.Pcdata[i].P)))
		}
		wrint(b, int64(len(pc.Funcdataoff)))
		for i := 0; i < len(pc.Funcdataoff); i++ {
			wrsym(b, pc.Funcdata[i])
		}
		for i := 0; i < len(pc.Funcdataoff); i++ {
			wrint(b, pc.Funcdataoff[i])
		}
		wrint(b, int64(len(pc.File)))
		for _, f := range pc.File {
			wrsym(b, f)
		}
	}
}

// Reusable buffer to avoid allocations.
// This buffer was responsible for 15% of gc's allocations.
var varintbuf [10]uint8

func wrint(b *Biobuf, sval int64) {
	var v uint64
	uv := (uint64(sval) << 1) ^ uint64(int64(sval>>63))
	p := varintbuf[:]
	for v = uv; v >= 0x80; v >>= 7 {
		p[0] = uint8(v | 0x80)
		p = p[1:]
	}
	p[0] = uint8(v)
	p = p[1:]
	b.Write(varintbuf[:len(varintbuf)-len(p)])
}

func wrstring(b *Biobuf, s string) {
	wrint(b, int64(len(s)))
	b.w.WriteString(s)
}

func wrsym(b *Biobuf, s *LSym) {
	if s == nil {
		wrint(b, 0)
		return
	}
	if s.RefIdx == 0 {
		log.Fatalln("writing an unreferenced symbol", s.Name)
	}
	wrint(b, int64(s.RefIdx))
}

// relocByOff sorts relocations by their offsets.
type relocByOff []Reloc

func (x relocByOff) Len() int           { return len(x) }
func (x relocByOff) Less(i, j int) bool { return x[i].Off < x[j].Off }
func (x relocByOff) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
