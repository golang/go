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
//	- sequence of sybol references used by the defined symbols
//	- byte 0xff (marks end of sequence)
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
//	- xadd [int]
//	- sym [symref index]
//	- xsym [symref index]
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
//	- The actual symbol memory images are interlaced with the symbol
//	  metadata. They should be separated, to reduce the I/O required to
//	  load just the metadata.

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
	var curtext, text, etext *LSym

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
				tmp6 := s.Seenglobl
				s.Seenglobl++
				if tmp6 != 0 {
					fmt.Printf("duplicate %v\n", p)
				}
				if s.Onlist != 0 {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Onlist = 1
				if ctxt.Data == nil {
					ctxt.Data = s
				} else {
					ctxt.Edata.Next = s
				}
				s.Next = nil
				s.Size = p.To.Offset
				if s.Type == 0 || s.Type == SXREF {
					s.Type = SBSS
				}
				flag := int(p.From3.Offset)
				if flag&DUPOK != 0 {
					s.Dupok = 1
				}
				if flag&RODATA != 0 {
					s.Type = SRODATA
				} else if flag&NOPTR != 0 {
					s.Type = SNOPTRBSS
				} else if flag&TLSBSS != 0 {
					s.Type = STLSBSS
				}
				ctxt.Edata = s
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
				if s.Onlist != 0 {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Onlist = 1
				if text == nil {
					text = s
				} else {
					etext.Next = s
				}
				etext = s
				flag := int(p.From3Offset())
				if flag&DUPOK != 0 {
					s.Dupok = 1
				}
				if flag&NOSPLIT != 0 {
					s.Nosplit = 1
				}
				if flag&REFLECTMETHOD != 0 {
					s.ReflectMethod = true
				}
				s.Next = nil
				s.Type = STEXT
				s.Text = p
				s.Etext = p
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
				continue
			}
			s := curtext
			s.Etext.Link = p
			s.Etext = p
		}
	}

	// Add reference to Go arguments for C or assembly functions without them.
	for s := text; s != nil; s = s.Next {
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
	for s := text; s != nil; s = s.Next {
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
			s.Etext = nil
		}
	}

	// Add to running list in ctxt.
	if text != nil {
		if ctxt.Text == nil {
			ctxt.Text = text
		} else {
			ctxt.Etext.Next = text
		}
		ctxt.Etext = etext
	}
	ctxt.Plist = nil
	ctxt.Plast = nil
	ctxt.Curp = nil
	if freeProgs {
		ctxt.freeProgs()
	}
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

	// Emit symbol references.
	for s := ctxt.Text; s != nil; s = s.Next {
		writerefs(ctxt, b, s)
	}
	for s := ctxt.Data; s != nil; s = s.Next {
		writerefs(ctxt, b, s)
	}
	Bputc(b, 0xff)

	// Emit symbols.
	for s := ctxt.Text; s != nil; s = s.Next {
		writesym(ctxt, b, s)
	}
	for s := ctxt.Data; s != nil; s = s.Next {
		writesym(ctxt, b, s)
	}

	// Emit footer.
	Bputc(b, 0xff)

	Bputc(b, 0xff)
	fmt.Fprintf(b, "go13ld")
}

func wrref(ctxt *Link, b *Biobuf, s *LSym, isPath bool) {
	if s == nil || s.RefIdx != 0 {
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
		if s.Dupok != 0 {
			fmt.Fprintf(ctxt.Bso, "dupok ")
		}
		if s.Cfunc != 0 {
			fmt.Fprintf(ctxt.Bso, "cfunc ")
		}
		if s.Nosplit != 0 {
			fmt.Fprintf(ctxt.Bso, "nosplit ")
		}
		fmt.Fprintf(ctxt.Bso, "size=%d value=%d", int64(s.Size), int64(s.Value))
		if s.Type == STEXT {
			fmt.Fprintf(ctxt.Bso, " args=%#x locals=%#x", uint64(s.Args), uint64(s.Locals))
			if s.Leaf != 0 {
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
	flags := int64(s.Dupok)
	if s.Local {
		flags |= 2
	}
	wrint(b, flags)
	wrint(b, s.Size)
	wrsym(b, s.Gotype)
	wrdata(b, s.P)

	wrint(b, int64(len(s.R)))
	var r *Reloc
	for i := 0; i < len(s.R); i++ {
		r = &s.R[i]
		wrint(b, int64(r.Off))
		wrint(b, int64(r.Siz))
		wrint(b, int64(r.Type))
		wrint(b, r.Add)
		wrint(b, 0) // Xadd, ignored
		wrsym(b, r.Sym)
		wrsym(b, nil) // Xsym, ignored
	}

	if s.Type == STEXT {
		wrint(b, int64(s.Args))
		wrint(b, int64(s.Locals))
		wrint(b, int64(s.Nosplit))
		flags := int64(s.Leaf) | int64(s.Cfunc)<<1
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
		wrdata(b, pc.Pcsp.P)
		wrdata(b, pc.Pcfile.P)
		wrdata(b, pc.Pcline.P)
		wrint(b, int64(len(pc.Pcdata)))
		for i := 0; i < len(pc.Pcdata); i++ {
			wrdata(b, pc.Pcdata[i].P)
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

func wrdata(b *Biobuf, v []byte) {
	wrint(b, int64(len(v)))
	b.Write(v)
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
