// Copyright 2013 The Go Authors.  All rights reserved.
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
// An empty name corresponds to a nil LSym* pointer.
//
// Each symbol is laid out as the following fields (taken from LSym*):
//
//	- byte 0xfe (sanity check for synchronization)
//	- type [int]
//	- name [string]
//	- version [int]
//	- flags [int]
//		1 dupok
//	- size [int]
//	- gotype [symbol reference]
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
//		1 leaf
//		2 C function
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
//	- sym [symbol reference]
//	- xsym [symbol reference]
//
// Each local has the encoding:
//
//	- asym [symbol reference]
//	- offset [int]
//	- type [int]
//	- gotype [symbol reference]
//
// The pcln table has the encoding:
//
//	- pcsp [data block]
//	- pcfile [data block]
//	- pcline [data block]
//	- npcdata [int]
//	- pcdata [npcdata data blocks]
//	- nfuncdata [int]
//	- funcdata [nfuncdata symbol references]
//	- funcdatasym [nfuncdata ints]
//	- nfile [int]
//	- file [nfile symbol references]
//
// The file layout and meaning of type integers are architecture-independent.
//
// TODO(rsc): The file format is good for a first pass but needs work.
//	- There are SymID in the object file that should really just be strings.
//	- The actual symbol memory images are interlaced with the symbol
//	  metadata. They should be separated, to reduce the I/O required to
//	  load just the metadata.
//	- The symbol references should be shortened, either with a symbol
//	  table or by using a simple backward index to an earlier mentioned symbol.

package obj

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"
)

// The Go and C compilers, and the assembler, call writeobj to write
// out a Go object file.  The linker does not call this; the linker
// does not write out object files.
func Writeobjdirect(ctxt *Link, b *Biobuf) {
	Flushplist(ctxt)
	Writeobjfile(ctxt, b)
}

func Flushplist(ctxt *Link) {
	var flag int
	var s *LSym
	var p *Prog
	var plink *Prog
	var a *Auto

	// Build list of symbols, and assign instructions to lists.
	// Ignore ctxt->plist boundaries. There are no guarantees there,
	// and the assemblers just use one big list.
	var curtext *LSym
	var text *LSym
	var etext *LSym

	for pl := ctxt.Plist; pl != nil; pl = pl.Link {
		for p = pl.Firstpc; p != nil; p = plink {
			if ctxt.Debugasm != 0 && ctxt.Debugvlog != 0 {
				fmt.Printf("obj: %v\n", p)
			}
			plink = p.Link
			p.Link = nil

			if p.As == AEND {
				continue
			}

			if p.As == ATYPE {
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
				a = new(Auto)
				a.Asym = p.From.Sym
				a.Aoffset = int32(p.From.Offset)
				a.Name = int16(p.From.Name)
				a.Gotype = p.From.Gotype
				a.Link = curtext.Autom
				curtext.Autom = a
				continue
			}

			if p.As == AGLOBL {
				s = p.From.Sym
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
				flag = int(p.From3.Offset)
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
			}

			if p.As == ADATA {
				savedata(ctxt, p.From.Sym, p, "<input>")
				continue
			}

			if p.As == ATEXT {
				s = p.From.Sym
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
				flag = int(p.From3Offset())
				if flag&DUPOK != 0 {
					s.Dupok = 1
				}
				if flag&NOSPLIT != 0 {
					s.Nosplit = 1
				}
				s.Next = nil
				s.Type = STEXT
				s.Text = p
				s.Etext = p
				curtext = s
				continue
			}

			if p.As == AFUNCDATA {
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
			s = curtext
			s.Etext.Link = p
			s.Etext = p
		}
	}

	// Add reference to Go arguments for C or assembly functions without them.
	var found int
	for s := text; s != nil; s = s.Next {
		if !strings.HasPrefix(s.Name, "\"\".") {
			continue
		}
		found = 0
		for p = s.Text; p != nil; p = p.Link {
			if p.As == AFUNCDATA && p.From.Type == TYPE_CONST && p.From.Offset == FUNCDATA_ArgsPointerMaps {
				found = 1
				break
			}
		}

		if found == 0 {
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
		ctxt.Arch.Follow(ctxt, s)
		ctxt.Arch.Preprocess(ctxt, s)
		ctxt.Arch.Assemble(ctxt, s)
		linkpcln(ctxt, s)
	}

	// Add to running list in ctxt.
	if ctxt.Etext == nil {
		ctxt.Text = text
	} else {
		ctxt.Etext.Next = text
	}
	ctxt.Etext = etext
	ctxt.Plist = nil
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

		var r *Reloc
		var name string
		for i := 0; i < len(s.R); i++ {
			r = &s.R[i]
			name = ""
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
	wrstring(b, s.Name)
	wrint(b, int64(s.Version))
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
		wrint(b, int64(s.Leaf)|int64(s.Cfunc)<<1)
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
		for i := 0; i < len(pc.File); i++ {
			wrpathsym(ctxt, b, pc.File[i])
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

// wrpath writes a path just like a string, but on windows, it
// translates '\\' to '/' in the process.
func wrpath(ctxt *Link, b *Biobuf, p string) {
	wrstring(b, filepath.ToSlash(p))
}

func wrdata(b *Biobuf, v []byte) {
	wrint(b, int64(len(v)))
	b.Write(v)
}

func wrpathsym(ctxt *Link, b *Biobuf, s *LSym) {
	if s == nil {
		wrint(b, 0)
		wrint(b, 0)
		return
	}

	wrpath(ctxt, b, s.Name)
	wrint(b, int64(s.Version))
}

func wrsym(b *Biobuf, s *LSym) {
	if s == nil {
		wrint(b, 0)
		wrint(b, 0)
		return
	}

	wrstring(b, s.Name)
	wrint(b, int64(s.Version))
}
