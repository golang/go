// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"fmt"
	"log"
	"path/filepath"
	"strings"
)

var outfile string

// The Go and C compilers, and the assembler, call writeobj to write
// out a Go object file.  The linker does not call this; the linker
// does not write out object files.
func Writeobjdirect(ctxt *Link, b *Biobuf) {

	var flag int
	var found int
	var h *Hist
	var s *LSym
	var text *LSym
	var etext *LSym
	var curtext *LSym
	var data *LSym
	var edata *LSym
	var pl *Plist
	var p *Prog
	var plink *Prog
	var a *Auto

	// Build list of symbols, and assign instructions to lists.
	// Ignore ctxt->plist boundaries. There are no guarantees there,
	// and the C compilers and assemblers just use one big list.
	text = nil

	curtext = nil
	data = nil
	etext = nil
	edata = nil
	for pl = ctxt.Plist; pl != nil; pl = pl.Link {
		for p = pl.Firstpc; p != nil; p = plink {
			if ctxt.Debugasm != 0 && ctxt.Debugvlog != 0 {
				fmt.Printf("obj: %p %v\n", p, p)
			}
			plink = p.Link
			p.Link = nil

			if int(p.As) == ctxt.Arch.AEND {
				continue
			}

			if int(p.As) == ctxt.Arch.ATYPE {
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
				a.Type = int16(ctxt.Arch.Symtype(&p.From))
				a.Gotype = p.From.Gotype
				a.Link = curtext.Autom
				curtext.Autom = a
				continue
			}

			if int(p.As) == ctxt.Arch.AGLOBL {
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
				if data == nil {
					data = s
				} else {

					edata.Next = s
				}
				s.Next = nil
				s.Size = p.To.Offset
				if s.Type == 0 || s.Type == SXREF {
					s.Type = SBSS
				}
				flag = ctxt.Arch.Textflag(p)
				if flag&DUPOK != 0 {
					s.Dupok = 1
				}
				if flag&RODATA != 0 {
					s.Type = SRODATA
				} else if flag&NOPTR != 0 {
					s.Type = SNOPTRBSS
				}
				edata = s
				continue
			}

			if int(p.As) == ctxt.Arch.ADATA {
				savedata(ctxt, p.From.Sym, p, "<input>")
				continue
			}

			if int(p.As) == ctxt.Arch.ATEXT {
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
				flag = ctxt.Arch.Textflag(p)
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

			if int(p.As) == ctxt.Arch.AFUNCDATA {
				// Rewrite reference to go_args_stackmap(SB) to the Go-provided declaration information.
				if curtext == nil { // func _() {}
					continue
				}
				if p.To.Sym.Name == "go_args_stackmap" {
					if int(p.From.Type) != ctxt.Arch.D_CONST || p.From.Offset != FUNCDATA_ArgsPointerMaps {
						ctxt.Diag("FUNCDATA use of go_args_stackmap(SB) without FUNCDATA_ArgsPointerMaps")
					}
					p.To.Sym = Linklookup(ctxt, string(fmt.Sprintf("%s.args_stackmap", curtext.Name)), int(curtext.Version))
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
	for s = text; s != nil; s = s.Next {

		if !strings.HasPrefix(s.Name, "\"\".") {
			continue
		}
		found = 0
		for p = s.Text; p != nil; p = p.Link {
			if int(p.As) == ctxt.Arch.AFUNCDATA && int(p.From.Type) == ctxt.Arch.D_CONST && p.From.Offset == FUNCDATA_ArgsPointerMaps {
				found = 1
				break
			}
		}

		if !(found != 0) {
			p = Appendp(ctxt, s.Text)
			p.As = int16(ctxt.Arch.AFUNCDATA)
			p.From.Type = int16(ctxt.Arch.D_CONST)
			p.From.Offset = FUNCDATA_ArgsPointerMaps
			if ctxt.Arch.Thechar == '6' || ctxt.Arch.Thechar == '8' {
				p.To.Type = int16(ctxt.Arch.D_EXTERN)
			} else {

				p.To.Type = int16(ctxt.Arch.D_OREG)
				p.To.Name = int8(ctxt.Arch.D_EXTERN)
			}

			p.To.Sym = Linklookup(ctxt, string(fmt.Sprintf("%s.args_stackmap", s.Name)), int(s.Version))
		}
	}

	// Turn functions into machine code images.
	for s = text; s != nil; s = s.Next {

		mkfwd(s)
		linkpatch(ctxt, s)
		ctxt.Arch.Follow(ctxt, s)
		ctxt.Arch.Addstacksplit(ctxt, s)
		ctxt.Arch.Assemble(ctxt, s)
		linkpcln(ctxt, s)
	}

	// Emit header.
	Bputc(b, 0)

	Bputc(b, 0)
	fmt.Fprintf(b, "go13ld")
	Bputc(b, 1) // version

	// Emit autolib.
	for h = ctxt.Hist; h != nil; h = h.Link {

		if h.Offset < 0 {
			wrstring(b, h.Name)
		}
	}
	wrstring(b, "")

	// Emit symbols.
	for s = text; s != nil; s = s.Next {

		writesym(ctxt, b, s)
	}
	for s = data; s != nil; s = s.Next {
		writesym(ctxt, b, s)
	}

	// Emit footer.
	Bputc(b, 0xff)

	Bputc(b, 0xff)
	fmt.Fprintf(b, "go13ld")
}

func writesym(ctxt *Link, b *Biobuf, s *LSym) {
	var r *Reloc
	var i int
	var j int
	var c int
	var n int
	var pc *Pcln
	var p *Prog
	var a *Auto
	var name string

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
		for p = s.Text; p != nil; p = p.Link {
			fmt.Fprintf(ctxt.Bso, "\t%#04x %v\n", uint(int(p.Pc)), p)
		}
		for i = 0; i < len(s.P); {
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

		for i = 0; i < len(s.R); i++ {
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
	wrint(b, int64(s.Dupok))
	wrint(b, s.Size)
	wrsym(b, s.Gotype)
	wrdata(b, s.P)

	wrint(b, int64(len(s.R)))
	for i = 0; i < len(s.R); i++ {
		r = &s.R[i]
		wrint(b, int64(r.Off))
		wrint(b, int64(r.Siz))
		wrint(b, int64(r.Type))
		wrint(b, r.Add)
		wrint(b, r.Xadd)
		wrsym(b, r.Sym)
		wrsym(b, r.Xsym)
	}

	if s.Type == STEXT {
		wrint(b, int64(s.Args))
		wrint(b, int64(s.Locals))
		wrint(b, int64(s.Nosplit))
		wrint(b, int64(s.Leaf)|int64(s.Cfunc)<<1)
		n = 0
		for a = s.Autom; a != nil; a = a.Link {
			n++
		}
		wrint(b, int64(n))
		for a = s.Autom; a != nil; a = a.Link {
			wrsym(b, a.Asym)
			wrint(b, int64(a.Aoffset))
			if int(a.Type) == ctxt.Arch.D_AUTO {
				wrint(b, A_AUTO)
			} else if int(a.Type) == ctxt.Arch.D_PARAM {
				wrint(b, A_PARAM)
			} else {

				log.Fatalf("%s: invalid local variable type %d", s.Name, a.Type)
			}
			wrsym(b, a.Gotype)
		}

		pc = s.Pcln
		wrdata(b, pc.Pcsp.P)
		wrdata(b, pc.Pcfile.P)
		wrdata(b, pc.Pcline.P)
		wrint(b, int64(len(pc.Pcdata)))
		for i = 0; i < len(pc.Pcdata); i++ {
			wrdata(b, pc.Pcdata[i].P)
		}
		wrint(b, int64(len(pc.Funcdataoff)))
		for i = 0; i < len(pc.Funcdataoff); i++ {
			wrsym(b, pc.Funcdata[i])
		}
		for i = 0; i < len(pc.Funcdataoff); i++ {
			wrint(b, pc.Funcdataoff[i])
		}
		wrint(b, int64(len(pc.File)))
		for i = 0; i < len(pc.File); i++ {
			wrpathsym(ctxt, b, pc.File[i])
		}
	}
}

func wrint(b *Biobuf, sval int64) {
	var uv uint64
	var v uint64
	var buf [10]uint8
	var p []uint8
	uv = (uint64(sval) << 1) ^ uint64(int64(sval>>63))
	p = buf[:]
	for v = uv; v >= 0x80; v >>= 7 {
		p[0] = uint8(v | 0x80)
		p = p[1:]
	}
	p[0] = uint8(v)
	p = p[1:]
	Bwrite(b, buf[:len(buf)-len(p)])
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
	Bwrite(b, v)
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

var startmagic string = "\x00\x00go13ld"

var endmagic string = "\xff\xffgo13ld"
