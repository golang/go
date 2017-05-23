// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"fmt"
	"log"
	"strings"
)

type Plist struct {
	Name    *LSym
	Firstpc *Prog
	Recur   int
	Link    *Plist
}

/*
 * start a new Prog list.
 */
func Linknewplist(ctxt *Link) *Plist {
	pl := new(Plist)
	if ctxt.Plist == nil {
		ctxt.Plist = pl
	} else {
		ctxt.Plast.Link = pl
	}
	ctxt.Plast = pl
	return pl
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
