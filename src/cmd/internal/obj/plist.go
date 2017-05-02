// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"cmd/internal/objabi"
	"fmt"
	"strings"
)

type Plist struct {
	Firstpc *Prog
	Curfn   interface{} // holds a *gc.Node, if non-nil
}

// ProgAlloc is a function that allocates Progs.
// It is used to provide access to cached/bulk-allocated Progs to the assemblers.
type ProgAlloc func() *Prog

func Flushplist(ctxt *Link, plist *Plist, newprog ProgAlloc) {
	// Build list of symbols, and assign instructions to lists.
	var curtext *LSym
	var etext *Prog
	var text []*LSym

	var plink *Prog
	for p := plist.Firstpc; p != nil; p = plink {
		if ctxt.Debugasm && ctxt.Debugvlog {
			fmt.Printf("obj: %v\n", p)
		}
		plink = p.Link
		p.Link = nil

		switch p.As {
		case AEND:
			continue

		case ATEXT:
			s := p.From.Sym
			if s == nil {
				// func _() { }
				curtext = nil
				continue
			}
			text = append(text, s)
			etext = p
			curtext = s
			continue

		case AFUNCDATA:
			// Rewrite reference to go_args_stackmap(SB) to the Go-provided declaration information.
			if curtext == nil { // func _() {}
				continue
			}
			if p.To.Sym.Name == "go_args_stackmap" {
				if p.From.Type != TYPE_CONST || p.From.Offset != objabi.FUNCDATA_ArgsPointerMaps {
					ctxt.Diag("FUNCDATA use of go_args_stackmap(SB) without FUNCDATA_ArgsPointerMaps")
				}
				p.To.Sym = ctxt.LookupDerived(curtext, curtext.Name+".args_stackmap")
			}

		}

		if curtext == nil {
			etext = nil
			continue
		}
		etext.Link = p
		etext = p
	}

	if newprog == nil {
		newprog = ctxt.NewProg
	}

	// Add reference to Go arguments for C or assembly functions without them.
	for _, s := range text {
		if !strings.HasPrefix(s.Name, "\"\".") {
			continue
		}
		found := false
		for p := s.Func.Text; p != nil; p = p.Link {
			if p.As == AFUNCDATA && p.From.Type == TYPE_CONST && p.From.Offset == objabi.FUNCDATA_ArgsPointerMaps {
				found = true
				break
			}
		}

		if !found {
			p := Appendp(s.Func.Text, newprog)
			p.As = AFUNCDATA
			p.From.Type = TYPE_CONST
			p.From.Offset = objabi.FUNCDATA_ArgsPointerMaps
			p.To.Type = TYPE_MEM
			p.To.Name = NAME_EXTERN
			p.To.Sym = ctxt.LookupDerived(s, s.Name+".args_stackmap")
		}
	}

	// Turn functions into machine code images.
	for _, s := range text {
		mkfwd(s)
		linkpatch(ctxt, s, newprog)
		ctxt.Arch.Preprocess(ctxt, s, newprog)
		ctxt.Arch.Assemble(ctxt, s, newprog)
		linkpcln(ctxt, s)
		ctxt.populateDWARF(plist.Curfn, s)
	}
}

func (ctxt *Link) InitTextSym(s *LSym, flag int) {
	if s == nil {
		// func _() { }
		return
	}
	if s.Func != nil {
		ctxt.Diag("InitTextSym double init for %s", s.Name)
	}
	s.Func = new(FuncInfo)
	if s.Func.Text != nil {
		ctxt.Diag("duplicate TEXT for %s", s.Name)
	}
	if s.OnList() {
		ctxt.Diag("symbol %s listed multiple times", s.Name)
	}
	s.Set(AttrOnList, true)
	s.Set(AttrDuplicateOK, flag&DUPOK != 0)
	s.Set(AttrNoSplit, flag&NOSPLIT != 0)
	s.Set(AttrReflectMethod, flag&REFLECTMETHOD != 0)
	s.Set(AttrWrapper, flag&WRAPPER != 0)
	s.Set(AttrNeedCtxt, flag&NEEDCTXT != 0)
	s.Set(AttrNoFrame, flag&NOFRAME != 0)
	s.Type = objabi.STEXT
	ctxt.Text = append(ctxt.Text, s)

	// Set up DWARF entries for s.
	dsym, drsym := ctxt.dwarfSym(s)
	dsym.Type = objabi.SDWARFINFO
	dsym.Set(AttrDuplicateOK, s.DuplicateOK())
	drsym.Type = objabi.SDWARFRANGE
	drsym.Set(AttrDuplicateOK, s.DuplicateOK())
	ctxt.Data = append(ctxt.Data, dsym)
	ctxt.Data = append(ctxt.Data, drsym)

	// Set up the function's gcargs and gclocals.
	// They will be filled in later if needed.
	gcargs := &s.Func.GCArgs
	gcargs.Set(AttrDuplicateOK, true)
	gcargs.Type = objabi.SRODATA
	gclocals := &s.Func.GCLocals
	gclocals.Set(AttrDuplicateOK, true)
	gclocals.Type = objabi.SRODATA
}

func (ctxt *Link) Globl(s *LSym, size int64, flag int) {
	if s.SeenGlobl() {
		fmt.Printf("duplicate %v\n", s)
	}
	s.Set(AttrSeenGlobl, true)
	if s.OnList() {
		ctxt.Diag("symbol %s listed multiple times", s.Name)
	}
	s.Set(AttrOnList, true)
	ctxt.Data = append(ctxt.Data, s)
	s.Size = size
	if s.Type == 0 {
		s.Type = objabi.SBSS
	}
	if flag&DUPOK != 0 {
		s.Set(AttrDuplicateOK, true)
	}
	if flag&RODATA != 0 {
		s.Type = objabi.SRODATA
	} else if flag&NOPTR != 0 {
		if s.Type == objabi.SDATA {
			s.Type = objabi.SNOPTRDATA
		} else {
			s.Type = objabi.SNOPTRBSS
		}
	} else if flag&TLSBSS != 0 {
		s.Type = objabi.STLSBSS
	}
}
