// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
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
				if p.From.Type != TYPE_CONST || p.From.Offset != FUNCDATA_ArgsPointerMaps {
					ctxt.Diag("FUNCDATA use of go_args_stackmap(SB) without FUNCDATA_ArgsPointerMaps")
				}
				p.To.Sym = ctxt.Lookup(fmt.Sprintf("%s.args_stackmap", curtext.Name), int(curtext.Version))
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
		for p := s.Text; p != nil; p = p.Link {
			if p.As == AFUNCDATA && p.From.Type == TYPE_CONST && p.From.Offset == FUNCDATA_ArgsPointerMaps {
				found = true
				break
			}
		}

		if !found {
			p := Appendp(s.Text, newprog)
			p.As = AFUNCDATA
			p.From.Type = TYPE_CONST
			p.From.Offset = FUNCDATA_ArgsPointerMaps
			p.To.Type = TYPE_MEM
			p.To.Name = NAME_EXTERN
			p.To.Sym = ctxt.Lookup(fmt.Sprintf("%s.args_stackmap", s.Name), int(s.Version))
		}
	}

	// Turn functions into machine code images.
	for _, s := range text {
		mkfwd(s)
		linkpatch(ctxt, s, newprog)
		ctxt.Arch.Preprocess(ctxt, s, newprog)
		ctxt.Arch.Assemble(ctxt, s, newprog)
		linkpcln(ctxt, s)
		makeFuncDebugEntry(ctxt, plist.Curfn, s)
	}

	// Add to running list in ctxt.
	ctxt.Text = append(ctxt.Text, text...)
}

func (ctxt *Link) InitTextSym(p *Prog) {
	if p.As != ATEXT {
		ctxt.Diag("InitTextSym non-ATEXT: %v", p)
	}
	s := p.From.Sym
	if s == nil {
		// func _() { }
		return
	}
	if s.FuncInfo != nil {
		ctxt.Diag("InitTextSym double init for %s", s.Name)
	}
	s.FuncInfo = new(FuncInfo)
	if s.Text != nil {
		ctxt.Diag("duplicate TEXT for %s", s.Name)
	}
	if s.OnList() {
		ctxt.Diag("symbol %s listed multiple times", s.Name)
	}
	s.Set(AttrOnList, true)
	flag := int(p.From3Offset())
	if flag&DUPOK != 0 {
		s.Set(AttrDuplicateOK, true)
	}
	if flag&NOSPLIT != 0 {
		s.Set(AttrNoSplit, true)
	}
	if flag&REFLECTMETHOD != 0 {
		s.Set(AttrReflectMethod, true)
	}
	s.Type = STEXT
	s.Text = p
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
	if s.Type == 0 || s.Type == SXREF {
		s.Type = SBSS
	}
	if flag&DUPOK != 0 {
		s.Set(AttrDuplicateOK, true)
	}
	if flag&RODATA != 0 {
		s.Type = SRODATA
	} else if flag&NOPTR != 0 {
		s.Type = SNOPTRBSS
	} else if flag&TLSBSS != 0 {
		s.Type = STLSBSS
	}
}
