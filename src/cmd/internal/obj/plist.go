// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package obj

import (
	"cmd/internal/objabi"
	"cmd/internal/src"
	"fmt"
	"internal/abi"
	"strings"
)

type Plist struct {
	Firstpc *Prog
	Curfn   Func
}

// ProgAlloc is a function that allocates Progs.
// It is used to provide access to cached/bulk-allocated Progs to the assemblers.
type ProgAlloc func() *Prog

func Flushplist(ctxt *Link, plist *Plist, newprog ProgAlloc) {
	if ctxt.Pkgpath == "" {
		panic("Flushplist called without Pkgpath")
	}

	// Build list of symbols, and assign instructions to lists.
	var curtext *LSym
	var etext *Prog
	var text []*LSym

	var plink *Prog
	for p := plist.Firstpc; p != nil; p = plink {
		if ctxt.Debugasm > 0 && ctxt.Debugvlog {
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
			switch p.To.Sym.Name {
			case "go_args_stackmap":
				if p.From.Type != TYPE_CONST || p.From.Offset != abi.FUNCDATA_ArgsPointerMaps {
					ctxt.Diag("%s: FUNCDATA use of go_args_stackmap(SB) without FUNCDATA_ArgsPointerMaps", p.Pos)
				}
				p.To.Sym = ctxt.LookupDerived(curtext, curtext.Name+".args_stackmap")
			case "no_pointers_stackmap":
				if p.From.Type != TYPE_CONST || p.From.Offset != abi.FUNCDATA_LocalsPointerMaps {
					ctxt.Diag("%s: FUNCDATA use of no_pointers_stackmap(SB) without FUNCDATA_LocalsPointerMaps", p.Pos)
				}
				// funcdata for functions with no local variables in frame.
				// Define two zero-length bitmaps, because the same index is used
				// for the local variables as for the argument frame, and assembly
				// frames have two argument bitmaps, one without results and one with results.
				// Write []uint32{2, 0}.
				b := make([]byte, 8)
				ctxt.Arch.ByteOrder.PutUint32(b, 2)
				s := ctxt.GCLocalsSym(b)
				if !s.OnList() {
					ctxt.Globl(s, int64(len(s.P)), int(RODATA|DUPOK))
				}
				p.To.Sym = s
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

	// Add reference to Go arguments for assembly functions without them.
	if ctxt.IsAsm {
		pkgPrefix := objabi.PathToPrefix(ctxt.Pkgpath) + "."
		for _, s := range text {
			if !strings.HasPrefix(s.Name, pkgPrefix) {
				continue
			}
			// The current args_stackmap generation in the compiler assumes
			// that the function in question is ABI0, so avoid introducing
			// an args_stackmap reference if the func is not ABI0 (better to
			// have no stackmap than an incorrect/lying stackmap).
			if s.ABI() != ABI0 {
				continue
			}
			// runtime.addmoduledata is a host ABI function, so it doesn't
			// need FUNCDATA anyway. Moreover, cmd/link has special logic
			// for linking it in eccentric build modes, which breaks if it
			// has FUNCDATA references (e.g., cmd/cgo/internal/testplugin).
			//
			// TODO(cherryyz): Fix cmd/link's handling of plugins (see
			// discussion on CL 523355).
			if s.Name == "runtime.addmoduledata" {
				continue
			}
			foundArgMap, foundArgInfo := false, false
			for p := s.Func().Text; p != nil; p = p.Link {
				if p.As == AFUNCDATA && p.From.Type == TYPE_CONST {
					if p.From.Offset == abi.FUNCDATA_ArgsPointerMaps {
						foundArgMap = true
					}
					if p.From.Offset == abi.FUNCDATA_ArgInfo {
						foundArgInfo = true
					}
					if foundArgMap && foundArgInfo {
						break
					}
				}
			}
			if !foundArgMap {
				p := Appendp(s.Func().Text, newprog)
				p.As = AFUNCDATA
				p.From.Type = TYPE_CONST
				p.From.Offset = abi.FUNCDATA_ArgsPointerMaps
				p.To.Type = TYPE_MEM
				p.To.Name = NAME_EXTERN
				p.To.Sym = ctxt.LookupDerived(s, s.Name+".args_stackmap")
			}
			if !foundArgInfo {
				p := Appendp(s.Func().Text, newprog)
				p.As = AFUNCDATA
				p.From.Type = TYPE_CONST
				p.From.Offset = abi.FUNCDATA_ArgInfo
				p.To.Type = TYPE_MEM
				p.To.Name = NAME_EXTERN
				p.To.Sym = ctxt.LookupDerived(s, fmt.Sprintf("%s.arginfo%d", s.Name, s.ABI()))
			}
		}
	}

	// Turn functions into machine code images.
	for _, s := range text {
		mkfwd(s)
		if ctxt.Arch.ErrorCheck != nil {
			ctxt.Arch.ErrorCheck(ctxt, s)
		}
		linkpatch(ctxt, s, newprog)
		ctxt.Arch.Preprocess(ctxt, s, newprog)
		ctxt.Arch.Assemble(ctxt, s, newprog)
		if ctxt.Errors > 0 {
			continue
		}
		linkpcln(ctxt, s)
		ctxt.populateDWARF(plist.Curfn, s)
		if ctxt.Headtype == objabi.Hwindows && ctxt.Arch.SEH != nil {
			s.Func().sehUnwindInfoSym = ctxt.Arch.SEH(ctxt, s)
		}
	}
}

func (ctxt *Link) InitTextSym(s *LSym, flag int, start src.XPos) {
	if s == nil {
		// func _() { }
		return
	}
	if s.Func() != nil {
		ctxt.Diag("%s: symbol %s redeclared\n\t%s: other declaration of symbol %s", ctxt.PosTable.Pos(start), s.Name, ctxt.PosTable.Pos(s.Func().Text.Pos), s.Name)
		return
	}
	s.NewFuncInfo()
	if s.OnList() {
		ctxt.Diag("%s: symbol %s redeclared", ctxt.PosTable.Pos(start), s.Name)
		return
	}
	if strings.HasPrefix(s.Name, `"".`) {
		ctxt.Diag("%s: unqualified symbol name: %s", ctxt.PosTable.Pos(start), s.Name)
	}

	// startLine should be the same line number that would be displayed via
	// pcln, etc for the declaration (i.e., relative line number, as
	// adjusted by //line).
	_, startLine := ctxt.getFileIndexAndLine(start)

	s.Func().FuncID = objabi.GetFuncID(s.Name, flag&WRAPPER != 0 || flag&ABIWRAPPER != 0)
	s.Func().FuncFlag = ctxt.toFuncFlag(flag)
	s.Func().StartLine = startLine
	s.Set(AttrOnList, true)
	s.Set(AttrDuplicateOK, flag&DUPOK != 0)
	s.Set(AttrNoSplit, flag&NOSPLIT != 0)
	s.Set(AttrReflectMethod, flag&REFLECTMETHOD != 0)
	s.Set(AttrWrapper, flag&WRAPPER != 0)
	s.Set(AttrABIWrapper, flag&ABIWRAPPER != 0)
	s.Set(AttrNeedCtxt, flag&NEEDCTXT != 0)
	s.Set(AttrNoFrame, flag&NOFRAME != 0)
	s.Set(AttrPkgInit, flag&PKGINIT != 0)
	s.Type = objabi.STEXT
	ctxt.Text = append(ctxt.Text, s)

	// Set up DWARF entries for s
	ctxt.dwarfSym(s)
}

func (ctxt *Link) toFuncFlag(flag int) abi.FuncFlag {
	var out abi.FuncFlag
	if flag&TOPFRAME != 0 {
		out |= abi.FuncFlagTopFrame
	}
	if ctxt.IsAsm {
		out |= abi.FuncFlagAsm
	}
	return out
}

func (ctxt *Link) Globl(s *LSym, size int64, flag int) {
	ctxt.GloblPos(s, size, flag, src.NoXPos)
}
func (ctxt *Link) GloblPos(s *LSym, size int64, flag int, pos src.XPos) {
	if s.OnList() {
		// TODO: print where the first declaration was.
		ctxt.Diag("%s: symbol %s redeclared", ctxt.PosTable.Pos(pos), s.Name)
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

// EmitEntryLiveness generates PCDATA Progs after p to switch to the
// liveness map active at the entry of function s. It returns the last
// Prog generated.
func (ctxt *Link) EmitEntryLiveness(s *LSym, p *Prog, newprog ProgAlloc) *Prog {
	pcdata := ctxt.EmitEntryStackMap(s, p, newprog)
	pcdata = ctxt.EmitEntryUnsafePoint(s, pcdata, newprog)
	return pcdata
}

// Similar to EmitEntryLiveness, but just emit stack map.
func (ctxt *Link) EmitEntryStackMap(s *LSym, p *Prog, newprog ProgAlloc) *Prog {
	pcdata := Appendp(p, newprog)
	pcdata.Pos = s.Func().Text.Pos
	pcdata.As = APCDATA
	pcdata.From.Type = TYPE_CONST
	pcdata.From.Offset = abi.PCDATA_StackMapIndex
	pcdata.To.Type = TYPE_CONST
	pcdata.To.Offset = -1 // pcdata starts at -1 at function entry

	return pcdata
}

// Similar to EmitEntryLiveness, but just emit unsafe point map.
func (ctxt *Link) EmitEntryUnsafePoint(s *LSym, p *Prog, newprog ProgAlloc) *Prog {
	pcdata := Appendp(p, newprog)
	pcdata.Pos = s.Func().Text.Pos
	pcdata.As = APCDATA
	pcdata.From.Type = TYPE_CONST
	pcdata.From.Offset = abi.PCDATA_UnsafePoint
	pcdata.To.Type = TYPE_CONST
	pcdata.To.Offset = -1

	return pcdata
}

// StartUnsafePoint generates PCDATA Progs after p to mark the
// beginning of an unsafe point. The unsafe point starts immediately
// after p.
// It returns the last Prog generated.
func (ctxt *Link) StartUnsafePoint(p *Prog, newprog ProgAlloc) *Prog {
	pcdata := Appendp(p, newprog)
	pcdata.As = APCDATA
	pcdata.From.Type = TYPE_CONST
	pcdata.From.Offset = abi.PCDATA_UnsafePoint
	pcdata.To.Type = TYPE_CONST
	pcdata.To.Offset = abi.UnsafePointUnsafe

	return pcdata
}

// EndUnsafePoint generates PCDATA Progs after p to mark the end of an
// unsafe point, restoring the register map index to oldval.
// The unsafe point ends right after p.
// It returns the last Prog generated.
func (ctxt *Link) EndUnsafePoint(p *Prog, newprog ProgAlloc, oldval int64) *Prog {
	pcdata := Appendp(p, newprog)
	pcdata.As = APCDATA
	pcdata.From.Type = TYPE_CONST
	pcdata.From.Offset = abi.PCDATA_UnsafePoint
	pcdata.To.Type = TYPE_CONST
	pcdata.To.Offset = oldval

	return pcdata
}

// MarkUnsafePoints inserts PCDATAs to mark nonpreemptible and restartable
// instruction sequences, based on isUnsafePoint and isRestartable predicate.
// p0 is the start of the instruction stream.
// isUnsafePoint(p) returns true if p is not safe for async preemption.
// isRestartable(p) returns true if we can restart at the start of p (this Prog)
// upon async preemption. (Currently multi-Prog restartable sequence is not
// supported.)
// isRestartable can be nil. In this case it is treated as always returning false.
// If isUnsafePoint(p) and isRestartable(p) are both true, it is treated as
// an unsafe point.
func MarkUnsafePoints(ctxt *Link, p0 *Prog, newprog ProgAlloc, isUnsafePoint, isRestartable func(*Prog) bool) {
	if isRestartable == nil {
		// Default implementation: nothing is restartable.
		isRestartable = func(*Prog) bool { return false }
	}
	prev := p0
	prevPcdata := int64(-1) // entry PC data value
	prevRestart := int64(0)
	for p := prev.Link; p != nil; p, prev = p.Link, p {
		if p.As == APCDATA && p.From.Offset == abi.PCDATA_UnsafePoint {
			prevPcdata = p.To.Offset
			continue
		}
		if prevPcdata == abi.UnsafePointUnsafe {
			continue // already unsafe
		}
		if isUnsafePoint(p) {
			q := ctxt.StartUnsafePoint(prev, newprog)
			q.Pc = p.Pc
			q.Link = p
			// Advance to the end of unsafe point.
			for p.Link != nil && isUnsafePoint(p.Link) {
				p = p.Link
			}
			if p.Link == nil {
				break // Reached the end, don't bother marking the end
			}
			p = ctxt.EndUnsafePoint(p, newprog, prevPcdata)
			p.Pc = p.Link.Pc
			continue
		}
		if isRestartable(p) {
			val := int64(abi.UnsafePointRestart1)
			if val == prevRestart {
				val = abi.UnsafePointRestart2
			}
			prevRestart = val
			q := Appendp(prev, newprog)
			q.As = APCDATA
			q.From.Type = TYPE_CONST
			q.From.Offset = abi.PCDATA_UnsafePoint
			q.To.Type = TYPE_CONST
			q.To.Offset = val
			q.Pc = p.Pc
			q.Link = p

			if p.Link == nil {
				break // Reached the end, don't bother marking the end
			}
			if isRestartable(p.Link) {
				// Next Prog is also restartable. No need to mark the end
				// of this sequence. We'll just go ahead mark the next one.
				continue
			}
			p = Appendp(p, newprog)
			p.As = APCDATA
			p.From.Type = TYPE_CONST
			p.From.Offset = abi.PCDATA_UnsafePoint
			p.To.Type = TYPE_CONST
			p.To.Offset = prevPcdata
			p.Pc = p.Link.Pc
		}
	}
}
