// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import (
	"bytes"
	"cmd/internal/obj"
	"cmd/internal/objabi"
)

const (
	ContFuncType = 1
	ContType     = 2
	ContTag      = 0
	ContTable    = 1

	BlockContRef = -1
)

type stackSwitch struct {
	ctxt    *obj.Link
	s       *obj.LSym
	frame   *obj.Prog
	restart *obj.Prog
	outer   *obj.Prog
	target  bool
}

func init() {
	for _, as := range []obj.As{ATableGet, ATableSet, ATableGrow, ATableSize, ARefFunc, AResume, ACALLCONT} {
		unaryDst[as] = true
	}
}

func newStackSwitch(ctxt *obj.Link, s *obj.LSym) *stackSwitch {
	if !ctxt.WasmStackSwitch {
		return nil
	}
	ss := &stackSwitch{ctxt: ctxt, s: s, frame: s.Func().Text}
	if p := s.Func().Text; p.Link != nil {
		for q, i := p.Link, 0; q != nil && i < 4; q, i = q.Link, i+1 {
			if q.Spadj != 0 {
				ss.frame = q
				break
			}
		}
	}
	ss.keepSwitchTargets()
	return ss
}

func switchesBack(sym *obj.LSym) bool {
	return sym != nil && sym.Name == "runtime.mcall"
}

func (ss *stackSwitch) keepSwitchTargets() {
	for p := ss.s.Func().Text; p != nil; p = p.Link {
		if p.As == ACALLCONT && p.To.Type == obj.TYPE_MEM && switchesBack(p.To.Sym) {
			p.As = obj.ACALL
		}
	}
}

func (ss *stackSwitch) restartLoop(p *obj.Prog, appendp appendFunc) *obj.Prog {
	if ss == nil {
		return p
	}
	p = appendp(p, ALoop)
	ss.restart = p
	return p
}

func (ss *stackSwitch) restartEnd(p *obj.Prog, appendp appendFunc) *obj.Prog {
	if ss == nil || ss.restart == nil {
		return p
	}
	return appendp(p, AEnd)
}

func (ss *stackSwitch) isRestart(call *obj.Prog) bool {
	return ss != nil && ss.restart != nil && (call.To.Sym == morestack || call.To.Sym == morestackNoCtxt)
}

func (ss *stackSwitch) restartBranch(p *obj.Prog, appendp appendFunc) *obj.Prog {
	p = appendp(p, AIf)
	p = appendp(p, ABr)
	p.To = obj.Addr{Type: obj.TYPE_BRANCH, Val: ss.restart}
	return appendp(p, AEnd)
}

func (ss *stackSwitch) depth(p *obj.Prog, appendp appendFunc) *obj.Prog {
	p = appendp(p, AGet, regAddr(REGG))
	p = appendp(p, AI32WrapI64)
	p = appendp(p, AI32Load, constAddr(int64(ss.ctxt.Arch.PtrSize)))
	p = appendp(p, AGet, regAddr(REG_SP))
	return appendp(p, AI32Sub)
}

func (ss *stackSwitch) planTarget(numResumePoints, numUnwindExits int, appendp appendFunc) {
	if ss == nil || ss.ctxt.IsAsm || numResumePoints == 0 || numUnwindExits == 0 {
		return
	}
	if ss.s.Func().WasmImport != nil || ss.s.Func().WasmExport != nil {
		return
	}
	ss.target = true
	last := appendp(ss.depth(ss.frame, appendp), ASet, regAddr(REG_MYSP))
	if ss.frame.Spadj != 0 {
		for q := ss.frame.Link; ; q = q.Link {
			q.Pc = ss.frame.Pc + 1
			if q == last {
				break
			}
		}
	}
}

func (ss *stackSwitch) outerLoop(p *obj.Prog, appendp appendFunc) *obj.Prog {
	if ss == nil || !ss.target {
		return p
	}
	p = appendp(p, ALoop)
	ss.outer = p
	return p
}

func (ss *stackSwitch) targetCheck(p *obj.Prog, appendp appendFunc) *obj.Prog {
	if ss == nil || !ss.target {
		return p
	}
	p = appendp(p, AGet, regAddr(REG_PAUSE))
	p = appendp(p, AI32Eqz)
	p = ss.depth(p, appendp)
	p = appendp(p, AGet, regAddr(REG_MYSP))
	p = appendp(p, AI32Eq)
	p = appendp(p, AI32And)
	p = appendp(p, AIf)
	p = appendp(p, AGet, regAddr(REG_SP))
	p = appendp(p, AI32Const, constAddr(8))
	p = appendp(p, AI32Sub)
	p = appendp(p, AI64Load, constAddr(0))
	p = appendp(p, AI64Const, constAddr(16))
	p = appendp(p, AI64ShrU)
	p = appendp(p, AI64Const, obj.Addr{Type: obj.TYPE_ADDR, Name: obj.NAME_EXTERN, Sym: ss.s})
	p = appendp(p, AI64Const, constAddr(16))
	p = appendp(p, AI64ShrU)
	p = appendp(p, AI64Eq)
	p = appendp(p, AIf)
	p = appendp(p, AGet, regAddr(REG_SP))
	p = appendp(p, AI32Const, constAddr(8))
	p = appendp(p, AI32Sub)
	p = appendp(p, AI32Load16U, constAddr(0))
	p = appendp(p, ASet, regAddr(REG_PC_B))
	p = appendp(p, AGet, regAddr(REG_PC_B))
	p = appendp(p, ABrIf)
	p.To = obj.Addr{Type: obj.TYPE_BRANCH, Val: ss.outer}
	p = appendp(p, AEnd)
	p = appendp(p, AEnd)
	return appendp(p, AEnd)
}

type appendFunc = func(p *obj.Prog, as obj.As, args ...obj.Addr) *obj.Prog

func encodeStackSwitch(ctxt *obj.Link, s *obj.LSym, w *bytes.Buffer, p *obj.Prog) bool {
	switch p.As {
	case ABlock:
		if p.From.Type != obj.TYPE_CONST || p.From.Offset != BlockContRef {
			return false
		}
		writeOpcode(w, ABlock)
		w.WriteByte(0x64)
		writeUleb128(w, ContType)
	case ATableGet, ATableSet, ATableGrow, ATableSize:
		writeOpcode(w, p.As)
		writeUleb128(w, uint64(p.To.Offset))
	case ARefNullCont:
		w.WriteByte(0xd0)
		writeUleb128(w, ContType)
	case ARefIsNull:
		w.WriteByte(0xd1)
	case ARefFunc:
		if p.To.Type != obj.TYPE_MEM || p.To.Sym == nil {
			ctxt.Diag("RefFunc needs a function symbol: %v", p)
			return true
		}
		w.WriteByte(0xd2)
		s.AddRel(ctxt, obj.Reloc{Type: objabi.R_CALL, Off: int32(w.Len()), Siz: 1, Sym: p.To.Sym})
	case AContNew:
		w.WriteByte(0xe0)
		writeUleb128(w, ContType)
	case ASuspend:
		w.WriteByte(0xe2)
		writeUleb128(w, ContTag)
	case AResume:
		w.WriteByte(0xe3)
		writeUleb128(w, ContType)
		writeUleb128(w, 1)
		w.WriteByte(0x00)
		writeUleb128(w, ContTag)
		writeUleb128(w, uint64(p.To.Offset))
	default:
		return false
	}
	return true
}
