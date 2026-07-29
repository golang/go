// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/objw"
	"cmd/internal/obj"
	"cmd/internal/obj/arm64"
	"cmd/internal/src"
)

// movdImmMemInfo describes a plain "MOVD reg, off(base)" or "MOVD off(base), reg"
// instruction. base, off, name, and sym record the memory operand's base
// register, offset, addressing class (NAME_AUTO or NAME_PARAM), and symbol;
// reg is the register stored or loaded, and isLoad distinguishes the two
// forms.
type movdImmMemInfo struct {
	base, reg int16
	off       int64
	name      obj.AddrName
	sym       *obj.LSym
	isLoad    bool
}

// movdImmMem decodes p into a movdImmMemInfo. ok reports whether p is a
// plain MOVD load or store using an immediate-offset addressing form
// suitable for LDP/STP coalescing.
func movdImmMem(p *obj.Prog) (info movdImmMemInfo, ok bool) {
	if p.As != arm64.AMOVD || p.Scond != 0 || p.Reg != 0 {
		return movdImmMemInfo{}, false
	}
	var a *obj.Addr
	switch {
	case p.From.Type == obj.TYPE_MEM && p.To.Type == obj.TYPE_REG:
		// Plain load: MOVD mem, reg.
		a = &p.From
		info.reg = p.To.Reg
		info.isLoad = true
	case p.From.Type == obj.TYPE_REG && p.To.Type == obj.TYPE_MEM:
		// Plain store: MOVD reg, mem.
		a = &p.To
		info.reg = p.From.Reg
	default:
		return movdImmMemInfo{}, false
	}
	if a.Index != 0 || (a.Name != obj.NAME_AUTO && a.Name != obj.NAME_PARAM) {
		return movdImmMemInfo{}, false
	}
	info.base, info.off, info.name, info.sym = a.Reg, a.Offset, a.Name, a.Sym
	return info, true
}

// ssaGenFinish runs after genssa has emitted all of a function's Progs,
// resolved its branch and jump-table targets, and finalized the frame size
// in defframe.
func ssaGenFinish(pp *objw.Progs) {
	if base.Flag.N != 0 {
		// Keep unoptimized builds unoptimized.
		return
	}
	pairSpills(pp.Text, pp.Text.To.Offset, pp.CurFunc.LSym.Func().JumpTables)
}

// pairSpills fuses strictly-adjacent MOVD load or store pairs that target the
// same base register at consecutive 8-byte offsets into a single LDP or STP.
// text is the function's first Prog, framesize its final frame size, and
// jumpTables its resolved jump tables.
//
// This recovers spill/reload pairings that the SSA-level pair pass
// (cmd/compile/internal/ssa/pair.go) cannot perform: that pass runs before
// regalloc and only sees source-level loads/stores, so the STR/LDR pairs that
// regalloc later inserts for OpStoreReg/OpLoadReg never get a chance to be
// paired. Doing the fusion here, as the last step of code generation, keeps the
// optimization in the compiler so the assembler can stay a simple translator.
func pairSpills(text *obj.Prog, framesize int64, jumpTables []obj.JumpTable) {
	// Branch and jump-table targets, collected lazily once a candidate pair
	// survives the cheaper checks; most functions have no candidates.
	// Fusing into the first of a pair whose second instruction is a target
	// would silently change control flow: paths that jump directly to the
	// second instruction would skip the work the LDP/STP now does at the
	// first instruction's position.
	var isTarget map[*obj.Prog]bool
	targets := func() map[*obj.Prog]bool {
		if isTarget == nil {
			isTarget = map[*obj.Prog]bool{}
			for p := text; p != nil; p = p.Link {
				if t := p.To.Target(); t != nil {
					isTarget[t] = true
				}
			}
			for _, jt := range jumpTables {
				for _, t := range jt.Targets {
					isTarget[t] = true
				}
			}
		}
		return isTarget
	}
	for p := text; p != nil; p = p.Link {
		q := p.Link
		if q == nil {
			continue
		}
		a, ok := movdImmMem(p)
		if !ok {
			continue
		}
		b, ok := movdImmMem(q)
		if !ok || a.isLoad != b.isLoad {
			continue
		}
		if a.base != b.base || a.name != b.name {
			continue
		}
		if a.reg == b.reg {
			// LDP with Rt1 == Rt2 is CONSTRAINED UNPREDICTABLE. (STP with
			// identical source registers would be fine, but spills never
			// store the same register twice, so don't bother.)
			continue
		}
		if a.isLoad && a.reg == a.base {
			// The first load overwrites the base register: executed
			// sequentially, the second load computes its address from the
			// just-loaded value, while LDP computes both addresses from
			// the original base. (The second load overwriting the base is
			// fine; no address depends on it.)
			continue
		}
		if q.Pos.IsStmt() == src.PosIsStmt {
			// q carries a statement boundary, and may also be registered
			// as an inline mark: genssa promotes instructions it reuses
			// as inline marks to statements. Reducing q to a 0-byte ANOP
			// would drop the statement from the line tables and violate
			// the invariant that inline marks are never zero-sized (they
			// are identified by PC).
			continue
		}
		lo, hi := a, b
		if lo.off > hi.off {
			lo, hi = hi, lo
		}
		if hi.off != lo.off+8 {
			continue
		}
		// Fuse only when the result will encode as a single LDP/STP, whose
		// signed 7-bit immediate scaled by 8 covers offsets [-512, 504].
		// The assembler resolves a NAME_AUTO offset to off+framesize+8
		// (the 8 is the saved LR) and a NAME_PARAM offset to
		// off+framesize+24 or +32 depending on frame alignment padding —
		// or off+8 in a frameless leaf, which is not decided until
		// assembly, so a frameless PARAM must fit both. An out-of-range
		// LDP/STP needs an assembler-synthesized address (ADD + LDP),
		// which is no smaller than the original pair and serializes the
		// accesses through REGTMP.
		var biasLo, biasHi int64
		switch {
		case lo.name == obj.NAME_AUTO:
			// Autos exist only in functions with a frame, and their offsets
			// are negative from FP, so framesize+8 rebases them onto the
			// non-negative SP-relative range the LDP/STP immediate must reach.
			biasLo = framesize + 8
			biasHi = biasLo
		case framesize == 0:
			biasLo, biasHi = 8, 24
		case framesize%16 == 0:
			biasLo = framesize + 24
			biasHi = biasLo
		default:
			biasLo = framesize + 32
			biasHi = biasLo
		}
		if lo.off%8 != 0 || lo.off+biasLo < -512 || lo.off+biasHi > 504 {
			continue
		}
		if targets()[q] {
			continue
		}
		mem := obj.Addr{Type: obj.TYPE_MEM, Reg: lo.base, Offset: lo.off, Name: lo.name, Sym: lo.sym}
		regs := obj.Addr{Type: obj.TYPE_REGREG, Reg: lo.reg, Offset: int64(hi.reg)}
		if a.isLoad {
			p.As = arm64.ALDP
			p.From, p.To = mem, regs
		} else {
			p.As = arm64.ASTP
			p.From, p.To = regs, mem
		}
		// Reduce q to a 0-byte ANOP rather than unlinking it so that
		// branch targets referencing q remain valid.
		obj.Nopout(q)
	}
}
